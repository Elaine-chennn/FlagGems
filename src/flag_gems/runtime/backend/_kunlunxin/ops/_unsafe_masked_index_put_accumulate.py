# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _dests_per_program(out_numel: int) -> int:
    """每个 program 负责多少个输出元素。

    让一个 program 复用一次源数组读入服务 DESTS 个目标，派发数与访存量同降 DESTS 倍。
    上限 32 是为了压住 `tl.static_range` 的展开体积。
    """
    if out_numel >= 512:
        return 32
    return max(1, min(32, triton.next_power_of_2(out_numel) // 16))


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _unsafe_masked_index_put_accumulate_kernel(
    out_ptr,
    inp_ptr,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    DESTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    dest_base = ext.program_id(0) * DESTS

    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < mask_numel
    keep = tl.load(mask + offsets, mask=active, other=0) != 0

    # 先算每个 source 的扁平目的地偏移。越界/负下标先 clamp 再用（该后端在 mask 生效前
    # 就把地址交给 gm2lm，靠 mask 屏蔽非法下标会真读越界并打挂整卡）；clamp 到
    # [0, SHAPE-1] 与 torch 的 `index.clamp(-size, size-1)` 在非负下标上等价。
    i0 = tl.load(index0 + offsets, mask=active, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offsets, mask=active, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offsets, mask=active, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    # mask 掉的源与 tile 尾部空 lane 一律把 update 置 0，下面的匹配不必再带 mask。
    update = tl.load(values + offsets, mask=active, other=0.0).to(tl.float32)
    update = tl.where(keep & active, update, 0.0)

    for c in tl.static_range(DESTS):
        out_off = dest_base + c
        # out 缓冲区尾部留了 DESTS 个哨兵元素，store 不带 mask（离散 store 的 mask
        # 在地址碰撞时不可依赖，宁可写进合法的填充区）。
        acc = tl.sum(tl.where(dest == out_off, update, 0.0), axis=0)
        in_off = tl.minimum(out_off, out_numel - 1)
        base = tl.load(inp_ptr + in_off).to(tl.float32)
        tl.store(out_ptr + out_off, base + acc)


# ---------------------------------------------------------------------------
# 多轮「胜者循环」路径（大规模）
#
# match kernel 是 O(out_numel * mask_numel)，大规模下结构性不可达，必须换成 O(mask_numel)。
# 该后端上 atomic_add（~192 ns/elem，且 mask 全 false 也照收全价）与排序/前缀和（多 kernel
# 且踩 TritonXPU 崩溃点）两条路都被堵死，于是用「无 atomic 的胜者循环」：靠离散 store 的
# 天然单胜者语义每轮从每个目标挑一个源，R 轮覆盖到最大重数。
#
# 核心约束：离散访存的代价只取决于「多少 lane 打在同一地址」（同址碰撞被串行化，
# ~192 ns/lane；随机地址 ~1 ns/elem）。因此本实现绝不让两个 lane 共享一个哨兵槽：
#   * 源 i 退休/被 mask 掉时写它私有的槽 `out_numel + i`，不是公共的 out_numel；
#   * 目标 d「本轮无人获胜」标记为它私有的值 `d`（借 val_lookup 翻译成 0），不是公共的 0。
#
# tag 地址空间（每轮一行，行长 row = out_numel + pad）：
#   [0, out_numel)              目标槽；值 < out_numel 表示「本轮该目标无人获胜」
#   [out_numel, out_numel+pad)  源私有槽；源 i 的 marker = out_numel + i
# tag 只要全 0 初始化即可（不能用 torch.arange(int32)，该后端会 ASSERT-FAIL 返回垃圾）。
# val_lookup 与 tag 同长：[0, out_numel) 恒 0；尾部 = values。combine 把「无人」的下标改写成
# `offs`（互不相同且 val_lookup[offs]==0），一次零碰撞的 gather 同时拿到「有没有胜者 + value」。
# ---------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["mask_numel", "out_numel"])
def _umipa_prep_kernel(
    dest_buf,
    val_lookup,
    mask,
    index0,
    index1,
    index2,
    values,
    mask_numel,
    out_numel,
    SHAPE0: tl.constexpr,
    SHAPE1: tl.constexpr,
    SHAPE2: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """把 (index0..2, mask) 压成扁平 int32 目的地数组，并把 values 搬到 val_lookup 尾部。

    下标先 clamp 再用；被 mask 掉的源写它私有的槽 `out_numel + i`（此后 round kernel 不必再看
    mask）；values 搬到 `val_lookup[out_numel + i]`，使 marker 既是 tag 私有槽下标又是取值下标。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < mask_numel

    i0 = tl.load(index0 + offs, mask=inb, other=0).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), SHAPE0 - 1)
    dest = i0 * STRIDE0
    if RANK >= 2:
        i1 = tl.load(index1 + offs, mask=inb, other=0).to(tl.int32)
        i1 = tl.minimum(tl.maximum(i1, 0), SHAPE1 - 1)
        dest += i1 * STRIDE1
    if RANK >= 3:
        i2 = tl.load(index2 + offs, mask=inb, other=0).to(tl.int32)
        i2 = tl.minimum(tl.maximum(i2, 0), SHAPE2 - 1)
        dest += i2 * STRIDE2

    keep = tl.load(mask + offs, mask=inb, other=0) != 0
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(inb & keep, dest, marker))
    v = tl.load(values + offs, mask=inb, other=0.0)
    tl.store(val_lookup + out_numel + offs, v)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_round_kernel(dest_buf, tag_prev, tag_cur, out_numel, BLOCK: tl.constexpr):
    """一轮胜者循环（融合版）。

    上一轮谁的 marker 留在 tag_prev[dest] 上谁就是胜者：把它从 dest_buf 摘掉（地址改成它的
    私有槽 marker），并往 tag_cur[dest] 写 `dest`（< out_numel）表示「本轮暂无人」；还活着的
    lane 往 tag_cur[dest] 写自己的 marker 申领本轮。离散 store 保证每个目标最多留一个 marker。

    已知次优：退休的胜者也往 tag_cur[dest] 写，会冲掉同目标其它活跃 lane 的申领 ⇒ 每级重数约
    烧两轮。正解是让胜者只写私有槽，但 `tl.store(tag_cur + tl.where(win, marker, d), marker)`
    在本后端稳定触发 721（illegal address）并 wedge 整卡，故大规模改用 retire+claim 拆分臂
    （见 `_ROUND_SPLIT_MIN_MASK_NUMEL`）；小规模轮数够，保留本融合版。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    win = w == marker
    tl.store(dest_buf + offs, tl.where(win, marker, d))
    tl.store(tag_cur + d, tl.where(win, d, marker))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_retire_kernel(dest_buf, tag_prev, out_numel, BLOCK: tl.constexpr):
    """拆分臂前半：只让上一轮胜者退休，只写连续的 dest_buf、不碰 tag。

    避开了融合版那个由 tl.where 算出的 store 地址（打 721）。r=0 时 tag_prev 全 0，等于不退休。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_prev + d)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(dest_buf + offs, tl.where(w == marker, marker, d))


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_claim_kernel(dest_buf, tag_cur, out_numel, BLOCK: tl.constexpr):
    """拆分臂后半：还活着的 lane 申领本轮。

    此时 d 已是退休后的值：已退休的源手里是私有槽，store 落在 tag_cur[out_numel+i] 上，天然不
    触碰任何目标、不冲刷别人的申领。store 地址是 load 出来的 d，是已验证可用的地址形式。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    marker = (out_numel + offs).to(tl.int32)
    tl.store(tag_cur + d, marker)


@libentry()
@triton.jit(do_not_specialize=["out_numel"])
def _umipa_finish_kernel(dest_buf, tag_last, alive, out_numel, BLOCK: tl.constexpr):
    """收掉最后一轮的胜者（下一轮才会摘，故补这一步），并按 program 统计剩余活跃源。

    `d < out_numel` 即「还活着」。
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dest_buf + offs)
    w = tl.load(tag_last + d)
    marker = (out_numel + offs).to(tl.int32)
    d = tl.where(w == marker, marker, d)
    tl.store(dest_buf + offs, d)
    tl.store(alive + pid, tl.sum((d < out_numel).to(tl.int32), axis=0))


@libentry()
@triton.jit(do_not_specialize=["out_numel", "row"])
def _umipa_combine_kernel(
    out_ptr,
    src_ptr,
    tag,
    val_lookup,
    out_numel,
    row,
    ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """目的地域求和：out[d] = src[d] + sum_r val_lookup[tag[r][d]]。

    `tag[r][d] >= out_numel` 表示第 r 轮 d 上有胜者，该值即胜者 marker，val_lookup[marker] 是它的
    value；否则改用 `offs` 当下标（val_lookup[offs]==0）。「无人」对每个 d 用互不相同的下标，避免
    同址碰撞。tag 的 gather 不带 mask：tail lane 读到私有槽区，地址合法且互不相同，结果被 store
    mask 丢掉。
    """
    offs = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < out_numel
    acc = tl.load(src_ptr + offs, mask=inb, other=0.0).to(tl.float32)
    limit = out_numel.to(tl.int32)
    self_idx = offs.to(tl.int32)
    for r in tl.static_range(ROUNDS):
        tid = tl.load(tag + (r + 1) * row + offs)
        tid = tl.where(tid >= limit, tid, self_idx)
        acc += tl.load(val_lookup + tid).to(tl.float32)
    tl.store(out_ptr + offs, acc, mask=inb)


# 一批跑多少轮。融合版胜者会冲刷同目标其它申领，每级重数约烧两轮，8 轮够典型输入
# （Poisson 重数 4~9），轮数不够时收敛循环会自动再跑一批。
_ROUNDS_PER_BATCH = 8
# 多轮四个 kernel 都不带 isCloseVectorization / buffer_size_limit（实测对本瓶颈无影响，
# 瓶颈是同址碰撞）；留成常量只为参数型 A/B 好切臂。
_ROUND_LAUNCH_KW = {}
# match 路径 ~ out_numel*mask_numel*74ps；多轮路径有 ~11 次 launch 的地板，两者在 ~4e6 交叉。
_MULTI_ROUND_MIN_WORK = 4_000_000
# 每轮拆成 retire+claim 的规模门槛（mask_numel >= 本值才拆）。拆开后轮数 = 最大重数、离散访存
# 减半，但每批多付 rounds 次 launch，交叉点 mask_numel ≈ 14000 ⇒ 取 16384。benchmark 四个 shape
# 里只有 (2,1024,64)(mask=131072) 触发拆分；(4096,) 保持融合。正确性不依赖本值，只影响性能。
_ROUND_SPLIT_MIN_MASK_NUMEL = 16384


def _unsafe_masked_index_put_accumulate_multi_round(
    inp, mask_c, idx_c, values_c, rank, shape, strides
):
    out_numel = inp.numel()
    mask_numel = mask_c.numel()
    rounds = _ROUNDS_PER_BATCH
    round_split = mask_numel >= _ROUND_SPLIT_MIN_MASK_NUMEL

    block = max(64, min(2048, triton.next_power_of_2(mask_numel)))
    grid_m = (triton.cdiv(mask_numel, block),)
    m_pad = grid_m[0] * block
    block_n = max(64, min(2048, triton.next_power_of_2(out_numel)))
    grid_n = (triton.cdiv(out_numel, block_n),)

    # 行长 row = out_numel 个目标槽 + 每个源一个私有槽；再保证 combine 那次不带 mask 的
    # tag gather（offs 最大到 grid_n*block_n-1）在界内。
    pad = max(m_pad, grid_n[0] * block_n - out_numel)
    row = out_numel + pad
    dev = inp.device

    # 每行全 0 即可（0 < out_numel 恒表示「无人」，且永不等于任何 marker）。
    # 不能用 torch.arange(int32)：该后端会 ASSERT-FAIL 并返回垃圾。
    tag = torch.zeros((rounds + 1) * row, dtype=torch.int32, device=dev)

    dest_buf = torch.empty(m_pad, dtype=torch.int32, device=dev)
    # 前 out_numel 个恒为 0（该目标本轮无贡献），尾部由 prep 填入 values。
    val_lookup = torch.zeros(row, dtype=inp.dtype, device=dev)
    alive = torch.empty(grid_m[0], dtype=torch.int32, device=dev)
    out = torch.empty_like(inp)

    with torch_device_fn.device(dev):
        _umipa_prep_kernel[grid_m](
            dest_buf,
            val_lookup,
            mask_c,
            idx_c[0],
            idx_c[1],
            idx_c[2],
            values_c,
            mask_numel,
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            BLOCK=block,
            **_ROUND_LAUNCH_KW,
        )
        src = inp
        for batch in range(64):
            if batch:
                tag.zero_()
            for r in range(rounds):
                if round_split:
                    _umipa_retire_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                    _umipa_claim_kernel[grid_m](
                        dest_buf,
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
                else:
                    _umipa_round_kernel[grid_m](
                        dest_buf,
                        tag[r * row :],
                        tag[(r + 1) * row :],
                        out_numel,
                        BLOCK=block,
                        **_ROUND_LAUNCH_KW,
                    )
            _umipa_finish_kernel[grid_m](
                dest_buf,
                tag[rounds * row :],
                alive,
                out_numel,
                BLOCK=block,
                **_ROUND_LAUNCH_KW,
            )
            _umipa_combine_kernel[grid_n](
                out,
                src,
                tag,
                val_lookup,
                out_numel,
                row,
                ROUNDS=rounds,
                BLOCK=block_n,
                **_ROUND_LAUNCH_KW,
            )
            src = out
            # 唯一的 device->host 同步：一批 rounds 结束后读一次 alive。归约必须在 host 侧做：
            # 本后端 gems 设备端 sum（use_gems 下 alive.sum() 会派发到它）在小张量上会非法访问
            # （error 700）并 wedge 整卡；.cpu() 只是 D2H 拷贝、随后在 CPU 上求和，规避该缺陷。
            if int(alive.cpu().sum()) == 0:
                break
        else:
            raise RuntimeError(
                "Kunlunxin _unsafe_masked_index_put_accumulate did not converge"
            )
    return out


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("GEMS_KUNLUNXIN _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    rank = input.ndim
    if rank < 1 or rank > 3 or len(indices) != rank:
        raise RuntimeError(
            "Kunlunxin _unsafe_masked_index_put_accumulate supports ranks 1 to 3"
        )
    # 该 aten 算子是函数式的（self 不可变，参考实现是 clone 后 index_put_），必须返回新张量。
    if input.numel() == 0 or mask.numel() == 0:
        return input.clone()

    inp = input if input.is_contiguous() else input.contiguous()
    mask_contiguous = mask.contiguous()
    values_contiguous = values.contiguous()
    contiguous_indices = [index.contiguous() for index in indices]
    while len(contiguous_indices) < 3:
        contiguous_indices.append(contiguous_indices[0])

    shape = list(inp.shape) + [1] * (3 - rank)
    strides = list(inp.stride()) + [0] * (3 - rank)
    out_numel = inp.numel()

    # 规模分流：小规模 O(N*M) 的 match 只要一次 launch；大规模 match 结构性不可达，走多轮路径。
    if out_numel * mask.numel() >= _MULTI_ROUND_MIN_WORK:
        return _unsafe_masked_index_put_accumulate_multi_round(
            inp, mask_contiguous, contiguous_indices, values_contiguous,
            rank, shape, strides,
        ).view(inp.shape)

    dests = _dests_per_program(out_numel)
    grid = (triton.cdiv(out_numel, dests),)
    # 尾部哨兵：grid*dests 可能大于 out_numel，多出的 lane 写进填充区而不是被 mask 掉。
    out_buf = torch.empty(grid[0] * dests, dtype=inp.dtype, device=inp.device)
    out = out_buf[:out_numel].view(inp.shape)
    block_size = triton.next_power_of_2(mask.numel())

    with torch_device_fn.device(input.device):
        _unsafe_masked_index_put_accumulate_kernel[grid](
            out_buf,
            inp,
            mask_contiguous,
            contiguous_indices[0],
            contiguous_indices[1],
            contiguous_indices[2],
            values_contiguous,
            mask.numel(),
            out_numel,
            SHAPE0=shape[0],
            SHAPE1=shape[1],
            SHAPE2=shape[2],
            STRIDE0=strides[0],
            STRIDE1=strides[1],
            STRIDE2=strides[2],
            RANK=rank,
            DESTS=dests,
            BLOCK_SIZE=block_size,
            isCloseVectorization=True,
            buffer_size_limit=2048,
        )
    return out
