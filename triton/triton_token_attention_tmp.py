import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_token_att1(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    scale,
    block_size,
    x,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h, stried_k_d, stride_k_c, stride_k_x,
    stride_b_s, stride_b_c,
    stride_o_s, stride_o_head, stride_o_c,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    # initialize offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + cur_head)
    context_len = tl.load(context_lens + cur_batch)

    block_mask = tl.where(start_n * BLOCK_N < context_len, 1, 0)

    for start_mark in range(0, block_mask, 1):

        off_q = cur_batch * stride_q_s + cur_head * stride_q_h + offs_d * stride_q_d  # [BLOCK_DMODEL]
        block_table_ptrs = block_tables + cur_batch * stride_b_s + offs_n // block_size * stride_b_c  # [BLOCK_N]
        physical_block_idx = tl.load(block_table_ptrs)  # [BLOCK_N]
        offs_page = physical_block_idx * stride_k_b
        off_k = kv_head_id * stride_k_h + (offs_d[:, None] // x) * stried_k_d + offs_page[None, :] + \
            (offs_n % block_size)[None, :] * stride_k_c + (offs_d[:, None] % x) * stride_k_x  # [BLOCK_DMODEL, BLOCK_N]

        q = tl.load(Q + off_q + start_mark)  # [BLOCK_DMODEL]
        # k = tl.load(K + off_k, mask=offs_n[None, :] < context_len, other=0.0)
        k = tl.load(K + off_k)  # [BLOCK_DMODEL, BLOCK_N]
        att_value = tl.sum(q[:, None] * k, 0)
        att_value *= scale

        off_o = cur_batch * stride_o_s * block_size + cur_head * stride_o_head + offs_n * stride_o_c  # [BLOCK_N]
        o_mask = offs_n < context_len
        tl.store(Out + off_o, att_value, mask=o_mask)


@torch.no_grad()
def token_att_fwd(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    scale,
    num_seqs,
    num_blocks,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
):
    BLOCK = 32
    assert head_size in {16, 32, 64, 128}
    grid = (num_seqs, num_heads, triton.cdiv(max_num_blocks_per_seq * block_size, BLOCK))

    # num_warps = 4 if head_size <= 64 else 8
    num_warps = 2

    _fwd_kernel_token_att1[grid](
        Q, K, head_mapping, context_lens, block_tables, Out,
        scale, block_size, x,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3), K.stride(4),
        block_tables.stride(0), block_tables.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(3),
        BLOCK_DMODEL=head_size, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )


@triton.jit
def _fwd_kernel_token_att1(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    scale,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + cur_head)
    context_len = tl.load(context_lens + cur_batch)

    off_q = cur_batch * num_heads * head_size + cur_head * head_size + offs_d  # [BLOCK_DMODEL]
    offs_block_table = block_tables + cur_batch * max_num_blocks_per_seq  # [BLOCK_N]
    offs_k = kv_head_id * head_size * block_size + (offs_d[:, None] // x) * block_size * x + \
        (offs_n % block_size)[None, :] * x + (offs_d[:, None] % x)  # [BLOCK_DMODEL, BLOCK_N]
    offs_o = cur_batch * num_heads * max_num_blocks_per_seq * block_size + \
        cur_head * max_num_blocks_per_seq * block_size + offs_n  # [BLOCK_N]

    q = tl.load(Q + off_q)  # [BLOCK_DMODEL]

    for start_n in range(0, context_len, BLOCK_N):
        physical_block_idx = tl.load(offs_block_table + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size
        # offs_page = offs_n

        k = tl.load(K + offs_k + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        att_value = tl.sum(q[:, None] * k, 0)
        att_value *= scale

        tl.store(Out + offs_o + start_n, att_value)


@torch.no_grad()
def token_att_fwd(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    scale,
    num_seqs,
    num_blocks,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
):
    if triton.__version__ >= "2.1.0":
        BLOCK = 128
    else:
        BLOCK = 64

    grid = (num_seqs, num_heads)
    num_warps = 4

    _fwd_kernel_token_att1[grid](
        Q, K, head_mapping, context_lens, block_tables, Out, scale,
        max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        BLOCK_DMODEL=head_size, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )
