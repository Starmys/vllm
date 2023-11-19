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
    start_n = tl.program_id(2)

    # initialize offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + cur_head)
    context_len = tl.load(context_lens + cur_batch)

    block_mask = tl.where(start_n * BLOCK_N < context_len, 1, 0)

    for start_mark in range(0, block_mask, 1):

        off_q = cur_batch * num_heads * head_size + cur_head * head_size + offs_d  # [BLOCK_DMODEL]
        offs_block_table = block_tables + cur_batch * max_num_blocks_per_seq + offs_n // block_size  # [BLOCK_N]
        physical_block_idx = tl.load(offs_block_table)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size
        off_k = kv_head_id * head_size * block_size + (offs_d[:, None] // x) * block_size * x + offs_page[None, :] + \
            (offs_n % block_size)[None, :] * x + (offs_d[:, None] % x)  # [BLOCK_DMODEL, BLOCK_N]

        q = tl.load(Q + off_q + start_mark)  # [BLOCK_DMODEL]
        # k = tl.load(K + off_k, mask=offs_n[None, :] < context_len, other=0.0)
        k = tl.load(K + off_k)  # [BLOCK_DMODEL, BLOCK_N]
        att_value = tl.sum(q[:, None] * k, 0)
        att_value *= scale

        off_o = cur_batch * num_heads * max_num_blocks_per_seq * block_size + \
            cur_head * max_num_blocks_per_seq * block_size + offs_n  # [BLOCK_N]
        # tl.store(Out + off_o, att_value, mask=offs_n < context_len)
        tl.store(Out + off_o, att_value)


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
        Q, K, head_mapping, context_lens, block_tables, Out, scale,
        max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        BLOCK_DMODEL=head_size, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )


@triton.jit
def _fwd_kernel_token_softmax(
    Logics,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    context_lens,  # [num_seqs]
    Prob_Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    num_heads,
    max_num_blocks_per_seq,
    block_size,
    BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK)
    context_len = tl.load(context_lens + cur_batch)

    offsets = cur_batch * num_heads * max_num_blocks_per_seq * block_size + \
        cur_head * max_num_blocks_per_seq * block_size + col_offsets
    mask = col_offsets < context_len

    row = tl.load(Logics + offsets, mask=mask, other=-float('inf')).to(tl.float32)

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # tl.store(Prob_Out + offsets, softmax_output, mask=mask)
    tl.store(Prob_Out + offsets, softmax_output)


@torch.no_grad()
def token_softmax_fwd(
    Logics,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    context_lens,  # [num_seqs]
    Prob_Out,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    num_seqs,
    num_heads,
    max_num_blocks_per_seq,
    block_size,
):
    BLOCK_SIZE = triton.next_power_of_2(max_num_blocks_per_seq * block_size)
    grid = (num_seqs, num_heads)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _fwd_kernel_token_softmax[grid](
        Logics, context_lens, Prob_Out,
        num_heads, max_num_blocks_per_seq, block_size,
        BLOCK=BLOCK_SIZE, num_warps=num_warps,
    )


@triton.jit
def _fwd_kernel_token_att2(
    Prob,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + cur_head)
    context_len = tl.load(context_lens + cur_batch)

    offs_block_table = block_tables + cur_batch * max_num_blocks_per_seq
    offs_v = kv_head_id * head_size * block_size + offs_d[None, :] * block_size + \
        (offs_n % block_size)[:, None]  # [BLOCK_N, BLOCK_DMODEL]
    offs_p = cur_batch * num_heads * max_num_blocks_per_seq * block_size + \
        cur_head * max_num_blocks_per_seq * block_size + offs_n  # [BLOCK_N]

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(offs_block_table + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        # offs_page = offs_n
        p_value = tl.load(Prob + offs_p + start_n)
        v_value = tl.load(V + offs_v + offs_page[:, None])
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(tl.float16)
    off_o = cur_batch * num_heads * head_size + cur_head * head_size + offs_d  # [BLOCK_DMODEL]
    tl.store(Out + off_o, acc)


@torch.no_grad()
def token_att_fwd2(
    Prob,  # [num_seqs, num_heads, max_num_blocks_per_seq * block_size]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    num_seqs,
    num_blocks,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
):
    if triton.__version__ >= "2.1.0":
        BLOCK = 128
    else:
        BLOCK = 64

    grid = (num_seqs, num_heads)
    num_warps = 4

    _fwd_kernel_token_att2[grid](
        Prob, V, head_mapping, context_lens, block_tables, Out,
        max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size,
        BLOCK_DMODEL=head_size, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )


def triton_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}
    _, _, _, _, x = key_cache.shape
    num_blocks, num_kv_heads, _, block_size = value_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    # max_len_in_batch = max_num_blocks_per_seq * block_size
    scale = head_size ** -0.5

    att_m_tensor = torch.empty(
        (num_seqs, num_heads, max_num_blocks_per_seq, block_size),
        dtype=query.dtype, device=query.device,
    )
    token_att_fwd(
        query, key_cache, head_mapping, context_lens, block_tables, att_m_tensor, scale,
        num_seqs, num_blocks, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
    )
    prob = torch.zeros_like(att_m_tensor)
    token_softmax_fwd(
        att_m_tensor, context_lens, prob,
        num_seqs, num_heads, max_num_blocks_per_seq, block_size,
    )
    att_m_tensor = None
    output = torch.empty_like(query)
    token_att_fwd2(
        prob, value_cache, head_mapping, context_lens, block_tables, output,
        num_seqs, num_blocks, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size,
    )
    prob = None
    return output
