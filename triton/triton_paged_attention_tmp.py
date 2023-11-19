import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + head_id)  # 1
    context_len = tl.load(context_lens + offs_m)  # BLOCK_M
    offs_block = offs_n % block_size  # BLOCK_N
    block_table_ptrs = block_tables + offs_m * max_num_blocks_per_seq + offs_n // block_size  # [BLOCK_M, BLOCK_N]

    off_q = offs_m[:, None] * num_heads * head_size + head_id * head_size + offs_d[None, :]  # [BLOCK_M, BLOCK_DMODEL]
    off_k = kv_head_id * head_size * block_size + (offs_d[:, None] // x) * block_size * x + \
        offs_block[None, :] * x + (offs_d[:, None] % x)  # [BLOCK_DMODEL, BLOCK_N]
    off_v = kv_head_id * head_size * block_size + offs_d[None, :] * block_size + offs_block[:, None]  # [BLOCK_N, BLOCK_DMODEL]

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    # for start_n in range(0, tl.max(context_len, 0), BLOCK_N):
    for start_n in range(0, max_num_blocks_per_seq, BLOCK_N):
        # load block table
        physical_block_idx = tl.load(block_table_ptrs)  # [BLOCK_M, BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size
        # compute qk
        k = tl.load(k_ptrs + offs_page[None, :])
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(start_n + offs_n[None, :] < context_len[:, None], qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs + offs_page[:, None])
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        block_table_ptrs += BLOCK_N // block_size

    # initialize pointers to output
    out_ptrs = Out + off_q
    tl.store(out_ptrs, acc)


def triton_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    # BLOCK_M, BLOCK_N = 64, 256
    BLOCK_M, BLOCK_N = 128, 128
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}
    _, num_kv_heads, _, block_size, x = key_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    scale = head_size ** -0.5
    output = torch.empty_like(query)
    L = torch.empty((num_seqs, num_heads), device=query.device, dtype=torch.float32)
    M = torch.empty((num_seqs, num_heads), device=query.device, dtype=torch.float32)
    grid = (triton.cdiv(num_seqs, BLOCK_M), num_heads, 1)
    num_warps = 4 if head_size <= 64 else 8
    _fwd_kernel[grid](
        query, key_cache, value_cache, head_mapping, context_lens, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_size,
        num_warps=num_warps, num_stages=2,
    )
    return output


@triton.jit
def _fwd_kernel(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_n = tl.arange(0, MAX_SEQ_LEN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + seq_id)

    off_block_table = seq_id * max_num_blocks_per_seq  # 1
    offs_q = seq_id * num_heads * head_size + head_id * head_size + offs_d  # [BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[:, None] // x) * block_size * x + \
        (offs_n[None, :] % block_size) * x + (offs_d[:, None] % x)  # [BLOCK_DMODEL, MAX_SEQ_LEN]
    offs_v = kv_head_id * head_size * block_size + offs_d[None, :] * block_size + \
        (offs_n[:, None] % block_size)  # [MAX_SEQ_LEN, BLOCK_DMODEL]

    q = tl.load(Q + offs_q) * sm_scale  # [BLOCK_DMODEL]

    # S = Q @ K
    physical_block_idx = tl.load(block_tables + off_block_table + offs_n)  # [MAX_SEQ_LEN]
    offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [MAX_SEQ_LEN]
    k = tl.load(K + offs_k + offs_page[None, :])  # [BLOCK_DMODEL, MAX_SEQ_LEN]
    s = tl.sum(q[:, None] * k, 0).to(tl.float32)  # [MAX_SEQ_LEN]

    # P = Softmax(S)
    s_max = tl.max(s, 0)
    s = tl.exp(s - s_max)
    s_sum = tl.sum(s, 0)
    s /= s_sum

    # O = P @ V
    o = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    physical_block_idx = tl.load(block_tables + off_block_table + offs_n)  # [MAX_SEQ_LEN]
    offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [MAX_SEQ_LEN]
    v = tl.load(V + offs_v + offs_page[:, None])  # [MAX_SEQ_LEN, BLOCK_DMODEL]
    o += tl.sum(s[:, None] * v, 0)  # [BLOCK_DMODEL]

    tl.store(Out + offs_q, o.to(tl.float16))


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
    _, num_kv_heads, _, block_size, x = key_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    max_seq_len = max_num_blocks_per_seq * block_size
    scale = head_size ** -0.5
    output = torch.empty_like(query)
    grid = (num_seqs, num_heads)
    num_warps = 4
    _fwd_kernel[grid](
        query, key_cache, value_cache, head_mapping, context_lens, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        MAX_SEQ_LEN=max_seq_len, BLOCK_DMODEL=head_size,
        num_warps=num_warps, num_stages=2,
    )
    return output
