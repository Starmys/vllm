import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_v1(
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
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + seq_id)

    off_block_table = seq_id * max_num_blocks_per_seq
    offs_q = seq_id * num_heads * head_size + head_id * head_size + offs_d  # [BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[None, :] // x) * block_size * x + \
        (offs_n[:, None] % block_size) * x + (offs_d[None, :] % x)  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = kv_head_id * head_size * block_size + offs_d[:, None] * block_size + \
        (offs_n[None, :] % block_size)  # [BLOCK_DMODEL, BLOCK_N]

    m_prev = float("-inf")
    l_prev = 0.0
    q = tl.load(Q + offs_q)  # [BLOCK_DMODEL]
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(block_tables + off_block_table + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        # offs_page = (start_n + offs_n) // block_size * num_kv_heads * head_size * block_size  # [BLOCK_N]
        k = tl.load(K + offs_k + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        qk = tl.zeros([BLOCK_N], dtype=tl.float32)
        qk += tl.sum(q[None, :] * k, 1)
        qk *= sm_scale  # [BLOCK_N]
        qk = tl.where(start_n + offs_n < context_len, qk, float("-inf"))
        m_curr = tl.maximum(tl.max(qk, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_rcp = 1. / l_curr
        p *= l_rcp
        acc *= (l_prev * l_rcp)
        p = p.to(tl.float16)
        v = tl.load(V + offs_v + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        acc += tl.sum(p[None, :] * v, 1)
        l_prev = l_curr
        m_prev = m_curr

    tl.store(Out + offs_q, acc.to(tl.float16))


@triton.jit
def _fwd_kernel_v2(
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
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + seq_id)

    off_block_table = seq_id * max_num_blocks_per_seq
    offs_q = seq_id * num_heads * head_size + head_id * head_size + offs_d  # [BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[None, :] // x) * block_size * x + \
        (offs_n[:, None] % block_size) * x + (offs_d[None, :] % x)  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = kv_head_id * head_size * block_size + offs_d[:, None] * block_size + \
        (offs_n[None, :] % block_size)  # [BLOCK_DMODEL, BLOCK_N]

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    # initialize pointer to m and l
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q + offs_q)  # [BLOCK_DMODEL]
    q = (q * qk_scale).to(tl.float16)

    for start_n in range(0, context_len, BLOCK_N):
        # -- load block table --
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(block_tables + off_block_table + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        # offs_page = (start_n + offs_n) // block_size
        # -- load k, v --
        k = tl.load(K + offs_k + offs_page[:, None])
        v = tl.load(V + offs_v + offs_page[None, :])
        # -- compute qk ---
        qk = tl.zeros([BLOCK_N], dtype=tl.float32)
        qk = tl.where(start_n + offs_n < context_len, qk, float("-inf"))
        qk += tl.sum(q[None, :] * k, 1)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new)
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale
        p = p.to(tl.float16)
        acc += tl.sum(p[None, :] * v, 1)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_i_new

    acc /= l_i
    tl.store(Out + offs_q, acc.to(tl.float16))


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
    scale = head_size ** -0.5
    output = torch.empty_like(query)
    grid = (num_seqs, num_heads)
    _fwd_kernel_v2[grid](
        query, key_cache, value_cache, head_mapping, context_lens, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        BLOCK_N=64, BLOCK_DMODEL=head_size,
        num_warps=4, num_stages=4,
    )
    return output
