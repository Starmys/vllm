import torch
import triton
import triton.language as tl


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
    kv_head_id = tl.program_id(1)
    head_id_1 = kv_head_id * 2 + 0
    head_id_2 = kv_head_id * 2 + 1

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    context_len = tl.load(context_lens + seq_id)

    off_block_table = seq_id * max_num_blocks_per_seq
    offs_q = seq_id * num_heads * head_size + offs_d  # [BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[None, :] // x) * block_size * x + \
        (offs_n[:, None] % block_size) * x + (offs_d[None, :] % x)  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = kv_head_id * head_size * block_size + offs_d[:, None] * block_size + \
        (offs_n[None, :] % block_size)  # [BLOCK_DMODEL, BLOCK_N]

    # initialize pointer to m and l
    m_i_1 = float("-inf")
    m_i_2 = float("-inf")
    l_i_1 = 0.0
    l_i_2 = 0.0
    acc_1 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    q_1 = tl.load(Q + offs_q + head_id_1 * head_size)  # [BLOCK_DMODEL]
    q_1 = (q_1 * qk_scale).to(tl.float16)
    q_2 = tl.load(Q + offs_q + head_id_2 * head_size)  # [BLOCK_DMODEL]
    q_2 = (q_2 * qk_scale).to(tl.float16)

    for start_n in range(0, context_len, BLOCK_N):
        # -- load block table --
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(block_tables + off_block_table + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        qk_mask = start_n + offs_n < context_len
        # -- load k, v --
        k = tl.load(K + offs_k + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V + offs_v + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        # -- initialize qk ---
        qk_1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        qk_1 = tl.where(qk_mask, qk_1, float("-inf"))
        qk_1 += tl.sum(q_1[None, :] * k, 1)
        qk_2 = tl.zeros([BLOCK_N], dtype=tl.float32)
        qk_2 = tl.where(qk_mask, qk_2, float("-inf"))
        qk_2 += tl.sum(q_2[None, :] * k, 1)
        # -- compute scaling constant ---
        m_i_new_1 = tl.maximum(m_i_1, tl.max(qk_1, 0))
        m_i_new_2 = tl.maximum(m_i_2, tl.max(qk_2, 0))
        alpha_1 = tl.math.exp2(m_i_1 - m_i_new_1)
        alpha_2 = tl.math.exp2(m_i_2 - m_i_new_2)
        p_1 = tl.math.exp2(qk_1 - m_i_new_1).to(tl.float16)  # [BLOCK_N]
        p_2 = tl.math.exp2(qk_2 - m_i_new_2).to(tl.float16)  # [BLOCK_N]
        # -- scale and update acc --
        acc_1 *= alpha_1
        acc_2 *= alpha_2
        acc_1 += tl.sum(p_1[None, :] * v, 1)
        acc_2 += tl.sum(p_2[None, :] * v, 1)
        # -- update m_i and l_i --
        l_i_1 = l_i_1 * alpha_1 + tl.sum(p_1, 0)
        l_i_2 = l_i_2 * alpha_2 + tl.sum(p_2, 0)
        m_i_1 = m_i_new_1
        m_i_2 = m_i_new_2

    acc_1 /= l_i_1
    acc_2 /= l_i_2
    tl.store(Out + offs_q + head_id_1 * head_size, acc_1.to(tl.float16))
    tl.store(Out + offs_q + head_id_2 * head_size, acc_2.to(tl.float16))


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
    grid = (num_seqs, num_kv_heads)
    _fwd_kernel_v2[grid](
        query, key_cache, value_cache, head_mapping, context_lens, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        BLOCK_N=64, BLOCK_DMODEL=head_size,
        num_warps=4, num_stages=4,
    )
    return output
