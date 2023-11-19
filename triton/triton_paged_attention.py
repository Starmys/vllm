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
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + seq_id)

    off_block_table = seq_id * max_num_blocks_per_seq  # 1
    offs_q = seq_id * num_heads * head_size + head_id * head_size + offs_d  # [BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[:, None] // x) * block_size * x + \
        (offs_n[None, :] % block_size) * x + (offs_d[:, None] % x)  # [BLOCK_DMODEL, BLOCK_N]
    offs_v = kv_head_id * head_size * block_size + offs_d[None, :] * block_size + \
        (offs_n[:, None] % block_size)  # [BLOCK_N, BLOCK_DMODEL]

    s = tl.zeros([MAX_SEQ_LEN], dtype=tl.float32)
    q = tl.load(Q + offs_q)  # [BLOCK_DMODEL]

    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(block_tables + off_block_table + start_n + offs_n)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        k = tl.load(K + offs_k + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        p = tl.sum(q[:, None] * k, 0).to(tl.float32) * sm_scale  # [BLOCK_N]
        tl.store(s.ptr[start_n + offs_n], p)

    s_max = tl.max(s, 0)

    # P = Softmax(S)
    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p = tl.load(s.ptr[start_n + offs_n])
        tl.store(s.ptr[start_n + offs_n], tl.exp(p - s_max))

    s_sum = tl.sum(s, 0)

    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p = tl.load(s.ptr[start_n + offs_n])
        tl.store(s.ptr[start_n + offs_n], p / s_sum)

    o = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # O = P @ V
    for start_n in range(0, context_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        physical_block_idx = tl.load(block_tables + off_block_table + start_n + offs_n)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        p = tl.load(s.ptr[start_n + offs_n])  # [BLOCK_N]
        v = tl.load(V + offs_v + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        o += tl.sum(p[:, None] * v, 0)  # [BLOCK_DMODEL]

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
    BLOCK_N = 32 * num_warps
    _fwd_kernel[grid](
        query, key_cache, value_cache, head_mapping, context_lens, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_heads, num_kv_heads, head_size, x,
        MAX_SEQ_LEN=max_seq_len, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_size,
        num_warps=num_warps, num_stages=2,
    )
    return output
