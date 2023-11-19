"""In this bench we don't consider multi-heads.
Parameters: batch_size, sequence_length, hidden_dim
"""


import time
import torch
import triton
import triton.language as tl
from vllm import attention_ops

from reference_paged_attention import ref_paged_attention


@triton.jit
def _fwd_kernel_v2(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    qk_max,  # [num_seqs, num_heads]
    exp_sum,  # [num_seqs, num_heads]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOAD_MID_RESULTS: tl.constexpr,
):
    seq_group_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    start_m = seq_group_id * BLOCK_M

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + start_m + offs_m)  # [BLOCK_M]

    offs_q = (start_m + offs_m[:, None]) * num_heads * head_size + \
        head_id * head_size + offs_d[None, :]  # [BLOCK_M, BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[None, :] // x) * block_size * x + \
        (offs_n[:, None] % block_size) * x + (offs_d[None, :] % x)  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = kv_head_id * head_size * block_size + offs_d[:, None] * block_size + \
        (offs_n[None, :] % block_size)  # [BLOCK_DMODEL, BLOCK_N]

    if LOAD_MID_RESULTS:
        m_i = tl.load(qk_max + (start_m + offs_m) * num_heads + head_id)
        l_i = tl.load(exp_sum + (start_m + offs_m) * num_heads + head_id)
        acc = tl.load(Out + offs_q).to(tl.float32) * l_i[:, None]
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q + offs_q)  # [BLOCK_M, BLOCK_DMODEL]
    q = (q * qk_scale).to(tl.float16)

    for start_n in range(0, tl.max(context_len), BLOCK_N):
        # -- load block table --
        physical_block_idx = tl.load(block_tables + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        # -- load k, v --
        k = tl.load(K + offs_k + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V + offs_v + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(start_n + offs_n[None, :] < context_len[:, None], qk, float("-inf"))
        qk += tl.dot(q, k.T)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])  # [BLOCK_M, BLOCK_N]
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v.T)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    if not LOAD_MID_RESULTS:
        tl.store(qk_max + (start_m + offs_m) * num_heads + head_id, m_i)
        tl.store(exp_sum + (start_m + offs_m) * num_heads + head_id, l_i)

    acc /= l_i[:, None]
    tl.store(Out + offs_q, acc.to(tl.float16))


def triton_flash_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    qk_max: torch.Tensor,  # [num_seqs, num_heads]
    exp_sum: torch.Tensor,  # [num_seqs, num_heads]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    output: torch.Tensor,  # [num_seqs, num_heads, head_size]
    load_mid_results: bool,
):
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}
    _, num_kv_heads, _, block_size, x = key_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    scale = head_size ** -0.5
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(num_seqs, BLOCK_M), num_heads)
    _fwd_kernel_v2[grid](
        query, key_cache, value_cache, head_mapping, context_lens, qk_max, exp_sum, block_tables, output,
        scale, max_num_blocks_per_seq, block_size, num_seqs, num_heads, num_kv_heads, head_size, x,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_size, LOAD_MID_RESULTS=load_mid_results,
        num_warps=4, num_stages=4,
    )


def profile(
    batch_size: int,
    shared_sequence_length: int,
    diverged_sequence_length: int,
    head_size: int,
    head_num: int = 32,
    block_size: int = 16,
    x: int = 8,
    dtype: torch.dtype = torch.float16,
    device: torch.device = "cuda",
    copy: str = False,
    flash: str = 'none',
    warmups: int = 20,
    iters: int = 100,
    seed: int = 2023,
):
    torch.manual_seed(seed)

    shared_block_count = shared_sequence_length // block_size
    diverged_block_count = diverged_sequence_length // block_size

    q = torch.randn([batch_size, head_num, head_size], dtype=dtype, device=device)
    if copy:
        k_cache_shared = torch.randn(
            [shared_block_count, head_num, head_size // x, block_size, x], dtype=dtype, device=device
        ).repeat((batch_size, 1, 1, 1, 1))
        v_cache_shared = torch.randn(
            [shared_block_count, head_num, head_size, block_size], dtype=dtype, device=device
        ).repeat((batch_size, 1, 1, 1))
    else:
        k_cache_shared = torch.randn(
            [shared_block_count, head_num, head_size // x, block_size, x], dtype=dtype, device=device
        )
        v_cache_shared = torch.randn(
            [shared_block_count, head_num, head_size, block_size], dtype=dtype, device=device
        )
    k_cache_diverged = torch.randn(
        [diverged_block_count * batch_size, head_num, head_size // x, block_size, x], dtype=dtype, device=device
    )
    v_cache_diverged = torch.randn(
        [diverged_block_count * batch_size, head_num, head_size, block_size], dtype=dtype, device=device
    )
    k_cache = torch.concat([k_cache_shared, k_cache_diverged])
    v_cache = torch.concat([v_cache_shared, v_cache_diverged])

    head_mapping = torch.arange(head_num, dtype=torch.int32, device=device)
    shared_context_lens = torch.tensor(
        [shared_sequence_length] * batch_size, dtype=torch.int32, device=device
    )
    diverged_context_lens = torch.tensor(
        [diverged_sequence_length] * batch_size, dtype=torch.int32, device=device
    )
    context_lens = shared_context_lens + diverged_context_lens
    qk_max = torch.zeros([batch_size, head_num], dtype=torch.float32, device=device)
    exp_sum = torch.zeros([batch_size, head_num], dtype=torch.float32, device=device)

    if copy:
        shared_block_tables = torch.tensor(
            list(range(shared_block_count * batch_size)), dtype=torch.int32, device=device
        ).reshape(batch_size, shared_block_count)
    else:
        shared_block_tables = torch.tensor(
            list(range(shared_block_count)) * batch_size, dtype=torch.int32, device=device
        ).reshape(batch_size, shared_block_count)
    diverged_block_tables = torch.tensor(
        list(range(diverged_block_count * batch_size)), dtype=torch.int32, device=device
    ).reshape(batch_size, diverged_block_count) + shared_block_count
    block_tables = torch.concat([shared_block_tables, diverged_block_tables], dim=-1)
    # print(block_tables)

    max_context_len = block_tables.shape[-1] * block_size

    if flash == 'first':
        def run_kernel():
            output = torch.empty_like(q)
            triton_flash_attention(
                q,
                k_cache,
                v_cache,
                head_mapping,
                shared_context_lens,
                qk_max,
                exp_sum,
                shared_block_tables,
                output,
                load_mid_results=False,
            )
            attention_ops.single_query_cached_kv_post_attention(
                output,
                q,
                k_cache,
                v_cache,
                head_mapping,
                head_size**-0.5,
                diverged_block_tables,
                diverged_context_lens,
                qk_max,
                exp_sum,
                block_size,
                max_context_len,
                None,  # alibi_slopes
            )
            return output
    elif flash == 'second':
        def run_kernel():
            output = torch.empty_like(q)
            attention_ops.single_query_cached_kv_prev_attention(
                output,
                q,
                k_cache,
                v_cache,
                head_mapping,
                head_size**-0.5,
                diverged_block_tables,
                diverged_context_lens,
                qk_max,
                exp_sum,
                block_size,
                max_context_len,
                None,  # alibi_slopes
            )
            triton_flash_attention(
                q,
                k_cache,
                v_cache,
                head_mapping,
                shared_context_lens,
                qk_max,
                exp_sum,
                shared_block_tables,
                output,
                load_mid_results=True,
            )
            return output
    else:
        def run_kernel():
            output = torch.empty_like(q)
            attention_ops.single_query_cached_kv_attention(
                output,
                q,
                k_cache,
                v_cache,
                head_mapping,
                head_size**-0.5,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,  # alibi_slopes
            )
            return output

    ref_out = ref_paged_attention(
        q,
        k_cache,
        v_cache,
        head_mapping,
        context_lens,
        block_tables,
    )
    output = run_kernel()
    torch.testing.assert_close(ref_out, output, atol=1e-2, rtol=1e-2)

    for _ in range(warmups):
        output = run_kernel()
    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for _ in range(iters):
        output = run_kernel()
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    return (ed - st) / iters / 1e3


if __name__ == "__main__":
    shape = {
        'batch_size': 128,
        'shared_sequence_length': 512,
        'diverged_sequence_length': 512,
        'head_size': 128,
        'head_num': 32,
        'block_size': 16,
    }
    print(', '.join([f'{k}={v}' for k, v in shape.items()]))
    latency = profile(**shape, copy=True, flash='none')
    print(f"   [Paged Copy] Latency: {latency:.3f} us", flush=True)
    latency = profile(**shape, copy=False, flash='none')
    print(f" [Paged Shared] Latency: {latency:.3f} us", flush=True)
    latency = profile(**shape, copy=False, flash='first')
    print(f"[Flash + Paged] Latency: {latency:.3f} us", flush=True)
    latency = profile(**shape, copy=False, flash='second')
    print(f"[Paged + Flash] Latency: {latency:.3f} us", flush=True)
