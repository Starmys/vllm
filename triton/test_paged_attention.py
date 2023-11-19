import torch
from vllm import attention_ops
from sparta.testing import profile
# from triton_token_attention import triton_paged_attention
# from triton_flash_attention import triton_paged_attention
# from triton_paged_attention import triton_paged_attention
from triton_multi_query_attention import triton_paged_attention

from reference_paged_attention import ref_paged_attention


'''
MxNxHxD = 32x(512/1024)x16x64:
              VLLM error   = 0.6612008213996887
    (Triton) Flash error   = 1.56293123960495
    (Triton) Token error   = 1.1697959303855896
              VLLM latency = 0.3549327392578125 ms
    (Triton) Flash latency = 0.4850913391113281 ms
    (Triton) Token latency = 0.7203553466796875 ms

MxNxHxD = 32x(512/2048)x16x64:
              VLLM error   = 0.582388162612915
    (Triton) Flash error   = 1.5092443823814392
    (Triton) Token error   = 1.0974004864692688
              VLLM latency = 0.4411177062988281 ms
    (Triton) Flash latency = 0.44918988037109375 ms
    (Triton) Token latency = 0.6300282592773437 ms

MxNxHxD = 32x(512/4096)x16x64:
              VLLM error   = 0.49213141202926636
    (Triton) Flash error   = 1.2754526734352112
    (Triton) Token error   = 0.9095631837844849
              VLLM latency = 0.5230960693359376 ms
    (Triton) Flash latency = 0.49032192993164064 ms
    (Triton) Token latency = 0.6875125732421875 ms
'''


def generate_test_data(
    num_seqs: int = 32,
    num_blocks: int = 4096,
    max_num_blocks_per_seq: int = 512,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_size: int = 128,
    block_size: int = 16,
    seed: int = 2023,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
):
    torch.manual_seed(seed)

    query_shape = [num_seqs, num_heads, head_size]
    query = torch.randn(size=query_shape, dtype=dtype, device=device)
    x = 16 // query.element_size()
    key_shape = [num_blocks, num_kv_heads, head_size // x, block_size, x]
    key = torch.randn(size=key_shape, dtype=dtype, device=device)
    value_shape = [num_blocks, num_kv_heads, head_size, block_size]
    value = torch.randn(size=value_shape, dtype=dtype, device=device)

    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device=device),
        num_heads // num_kv_heads,
    )
    max_context_len = max_num_blocks_per_seq * block_size
    context_lens = torch.randint(1, max_context_len, size=(num_seqs, ), dtype=torch.int32, device=device)
    block_tables = torch.concat([
        torch.randperm(num_blocks, dtype=torch.int32, device=device)[:max_num_blocks_per_seq].unsqueeze(0)
        for _ in range(num_seqs)
    ])

    return query, key, value, head_mapping, context_lens, block_tables


def vllm_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
):
    scale = query.shape[-1] ** -0.5
    block_size = value_cache.shape[-1]
    max_context_len = block_tables.shape[-1] * block_size
    output = torch.empty_like(query)
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # alibi_slopes
    )
    return output

if __name__ == '__main__':
    data = generate_test_data()
    ref_results = ref_paged_attention(*data)
    vllm_results = vllm_paged_attention(*data)
    triton_results = triton_paged_attention(*data)
    # import ipdb; ipdb.set_trace()
    print(f'VLLM error = {(ref_results - vllm_results).to(torch.float64).abs().sum().item()}')
    print(f'Triton error = {(ref_results - triton_results).to(torch.float64).abs().sum().item()}')
    try:
        torch.testing.assert_close(vllm_results, ref_results, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(triton_results, ref_results, atol=1e-3, rtol=1e-3)
    except AssertionError as err:
        print('[Error]', err)
        # import ipdb; ipdb.set_trace()
    vllm_latency = profile(vllm_paged_attention, data, num_warmups=20, num_iters=100)
    print(f'VLLM latency = {vllm_latency} ms')
    triton_latency = profile(triton_paged_attention, data, num_warmups=20, num_iters=100)
    print(f'Triton latency = {triton_latency} ms')
