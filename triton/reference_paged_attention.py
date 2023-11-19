import torch


def ref_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    block_tables_long = block_tables.to(torch.long)
    head_mapping_long = head_mapping.to(torch.long)
    _, num_heads, head_size = query.shape
    scale = head_size ** -0.5
    output = []
    for q, context_len, block_table in zip(query, context_lens, block_tables_long):
        v = value_cache[block_table]
        k = key_cache[block_table].swapaxes(-1, -2).reshape(v.shape)
        v = v[:, head_mapping_long].swapaxes(1, -1).reshape(-1, head_size, num_heads)
        k = k[:, head_mapping_long].swapaxes(1, -1).reshape(-1, head_size, num_heads)
        p = torch.einsum('hd, ndh -> hn', q * scale, k).reshape((num_heads, -1))
        p[:, context_len:] = -torch.inf
        # p[:, context_len:] = 0.0
        s = torch.softmax(p, dim=-1)
        o = torch.einsum('hn, ndh -> hd', s, v)
        output.append(o.unsqueeze(0))
        # output.append(p.reshape((-1, block_tables.shape[-1], value_cache.shape[-1])).unsqueeze(0))
    return torch.concat(output)
