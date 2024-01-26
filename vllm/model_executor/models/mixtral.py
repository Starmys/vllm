# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Mixtral model."""
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from transformers import MistralConfig

import triton
import triton.language as tl

try:
    import stk
except ImportError:
    print(
        "STK not found: please see https://github.com/stanford-futuredata/stk")

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.wqkv = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=False,  # weights not in HF format
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.wqkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata,
                                cache_event)
        output, _ = self.wo(attn_output)
        return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def mixtral_linear_kernel(
    a_ptr,  # [M, K] or [T * M, K]
    b_ptr,  # [E, N, K]
    c_ptr,  # [T * M, N]
    cnt_ptr,  # [E] in [0, M)
    idx_ptr,  # [E, M] in [0, T * M)
    M, N, K, E, T,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NAME: tl.constexpr
):
    eid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    cnt = tl.load(cnt_ptr + eid)
    if pid_m * BLOCK_SIZE_M >= cnt:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask = offs_m < cnt

    idx = tl.load(idx_ptr + eid * M + offs_m, mask=mask)
    if NAME == 'w2':
        a_ptrs = a_ptr + (idx[:, None] * stride_am + offs_k[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + ((idx // T)[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + eid * stride_be + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if NAME == "w1":
        accumulator = silu(accumulator)
    c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + (idx[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    if NAME == "w3":
        c *= tl.load(c_ptrs, mask=mask[:, None])
    tl.store(c_ptrs, c, mask=mask[:, None])


@triton.jit
def silu(x):
    return x * (tl.exp(x) / (1 + tl.exp(x)))


class BlockSparseMoE(torch.nn.Module):

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, device=torch.cuda.current_device())

        tp_size = get_tensor_model_parallel_world_size()
        assert self.ffn_dim % tp_size == 0
        self.ffn_dim_per_partition = self.ffn_dim // tp_size
        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim_per_partition, device=torch.cuda.current_device())
        )
        set_weight_attrs(self.w1, {"weight_loader": self.moe_weight_loader_transpose})
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.ffn_dim_per_partition, self.hidden_dim, device=torch.cuda.current_device())
        )
        set_weight_attrs(self.w2, {"weight_loader": self.moe_weight_loader})
        self.w3 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim_per_partition, device=torch.cuda.current_device())
        )
        set_weight_attrs(self.w3, {"weight_loader": self.moe_weight_loader_transpose})

    def moe_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """
        Load the weights for the MoE linear layer.
        """
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.ffn_dim_per_partition
        loaded_weight = loaded_weight.view(self.num_experts, self.ffn_dim, -1)
        loaded_weight = loaded_weight[:, shard_size * tp_rank:shard_size * (tp_rank + 1)]
        loaded_weight = loaded_weight.reshape_as(param)
        param.data.copy_(loaded_weight)

    def moe_weight_loader_transpose(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """
        Load the weights for the MoE linear layer.
        """
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.ffn_dim_per_partition
        loaded_weight = loaded_weight.view(self.num_experts, self.ffn_dim, -1).transpose_(-1, -2)
        loaded_weight = loaded_weight[:, :, shard_size * tp_rank:shard_size * (tp_rank + 1)]
        loaded_weight = loaded_weight.reshape_as(param)
        param.data.copy_(loaded_weight)

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # sparse index of tokens for each expert
        # TODO: custom kernel
        num_tokens = batch_size * sequence_length
        expert_mask = torch.nn.functional.one_hot(selected_experts.flatten(), num_classes=self.num_experts)
        expert_cnt = expert_mask.sum(dim=0)
        expert_idx = expert_mask.argsort(dim=0, descending=True)[:num_tokens].T.contiguous()

        torch.cuda.set_device(hidden_states.device)
        intermediate_output = torch.empty(
            size=(num_tokens * self.top_k, self.ffn_dim_per_partition),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        moe_output = torch.empty(
            size=(num_tokens * self.top_k, hidden_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(self.ffn_dim_per_partition, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            hidden_states, self.w1, intermediate_output, expert_cnt, expert_idx,
            num_tokens, self.ffn_dim_per_partition, hidden_dim, self.num_experts, self.top_k,
            hidden_states.stride(0), hidden_states.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            intermediate_output.stride(0), intermediate_output.stride(1),
            NAME='w1'
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(self.ffn_dim_per_partition, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            hidden_states, self.w3, intermediate_output, expert_cnt, expert_idx,
            num_tokens, self.ffn_dim_per_partition, hidden_dim, self.num_experts, self.top_k,
            hidden_states.stride(0), hidden_states.stride(1),
            self.w3.stride(0), self.w3.stride(1), self.w3.stride(2),
            intermediate_output.stride(0), intermediate_output.stride(1),
            NAME='w3'
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(hidden_dim, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            intermediate_output, self.w2, moe_output, expert_cnt, expert_idx,
            num_tokens, hidden_dim, self.ffn_dim_per_partition, self.num_experts, self.top_k,
            intermediate_output.stride(0), intermediate_output.stride(1),
            self.w2.stride(0), self.w2.stride(1), self.w2.stride(2),
            moe_output.stride(0), moe_output.stride(1),
            NAME='w2'
        )

        final_hidden_states = (moe_output.reshape(num_tokens, self.top_k, hidden_dim) * routing_weights[:, :, None]).sum(dim=1)
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MistralConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.attention = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window)
        self.block_sparse_moe = BlockSparseMoE(
            hidden_dim=self.hidden_size,
            ffn_dim=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )
        self.attention_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        r = self.attention(
            positions=positions,
            hidden_states=self.attention_norm(x),
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        h = x + r
        r = self.block_sparse_moe(self.ffn_norm(h))
        out = h + r
        return out


class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MistralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        assert linear_method is None
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.tok_embeddings(input_ids)

        # forward
        for i in range(len(self.layers)):
            cache_event = None if cache_events is None else cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def sample(
        self,
        hidden_states: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.output.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("wqkv", "wq", "q"),
            ("wqkv", "wk", "k"),
            ("wqkv", "wv", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
