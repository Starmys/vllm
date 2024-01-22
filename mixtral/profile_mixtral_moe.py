import time
import torch
from transformers import AutoConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
# from vllm.model_executor.models.mixtral import MixtralMoE, MixtralConfig


MODEL_PATH = 'mistralai/Mixtral-8x7B-v0.1'
WEIGHT_PATH = './layer_0.pt'
DEVICE = 'cuda:1'
torch.cuda.set_device(DEVICE)

config = AutoConfig.from_pretrained(MODEL_PATH)
moe = MixtralSparseMoeBlock(config)
moe.load_state_dict(torch.load(WEIGHT_PATH))

moe.to(torch.float16).to(DEVICE)

batch_size, sequence_length, hidden_dim = 8, 1024, 4096
hidden_states = torch.randn((batch_size, sequence_length, hidden_dim), device=DEVICE, requires_grad=False, dtype=torch.float16)

ref, _ = moe.forward(hidden_states)

with torch.no_grad():
    for _ in range(10):
        ref, _ = moe.forward(hidden_states)
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(10):
        ref, _ = moe.forward(hidden_states)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

latency = (end - start) / 1e6 / 10
print(f'MixtralSparseMoeBlock: {latency:.3f} ms')  # 394.413


import triton
import triton.language as tl


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
    stride_be, stride_bn, stride_bk,
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

    # offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < cnt
    mask_n = offs_n < N

    idx = tl.load(idx_ptr + eid * M + offs_m, mask=mask_m)
    if NAME == 'w2':
        a_ptrs = a_ptr + (idx[:, None] * stride_am + offs_k[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + ((idx // T)[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + eid * stride_be + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        mask_k = k + offs_k < K
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0.0)
        accumulator += tl.dot(a, b.T)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if NAME == "w1":
        accumulator = silu(accumulator)
    c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + (idx[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    if NAME == "w3":
        c *= tl.load(c_ptrs, mask=(mask_m[:, None] & mask_n[None, :]))
    tl.store(c_ptrs, c, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def silu(x):
    return x * (tl.exp(x) / (1 + tl.exp(x)))


def f(x, w1, w2, w3):
    o1 = torch.nn.functional.silu(torch.matmul(x, w1.T))
    o2 = o1 * torch.matmul(x, w3.T)
    o3 = torch.matmul(o2, w2.T)
    return o1, o2, o3


class MixtralSparTAMoeBlock(torch.nn.Module):

    def __init__(self, raw_moe_block: MixtralSparseMoeBlock):
        super().__init__()
        self.hidden_dim = raw_moe_block.hidden_dim
        self.ffn_dim = raw_moe_block.ffn_dim
        self.num_experts = raw_moe_block.num_experts
        self.top_k = raw_moe_block.top_k

        self.gate = raw_moe_block.gate
        self.w1 = torch.nn.Parameter(torch.stack([mlp.w1.weight for mlp in raw_moe_block.experts]).contiguous(), requires_grad=False)
        self.w2 = torch.nn.Parameter(torch.stack([mlp.w2.weight for mlp in raw_moe_block.experts]).contiguous(), requires_grad=False)
        self.w3 = torch.nn.Parameter(torch.stack([mlp.w3.weight for mlp in raw_moe_block.experts]).contiguous(), requires_grad=False)

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
        expert_mask = torch.nn.functional.one_hot(selected_experts.flatten(), num_classes=8)
        expert_cnt = expert_mask.sum(dim=0)
        expert_idx = expert_mask.argsort(dim=0, descending=True)[:num_tokens].T.contiguous()

        intermediate_output = torch.empty(
            size=(num_tokens * self.top_k, self.ffn_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        output = torch.empty(
            size=(num_tokens * self.top_k, hidden_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(self.ffn_dim, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            hidden_states, self.w1, intermediate_output, expert_cnt, expert_idx,
            num_tokens, self.ffn_dim, hidden_dim, self.num_experts, self.top_k,
            hidden_states.stride(0), hidden_states.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            intermediate_output.stride(0), intermediate_output.stride(1),
            NAME='w1'
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(self.ffn_dim, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            hidden_states, self.w3, intermediate_output, expert_cnt, expert_idx,
            num_tokens, self.ffn_dim, hidden_dim, self.num_experts, self.top_k,
            hidden_states.stride(0), hidden_states.stride(1),
            self.w3.stride(0), self.w3.stride(1), self.w3.stride(2),
            intermediate_output.stride(0), intermediate_output.stride(1),
            NAME='w3',
        )
        mixtral_linear_kernel[lambda META: (
            triton.cdiv(num_tokens, META['BLOCK_SIZE_M']) * triton.cdiv(hidden_dim, META['BLOCK_SIZE_N']),
            self.num_experts
        )](
            intermediate_output, self.w2, output, expert_cnt, expert_idx,
            num_tokens, hidden_dim, self.ffn_dim, self.num_experts, self.top_k,
            intermediate_output.stride(0), intermediate_output.stride(1),
            self.w2.stride(0), self.w2.stride(1), self.w2.stride(2),
            output.stride(0), output.stride(1),
            NAME='w2',
        )

        final_hidden_states = (output.reshape(num_tokens, self.top_k, hidden_dim) * routing_weights[:, :, None]).sum(dim=1)

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


sparta_moe = MixtralSparTAMoeBlock(moe)
sparta_moe.to(DEVICE)

out, _ = sparta_moe.forward(hidden_states)

torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

with torch.no_grad():
    for _ in range(10):
        out, _ = sparta_moe.forward(hidden_states)
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(10):
        out, _ = sparta_moe.forward(hidden_states)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()

latency = (end - start) / 1e6 / 10
print(f'MixtralSparseMoeBlock: {latency:.3f} ms')  # 394.413
