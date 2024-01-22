import time
import json
from vllm import LLM, SamplingParams


# DATA_PATH = '/data/cz/bing_chat_dataset.jsonl'
# with open(DATA_PATH, encoding='utf8') as f:
#     prompts = [json.loads(line)['prompt'] for line in f.readlines()]
# prompts = prompts[:8]

# prompts = ['The best AI company is']
prompts = ['The best AI company is'] * 8
# prompts = ["I've been reading books of old, "]
print('Prompts:', len(prompts))

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
sampling_params = SamplingParams(temperature=0.8, max_tokens=16)

llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=4,
    enforce_eager=True,
    # max_num_batched_tokens=4096,
    max_num_seqs=8,
)
# llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=4)
# llm.llm_engine.driver_worker.model_runner.model.model.layers[0].block_sparse_moe
# llm.llm_engine.driver_worker.model_runner.model.model.layers[0].block_sparse_moe.experts[0].w1.weight.device
# import ipdb; ipdb.set_trace()

start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

print(f'Latency: {end - start:.3f} s')  # 29.456 s, 667.196 s

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
