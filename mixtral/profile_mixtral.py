import json
from vllm import LLM, SamplingParams


DATA_PATH = '/home/chengzhang/LLMOS-Parrot/bing_chat_dataset.jsonl'
with open(DATA_PATH, encoding='utf8') as f:
    prompts = [json.loads(line)['prompt'] for line in f.readlines()]
prompts = prompts[:1]
print('Prompts:', len(prompts))

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="mistralai/Mixtral-8x7B-v0.1")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
