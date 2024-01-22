import time
import torch
from transformers.models.mixtral import MixtralForCausalLM
from transformers import AutoTokenizer


MODEL_PATH = 'mistralai/Mixtral-8x7B-v0.1'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = MixtralForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# import ipdb; ipdb.set_trace()
# torch.save(model.model.layers[0].block_sparse_moe.state_dict(), 'layer_0.pt')

# prompts = ['The best AI company is']
prompts = ['The best AI company is'] * 8
input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to('cuda')

start = time.time()
outputs = model.generate(
    input_ids,
    use_cache=False,
    temperature=0.0,
    max_new_tokens=16,
    do_sample=False,
    # top_p=0.95,
)
end = time.time()

outputs = tokenizer.batch_decode(outputs)

print(f'Latency: {end - start:.3f} s')  # 29.456 s, 667.196 s

# Print the outputs.
for output in outputs:
    print(output)

# The best AI company is one that can provide you with the best AI solutions for your business. There are
