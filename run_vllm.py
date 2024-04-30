from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
model_path = '/home/deepspeed/xiaoxia/fp16'
llm = LLM(model=f"{model_path}/mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=1, quantization="fp8")

outputs = llm.generate(prompts, )#sampling_params

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
