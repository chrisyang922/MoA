import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

def run_vllm(prompted_dataset, output_dir, model_name = "meta-llama/Llama-3.1-8B-Instruct",
             temperature = 0.8, top_p = 0.9, max_new_tokens = 1024, batch_size=1):
    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
        model = model_name,
        max_model_len = 12288,      
        gpu_memory_utilization = 0.95,
        max_num_batched_tokens = 16384,
        max_num_seqs = 32
    )

    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = top_p,
        max_tokens = max_new_tokens
    )

    prompts = [ex["source"] for ex in prompted_dataset]
    generations = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for prompt, out in zip(batch, outputs):
            text = out.outputs[0].text if out.outputs else ""
            generations.append({"input": prompt, "output": text})

    result_path = os.path.join(output_dir, "results.jsonl")
    with open(result_path, "w") as f:
        for item in generations:
            f.write(json.dumps(item) + "\n")
    return generations


def perform(eval_dataset, model_name, output_dir):
    return run_vllm(eval_dataset, output_dir, model_name=model_name)


def run(eval_dataset, model_name, output_dir):
    return run_vllm(eval_dataset, output_dir, model_name=model_name)
