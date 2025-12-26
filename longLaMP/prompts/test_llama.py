import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

def run_vllm(prompted_dataset, output_dir, model_name = "meta-llama/Llama-3.2-1B-Instruct", temperature = 0.8, top_p = 0.9, max_new_tokens = 512, batch_size = 1):
    os.makedirs(output_dir, exist_ok = True)
    llm = LLM(model = model_name)
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = top_p,
        max_tokens = max_new_tokens
    )
 
    prompts = [ex["source"] for ex in prompted_dataset]
    generations = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
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