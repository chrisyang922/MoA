import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_LOG"] = "0"

import json
import numpy as np
import evaluate




def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]  # HF BLEU expects list-of-lists
    return preds, labels



def compute_metrics(decoded_preds, decoded_labels):
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    result_bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result_rouge = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result_meteor = meteor_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "bleu": result_bleu["bleu"],
        "rouge-1": result_rouge["rouge1"],
        "rouge-2": result_rouge["rouge2"],
        "rouge-L": result_rouge["rougeL"],
        "rouge-LSum": result_rouge["rougeLsum"],
        "meteor": result_meteor["meteor"],
    }




def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"\n❌ JSON ERROR on line {i}:")
                print(line)
                raise e
    return data



# LOAD DATA
ref_path = "top_200_profile_length_abstract_generation_temporal.jsonl"
pred_path = "abstract_generation_temporal/top_1000_multi_agent/fused_l3_1-1000.jsonl"

print("Loading reference labels...")
references = load_jsonl(ref_path)

# Build reference dict: reviewerId → output text
ref_dict = {str(r["name"]): r["output"] for r in references}


print("Loading predictions...")
predictions = load_jsonl(pred_path)


# ALIGN PREDICTIONS & LABELS
decoded_preds = []
decoded_labels = []

for p in predictions:  
    pid = str(p["name"])

    if pid not in ref_dict:
        print("Warning: missing ID in references:", pid)
        continue

    #prediction text key: p["output"]
    decoded_preds.append(p["output"])
    decoded_labels.append(ref_dict[pid])

print(f"Aligned {len(decoded_preds)} prediction–label pairs.")


# METRIC COMPUTATION
print("Computing metrics...")
metrics = compute_metrics(decoded_preds, decoded_labels)

print("\n==== FINAL METRICS ====")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save metrics
out_path = "abstract_generation_temporal/top_1000_multi_agent/l1_a_200_eval3.jsonl"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved metrics to:", out_path)
