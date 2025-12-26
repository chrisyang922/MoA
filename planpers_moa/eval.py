import json
import argparse
import evaluate


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]  # BLEU expects list-of-lists
    return preds, labels


def compute_metrics(decoded_preds, decoded_labels):
    """
    Compute BLEU, ROUGE, METEOR using HF evaluate library.
    """

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
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference JSONL file")
    parser.add_argument("--pred", required=True, help="Model predictions JSONL file")
    parser.add_argument("--out", required=True, help="Output JSON to save metrics")
    parser.add_argument("--id_field", default="id",
                        help="ID field name (use 'name' for the user dataset)")
    args = parser.parse_args()

    # ----- Load reference -----
    print("Loading reference labels...")
    refs = load_jsonl(args.ref)

    # Build reference dictionary
    ref_dict = {}
    for r in refs:
        key = str(r.get(args.id_field))
        if key:
            ref_dict[key] = r["output"]

    # ----- Load predictions -----
    print("Loading predictions...")
    preds = load_jsonl(args.pred)

    decoded_preds = []
    decoded_labels = []

    for p in preds:
        pid = str(p.get(args.id_field))
        if pid in ref_dict:
            decoded_preds.append(p["output"])
            decoded_labels.append(ref_dict[pid])
        else:
            print(f"Warning: missing ID in reference: {pid}")

    print(f"\nAligned {len(decoded_preds)} prediction–label pairs.\n")

    # ----- Compute metrics -----
    print("Computing metrics...")
    metrics = compute_metrics(decoded_preds, decoded_labels)

    print("\n==== FINAL METRICS ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ----- Save -----
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics to:", args.out)


if __name__ == "__main__":
    main()
