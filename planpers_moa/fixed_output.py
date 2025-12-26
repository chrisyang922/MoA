import json
import os
import glob

input_path = "product_review_temporal/data/top_1000_profile_length_product_review_temporal.jsonl"
output_dir = "product_review_temporal/top_1000"

# Load reviewer IDs once
input_rows = []
with open(input_path, "r") as f:
    for line in f:
        if line.strip():
            input_rows.append(json.loads(line))

print(f"Loaded {len(input_rows)} input rows.")

# Find all output files containing "1-500"
files_to_fix = glob.glob(os.path.join(output_dir, "*1-500*.jsonl"))

print("Found files:")
for f in files_to_fix:
    print(" -", f)

# Process each matching file
for file_path in files_to_fix:
    fixed_path = file_path.replace(".jsonl", "_fixed.jsonl")

    output_rows = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                output_rows.append(json.loads(line))

    fixed_rows = []
    for idx, row in enumerate(output_rows):
        reviewer_id = input_rows[idx]["reviewerId"]
        fixed_rows.append({
            "id": reviewer_id,
            "output": row["output"]
        })

    with open(fixed_path, "w") as f:
        for row in fixed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✔ Fixed: {file_path} → {fixed_path}")

print("\nAll matching files processed.")
