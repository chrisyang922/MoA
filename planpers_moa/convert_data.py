import json
from datasets import load_dataset
from tqdm import tqdm
import os

# Load dataset
dataset = load_dataset("LongLaMP/LongLaMP", "abstract_generation_temporal")

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), "abstract_generation_temporal")
os.makedirs(output_dir, exist_ok=True)

all_records = []

# Collect (profile_length, record)
for record in dataset["test"]:
    profile = record.get("profile", [])
    profile_size = len(profile)
    all_records.append((profile_size, record))

# Sort by profile length (descending: largest first)
all_records.sort(key=lambda x: x[0], reverse=True)

# Select the top 1000 longest profiles
top_1000 = all_records[:1000]

def write_records(records, filename):
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        for _, record in records:
            f.write(json.dumps(record) + "\n")

write_records(top_1000, "top_1000_profile_length_abstract_generation_temporal.jsonl")

print("Done! Saved top_1000 profile-length subset.")
