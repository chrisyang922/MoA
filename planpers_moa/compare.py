import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)  # list of objects, ascending order

def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))  # descending order
    return items

json_path = "top_1000_ordered_product_review_temporal_test_with_id.json"
jsonl_path = "product_review_temporal/data/top_1000_profile_length_product_review_temporal.jsonl"

data_json  = load_json(json_path)
data_jsonl = load_jsonl(jsonl_path)

# Reverse the JSONL list so both orders match
data_jsonl_reversed = list(reversed(data_jsonl))

# Length check
if len(data_json) != len(data_jsonl_reversed):
    print("❌ LENGTH mismatch:", len(data_json), "vs", len(data_jsonl_reversed))
else:
    print("✓ Same number of samples:", len(data_json))

# Compare line-by-line
all_match = True
for i, (a, b) in enumerate(zip(data_json, data_jsonl_reversed)):
    if a != b:
        print(f"❌ Mismatch at index {i}")
        all_match = False
        break

if all_match:
    print("✅ Files contain identical samples, just reversed order.")
