# plot_expert.py
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

ids = []
warmup_ids = [43, 5, 7, 58] # The IDs from your profile run

print("Processing log file and filtering warmup data...")
with open("moe_routes.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if data["type"] == "route":
            # Only include tokens that aren't part of the repeating profile block
            if data["topk_ids"] != warmup_ids:
                ids.extend(data["topk_ids"])

if not ids:
    print("Error: No real routing data found (only warmup data detected)!")
    exit()

print(f"Captured {len(ids)} expert selections from real tokens.")

counts = Counter(ids)
# Qwen1.5-MoE has 64 experts
expert_ids = list(range(64))
usage = [counts.get(i, 0) for i in expert_ids]

# 1. Save Plot
plt.figure(figsize=(12, 6))
plt.bar(expert_ids, usage, color='darkorange', edgecolor='black', alpha=0.8)
plt.title("Expert Usage Histogram (GSM8K Real Tokens - Layer 0)")
plt.xlabel("Expert Index")
plt.ylabel("Frequency")
plt.xticks(range(0, 64, 4))
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig("expert_hist.png")

# 2. Calculate Metrics
total_selections = sum(usage)
top_3 = counts.most_common(3)
probs = np.array(usage) / total_selections
entropy = -np.sum(probs * np.log2(probs + 1e-10))

print("\n" + "="*40)
print("ANALYSIS FOR YOUR README")
print("="*40)
print(f"Top-3 Experts: {top_3}")
print(f"Entropy: {entropy:.4f} bits")
print(f"Total Real Expert Selections: {total_selections}")
print("="*40)
print("File 'expert_hist.png' has been generated.")