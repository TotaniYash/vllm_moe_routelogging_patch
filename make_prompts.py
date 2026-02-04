from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
prompts = [ex["question"] for ex in ds.select(range(25))]
with open("prompts.txt", "w") as f:
    f.write("\n\n---\n\n".join(prompts))
print("Created prompts.txt")