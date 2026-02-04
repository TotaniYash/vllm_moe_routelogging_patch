from vllm import LLM, SamplingParams
import os, json, time, random

random.seed(1234)
prompts = open("prompts.txt").read().split("\n\n---\n\n")
sp = SamplingParams(temperature=0.0, max_tokens=128)

# VRAM configuration to fit 14.3B model on a single card
llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B-Chat", 
          max_model_len=512, 
          gpu_memory_utilization=0.95, 
          enforce_eager=True)

t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()

log_mode = "log" if "VLLM_LOG_MOE" in os.environ else "no_log"
entry = {log_mode: {"wall_time_sec": t1 - t0,
                    "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outs)}}

# Load existing or create new timing.json
timing = {}
if os.path.exists("timing.json"):
    with open("timing.json", "r") as f:
        timing = json.load(f)

timing.update(entry)

with open("timing.json", "w") as f:
    json.dump(timing, f, indent=2)

print(f"Finished {log_mode} run.")