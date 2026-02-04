The patch was applied exclusively to fused_moe.py file in vllm/model_executor/layers/fused_moe. The changes are made in two function blocks, fused_experts_impl and the apply function towards the end. 
The exact changes are highlighted in the patch file.

Commands to run after implementing the patch and creating the python files attached in this repo. 
Dataset generation: python3 make_prompts.py
Baseline run: python3 run_generate.py 
Logged run: VLLM_LOG_MOE=moe_routes.jsonl python3 run_generate.py
Data plot: python3 plot_expert.py

Top-3 Experts: 
The most utilized experts in Layer 0 were Expert #42 (240 selections), Expert #20 (229 selections), and Expert #46 (222 selections).

Normalized Distribution: 
The selection distribution is remarkably flat. 
With 9,736 total selections across 64 experts, the top expert accounted for only ~2.46% of total selections, indicating a very well-balanced routing logic.

Routing Entropy:
The calculated entropy is 5.8758 bits.

Interpretation: 
Since the maximum possible entropy for 64 experts is 6.0, a score of 5.87 suggests that Qwen1.5-MoE has successfully avoided "expert collapse." 
The model utilizes a high degree of its total parameter capacity to solve GSM8K math problems rather than over-specializing in a small subset of experts.

AI Usage log
