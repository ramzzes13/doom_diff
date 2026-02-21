# Overnight Execution Rules

## CRITICAL CONSTRAINTS
- ALL code must be REAL and WORKING. Never mock, stub, or fake any functionality.
- Do NOT modify any files outside `/home/mekashirskiy/rom4ik/doom_diff/`
- Do NOT kill any processes you did not create. There are sglang servers and other services running.
- Commit frequently with meaningful messages after each milestone.
- This is a full autonomous overnight execution. Keep working until ALL goals from the research plan are reached.
- If a dependency install fails, try alternative approaches. Do not stop.
- If training takes time, wait for it. Do not skip or mock results.
- All models, data downloads, and training runs must be real.
- Use GPU resources available on the machine. Check with `nvidia-smi` before starting.

## Project Goal
Implement the memory-augmented diffusion world model for DOOM from `doom_diff_research_plan.md`:
1. ViZDoom environment setup and data collection
2. Memory-augmented action-conditioned diffusion model
3. Training pipeline with state-consistency supervision
4. Real-time inference with efficient sampling
5. Evaluation with state-sensitive metrics
