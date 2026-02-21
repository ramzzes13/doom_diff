# Research Plan: Memory-Augmented Real-Time Diffusion World Models for Interactive Game Simulation (Extending GameNGen on DOOM)

## Thesis framing (one-paragraph summary)

This research plan proposes a new scientific paper (and diploma/thesis) that extends **GameNGen** (“Diffusion Models Are Real-Time Game Engines”) by addressing its most consequential limitation: **short effective memory (≈3 seconds of history) leading to long-horizon state inconsistencies**. Building on 2025–2026 advances in **memory-augmented diffusion for long video** (e.g., MALT Diffusion), **flexible-history diffusion objectives** (History-Guided Video Diffusion / DFoT), **causalization and action guidance** for interactive world models (Vid2World, DWS), and **real-time streaming long-horizon diffusion** (Rolling Forcing, Matrix-Game 2.0), we will develop and evaluate a **memory-augmented, action-conditioned diffusion world model** that can maintain consistent game state over **minutes** while remaining feasible to train and run on **one consumer GPU** using public tools (ViZDoom).

---

### 1. Introduction

Interactive virtual worlds (games, simulators, interactive software) require a tight control loop: user actions update latent state, which is rendered to pixels at real-time frame rates. Recent diffusion-based video generators produce high-fidelity frames, but converting them into **interactive, action-conditioned, stable long-horizon simulators** remains difficult due to (i) **exposure bias** (teacher forcing vs autoregressive rollout), (ii) **error accumulation/drift**, (iii) insufficient **memory** for state that must persist beyond a few seconds, and (iv) the tension between **quality** and **latency**.

#### 1.1 What the original paper achieved
The original article, **“Diffusion Models Are Real-Time Game Engines”** (Valevski et al.; “GameNGen”), demonstrates the first fully neural “engine” that can simulate **DOOM** interactively in real time by:
- Training an RL agent (PPO) to play DOOM in ViZDoom and recording trajectories.
- Fine-tuning a pretrained **Stable Diffusion v1.4** U-Net to predict the next frame conditioned on a **sequence of past frames and actions** (no text conditioning).
- Stabilizing long rollouts via **conditioning noise augmentation** (corrupting context frames during training with a known noise level embedding) to reduce autoregressive drift.
- Improving visual fidelity (especially HUD text/numbers) via **VAE decoder fine-tuning**.
- Achieving **20 FPS** with **4 DDIM steps** on a TPU-v5; and showing a **50 FPS** variant via 1-step distillation (with some quality cost).

#### 1.2 Key limitations to target
From the paper’s Discussion/Limitation section and ablations, the most research-relevant limitations are:
- **Limited memory / context**: the model conditions on only 64 frames (~3 seconds), yet must maintain state over minutes; increasing context yields diminishing returns, suggesting an architectural change is needed.
- **Behavior distribution mismatch**: training data is largely from an RL agent; coverage gaps and non-human behavior cause failure modes in rarely visited areas/interactions.
- **Evaluation gaps**: pixel metrics (PSNR/LPIPS) become less informative once trajectories diverge; long-horizon “state correctness” is hard to quantify without explicit state probes.
- **Compute/replicability**: original training uses large-scale TPU resources and ~70M examples; reproducing this exactly is infeasible for a typical thesis setup.

#### 1.3 Goal and novel contribution of the new work
**Goal**: Create a new diffusion-based interactive world model for DOOM that maintains **state consistency over minute-scale horizons** with **real-time or near-real-time** inference on a single GPU, and that is evaluated with **state- and control-sensitive metrics** beyond pixel similarity.

**Core contribution (proposed paper)**:
1) **Memory-Augmented Diffusion World Model**: introduce a compact, learnable memory mechanism that extends effective history from seconds to minutes without linear growth in compute.
2) **State-consistency supervision and evaluation**: add auxiliary objectives and metrics that directly measure persistent game state (HUD variables, events, and controllability).
3) **Real-time streaming via efficient sampling**: integrate few-step sampling (and optional distillation / streaming window strategies) to meet real-time constraints on consumer hardware.

Working title:
- **“MemGameNGen: Memory-Augmented Causal Diffusion for Long-Horizon Real-Time Neural Game Engines”**

---

### 2. Literature Review

This section focuses on 2025–2026 work most relevant to extending GameNGen, plus a small set of essential precursors for context.

#### 2.1 Neural interactive environments and world models (context)
- **GameNGen / GameNGen paper**: Valevski et al., *Diffusion Models Are Real-Time Game Engines*, arXiv:2408.14837, 2024 (published ICLR 2025). Demonstrates real-time DOOM simulation via action-conditioned next-frame diffusion with noise augmentation and VAE decoder fine-tuning.
- **Genie**: Bruce et al., *Genie: Generative Interactive Environments*, arXiv:2402.15391, 2024. Interactive environments from learned models (broader, but not necessarily DOOM-like fidelity in real-time).

#### 2.2 Turning video generators into interactive world models (2025)
- **Vid2World**: Huang et al., *Vid2World: Crafting Video Diffusion Models to Interactive World Models*, arXiv:2505.14357, 2025. Key idea: **video diffusion causalization**—reshape architecture and training objective to enable autoregressive, action-conditioned generation; includes **causal action guidance** to improve controllability. Relevant for action-conditioning correctness and causal rollout stability.
- **Dynamic World Simulation (DWS)**: He et al., *Pre-Trained Video Generative Models as World Simulators*, arXiv:2502.07825, 2025. Adds a **lightweight action-conditioned module** plus **motion-reinforced loss** to enforce action-aligned dynamics; applicable to diffusion and autoregressive transformers. Relevant for improving controllability and dynamics (not just visual detail).

#### 2.3 Real-time streaming long-horizon diffusion (2025–2026)
- **Matrix-Game 2.0**: He et al., *Matrix-game 2.0: An open-source real-time and streaming interactive world model*, arXiv:2508.13009, 2025. Emphasizes **few-step autoregressive diffusion** + **causal architecture** and a large annotated dataset pipeline (Unreal/GTA5), enabling **25 FPS** minute-level streaming generation. Relevant as a practical reference for streaming constraints and causal design.
- **Rolling Forcing**: Liu et al., *Rolling Forcing: Autoregressive Long Video Diffusion in Real Time*, ICLR 2026. Key ideas: joint denoising of multi-frame windows to suppress error propagation; **attention sink** for global anchoring; few-step distillation over extended windows to enable **single-GPU real-time multi-minute streaming**. Highly relevant for stable long rollouts.

#### 2.4 Memory and flexible-history conditioning for long video (2025)
- **MALT Diffusion**: Yu et al., *MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation*, arXiv:2502.12632, 2025. Segment-level autoregressive generation with a compact **memory latent vector** maintained over time. Provides concrete mechanisms for scaling “effective history” without feeding thousands of frames.
- **History-Guided Video Diffusion / DFoT**: Song et al., *History-Guided Video Diffusion*, arXiv:2502.06764, 2025. Proposes DFoT architecture + training objective enabling **flexible number of history frames** and “history guidance” that can roll out extremely long videos. Relevant for exposure bias and variable-length history handling.

#### 2.5 Evaluation benchmarks emphasizing controllability and functionality (2025–2026)
- **WorldScore**: Duan et al., *WorldScore: A Unified Evaluation Benchmark for World Generation*, ICCV 2025 (benchmark site provides BibTeX). Emphasizes **controllability**, **multi-scene generation**, and dynamic metrics beyond single-clip fidelity.
- **WorldArena**: Zhu et al., *WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models*, arXiv HTML (ICML), 2026. Highlights the **perception–functionality gap** and evaluates world models as data engines / policy evaluators / action planners; motivates metrics that matter for downstream use.

#### 2.6 Instruction-driven interaction (late 2025–2026)
- **Hunyuan-GameCraft-2**: Tang et al., *Instruction-following Interactive Game World Model*, arXiv:2511.23429, 2026. Uses instruction-driven interaction and introduces **InterBench**. Relevant as an “upper bound” direction: adding semantic interaction beyond keyboard/mouse.
- **Genie 3 (system announcement)**: Parker-Holder & Fruchter, *Genie 3: A new frontier for world models*, Google DeepMind blog (Aug 2025). Demonstrates real-time 24 FPS 720p interactive worlds with minute-scale consistency; not a full paper, but a strong signal of emerging expectations for memory and controllability.

#### 2.7 Synthesis of gaps from the literature
Across these works, a clear gap remains for DOOM-like neural game engines:
- **Long-horizon state correctness** remains under-measured and under-optimized in pixel-only approaches.
- **Memory mechanisms** exist for long video, but adapting them to **action-conditioned, real-time interactive** settings (where actions stream in online) is still under-explored.
- **Streaming diffusion** techniques exist, but their interaction with **explicit game state** and controllability metrics needs systematic investigation.

---

### 3. Proposed Methodology

We propose a method that retains the key strengths of GameNGen (action-conditioned latent diffusion; noise augmentation; few-step sampling) while adding **explicit, compact memory** and **state-aware training/evaluation**.

#### 3.1 Research questions and hypotheses

- **RQ1 (memory)**: Can a diffusion-based neural engine maintain correct game state over minutes if given a compact learned memory rather than a short raw-frame context?
  - **H1**: A memory-augmented model will significantly improve long-horizon state consistency (minutes) at similar short-horizon frame quality, compared to a 64-frame context-only baseline.

- **RQ2 (controllability)**: Can we improve action-following without sacrificing realism using action-aligned objectives/guidance?
  - **H2**: Adding motion/action consistency losses or action guidance will reduce “action lag” and spurious dynamics under autoregressive rollout.

- **RQ3 (real-time feasibility)**: Can we reach interactive latency (≥15–30 FPS) on a single consumer GPU while maintaining long-horizon stability?
  - **H3**: With few-step sampling and windowed/streaming generation (inspired by Rolling Forcing / Matrix-Game 2.0), real-time play is achievable with acceptable quality.

#### 3.2 Baseline to reproduce (scaled thesis baseline)

Re-implement a “thesis-scale” version of GameNGen:
- **Environment**: ViZDoom (DOOM).
- **Data**: trajectories from (i) PPO agent training + (ii) a small set of human-play sessions if feasible.
- **Model**: latent diffusion U-Net initialized from an open latent diffusion checkpoint (e.g., Stable Diffusion 1.4/1.5) or a smaller diffusion model if GPU memory requires.
- **Conditioning**:
  - Past frames: encoded latents concatenated along channels (as in original paper).
  - Actions: learned action-token embeddings fed via cross-attention (replace text tokens).
- **Stabilization**: conditioning noise augmentation (as in Section 3.2.1 of the paper).
- **Quality**: optional VAE decoder fine-tune to improve HUD.

This baseline anchors comparisons and ensures any improvement is attributable to the proposed contributions rather than implementation differences.

#### 3.3 MemGameNGen: memory-augmented action-conditioned diffusion

##### 3.3.1 Core idea: compact memory tokens / memory latent

The original model is limited by a fixed window of raw latent frames. We propose to maintain a compact memory \(m_t\) summarizing distant history, updated over time:

- **Memory representation**:
  - Option A (token memory): a fixed set of \(K\) memory tokens \(m_t \in \mathbb{R}^{K \times d}\).
  - Option B (single latent memory): one vector \(m_t \in \mathbb{R}^{d}\) (lighter; may underfit complex state).

- **Memory update** (online, causal):
  - A lightweight recurrent update network \(g_\psi\) that updates memory from the new predicted latent frame \(\hat{z}_t\), action \(a_t\), and previous memory:
    \[
    m_t = g_\psi(m_{t-1}, \hat{z}_t, a_t)
    \]
  - Update can be implemented as (i) GRU-style, (ii) cross-attention from memory to the new frame features, or (iii) a small transformer block with causal masking.

- **Memory usage during denoising**:
  - Inject memory into the diffusion U-Net via:
    - Cross-attention: U-Net attention blocks attend to \([ \text{action tokens} ; \text{memory tokens}]\).
    - FiLM/adapters: memory produces scale/shift conditioning at multiple U-Net resolutions.

This is inspired by **MALT Diffusion** (compact memory latent for long videos), but adapted for **online interactive** generation where actions arrive frame-by-frame.

##### 3.3.2 Retrieval-augmented memory (optional extension)

For state that reoccurs (returning to a location after a long time), add a small episodic memory:
- Store periodic keyframes and associated embeddings.
- At time \(t\), retrieve top-\(r\) relevant memories given current predicted frame embedding and memory state.
- Provide retrieved tokens as additional conditioning.

This makes “returning to a room after a minute” feasible without unbounded context.

##### 3.3.3 State-aware auxiliary supervision

Pixel losses alone are insufficient to ensure correct **game logic**. ViZDoom can expose structured variables, enabling direct supervision and evaluation:

- **State targets** (examples; choose those accessible in ViZDoom):
  - Health, armor, ammo counts, current weapon.
  - Kill count / damage dealt / items picked.
  - (If available) player position/orientation; or discretized room/sector ID.

- **Auxiliary heads**:
  - Attach a small “state head” \(h_\omega\) to latent features (or memory) to predict state variables.
  - Loss: cross-entropy for categorical, MSE for continuous, optionally focal/Huber.

- **Consistency regularization**:
  - Encourage memory \(m_t\) to be predictive of state: \(h_\omega(m_t)\).
  - Encourage temporal consistency for slowly varying variables (e.g., weapon): penalize impossible rapid flips.

This directly targets the GameNGen limitation where plausible pixels can hide incorrect underlying state.

##### 3.3.4 Action controllability objective / guidance

To reduce action-conditional failures (e.g., unintended motion, action lag), incorporate 2025 controllability ideas:
- **Motion-reinforced loss** (DWS): emphasize learning transitions aligned with action-induced motion.
- **Causal action guidance** (Vid2World): during sampling, apply guidance that increases alignment between predicted change and intended action (implemented as an auxiliary score or classifier-like guidance term).

Practical thesis implementation:
- Use optical flow between consecutive frames to compute a differentiable “motion signature”.
- For simple actions (turn left/right, move forward), enforce that motion direction/magnitude matches expected patterns using a lightweight predictor trained from real data.

##### 3.3.5 Streaming window generation (real-time stability)

Integrate a streaming strategy inspired by Rolling Forcing:
- Instead of sampling one frame at a time with strict causality, denoise a small window of \(W\) frames jointly with a structured noise schedule (e.g., higher noise for later frames).
- Maintain a persistent “anchor” representation (attention sink / memory tokens) that stabilizes global context.

This component is proposed as a **second-stage upgrade** after the memory model is validated in open-loop/short closed-loop settings.

#### 3.4 Proposed system diagram (textual description)

**Diagram** (to include in the paper/thesis):
- Inputs at time \(t\): last \(N\) latent frames \(\hat{z}_{t-N:t-1}\), last \(N\) actions \(a_{t-N:t-1}\), memory tokens \(m_{t-1}\).
- Denoiser U-Net receives noised latent \(x_t\), concatenated history latents, and attends to a context of \([A\_emb(a_{\le t}), m_{t-1}]\).
- Output: predicted velocity/noise for \(\hat{z}_t\).
- Memory update module consumes \(\hat{z}_t\), \(a_t\), and \(m_{t-1}\) to produce \(m_t\).
- State head predicts structured variables from \(m_t\) and/or U-Net features.

#### 3.5 Pseudocode (high-level)

```
Initialize memory m0 (learned or zeros)
Initialize history buffer H0 with real frames (warm start) or repeated first frame

for t in 1..T:
  # Diffusion sampling to generate next latent frame z_t
  z_t = sample_diffusion(
          unet=f_theta,
          conditioning={
            "history_latents": H_{t-1},
            "action_tokens": A_emb(a_{t-N:t-1}),
            "memory_tokens": m_{t-1}
          },
          steps=S  # e.g., 4
        )

  # Decode for display (optional)
  frame_t = VAE.decode(z_t)

  # Update memory causally
  m_t = g_psi(m_{t-1}, z_t, a_t)

  # Predict structured state for auxiliary loss/eval
  s_hat_t = h_omega(m_t)

  # Update history buffer with z_t
  H_t = update_buffer(H_{t-1}, z_t)
```

---

### 4. Experimental Design

This section specifies *concrete experiments* that can be executed with reasonable resources, while still supporting a publishable paper.

#### 4.1 Datasets

##### 4.1.1 Primary: DOOM via ViZDoom (interactive world simulation)
- **Environment**: ViZDoom (Wydmuch et al., 2019), using a curated set of 5–10 levels to balance diversity and training tractability.
- **Data sources**:
  - **Agent-play**: PPO agent (as in GameNGen) trained with a reward that encourages exploration and diverse interactions; record trajectories throughout training to capture varying skill levels.
  - **Human-play (optional but valuable)**: record 1–5 hours of human play to reduce behavior mismatch; even small amounts can help calibrate “human-like” action distributions.
  - **Random policy**: include a small subset for coverage of unusual transitions (as GameNGen found random surprisingly helpful, but limited by exploration).
- **Logging**:
  - RGB frames at 320×240 (pad to 320×256 if needed for latent size alignment).
  - Actions (key presses), including repeated-action modeling (apply one action for multiple frames).
  - Game variables from ViZDoom: health, ammo, armor, weapon id, etc.

##### 4.1.2 Secondary: Chrome Dino (sanity check + fast iteration)
Use the simpler “Chrome Dino” setup (GameNGen appendix) as a low-cost testbed for:
- rapid ablations,
- debugging action conditioning,
- testing memory mechanisms on a simpler state space.

##### 4.1.3 Optional generalization dataset (if time permits)
Choose *one* additional environment to test transfer/generalization:
- Atari (ALE) with an action-conditioned video dataset, or
- a simple 2D top-down environment where ground-truth state is easily measurable.

#### 4.2 Baselines

At minimum, compare against:
- **B0: Context-only diffusion (thesis-scale GameNGen reproduction)**:
  - 64-frame history, action tokens, noise augmentation, few-step sampling.
- **B1: + VAE decoder fine-tuning**:
  - Isolates the contribution of decoder fine-tuning for HUD/state readability.
- **B2: Longer raw context (if feasible)**:
  - 128 or 256-frame history using frame subsampling to control compute; tests whether brute-force longer context is enough.
- **B3: Memory-only (minimal history)**:
  - Reduce history to 8–16 frames but add memory tokens; tests whether memory can replace raw history.

If feasible, add:
- **B4: Action-controllability enhanced baseline**:
  - Add motion-reinforced loss / action guidance without memory.

#### 4.3 Metrics

##### 4.3.1 Standard perceptual/pixel metrics (short-horizon and teacher-forcing)
- **PSNR** and **LPIPS** under teacher forcing (predict \(o_t\) given ground-truth history).
- Report across multiple held-out levels and initial states.

##### 4.3.2 Distributional video metrics (short rollouts)
- **FVD** on 16- and 32-frame segments, following the original paper’s rationale.

##### 4.3.3 Long-horizon stability and state correctness (core contribution)

Define metrics that remain meaningful when trajectories diverge:

1) **HUD-variable accuracy**:
   - Extract ground-truth variables directly from ViZDoom during rollout under the same action sequence.
   - From generated frames, estimate variables via:
     - the model’s auxiliary state head \(h_\omega\), and/or
     - an external HUD OCR/recognizer trained on real frames (optional; robustness check).
   - Report accuracy/MSE over time, including after 1, 3, 5, and 10 minutes of autoregressive generation.

2) **Event consistency metrics**:
   - Define events detectable in engine state: enemy hit/kill, item pickup, door open, episode end/restart.
   - Measure event precision/recall vs ground truth under fixed action sequences (scripted scenarios).

3) **Controllability metrics**:
   - For action subsets (turn left/right, move forward, strafe), quantify:
     - expected vs observed camera motion (approximate via optical flow or learned pose estimator),
     - action-to-motion latency,
     - variance under repeated identical actions (stochasticity / instability).

4) **Functional utility (WorldArena-inspired, scaled to thesis)**:
   - Use the world model as an environment proxy to evaluate a small policy: compare policy success ranking in real ViZDoom vs in the world model (correlation).
   - Even a small study (3–5 policies with different skill) can show whether the model is “useful” beyond visuals.

##### 4.3.4 Performance metrics (real-time requirement)
- **FPS** and **end-to-end latency** (including decoding) on a single GPU.
- **VRAM usage**, model size, and sampling steps.

#### 4.4 Implementation details (realistic thesis constraints)

##### 4.4.1 Hardware target
- **Training**: single consumer GPU (e.g., 16–24GB VRAM); mixed precision (bf16/fp16).
- **Inference**: same GPU; target ≥15–30 FPS for interactive feel, with a “stretch goal” of 60 FPS.

##### 4.4.2 Software stack
- PyTorch + diffusers (or a custom latent diffusion training loop for action conditioning).
- ViZDoom for data collection and evaluation.
- Weights & Biases (or equivalent) for experiment tracking.

##### 4.4.3 Training strategy for feasibility
- Use parameter-efficient tuning when necessary:
  - LoRA/adapters in attention blocks,
  - small memory modules trained from scratch.
- Dataset scaling:
  - Start with 1M–5M frames for early experiments.
  - Move to 10M–30M frames if stable and beneficial (guided by scaling curves similar to GameNGen’s Appendix A.3).

#### 4.5 Experimental matrix (concrete experiments)

##### Experiment 1: Reproduce and validate a GameNGen-like baseline
- **Setup**: context-only, 64 frames, noise augmentation on context, 4-step DDIM sampling.
- **Evaluate**: PSNR/LPIPS (teacher forcing), FVD (16/32), FPS.
- **Expected result**: similar qualitative behavior; lower absolute metrics vs original paper due to reduced compute/data, but stable enough for controlled studies.

##### Experiment 2: Memory ablation study (main contribution)
Train and compare:
- **E2-A**: context-only (B0).
- **E2-B**: context-only with longer raw context (B2).
- **E2-C**: memory-augmented with same context (MemGameNGen).
- **E2-D**: memory-augmented with reduced context (B3).

**Ablate**:
- memory token count \(K \in \{8, 16, 32, 64\}\),
- memory update frequency (every frame vs every \(k\) frames),
- retrieval on/off (if implemented).

**Evaluate (core)**:
- HUD-variable accuracy over multi-minute rollouts,
- event consistency,
- controllability metrics.

**Expected results**:
- Memory-augmented models maintain higher state accuracy after 3–10 minutes.
- Longer raw context yields diminishing returns (consistent with original paper).

##### Experiment 3: Action-controllability objective/guidance
Compare:
- **E3-A**: MemGameNGen without motion/action loss.
- **E3-B**: + motion-reinforced loss (DWS-inspired).
- **E3-C**: + action guidance during sampling (Vid2World-inspired), if feasible.

**Evaluate**:
- action-to-motion alignment,
- reduction in spurious dynamics,
- effect on perceptual quality and FPS.

**Expected results**:
- Improved controllability metrics with minimal drop in PSNR/LPIPS.
- Best trade-off may be training-time loss rather than inference-time guidance (for latency reasons).

##### Experiment 4: Streaming/windowed denoising for long-horizon stability (stretch)
Implement a small window \(W \in \{2, 4, 8\}\) joint denoising scheme:
- Compare strict single-frame autoregression vs windowed denoising.
- Add “anchor/sink” tokens that persist across windows.

**Evaluate**:
- long-horizon drift rate,
- FPS/latency.

**Expected results**:
- Lower drift, especially for minute-scale rollouts, consistent with Rolling Forcing’s premise.

##### Experiment 5: Data distribution and coverage (agent vs human vs mixed)
Train MemGameNGen under:
- agent-only data,
- random-only data (small),
- mixed agent+random,
- mixed agent+small human.

**Evaluate**:
- OOD robustness: rarely visited rooms, unusual action sequences,
- event correctness in scripted tests.

**Expected results**:
- Mixed data improves robustness; even small human-play data improves “human-like” controllability and reduces failure in unseen interactions.

#### 4.6 Scripted evaluation scenarios (to make results reproducible)

Define 10–20 deterministic “unit tests” in ViZDoom:
- **Door test**: approach door, open, pass through, return after 60s.
- **Pickup test**: pick health/ammo; verify HUD change and persistence.
- **Combat test**: engage enemy, kill; verify enemy stays dead after leaving and returning.
- **Loop test**: run a fixed square path for 2 minutes; measure location consistency and scene re-encounter.
- **Idle test**: stand still for 2 minutes; measure visual drift vs stability.

Each test yields state traces and a standardized report.

---

### 5. Discussion and Future Work

#### 5.1 Potential pitfalls and mitigations
- **Memory collapse / shortcut learning**: the model might ignore memory or store non-causal artifacts.
  - Mitigation: ablate memory usage; add auxiliary state prediction from memory; enforce causal masking; monitor attention to memory tokens.
- **State head overfitting**: predicting variables may be easier than ensuring pixels reflect them.
  - Mitigation: evaluate both (i) state head accuracy and (ii) pixel-HUD recognizer accuracy; ensure consistency between them.
- **Trade-off between realism and controllability**: stronger action losses may reduce visual quality.
  - Mitigation: use multi-objective tuning; report Pareto front (quality vs controllability vs FPS).
- **Compute limits**: training from SD checkpoints may still be heavy.
  - Mitigation: use LoRA/adapters, smaller base models, lower resolution, curriculum training (start with fewer levels).
- **Evaluation difficulty**: long-horizon “correctness” can be subjective.
  - Mitigation: prioritize engine-state-based metrics and scripted tests; add a small human study only as supplementary evidence.

#### 5.2 Alternative interpretations
- Improvements may come from better regularization rather than memory per se.
  - Address via careful baselines (longer context, different noise augmentation, different model size).
- “State correctness” may improve even when the world model is less visually similar.
  - This is acceptable; functional utility is the goal. Use WorldArena-inspired framing: perception vs functionality.

#### 5.3 Future work beyond the thesis
- **Instruction-driven game editing**: incorporate natural language “world events” (direction aligned with Hunyuan-GameCraft-2, Genie 3).
- **New games / interactive software**: test on higher-resolution FPS games or non-game interactive software loops.
- **Hybrid latent-state engines**: combine diffusion renderer with an explicit learned latent dynamics model (e.g., discrete state + diffusion decoder).
- **Safety and misuse analysis**: neural engines that imitate proprietary games may raise IP and policy concerns; add responsible development guidance.

---

### 6. Conclusion

This plan outlines a diploma-level research program to extend GameNGen with **compact memory**, **state-aware training and evaluation**, and **real-time streaming** considerations. The expected outcome is a new scientific paper demonstrating that diffusion-based neural game engines can be made significantly more **long-horizon consistent and controllable**—measured not only by perceptual metrics but by **game-state correctness and functional utility**—while remaining feasible to train and run on **a single GPU** with public tooling.

---

### 7. References

*Original paper (primary source)*  
- Valevski, D., Leviathan, Y., Arar, M., & Fruchter, S. **Diffusion Models Are Real-Time Game Engines.** arXiv:2408.14837, 2024. (ICLR 2025 publication page available.)

*Interactive environments / neural world models*  
- Bruce, J., et al. **Genie: Generative Interactive Environments.** arXiv:2402.15391, 2024.

*2025–2026: interactive world models and streaming real-time diffusion*  
- Huang, S., Wu, J., Zhou, Q., Miao, S., & Long, M. **Vid2World: Crafting Video Diffusion Models to Interactive World Models.** arXiv:2505.14357, 2025.  
- He, H., Zhang, Y., Lin, L., Xu, Z., & Pan, L. **Pre-Trained Video Generative Models as World Simulators.** arXiv:2502.07825, 2025.  
- He, X., et al. **Matrix-game 2.0: An open-source real-time and streaming interactive world model.** arXiv:2508.13009, 2025.  
- Liu, K., Hu, W., Xu, J., Shan, Y., & Lu, S. **Rolling Forcing: Autoregressive Long Video Diffusion in Real Time.** ICLR 2026. (OpenReview id: IAyzXjbfwo.)

*2025: long-horizon memory / flexible history in diffusion video models*  
- Yu, S., Hahn, M., Kondratyuk, D., Shin, J., Gupta, A., Lezama, J., Essa, I., Ross, D., & Huang, J. **MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation.** arXiv:2502.12632, 2025.  
- Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., & Sitzmann, V. **History-Guided Video Diffusion.** arXiv:2502.06764, 2025.

*Benchmarks and evaluation (2025–2026)*  
- Duan, H., Yu, H.-X., Chen, S., Fei-Fei, L., & Wu, J. **WorldScore: A Unified Evaluation Benchmark for World Generation.** ICCV 2025. (Benchmark site provides BibTeX.)  
- Zhu, Y. S., et al. **WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models.** 2026. (Benchmark/leaderboard: `world-arena.ai`.)

*2026: instruction-driven interactive world modeling*  
- Tang, J., et al. **Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model.** arXiv:2511.23429, 2026.

*System announcement / perspective (non-peer-reviewed, but informative)*  
- Parker-Holder, J., & Fruchter, S. **Genie 3: A new frontier for world models.** Google DeepMind blog, Aug 2025.

*Foundational tools and classic references used in the original paper (for completeness)*  
- Wydmuch, M., Kempka, M., & Jaskowski, W. **ViZDoom Competitions: Playing Doom from Pixels.** IEEE Transactions on Games, 2019.  
- Schulman, J., et al. **Proximal Policy Optimization Algorithms.** arXiv:1707.06347, 2017.  
- Rombach, R., et al. **High-Resolution Image Synthesis with Latent Diffusion Models.** CVPR, 2022.  
- Zhang, R., et al. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).** CVPR, 2018.  
- Unterthiner, T., et al. **FVD: A New Metric for Video Generation.** ICLR 2019 Workshop.

