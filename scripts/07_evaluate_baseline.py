"""Step 7: Evaluate the trained baseline (no memory) diffusion model.

Runs teacher-forcing and autoregressive evaluation, then saves results
to results/baseline_evaluation_results.json for comparison with the
memory-augmented model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.trainer import MemGameNGenTrainer
from memgamengen.evaluation import MemGameNGenEvaluator, compute_psnr, compute_lpips
import torch.nn.functional as F
import time


def evaluate_teacher_forcing_baseline(model, dataloader, device, max_batches=100):
    """Evaluate baseline model with teacher forcing (ground truth context).

    Measures: PSNR, LPIPS
    """
    model.eval()
    psnr_values = []
    lpips_values = []

    # Try to load LPIPS
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
    except Exception as e:
        print(f"Warning: Could not load LPIPS model: {e}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        context_frames = batch["context_frames"].to(device)
        target_frame = batch["target_frame"].to(device)
        context_actions = batch["context_actions"].to(device)

        B = context_frames.shape[0]

        # No memory for baseline
        memory_state = None

        # Generate prediction
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model.generate_next_frame(
                    context_frames=context_frames,
                    context_actions=context_actions,
                    memory_state=memory_state,
                    num_inference_steps=4,
                )

        pred_frame = outputs["frame"]

        # Resize if needed to match target
        if pred_frame.shape != target_frame.shape:
            pred_frame = F.interpolate(
                pred_frame, size=target_frame.shape[-2:],
                mode='bilinear', align_corners=False
            )

        # PSNR
        psnr = compute_psnr(pred_frame, target_frame)
        psnr_values.append(psnr)

        # LPIPS
        if lpips_model is not None:
            lpips_val = compute_lpips(pred_frame, target_frame, lpips_model)
            lpips_values.append(lpips_val)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Teacher-forcing batch {batch_idx + 1}/{max_batches}")

    results = {
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "psnr_std": float(np.std(psnr_values)) if psnr_values else 0.0,
    }
    if lpips_values:
        results["lpips_mean"] = float(np.mean(lpips_values))
        results["lpips_std"] = float(np.std(lpips_values))

    return results


def evaluate_autoregressive_baseline(model, dataset, device, num_sequences=10, sequence_length=64):
    """Evaluate baseline model autoregressively.

    Feeds model's own predictions back as context (no ground truth).
    Compares generated frames to ground truth at each step.
    """
    model.eval()
    num_context = model.num_context_frames

    all_psnr_per_step = {}
    all_sequences_psnr = []

    for seq_idx in range(num_sequences):
        # Pick a random starting point from dataset
        start_idx = np.random.randint(0, max(1, len(dataset) - sequence_length))

        # Get initial context from ground truth
        first_sample = dataset[start_idx]
        context_frames_list = [first_sample["context_frames"][i] for i in range(num_context)]
        context_actions_list = [first_sample["context_actions"][i].item() for i in range(num_context)]

        # Collect ground truth frames for the entire sequence
        gt_frames = []
        gt_actions = []
        for offset in range(sequence_length):
            sample_idx = min(start_idx + offset, len(dataset) - 1)
            sample = dataset[sample_idx]
            gt_frames.append(sample["target_frame"])
            if offset < sequence_length - 1:
                # Use the last context action from next sample as the action
                next_sample_idx = min(start_idx + offset + 1, len(dataset) - 1)
                next_sample = dataset[next_sample_idx]
                gt_actions.append(next_sample["context_actions"][-1].item())
            else:
                gt_actions.append(0)

        # Run autoregressive generation
        seq_psnr = []
        context_buffer = list(context_frames_list)
        action_buffer = list(context_actions_list)

        for step in range(min(sequence_length, len(gt_frames))):
            # Prepare context tensor
            context = torch.stack(context_buffer[-num_context:]).unsqueeze(0).to(device)
            actions = torch.LongTensor(action_buffer[-num_context:]).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model.generate_next_frame(
                        context_frames=context,
                        context_actions=actions,
                        memory_state=None,  # No memory for baseline
                        num_inference_steps=4,
                    )

            pred_frame = outputs["frame"].cpu()

            # Compare with ground truth
            gt_frame = gt_frames[step].unsqueeze(0)

            # Resize if needed
            if pred_frame.shape != gt_frame.shape:
                pred_frame_resized = F.interpolate(
                    pred_frame, size=gt_frame.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                pred_frame_resized = pred_frame

            psnr = compute_psnr(pred_frame_resized, gt_frame)
            seq_psnr.append(psnr)

            if step not in all_psnr_per_step:
                all_psnr_per_step[step] = []
            all_psnr_per_step[step].append(psnr)

            # Feed prediction back as context (autoregressive)
            context_buffer.append(pred_frame.squeeze(0))
            action_buffer.append(gt_actions[step])

        all_sequences_psnr.append(np.mean(seq_psnr) if seq_psnr else 0.0)

        if (seq_idx + 1) % 5 == 0:
            print(f"  Autoregressive sequence {seq_idx + 1}/{num_sequences}, "
                  f"avg PSNR: {np.mean(seq_psnr):.2f}")

    # Aggregate results
    results = {
        "overall_psnr_mean": float(np.mean(all_sequences_psnr)),
        "overall_psnr_std": float(np.std(all_sequences_psnr)),
    }

    # PSNR at different horizons
    for horizon in [10, 20, 50]:
        horizon_psnr = []
        for t in range(min(horizon, max(all_psnr_per_step.keys()) + 1) if all_psnr_per_step else 0):
            if t in all_psnr_per_step:
                horizon_psnr.extend(all_psnr_per_step[t])
        if horizon_psnr:
            results[f"psnr_horizon_{horizon}"] = float(np.mean(horizon_psnr))

    return results


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use GPU with most free memory
    gpu_id = int(os.environ.get("EVAL_GPU", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (physical GPU {gpu_id})")

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
        print(f"GPU memory: {free_mem:.1f} GB free")

    # Create baseline model (must match training config exactly)
    print("\nCreating baseline MemGameNGen model (NO memory)...")
    model = MemGameNGen(
        pretrained_model="CompVis/stable-diffusion-v1-4",
        num_actions=8,
        num_context_frames=4,
        memory_enabled=False,
        num_memory_tokens=16,
        memory_dim=768,
        memory_update_type="gru",
        action_embed_dim=128,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        state_head_enabled=True,
        num_state_variables=2,
        noise_augmentation=True,
        noise_aug_max_level=0.7,
    )

    # Load checkpoint
    checkpoint_dir = os.path.join(project_dir, "checkpoints", "baseline")
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    ]) if os.path.exists(checkpoint_dir) else []

    if not checkpoints:
        print("ERROR: No baseline checkpoint found. Run 06_train_baseline.py first.")
        return

    # Prefer final checkpoint
    if "memgamengen_final.pt" in checkpoints:
        ckpt = "memgamengen_final.pt"
    else:
        ckpt = checkpoints[-1]
    ckpt_path = os.path.join(checkpoint_dir, ckpt)
    print(f"Loading checkpoint: {ckpt_path}")

    # Use trainer to load (handles LoRA properly)
    data_dir = os.path.join(project_dir, "data", "trajectories")
    dummy_dataset = DoomTrajectoryDataset(
        data_dir=data_dir,
        num_context_frames=4,
    )

    trainer = MemGameNGenTrainer(
        model=model,
        train_dataset=dummy_dataset,
        device=device,
        max_train_steps=1,
    )
    trainer.load_checkpoint(ckpt_path)
    model = trainer.model
    model.eval()

    # Use fp16 VAE to save GPU memory
    if hasattr(model, 'vae'):
        model.vae = model.vae.half()
        print("Converted VAE to fp16 to save memory")

    print(f"Loaded baseline model at step {trainer.global_step}")
    print(f"Memory enabled: {model.memory_enabled}")

    # Create evaluation dataset
    print(f"\nLoading evaluation dataset from {data_dir}...")
    eval_dataset = DoomTrajectoryDataset(
        data_dir=data_dir,
        num_context_frames=4,
        resolution=(120, 160),
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    results = {}

    # 1. Teacher-forcing evaluation
    print("\n" + "=" * 60)
    print("1. Teacher-Forcing Evaluation (Baseline)")
    print("=" * 60)
    tf_results = evaluate_teacher_forcing_baseline(
        model, eval_loader, device, max_batches=50
    )
    results["teacher_forcing"] = tf_results
    print(f"   PSNR: {tf_results['psnr_mean']:.2f} +/- {tf_results['psnr_std']:.2f}")
    if "lpips_mean" in tf_results:
        print(f"   LPIPS: {tf_results['lpips_mean']:.4f} +/- {tf_results['lpips_std']:.4f}")

    # 2. Autoregressive evaluation
    print("\n" + "=" * 60)
    print("2. Autoregressive Evaluation (Baseline)")
    print("=" * 60)
    ar_results = evaluate_autoregressive_baseline(
        model, eval_dataset, device,
        num_sequences=10,
        sequence_length=64,
    )
    results["autoregressive"] = ar_results
    print(f"   Overall PSNR: {ar_results['overall_psnr_mean']:.2f} +/- {ar_results['overall_psnr_std']:.2f}")
    for k, v in ar_results.items():
        if "horizon" in k:
            print(f"   {k}: {v:.2f}")

    # 3. Inference performance
    print("\n" + "=" * 60)
    print("3. Inference Performance (Baseline)")
    print("=" * 60)
    # Measure inference speed
    sample = eval_dataset[0]
    context = sample["context_frames"].unsqueeze(0).to(device)
    actions = sample["context_actions"].unsqueeze(0).to(device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                model.generate_next_frame(
                    context_frames=context,
                    context_actions=actions,
                    memory_state=None,
                    num_inference_steps=4,
                )

    torch.cuda.synchronize()
    latencies = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                model.generate_next_frame(
                    context_frames=context,
                    context_actions=actions,
                    memory_state=None,
                    num_inference_steps=4,
                )
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    avg_latency = np.mean(latencies)
    results["performance"] = {
        "inference_fps": float(1.0 / avg_latency) if avg_latency > 0 else 0,
        "latency_ms": float(avg_latency * 1000),
        "latency_std_ms": float(np.std(latencies) * 1000),
    }
    print(f"   Inference FPS: {results['performance']['inference_fps']:.1f}")
    print(f"   Latency: {results['performance']['latency_ms']:.1f} ms")

    # 4. Compare with memory model results
    memory_results_path = os.path.join(project_dir, "results", "evaluation_results.json")
    if os.path.exists(memory_results_path):
        with open(memory_results_path, "r") as f:
            memory_results = json.load(f)

        print("\n" + "=" * 60)
        print("COMPARISON: Baseline vs Memory-Augmented")
        print("=" * 60)

        results["comparison"] = {}

        # Teacher-forcing PSNR
        mem_psnr = memory_results.get("teacher_forcing", {}).get("psnr_mean", 0)
        base_psnr = tf_results.get("psnr_mean", 0)
        diff = base_psnr - mem_psnr
        print(f"  Teacher-Forcing PSNR:")
        print(f"    Baseline:  {base_psnr:.2f}")
        print(f"    Memory:    {mem_psnr:.2f}")
        print(f"    Delta:     {diff:+.2f} {'(memory better)' if diff < 0 else '(baseline better)'}")
        results["comparison"]["tf_psnr_baseline"] = base_psnr
        results["comparison"]["tf_psnr_memory"] = mem_psnr
        results["comparison"]["tf_psnr_delta"] = diff

        # Teacher-forcing LPIPS
        mem_lpips = memory_results.get("teacher_forcing", {}).get("lpips_mean", None)
        base_lpips = tf_results.get("lpips_mean", None)
        if mem_lpips is not None and base_lpips is not None:
            diff_lpips = base_lpips - mem_lpips
            print(f"  Teacher-Forcing LPIPS:")
            print(f"    Baseline:  {base_lpips:.4f}")
            print(f"    Memory:    {mem_lpips:.4f}")
            print(f"    Delta:     {diff_lpips:+.4f} {'(memory better)' if diff_lpips > 0 else '(baseline better)'}")
            results["comparison"]["tf_lpips_baseline"] = base_lpips
            results["comparison"]["tf_lpips_memory"] = mem_lpips
            results["comparison"]["tf_lpips_delta"] = diff_lpips

        # Inference FPS
        mem_fps = memory_results.get("performance", {}).get("inference_fps", 0)
        base_fps = results["performance"]["inference_fps"]
        print(f"  Inference FPS:")
        print(f"    Baseline:  {base_fps:.1f}")
        print(f"    Memory:    {mem_fps:.1f}")
        results["comparison"]["fps_baseline"] = base_fps
        results["comparison"]["fps_memory"] = mem_fps

    # Save results
    results_path = os.path.join(project_dir, "results", "baseline_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    for section, section_results in results.items():
        print(f"\n{section}:")
        if isinstance(section_results, dict):
            for k, v in section_results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    print("\nBaseline evaluation complete!")


if __name__ == "__main__":
    main()
