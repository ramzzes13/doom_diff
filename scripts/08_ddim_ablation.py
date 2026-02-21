"""DDIM steps ablation: measure quality and speed at different inference step counts."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import numpy as np
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.trainer import MemGameNGenTrainer
from memgamengen.evaluation import compute_psnr, compute_lpips
import torch.nn.functional as F


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use GPU 7 (has some free memory)
    gpu_id = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (physical GPU {gpu_id})")

    # Create model
    model = MemGameNGen(
        pretrained_model="CompVis/stable-diffusion-v1-4",
        num_actions=8,
        num_context_frames=4,
        memory_enabled=True,
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

    # Convert VAE to fp16 to save memory
    model.vae = model.vae.half()

    # Load checkpoint
    data_dir = os.path.join(project_dir, "data", "trajectories")
    dataset = DoomTrajectoryDataset(data_dir=data_dir, num_context_frames=4)

    trainer = MemGameNGenTrainer(
        model=model,
        train_dataset=dataset,
        device=device,
        max_train_steps=1,
    )
    ckpt_path = os.path.join(project_dir, "checkpoints", "diffusion", "memgamengen_final.pt")
    trainer.load_checkpoint(ckpt_path)
    model = trainer.model
    model.vae = model.vae.half()  # Re-apply after checkpoint load
    model.eval()
    print(f"Loaded model at step {trainer.global_step}")

    # Skip LPIPS to save GPU memory
    lpips_model = None

    # Test different DDIM step counts
    ddim_steps_list = [1, 2, 4, 8, 16]
    results = {}

    from torch.utils.data import DataLoader
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for num_steps in ddim_steps_list:
        print(f"\n{'='*50}")
        print(f"Testing {num_steps} DDIM steps")
        print(f"{'='*50}")

        # Quality: teacher-forcing PSNR/LPIPS
        psnr_values = []
        lpips_values = []
        max_batches = 50

        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break

            context_frames = batch["context_frames"].to(device)
            target_frame = batch["target_frame"].to(device)
            context_actions = batch["context_actions"].to(device)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model.generate_next_frame(
                        context_frames=context_frames,
                        context_actions=context_actions,
                        memory_state=None,
                        num_inference_steps=num_steps,
                    )

            pred_frame = outputs["frame"]
            if pred_frame.shape != target_frame.shape:
                pred_frame = F.interpolate(pred_frame, size=target_frame.shape[-2:],
                                           mode='bilinear', align_corners=False)

            psnr = compute_psnr(pred_frame, target_frame)
            psnr_values.append(psnr)

            if lpips_model is not None:
                lpips_val = compute_lpips(pred_frame, target_frame, lpips_model)
                lpips_values.append(lpips_val)

        # Speed: measure latency
        sample = dataset[0]
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
                        num_inference_steps=num_steps,
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
                        num_inference_steps=num_steps,
                    )
            torch.cuda.synchronize()
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies)
        fps = 1.0 / avg_latency if avg_latency > 0 else 0

        step_results = {
            "num_steps": num_steps,
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "fps": float(fps),
            "latency_ms": float(avg_latency * 1000),
        }
        if lpips_values:
            step_results["lpips_mean"] = float(np.mean(lpips_values))
            step_results["lpips_std"] = float(np.std(lpips_values))

        results[str(num_steps)] = step_results
        print(f"  PSNR: {step_results['psnr_mean']:.2f} ± {step_results['psnr_std']:.2f}")
        if 'lpips_mean' in step_results:
            print(f"  LPIPS: {step_results['lpips_mean']:.4f}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Latency: {avg_latency*1000:.1f} ms")

    # Save results
    results_path = os.path.join(project_dir, "results", "ddim_ablation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"DDIM Steps Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Steps':>6} {'PSNR':>8} {'LPIPS':>8} {'FPS':>8} {'Latency':>10}")
    for num_steps in ddim_steps_list:
        r = results[str(num_steps)]
        lpips_str = f"{r.get('lpips_mean', 0):.3f}" if 'lpips_mean' in r else "N/A"
        print(f"{num_steps:>6} {r['psnr_mean']:>8.2f} {lpips_str:>8} {r['fps']:>8.1f} {r['latency_ms']:>9.1f}ms")


if __name__ == "__main__":
    main()
