"""Step 5: Real-time interactive inference demo.

Runs MemGameNGen in a loop, generating frames from user keyboard actions
or a scripted action sequence, and saving output as a video.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import time
import cv2
import argparse

from memgamengen.doom_env import DoomEnvironment
from memgamengen.diffusion_model import MemGameNGen
from memgamengen.data_collector import DoomTrajectoryDataset
from memgamengen.trainer import MemGameNGenTrainer


def load_model(checkpoint_path: str, device: torch.device, project_dir: str) -> MemGameNGen:
    """Load trained MemGameNGen model using the trainer checkpoint loader."""
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

    # Use trainer to load (handles LoRA properly)
    dummy_dataset = DoomTrajectoryDataset(
        data_dir=os.path.join(project_dir, "data", "trajectories"),
        num_context_frames=4,
    )
    trainer = MemGameNGenTrainer(
        model=model,
        train_dataset=dummy_dataset,
        device=device,
        max_train_steps=1,
    )
    trainer.load_checkpoint(checkpoint_path)
    model = trainer.model

    model.to(device)
    model.eval()
    return model


def run_scripted_demo(
    model: MemGameNGen,
    env: DoomEnvironment,
    device: torch.device,
    output_path: str,
    num_frames: int = 300,
    action_pattern: str = "mixed",
):
    """Run a scripted demo and save the output as a video.

    Generates frames side-by-side: ground truth (ViZDoom) | predicted (MemGameNGen).
    """
    # Define action patterns
    patterns = {
        "forward": [4] * num_frames,  # MOVE_FORWARD
        "circle": [4, 4, 6, 6, 4, 4, 7, 7] * (num_frames // 8 + 1),  # Forward + turns
        "combat": [4, 4, 4, 0, 0, 6, 4, 4, 0, 7] * (num_frames // 10 + 1),  # Move + attack + turn
        "mixed": [],
    }

    if action_pattern == "mixed":
        # Generate a diverse action sequence
        rng = np.random.RandomState(42)
        segments = [
            ([4] * 20, "forward"),  # Walk forward
            ([6] * 10, "turn_left"),  # Turn left
            ([4] * 15, "forward"),
            ([0] * 5, "attack"),  # Attack
            ([7] * 10, "turn_right"),
            ([4] * 20, "forward"),
            ([5] * 10, "backward"),  # Move backward
            ([6] * 8, "turn_left"),
            ([4] * 25, "forward"),
            ([0, 0, 4, 4, 0, 0] * 5, "strafe_attack"),
        ]
        for seg_actions, _ in segments:
            patterns["mixed"].extend(seg_actions)
        # Fill remainder with random
        while len(patterns["mixed"]) < num_frames:
            patterns["mixed"].append(rng.randint(0, 8))

    actions = patterns[action_pattern][:num_frames]

    # Initialize environment
    obs, info = env.reset()
    num_context = model.num_context_frames

    # Collect initial context from ground truth
    context_frames = []
    gt_frames = [obs.copy()]
    frame = obs.astype(np.float32) / 127.5 - 1.0
    frame_chw = np.transpose(frame, (2, 0, 1))
    context_frames.append(torch.FloatTensor(frame_chw))

    for i in range(1, num_context):
        obs, _, done, _ = env.step(actions[i])
        if done:
            obs, _ = env.reset()
        gt_frames.append(obs.copy())
        frame = obs.astype(np.float32) / 127.5 - 1.0
        frame_chw = np.transpose(frame, (2, 0, 1))
        context_frames.append(torch.FloatTensor(frame_chw))

    # Initialize memory
    memory_state = None
    if model.memory is not None:
        memory_state = model.memory.get_initial_memory(1).to(device)

    # Set up video writer
    h, w = obs.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = 10  # Output video FPS
    video = cv2.VideoWriter(output_path, fourcc, fps_out, (w * 2, h))

    # Action names for overlay
    action_names = [
        "ATTACK", "USE", "LEFT", "RIGHT",
        "FORWARD", "BACKWARD", "TURN_L", "TURN_R",
    ]

    print(f"Generating {num_frames} frames...")
    print(f"Action pattern: {action_pattern}")

    gen_times = []
    pred_frames = []

    for step in range(num_context, num_frames):
        # Prepare context
        context = torch.stack(context_frames[-num_context:]).unsqueeze(0).to(device)

        # Get action context
        action_indices = actions[max(0, step - num_context):step]
        while len(action_indices) < num_context:
            action_indices = [0] + action_indices
        action_tensor = torch.LongTensor([action_indices]).to(device)
        last_action = torch.LongTensor([actions[step - 1]]).to(device)

        # Generate
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model.generate_next_frame(
                context_frames=context,
                context_actions=action_tensor,
                memory_state=memory_state,
                num_inference_steps=4,
                last_action=last_action,
            )
        gen_time = time.time() - t0
        gen_times.append(gen_time)

        pred_frame = outputs["frame"]
        memory_state = outputs.get("memory_state", memory_state)

        # Convert prediction to numpy image
        pred_np = pred_frame.squeeze(0).cpu().clamp(-1, 1)
        pred_np = ((pred_np + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()

        # Resize prediction to match ground truth
        pred_resized = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_frames.append(pred_resized)

        # Get ground truth
        obs, _, done, info = env.step(actions[step])
        if done:
            obs, _ = env.reset()
        gt_frames.append(obs.copy())

        # Update context buffer (autoregressive)
        pred_for_ctx = pred_frame.squeeze(0).cpu()
        context_frames.append(pred_for_ctx)

        # Create side-by-side frame
        gt_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_resized, cv2.COLOR_RGB2BGR)

        # Add labels
        cv2.putText(gt_bgr, "Ground Truth", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(pred_bgr, "MemGameNGen", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add action and state info
        act_name = action_names[actions[step]] if actions[step] < len(action_names) else str(actions[step])
        cv2.putText(gt_bgr, f"Action: {act_name}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(gt_bgr, f"Step: {step}", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        if "state_pred" in outputs:
            state = outputs["state_pred"].cpu().squeeze(0)
            cv2.putText(pred_bgr, f"HP: {state[0]:.0f} Ammo: {state[1]:.0f}", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        fps = 1.0 / gen_time if gen_time > 0 else 0
        cv2.putText(pred_bgr, f"FPS: {fps:.1f}", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # Combine
        combined = np.concatenate([gt_bgr, pred_bgr], axis=1)
        video.write(combined)

        if (step + 1) % 50 == 0:
            avg_fps = 1.0 / np.mean(gen_times[-50:])
            print(f"  Step {step + 1}/{num_frames}, Avg FPS: {avg_fps:.1f}")

    video.release()

    # Print summary
    avg_gen_time = np.mean(gen_times)
    avg_fps = 1.0 / avg_gen_time
    print(f"\nDemo complete!")
    print(f"  Total frames generated: {len(gen_times)}")
    print(f"  Average generation time: {avg_gen_time * 1000:.1f} ms/frame")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Video saved to: {output_path}")

    return {
        "num_frames": len(gen_times),
        "avg_fps": float(avg_fps),
        "avg_latency_ms": float(avg_gen_time * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description="MemGameNGen Interactive Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Number of frames to generate")
    parser.add_argument("--pattern", type=str, default="mixed",
                        choices=["forward", "circle", "combat", "mixed"],
                        help="Action pattern to use")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Default checkpoint path
    checkpoint_path = args.checkpoint or os.path.join(
        project_dir, "checkpoints", "diffusion", "memgamengen_final.pt"
    )

    # Default output path
    output_path = args.output or os.path.join(
        project_dir, "results", "demo_output.mp4"
    )

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device, project_dir)
    print(f"Model loaded. Trainable params: {model.count_trainable_params():,}")

    # Create environment
    print("Creating ViZDoom environment...")
    env = DoomEnvironment(
        scenario="basic",
        resolution=(160, 120),
        frame_skip=4,
        visible=False,
        game_variables=["HEALTH", "AMMO2"],
    )

    # Run demo
    stats = run_scripted_demo(
        model=model,
        env=env,
        device=device,
        output_path=output_path,
        num_frames=args.num_frames,
        action_pattern=args.pattern,
    )

    env.close()

    # Save demo stats
    import json
    stats_path = os.path.join(project_dir, "results", "demo_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
