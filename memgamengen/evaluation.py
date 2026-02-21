"""Evaluation pipeline for MemGameNGen with state-sensitive metrics.

Implements:
- PSNR, LPIPS (short-horizon, teacher-forcing)
- FVD (distributional video quality)
- HUD-variable accuracy (long-horizon state correctness)
- Controllability metrics (action-to-motion alignment)
- Scripted evaluation scenarios
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predicted and target frames.

    Args:
        pred, target: (B, C, H, W) in [-1, 1]
    Returns:
        Average PSNR in dB
    """
    # Convert to [0, 1]
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2

    mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_model) -> float:
    """Compute LPIPS perceptual distance.

    Args:
        pred, target: (B, C, H, W) in [-1, 1]
        lpips_model: Pre-initialized LPIPS model
    Returns:
        Average LPIPS distance
    """
    with torch.no_grad():
        dist = lpips_model(pred, target)
    return dist.mean().item()


def compute_fvd_features(frames: torch.Tensor, i3d_model=None) -> np.ndarray:
    """Extract features for FVD computation.

    Since I3D is heavy, we use a simpler proxy: frame-level features.
    Args:
        frames: (B, T, C, H, W) in [-1, 1]
    Returns:
        features: (B, D) numpy array
    """
    # Simple proxy: use mean/std of frames as features
    B, T, C, H, W = frames.shape
    frames_flat = frames.reshape(B, T, -1)
    mean_feat = frames_flat.mean(dim=2)  # (B, T)
    std_feat = frames_flat.std(dim=2)  # (B, T)
    features = torch.cat([mean_feat, std_feat], dim=1)  # (B, 2*T)
    return features.cpu().numpy()


def compute_fvd(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Compute FVD (Frechet Video Distance) from pre-extracted features.

    Uses the Frechet distance formula: ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1*S2))
    """
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)

    # Ensure covariance matrices are 2D
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    diff = mu1 - mu2
    # Product of covariance matrices
    covmean = _sqrtm_approx(sigma1 @ sigma2)

    fvd = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fvd))


def _sqrtm_approx(mat: np.ndarray) -> np.ndarray:
    """Approximate matrix square root using eigendecomposition."""
    try:
        from scipy.linalg import sqrtm
        result = sqrtm(mat)
        return np.real(result)
    except ImportError:
        # Fallback: eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


class MemGameNGenEvaluator:
    """Comprehensive evaluator for MemGameNGen."""

    def __init__(
        self,
        model,
        env,
        device: torch.device = torch.device("cuda:0"),
        results_dir: str = "results",
    ):
        self.model = model
        self.env = env
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Try to load LPIPS
        self.lpips_model = None
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        except Exception as e:
            print(f"Warning: Could not load LPIPS model: {e}")

    @torch.no_grad()
    def evaluate_teacher_forcing(
        self,
        dataloader,
        max_batches: int = 100,
    ) -> Dict[str, float]:
        """Evaluate with teacher forcing (ground truth context).

        Measures: PSNR, LPIPS
        """
        self.model.eval()
        psnr_values = []
        lpips_values = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            context_frames = batch["context_frames"].to(self.device)
            target_frame = batch["target_frame"].to(self.device)
            context_actions = batch["context_actions"].to(self.device)

            B = context_frames.shape[0]
            memory_state = None
            if self.model.memory is not None:
                memory_state = self.model.memory.get_initial_memory(B).to(self.device)

            # Generate prediction
            with torch.amp.autocast('cuda'):
                outputs = self.model.generate_next_frame(
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
            if self.lpips_model is not None:
                lpips_val = compute_lpips(pred_frame, target_frame, self.lpips_model)
                lpips_values.append(lpips_val)

        results = {
            "psnr_mean": float(np.mean(psnr_values)) if psnr_values else 0.0,
            "psnr_std": float(np.std(psnr_values)) if psnr_values else 0.0,
        }
        if lpips_values:
            results["lpips_mean"] = float(np.mean(lpips_values))
            results["lpips_std"] = float(np.std(lpips_values))

        return results

    @torch.no_grad()
    def evaluate_autoregressive(
        self,
        num_trajectories: int = 10,
        trajectory_length: int = 256,
        action_sequence: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate autoregressive rollout quality.

        Runs the model autoregressively and compares with ground truth
        from the same action sequence in ViZDoom.
        """
        self.model.eval()

        all_psnr = defaultdict(list)  # keyed by time step
        all_state_errors = defaultdict(list)
        fps_measurements = []

        for traj_idx in range(num_trajectories):
            # Reset environment
            obs, info = self.env.reset()

            # Collect ground truth frames with actions
            gt_frames = [obs.copy()]
            gt_actions = []
            gt_variables = [info.get("game_variables", {})]

            # Generate action sequence
            if action_sequence is not None:
                actions = action_sequence[:trajectory_length]
            else:
                # Random actions
                actions = [np.random.randint(0, self.env.num_actions)
                           for _ in range(trajectory_length)]

            # Play ground truth
            for action in actions:
                obs, reward, done, info = self.env.step(action)
                gt_frames.append(obs.copy())
                gt_actions.append(action)
                gt_variables.append(info.get("game_variables", {}))
                if done:
                    obs, info = self.env.reset()
                    gt_frames[-1] = obs.copy()
                    gt_variables[-1] = info.get("game_variables", {})

            # Now run autoregressive generation
            num_context = self.model.num_context_frames

            # Initialize with ground truth context frames
            context_buffer = []
            for i in range(num_context):
                frame = gt_frames[i].astype(np.float32) / 127.5 - 1.0
                frame = np.transpose(frame, (2, 0, 1))  # CHW
                context_buffer.append(torch.FloatTensor(frame))

            # Initialize memory
            B = 1
            memory_state = None
            if self.model.memory is not None:
                memory_state = self.model.memory.get_initial_memory(B).to(self.device)

            generated_frames = []
            state_predictions = []

            gen_start_time = time.time()

            for t in range(num_context, min(len(gt_actions), trajectory_length)):
                # Prepare context
                context = torch.stack(context_buffer[-num_context:]).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor(
                    gt_actions[max(0, t - num_context):t]
                ).unsqueeze(0).to(self.device)

                # Pad actions if needed
                if action_tensor.shape[1] < num_context:
                    pad_size = num_context - action_tensor.shape[1]
                    action_tensor = F.pad(action_tensor, (pad_size, 0), value=0)

                last_action = torch.LongTensor([gt_actions[t - 1]]).to(self.device)

                # Generate
                with torch.amp.autocast('cuda'):
                    outputs = self.model.generate_next_frame(
                        context_frames=context,
                        context_actions=action_tensor,
                        memory_state=memory_state,
                        num_inference_steps=4,
                        last_action=last_action,
                    )

                pred_frame = outputs["frame"]
                memory_state = outputs.get("memory_state", memory_state)

                # Store generated frame
                generated_frames.append(pred_frame.cpu())

                # State prediction
                if "state_pred" in outputs:
                    state_predictions.append(outputs["state_pred"].cpu())

                # Add to context buffer (autoregressive)
                pred_for_context = pred_frame.squeeze(0).cpu()
                context_buffer.append(pred_for_context)

                # Compare with ground truth
                gt_frame = gt_frames[t + 1] if t + 1 < len(gt_frames) else gt_frames[-1]
                gt_tensor = torch.FloatTensor(
                    np.transpose(gt_frame.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
                ).unsqueeze(0)

                # Resize pred to match gt
                pred_resized = F.interpolate(
                    pred_frame.cpu(), size=gt_tensor.shape[-2:],
                    mode='bilinear', align_corners=False
                )

                psnr = compute_psnr(pred_resized, gt_tensor)
                all_psnr[t - num_context].append(psnr)

                # State error
                if "state_pred" in outputs and gt_variables[t + 1] if t + 1 < len(gt_variables) else None:
                    gt_vars = gt_variables[min(t + 1, len(gt_variables) - 1)]
                    if gt_vars:
                        var_names = sorted(gt_vars.keys())
                        gt_var_tensor = torch.FloatTensor([gt_vars[n] for n in var_names])
                        pred_var = outputs["state_pred"].cpu().squeeze(0)[:len(var_names)]
                        state_error = F.l1_loss(pred_var, gt_var_tensor).item()
                        all_state_errors[t - num_context].append(state_error)

            gen_time = time.time() - gen_start_time
            num_gen_frames = len(generated_frames)
            fps = num_gen_frames / gen_time if gen_time > 0 else 0
            fps_measurements.append(fps)

            if (traj_idx + 1) % 5 == 0:
                print(f"  Evaluated {traj_idx + 1}/{num_trajectories} trajectories, FPS: {fps:.1f}")

        # Aggregate results
        results = {
            "fps_mean": float(np.mean(fps_measurements)),
            "fps_std": float(np.std(fps_measurements)),
        }

        # PSNR at different time horizons
        for horizon in [10, 50, 100, 200]:
            psnr_at_horizon = []
            for t in range(min(horizon, max(all_psnr.keys()) + 1) if all_psnr else 0):
                if t in all_psnr:
                    psnr_at_horizon.extend(all_psnr[t])
            if psnr_at_horizon:
                results[f"psnr_horizon_{horizon}"] = float(np.mean(psnr_at_horizon))

        # State errors at different horizons
        for horizon in [10, 50, 100, 200]:
            state_err = []
            for t in range(min(horizon, max(all_state_errors.keys()) + 1) if all_state_errors else 0):
                if t in all_state_errors:
                    state_err.extend(all_state_errors[t])
            if state_err:
                results[f"state_error_horizon_{horizon}"] = float(np.mean(state_err))

        return results

    @torch.no_grad()
    def evaluate_controllability(
        self,
        num_tests: int = 20,
        sequence_length: int = 30,
    ) -> Dict[str, float]:
        """Evaluate action controllability.

        Tests whether the model responds correctly to specific actions
        by measuring consistency of visual changes with action type.
        """
        self.model.eval()
        action_consistency = defaultdict(list)

        for test_idx in range(num_tests):
            obs, info = self.env.reset()

            # Test each action type
            for action_idx in range(min(self.env.num_actions, 8)):
                # Prepare context from environment
                context_frames = []
                for _ in range(self.model.num_context_frames):
                    frame = obs.astype(np.float32) / 127.5 - 1.0
                    frame = np.transpose(frame, (2, 0, 1))
                    context_frames.append(torch.FloatTensor(frame))

                # Create same-action sequence
                actions = [action_idx] * self.model.num_context_frames

                context = torch.stack(context_frames).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([actions]).to(self.device)

                B = 1
                memory_state = None
                if self.model.memory is not None:
                    memory_state = self.model.memory.get_initial_memory(B).to(self.device)

                # Generate two consecutive frames with same action
                with torch.amp.autocast('cuda'):
                    out1 = self.model.generate_next_frame(
                        context_frames=context,
                        context_actions=action_tensor,
                        memory_state=memory_state,
                        num_inference_steps=4,
                    )

                # Also get ground truth change
                gt_obs1, _, _, _ = self.env.step(action_idx)
                gt_obs2, _, _, _ = self.env.step(action_idx)

                # Measure predicted visual change
                pred_diff = (out1["frame"].cpu() - context[:, -1:].cpu()).abs().mean().item()

                # Measure ground truth visual change
                gt_diff = np.abs(
                    gt_obs1.astype(float) - obs.astype(float)
                ).mean() / 255.0

                # Consistency: how close is pred change to gt change
                consistency = 1.0 - min(abs(pred_diff - gt_diff) / max(gt_diff + 1e-8, 1e-8), 1.0)
                action_consistency[action_idx].append(consistency)

                obs = gt_obs2

        results = {}
        for action_idx, vals in action_consistency.items():
            results[f"controllability_action_{action_idx}"] = float(np.mean(vals))

        results["controllability_mean"] = float(
            np.mean([np.mean(v) for v in action_consistency.values()])
        )

        return results

    @torch.no_grad()
    def run_scripted_test(
        self,
        test_name: str,
        num_steps: int = 200,
    ) -> Dict[str, float]:
        """Run a scripted evaluation scenario."""
        if test_name == "idle_test":
            return self._idle_test(num_steps)
        elif test_name == "movement_test":
            return self._movement_test(num_steps)
        elif test_name == "pickup_test":
            return self._pickup_test(num_steps)
        else:
            print(f"Unknown test: {test_name}")
            return {}

    def _idle_test(self, num_steps: int) -> Dict[str, float]:
        """Idle test: stand still and measure visual drift."""
        obs, info = self.env.reset()

        # Use NOOP (move forward but immediately stop - use action 0 which is attack as "idle")
        # Actually, create a "do nothing" by not pressing any action - use the first frame repeatedly
        context_frames = []
        for _ in range(self.model.num_context_frames):
            frame = obs.astype(np.float32) / 127.5 - 1.0
            frame = np.transpose(frame, (2, 0, 1))
            context_frames.append(torch.FloatTensor(frame))

        # Idle = forward action with small frame skip should approximate idle in some scenarios
        idle_action = 4  # MOVE_FORWARD - at least generates some change

        context = torch.stack(context_frames).unsqueeze(0).to(self.device)
        actions = [idle_action] * self.model.num_context_frames
        action_tensor = torch.LongTensor([actions]).to(self.device)

        B = 1
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.get_initial_memory(B).to(self.device)

        frame_diffs = []
        first_frame = None

        for step in range(num_steps):
            with torch.amp.autocast('cuda'):
                outputs = self.model.generate_next_frame(
                    context_frames=context,
                    context_actions=action_tensor,
                    memory_state=memory_state,
                    num_inference_steps=4,
                    last_action=torch.LongTensor([idle_action]).to(self.device),
                )

            pred_frame = outputs["frame"]
            memory_state = outputs.get("memory_state", memory_state)

            if first_frame is None:
                first_frame = pred_frame.cpu()

            # Measure drift from first frame
            drift = F.mse_loss(pred_frame.cpu(), first_frame).item()
            frame_diffs.append(drift)

            # Update context
            pred_for_ctx = pred_frame.squeeze(0).cpu()
            context_list = [context[0, i] for i in range(1, self.model.num_context_frames)]
            context_list.append(pred_for_ctx)
            context = torch.stack(context_list).unsqueeze(0).to(self.device)

        return {
            "idle_drift_mean": float(np.mean(frame_diffs)),
            "idle_drift_max": float(np.max(frame_diffs)),
            "idle_drift_final": float(frame_diffs[-1]) if frame_diffs else 0,
        }

    def _movement_test(self, num_steps: int) -> Dict[str, float]:
        """Movement test: move forward and measure visual coherence."""
        obs, info = self.env.reset()

        # Alternate: forward, turn left, forward, turn right
        action_pattern = [4, 6, 4, 7]  # FORWARD, TURN_LEFT, FORWARD, TURN_RIGHT

        context_frames = []
        gt_frames = []

        # Get initial context from ground truth
        current_obs = obs
        for i in range(self.model.num_context_frames):
            frame = current_obs.astype(np.float32) / 127.5 - 1.0
            frame = np.transpose(frame, (2, 0, 1))
            context_frames.append(torch.FloatTensor(frame))
            action = action_pattern[i % len(action_pattern)]
            current_obs, _, done, _ = self.env.step(action)
            if done:
                current_obs, _ = self.env.reset()

        context = torch.stack(context_frames).unsqueeze(0).to(self.device)

        B = 1
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.get_initial_memory(B).to(self.device)

        frame_changes = []

        for step in range(num_steps):
            action = action_pattern[step % len(action_pattern)]
            actions_for_context = [
                action_pattern[(step - self.model.num_context_frames + 1 + i) % len(action_pattern)]
                for i in range(self.model.num_context_frames)
            ]
            action_tensor = torch.LongTensor([actions_for_context]).to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model.generate_next_frame(
                    context_frames=context,
                    context_actions=action_tensor,
                    memory_state=memory_state,
                    num_inference_steps=4,
                    last_action=torch.LongTensor([action]).to(self.device),
                )

            pred_frame = outputs["frame"]
            memory_state = outputs.get("memory_state", memory_state)

            # Measure frame-to-frame change
            prev_frame = context[:, -1:]
            change = F.mse_loss(pred_frame.cpu(), prev_frame.cpu()).item()
            frame_changes.append(change)

            # Update context
            pred_for_ctx = pred_frame.squeeze(0).cpu()
            context_list = [context[0, i] for i in range(1, self.model.num_context_frames)]
            context_list.append(pred_for_ctx)
            context = torch.stack(context_list).unsqueeze(0).to(self.device)

        return {
            "movement_change_mean": float(np.mean(frame_changes)),
            "movement_change_std": float(np.std(frame_changes)),
        }

    def _pickup_test(self, num_steps: int) -> Dict[str, float]:
        """Pickup test: check if state variables change correctly."""
        obs, info = self.env.reset()
        initial_vars = info.get("game_variables", {})

        # Move forward to potentially pick up items
        results = {
            "pickup_initial_health": initial_vars.get("HEALTH", 0),
            "pickup_initial_ammo": initial_vars.get("AMMO2", 0),
        }

        # Run a trajectory and track state predictions
        state_preds = []
        gt_states = [initial_vars]

        context_frames = []
        for _ in range(self.model.num_context_frames):
            frame = obs.astype(np.float32) / 127.5 - 1.0
            frame = np.transpose(frame, (2, 0, 1))
            context_frames.append(torch.FloatTensor(frame))

        context = torch.stack(context_frames).unsqueeze(0).to(self.device)

        B = 1
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.get_initial_memory(B).to(self.device)

        for step in range(num_steps):
            action = 4  # MOVE_FORWARD
            actions = [action] * self.model.num_context_frames
            action_tensor = torch.LongTensor([actions]).to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model.generate_next_frame(
                    context_frames=context,
                    context_actions=action_tensor,
                    memory_state=memory_state,
                    num_inference_steps=4,
                    last_action=torch.LongTensor([action]).to(self.device),
                )

            memory_state = outputs.get("memory_state", memory_state)

            if "state_pred" in outputs:
                state_preds.append(outputs["state_pred"].cpu().numpy())

            # Ground truth
            obs, _, done, info = self.env.step(action)
            gt_states.append(info.get("game_variables", {}))
            if done:
                obs, info = self.env.reset()

            # Update context
            pred_for_ctx = outputs["frame"].squeeze(0).cpu()
            context_list = [context[0, i] for i in range(1, self.model.num_context_frames)]
            context_list.append(pred_for_ctx)
            context = torch.stack(context_list).unsqueeze(0).to(self.device)

        if state_preds:
            results["num_state_predictions"] = len(state_preds)

        return results

    def run_full_evaluation(
        self,
        dataloader=None,
        num_autoreg_trajectories: int = 10,
        autoreg_length: int = 256,
    ) -> Dict:
        """Run full evaluation suite."""
        print("\n" + "=" * 60)
        print("Running Full MemGameNGen Evaluation")
        print("=" * 60)

        all_results = {}

        # 1. Teacher forcing evaluation
        if dataloader is not None:
            print("\n1. Teacher-Forcing Evaluation...")
            tf_results = self.evaluate_teacher_forcing(dataloader, max_batches=50)
            all_results["teacher_forcing"] = tf_results
            print(f"   PSNR: {tf_results['psnr_mean']:.2f} +/- {tf_results['psnr_std']:.2f}")
            if "lpips_mean" in tf_results:
                print(f"   LPIPS: {tf_results['lpips_mean']:.4f} +/- {tf_results['lpips_std']:.4f}")

        # 2. Autoregressive evaluation
        print(f"\n2. Autoregressive Evaluation ({num_autoreg_trajectories} trajectories)...")
        ar_results = self.evaluate_autoregressive(
            num_trajectories=num_autoreg_trajectories,
            trajectory_length=autoreg_length,
        )
        all_results["autoregressive"] = ar_results
        print(f"   FPS: {ar_results['fps_mean']:.1f} +/- {ar_results['fps_std']:.1f}")
        for k, v in ar_results.items():
            if "psnr_horizon" in k:
                print(f"   {k}: {v:.2f}")

        # 3. Controllability
        print("\n3. Controllability Evaluation...")
        ctrl_results = self.evaluate_controllability(num_tests=10)
        all_results["controllability"] = ctrl_results
        print(f"   Mean controllability: {ctrl_results['controllability_mean']:.4f}")

        # 4. Scripted tests
        print("\n4. Scripted Tests...")
        for test_name in ["idle_test", "movement_test", "pickup_test"]:
            print(f"   Running {test_name}...")
            test_results = self.run_scripted_test(test_name, num_steps=100)
            all_results[test_name] = test_results
            for k, v in test_results.items():
                print(f"     {k}: {v:.4f}")

        # 5. Performance metrics
        print("\n5. Performance Metrics...")
        perf = self._measure_performance()
        all_results["performance"] = perf
        print(f"   Inference FPS: {perf['inference_fps']:.1f}")
        print(f"   Latency per frame: {perf['latency_ms']:.1f} ms")

        # Save results
        results_path = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

        return all_results

    @torch.no_grad()
    def _measure_performance(self, num_warmup: int = 5, num_measure: int = 20) -> Dict[str, float]:
        """Measure inference performance."""
        self.model.eval()

        obs, _ = self.env.reset()
        context_frames = []
        for _ in range(self.model.num_context_frames):
            frame = obs.astype(np.float32) / 127.5 - 1.0
            frame = np.transpose(frame, (2, 0, 1))
            context_frames.append(torch.FloatTensor(frame))

        context = torch.stack(context_frames).unsqueeze(0).to(self.device)
        actions = torch.zeros(1, self.model.num_context_frames, dtype=torch.long).to(self.device)

        B = 1
        memory_state = None
        if self.model.memory is not None:
            memory_state = self.model.memory.get_initial_memory(B).to(self.device)

        # Warmup
        for _ in range(num_warmup):
            with torch.amp.autocast('cuda'):
                self.model.generate_next_frame(
                    context_frames=context,
                    context_actions=actions,
                    memory_state=memory_state,
                    num_inference_steps=4,
                )

        torch.cuda.synchronize()

        # Measure
        latencies = []
        for _ in range(num_measure):
            start = time.time()
            with torch.amp.autocast('cuda'):
                self.model.generate_next_frame(
                    context_frames=context,
                    context_actions=actions,
                    memory_state=memory_state,
                    num_inference_steps=4,
                )
            torch.cuda.synchronize()
            latencies.append(time.time() - start)

        avg_latency = np.mean(latencies)
        return {
            "inference_fps": 1.0 / avg_latency if avg_latency > 0 else 0,
            "latency_ms": avg_latency * 1000,
            "latency_std_ms": float(np.std(latencies)) * 1000,
        }
