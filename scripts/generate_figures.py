#!/usr/bin/env python3
"""Generate publication-quality figures from MemGameNGen experimental data.

Reads real training logs and evaluation results, produces PDF figures
suitable for a two-column ICML paper.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as patheffects

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,      # TrueType fonts in PDF
    "ps.fonttype": 42,
})

# Colorblind-friendly palette (Okabe-Ito)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"
C_YELLOW = "#F0E442"
C_BLACK = "#000000"

SINGLE_COL = 3.25   # inches
DOUBLE_COL = 6.75   # inches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "paper", "figures")
LOG_PATH = os.path.join(BASE, "results", "logs", "training_log.json")
EVAL_PATH = os.path.join(BASE, "results", "evaluation_results.json")


def running_average(values, window=11):
    """Compute a centred running average with the given window size."""
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


# ===================================================================
# Figure 1: Training loss curves
# ===================================================================
def make_training_curves(train_data, val_data, out_path):
    steps = np.array([d["step"] for d in train_data])
    diff_loss = np.array([d["diffusion_loss"] for d in train_data])
    state_loss = np.array([d["state_loss"] for d in train_data])

    val_steps = np.array([d["step"] for d in val_data])
    val_loss = np.array([d["val_loss"] for d in val_data])

    # Smooth
    diff_smooth = running_average(diff_loss, window=15)
    state_smooth = running_average(state_loss, window=15)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(SINGLE_COL, 3.8), sharex=True,
        gridspec_kw={"hspace": 0.12}
    )

    # --- Diffusion loss (top) ---
    ax1.plot(steps, diff_loss, color=C_BLUE, alpha=0.18, linewidth=0.6)
    ax1.plot(steps, diff_smooth, color=C_BLUE, linewidth=1.4, label="Diffusion loss")
    ax1.set_ylabel("Diffusion loss")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title("Training loss curves", fontsize=10, pad=4)

    # Overlay validation loss on a twin axis
    ax1v = ax1.twinx()
    ax1v.plot(val_steps, val_loss, "s-", color=C_ORANGE, markersize=3,
              linewidth=1.0, label="Validation loss")
    ax1v.set_ylabel("Validation loss", color=C_ORANGE)
    ax1v.tick_params(axis="y", colors=C_ORANGE)
    ax1v.spines["right"].set_visible(True)
    ax1v.spines["right"].set_color(C_ORANGE)
    ax1v.spines["right"].set_linewidth(0.6)
    ax1v.spines["top"].set_visible(False)
    ax1v.legend(frameon=False, loc="center right")

    # --- State loss (bottom, log-scale) ---
    ax2.plot(steps, state_loss, color=C_GREEN, alpha=0.18, linewidth=0.6)
    ax2.plot(steps, state_smooth, color=C_GREEN, linewidth=1.4, label="State loss")
    ax2.set_yscale("log")
    ax2.set_ylabel("State loss (log scale)")
    ax2.set_xlabel("Training step")
    ax2.legend(frameon=False, loc="upper right")

    fig.align_ylabels([ax1, ax2])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Figure 2: PSNR at different horizons
# ===================================================================
def make_evaluation_metrics(eval_data, out_path):
    ar = eval_data["autoregressive"]
    tf = eval_data["teacher_forcing"]

    horizons = [10, 50, 100, 200]
    psnr_vals = [ar[f"psnr_horizon_{h}"] for h in horizons]
    labels = [str(h) for h in horizons]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(len(horizons))
    bars = ax.bar(x, psnr_vals, width=0.55, color=C_BLUE, edgecolor="white",
                  linewidth=0.5, zorder=3)

    # Teacher-forcing reference line
    ax.axhline(tf["psnr_mean"], color=C_RED, linestyle="--", linewidth=1.0,
               label=f'Teacher-forcing ({tf["psnr_mean"]:.1f} dB)', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Prediction horizon (frames)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Autoregressive PSNR by horizon", fontsize=10, pad=4)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(bottom=14.0)

    # Value labels on bars
    for bar, v in zip(bars, psnr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Figure 3: Controllability per action
# ===================================================================
def make_controllability(eval_data, out_path):
    ctrl = eval_data["controllability"]
    action_names = [
        "ATTACK", "USE", "MOVE\nLEFT", "MOVE\nRIGHT",
        "MOVE\nFWD", "MOVE\nBACK", "TURN\nLEFT", "TURN\nRIGHT"
    ]
    scores = [ctrl[f"controllability_action_{i}"] for i in range(8)]
    mean_score = ctrl["controllability_mean"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    x = np.arange(len(action_names))
    colors = [C_BLUE if s >= mean_score else C_CYAN for s in scores]
    bars = ax.bar(x, scores, width=0.6, color=colors, edgecolor="white",
                  linewidth=0.5, zorder=3)

    ax.axhline(mean_score, color=C_RED, linestyle="--", linewidth=0.9,
               label=f"Mean ({mean_score:.2f})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(action_names, fontsize=7)
    ax.set_ylabel("Controllability score")
    ax.set_title("Per-action controllability", fontsize=10, pad=4)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 0.6)

    # Value labels
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Figure 4: Drift analysis (idle vs movement)
# ===================================================================
def make_drift_analysis(eval_data, out_path):
    idle = eval_data["idle_test"]
    move = eval_data["movement_test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4),
                                    gridspec_kw={"wspace": 0.35})

    # --- Left panel: idle drift ---
    metrics = ["Mean drift", "Max drift", "Final drift"]
    values = [idle["idle_drift_mean"], idle["idle_drift_max"],
              idle["idle_drift_final"]]
    x = np.arange(len(metrics))
    bars1 = ax1.bar(x, values, width=0.5, color=[C_BLUE, C_ORANGE, C_GREEN],
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=8)
    ax1.set_ylabel("Pixel-space drift (L1)")
    ax1.set_title("Idle-action stability", fontsize=10, pad=4)
    ax1.set_ylim(0, 0.25)
    for bar, v in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # --- Right panel: movement response ---
    vals = [move["movement_change_mean"], move["movement_change_std"]]
    labs = ["Mean change", "Std change"]
    x2 = np.arange(len(labs))
    bars2 = ax2.bar(x2, vals, width=0.45, color=[C_PURPLE, C_CYAN],
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labs, fontsize=8)
    ax2.set_ylabel("Pixel-space change (L1)")
    ax2.set_title("Movement response", fontsize=10, pad=4)
    ax2.set_ylim(0, 0.15)
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Figure 5: Architecture block diagram
# ===================================================================
def _box(ax, xy, w, h, text, color, text_color="white", fontsize=7.5,
         rounded=True, alpha=1.0):
    """Draw a rounded rectangle with centred text."""
    x, y = xy
    style = "round,pad=0.02" if rounded else "square,pad=0"
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                         facecolor=color, edgecolor="black",
                         linewidth=0.6, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=3,
            fontweight="bold")
    return (x, y, w, h)


def _arrow(ax, start, end, color="black", style="-|>", lw=0.8,
           connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(start, end,
                          arrowstyle=style, color=color,
                          linewidth=lw, mutation_scale=8,
                          connectionstyle=connectionstyle,
                          zorder=1)
    ax.add_patch(arr)


def make_architecture(out_path):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_aspect("equal")

    # Colors
    c_input = "#4A90D9"
    c_vae = "#2E7D32"
    c_unet = "#C62828"
    c_action = "#E65100"
    c_mem = "#6A1B9A"
    c_state = "#00695C"
    c_out = "#37474F"

    # --- Main path (left to right) ---
    # Input frames
    _box(ax, (0.1, 2.0), 1.3, 0.8, "Input\nFrames", c_input, fontsize=7)
    # VAE Encoder
    _box(ax, (2.0, 2.0), 1.3, 0.8, "VAE\nEncoder", c_vae, fontsize=7)
    # Context latents
    _box(ax, (3.9, 2.55), 1.3, 0.55, "Context\nLatents", c_vae,
         text_color="white", fontsize=6.5)
    # Noisy target
    _box(ax, (3.9, 1.85), 1.3, 0.55, "Noisy\nTarget", c_input,
         text_color="white", fontsize=6.5)
    # Input projection
    _box(ax, (5.8, 2.0), 1.2, 0.8, "Input\nProj", c_unet,
         text_color="white", fontsize=7)
    # UNet
    _box(ax, (7.5, 1.7), 1.4, 1.4, "UNet\n(LoRA)", c_unet,
         text_color="white", fontsize=8)
    # Output
    _box(ax, (9.2, 2.1), 0.7, 0.6, r"$\hat{\epsilon}$", c_out,
         text_color="white", fontsize=9)

    # --- Actions (bottom path) ---
    _box(ax, (2.0, 0.4), 1.3, 0.7, "Actions", c_action, fontsize=7)
    _box(ax, (3.9, 0.4), 1.3, 0.7, "Action\nEmbed", c_action, fontsize=7)
    # Cross-attention label
    ax.text(6.95, 0.95, "cross-attn", fontsize=5.5, ha="center",
            va="center", style="italic", color=c_action)

    # --- Memory (top path) ---
    _box(ax, (3.9, 3.7), 1.3, 0.7, "Memory\n(GRU)", c_mem, fontsize=7)
    ax.text(6.95, 3.5, "cross-attn", fontsize=5.5, ha="center",
            va="center", style="italic", color=c_mem)

    # State head branching from memory
    _box(ax, (6.0, 4.05), 1.1, 0.55, "State\nHead", c_state, fontsize=6.5)

    # --- Arrows: main path ---
    _arrow(ax, (1.4, 2.4), (2.0, 2.4))                   # frames -> VAE
    _arrow(ax, (3.3, 2.65), (3.9, 2.75))                  # VAE -> context
    _arrow(ax, (3.3, 2.2), (3.9, 2.1))                    # VAE -> noisy
    _arrow(ax, (5.2, 2.75), (5.8, 2.55))                  # context -> input proj
    _arrow(ax, (5.2, 2.1), (5.8, 2.25))                   # noisy -> input proj
    _arrow(ax, (7.0, 2.4), (7.5, 2.4))                    # input proj -> UNet
    _arrow(ax, (8.9, 2.4), (9.2, 2.4))                    # UNet -> output

    # --- Arrows: actions ---
    _arrow(ax, (3.3, 0.75), (3.9, 0.75))                  # actions -> embed
    _arrow(ax, (5.2, 0.75), (8.2, 1.7),                   # embed -> UNet bottom
           color=c_action,
           connectionstyle="arc3,rad=-0.15")

    # --- Arrows: memory ---
    _arrow(ax, (5.2, 4.05), (8.2, 3.1),                   # memory -> UNet top
           color=c_mem,
           connectionstyle="arc3,rad=0.15")
    # State head arrow
    _arrow(ax, (5.2, 4.15), (6.0, 4.3),
           color=c_state)

    # Title
    ax.set_title("MemGameNGen Architecture", fontsize=10, pad=6)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading data ...")
    with open(LOG_PATH) as f:
        log_data = json.load(f)
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    train_data = log_data["train_losses"]
    val_data = log_data["val_losses"]

    print("Generating figures:")

    make_training_curves(
        train_data, val_data,
        os.path.join(FIG_DIR, "training_curves.pdf")
    )

    make_evaluation_metrics(
        eval_data,
        os.path.join(FIG_DIR, "evaluation_metrics.pdf")
    )

    make_controllability(
        eval_data,
        os.path.join(FIG_DIR, "controllability.pdf")
    )

    make_drift_analysis(
        eval_data,
        os.path.join(FIG_DIR, "drift_analysis.pdf")
    )

    make_architecture(
        os.path.join(FIG_DIR, "architecture.pdf")
    )

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
