"""Phase 5: Generate publication-quality figures from experiment results.

Reads results JSON files from the results directory and produces four figures.
Also uploads figures to wandb as artifacts.

Usage:
    python plot_results.py
    python plot_results.py --results_dir /path/to/results --no_wandb
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    FIGURES_DIR,
    NUM_TOKENS,
    PATIENT_IDS,
    RESULTS_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

# --- Publication style ---
PALETTE = ["#3A86FF", "#FF006E", "#FB5607", "#8338EC", "#06D6A0"]
LIGHT_PALETTE = ["#9FC5FF", "#FF80B7", "#FD9B63", "#C19EF6", "#82EBD0"]
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#2B2B2B"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "white",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "sans-serif",
})


def load_single_patient_results() -> dict[str, float]:
    """Load individual cartridge evaluation results (k=1)."""
    accs = {}
    for pid in PATIENT_IDS:
        path = os.path.join(RESULTS_DIR, f"eval_{pid}.json")
        if os.path.exists(path):
            df = pd.read_json(path)
            accs[pid] = df["is_correct"].mean()
    return accs


def load_canonical_stack_results() -> dict[int, dict]:
    """Load canonical ordering results for k=2..5."""
    results = {}
    for k in range(2, len(PATIENT_IDS) + 1):
        ordering_str = "_".join(PATIENT_IDS[:k])
        path = os.path.join(RESULTS_DIR, f"eval_{ordering_str}.json")
        if os.path.exists(path):
            df = pd.read_json(path)
            per_patient = df.groupby("question_patient")["is_correct"].mean().to_dict()
            results[k] = {
                "overall": df["is_correct"].mean(),
                "per_patient": per_patient,
            }
    return results


def load_permutation_results() -> pd.DataFrame:
    """Load all permutation evaluation results."""
    files = glob.glob(os.path.join(RESULTS_DIR, "permutation_results*.json"))
    all_results = []
    for f in files:
        with open(f) as fh:
            all_results.extend(json.load(fh))
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def figure1_per_patient_accuracy(
    single_accs: dict[str, float],
    stack_results: dict[int, dict],
):
    """Figure 1: Per-patient accuracy — single cartridge vs 5-patient stack."""
    fig, ax = plt.subplots(figsize=(8, 5))

    k5 = stack_results.get(len(PATIENT_IDS), {})
    k5_per_patient = k5.get("per_patient", {})

    x = np.arange(len(PATIENT_IDS))
    width = 0.35

    single_vals = [single_accs.get(pid, 0) for pid in PATIENT_IDS]
    stack_vals = [k5_per_patient.get(pid, 0) for pid in PATIENT_IDS]

    bars1 = ax.bar(
        x - width / 2, single_vals, width,
        label="Single cartridge", color=PALETTE, edgecolor="white", linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, stack_vals, width,
        label="5-patient stack", color=LIGHT_PALETTE, edgecolor="white", linewidth=0.8,
    )

    # Mean lines
    if single_vals:
        ax.axhline(
            np.mean(single_vals), color=TEXT_COLOR, linestyle="--",
            alpha=0.4, linewidth=1, label=f"Single mean ({np.mean(single_vals):.1%})",
        )
    if stack_vals and any(v > 0 for v in stack_vals):
        ax.axhline(
            np.mean(stack_vals), color=PALETTE[3], linestyle=":",
            alpha=0.5, linewidth=1, label=f"Stack mean ({np.mean(stack_vals):.1%})",
        )

    ax.set_xlabel("Patient")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Patient Accuracy: Single Cartridge vs. 5-Patient Stack")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(PATIENT_IDS))])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right", framealpha=0.9)

    return fig


def figure2_accuracy_vs_stack_size(
    single_accs: dict[str, float],
    stack_results: dict[int, dict],
):
    """Figure 2: Average accuracy vs stack size (canonical ordering)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ks = []
    means = []

    # k=1: average of single cartridge results
    if single_accs:
        ks.append(1)
        means.append(np.mean(list(single_accs.values())))

    for k in sorted(stack_results.keys()):
        ks.append(k)
        means.append(stack_results[k]["overall"])

    ax.plot(ks, means, "o-", color=PALETTE[0], linewidth=2.5, markersize=8, zorder=5)

    # Fill area under curve
    ax.fill_between(ks, means, alpha=0.1, color=PALETTE[0])

    ax.set_xlabel("Number of Stacked Patients")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Accuracy Scaling with Stack Size (Canonical Ordering)")
    ax.set_xticks(range(1, len(PATIENT_IDS) + 1))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Annotate points
    for k, m in zip(ks, means):
        ax.annotate(
            f"{m:.1%}", (k, m), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9, fontweight="bold",
        )

    return fig


def figure3_permutation_scatter(
    single_accs: dict[str, float],
    perm_df: pd.DataFrame,
):
    """Figure 3: Average accuracy vs stack size with all permutation results."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # k=1 (no permutations, just individual results)
    if single_accs:
        single_vals = list(single_accs.values())
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(single_vals))
        ax.scatter(
            1 + jitter, single_vals,
            color=PALETTE[0], alpha=0.6, s=30, zorder=4,
        )
        ax.scatter(
            [1], [np.mean(single_vals)],
            color=PALETTE[0], s=120, marker="D", edgecolors="white",
            linewidth=1.5, zorder=6, label="Mean",
        )

    # k=2..5 from permutation results
    if not perm_df.empty:
        for k in sorted(perm_df["stack_size"].unique()):
            k_df = perm_df[perm_df["stack_size"] == k]
            accs = k_df["overall_accuracy"].values
            jitter = np.random.RandomState(42 + k).uniform(-0.2, 0.2, len(accs))

            color_idx = min(k - 1, len(PALETTE) - 1)
            ax.scatter(
                k + jitter, accs,
                color=PALETTE[color_idx], alpha=0.35, s=20, zorder=3,
            )

            # Box plot overlay
            bp = ax.boxplot(
                accs, positions=[k], widths=0.3, patch_artist=True,
                showfliers=False, zorder=5,
                boxprops=dict(facecolor=PALETTE[color_idx], alpha=0.3),
                medianprops=dict(color=TEXT_COLOR, linewidth=1.5),
                whiskerprops=dict(color=TEXT_COLOR, alpha=0.5),
                capprops=dict(color=TEXT_COLOR, alpha=0.5),
            )

            # Mean marker
            ax.scatter(
                [k], [accs.mean()],
                color=PALETTE[color_idx], s=120, marker="D",
                edgecolors="white", linewidth=1.5, zorder=6,
            )

    # Connect means
    all_ks = []
    all_means = []
    if single_accs:
        all_ks.append(1)
        all_means.append(np.mean(list(single_accs.values())))
    if not perm_df.empty:
        for k in sorted(perm_df["stack_size"].unique()):
            all_ks.append(k)
            all_means.append(perm_df[perm_df["stack_size"] == k]["overall_accuracy"].mean())
    if all_ks:
        ax.plot(all_ks, all_means, "--", color=TEXT_COLOR, alpha=0.4, linewidth=1.5, zorder=2)

    ax.set_xlabel("Number of Stacked Patients")
    ax.set_ylabel("Average Accuracy per Patient")
    ax.set_title("Accuracy vs. Stack Size (All Permutations)")
    ax.set_xticks(range(1, len(PATIENT_IDS) + 1))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Count annotations
    for k in range(1, len(PATIENT_IDS) + 1):
        if k == 1:
            n = len(single_accs) if single_accs else 0
        elif not perm_df.empty:
            n = len(perm_df[perm_df["stack_size"] == k])
        else:
            n = 0
        if n > 0:
            ax.text(
                k, -0.06, f"n={n}", ha="center", fontsize=8,
                color=TEXT_COLOR, alpha=0.6, transform=ax.get_xaxis_transform(),
            )

    return fig


def figure4_attention_heatmap():
    """Figure 4: Attention distribution heatmap (correct vs incorrect)."""
    # Load the canonical 5-stack results which have attention columns
    ordering_str = "_".join(PATIENT_IDS)
    path = os.path.join(RESULTS_DIR, f"eval_{ordering_str}.json")
    if not os.path.exists(path):
        return None

    df = pd.read_json(path)
    attn_cols = [c for c in df.columns if c.startswith("attn_to_patient_")]
    if not attn_cols:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.12 + 2)), sharey=True)

    for idx, (label, subset) in enumerate([
        ("Correct Answers", df[df["is_correct"]]),
        ("Incorrect Answers", df[~df["is_correct"]]),
    ]):
        ax = axes[idx]
        if len(subset) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        attn_matrix = subset[attn_cols].values
        col_labels = [c.replace("attn_to_", "").replace("patient_0", "P") for c in attn_cols]
        row_labels = [
            f"{r['question_patient'].replace('patient_0', 'P')}"
            for _, r in subset.iterrows()
        ]

        im = ax.imshow(attn_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_xlabel("Cartridge Region")
        ax.set_title(f"{label} (n={len(subset)})")

        if idx == 0:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_ylabel("Question (source patient)")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Attention Fraction")
    fig.suptitle("Attention Distribution: Correct vs. Incorrect", fontsize=14, y=1.02)

    return fig


def save_figure(fig, name: str, upload_to_wandb: bool = True):
    """Save figure as PNG and PDF."""
    if fig is None:
        return
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    plt.close(fig)

    if upload_to_wandb:
        try:
            import wandb
            wandb.log({f"figures/{name}": wandb.Image(os.path.join(FIGURES_DIR, f"{name}.png"))})
        except Exception:
            pass


def main(upload_to_wandb: bool = True):
    single_accs = load_single_patient_results()
    stack_results = load_canonical_stack_results()
    perm_df = load_permutation_results()

    if upload_to_wandb:
        import wandb
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name="stacking_figures",
            tags=["figures", "stacking", "longhealth"],
            config={"num_tokens": NUM_TOKENS},
        )

    fig1 = figure1_per_patient_accuracy(single_accs, stack_results)
    save_figure(fig1, "fig1_per_patient_accuracy", upload_to_wandb)

    fig2 = figure2_accuracy_vs_stack_size(single_accs, stack_results)
    save_figure(fig2, "fig2_accuracy_vs_stack_size", upload_to_wandb)

    fig3 = figure3_permutation_scatter(single_accs, perm_df)
    save_figure(fig3, "fig3_permutation_scatter", upload_to_wandb)

    fig4 = figure4_attention_heatmap()
    save_figure(fig4, "fig4_attention_heatmap", upload_to_wandb)

    if upload_to_wandb:
        wandb.finish()

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if args.results_dir:
        import config
        config.RESULTS_DIR = args.results_dir

    main(upload_to_wandb=not args.no_wandb)
