"""Generate publication-quality figures for the intrinsic vs extrinsic safety paper.

Extended for multi-algorithm design (PPO, SAC, TD3, Lagrangian PPO) and
passivity analysis.

Figures:
    1. Main results — grouped bar chart (per algorithm, reward × gateway)
    2. Safety metrics — rejection rates and violation counts
    3. Risk-return trade-off scatter plot
    4. Factorial interaction plots (reward_type × gateway)
    5. Boxplots per seed
    6. Percentage reduction waterfall
    7. Cross-algorithm consistency (NEW)
    8. Passivity analysis (NEW)
    9. Lagrangian PPO comparison (NEW)

Usage:
    python scripts/generate_figures.py [--results output/experiments/raw_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

REWARD_LABELS = {"std": "Standard", "risk": "Risk-Adjusted", "sharpe": "Sharpe-Inspired"}
ALGO_LABELS = {"ppo": "PPO", "sac": "SAC", "td3": "TD3", "lag_ppo": "Lag-PPO"}
GW_LABELS = {False: "No Gateway", True: "With Gateway"}
COLORS = {
    "std": "#4C72B0",
    "risk": "#DD8452",
    "sharpe": "#55A868",
}
ALGO_COLORS = {
    "ppo": "#4C72B0",
    "sac": "#DD8452",
    "td3": "#55A868",
    "lag_ppo": "#C44E52",
}
ALGO_MARKERS = {"ppo": "o", "sac": "s", "td3": "D", "lag_ppo": "^"}
GW_HATCHES = {False: "", True: "///"}


def load_results(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def detect_algorithms(results: list[dict]) -> list[str]:
    algos = sorted(set(
        r.get("algorithm", "ppo") for r in results
        if r.get("algorithm", "n/a") != "n/a"
    ))
    return algos if algos else ["ppo"]


def per_seed_means(results: list[dict], metric: str) -> dict[str, list[float]]:
    by_cond_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        val = r.get(metric, 0.0)
        if val is None:
            val = 0.0
        by_cond_seed[r["condition"]][r["seed"]].append(float(val))
    out: dict[str, list[float]] = {}
    for cond, seed_dict in by_cond_seed.items():
        out[cond] = [float(np.mean(vals)) for vals in seed_dict.values()]
    return out


def ci95(vals):
    a = np.array(vals, dtype=float)
    if len(a) < 2:
        return float(np.mean(a)), 0.0
    se = scipy_stats.sem(a)
    h = se * scipy_stats.t.ppf(0.975, len(a) - 1)
    return float(np.mean(a)), float(h)


def _cond_name(algo: str, reward: str, gw: bool) -> str:
    """Build condition name from components."""
    gw_str = "riskgw" if gw else "naked"
    return f"{algo}_{reward}_{gw_str}"


def _save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name} saved.")


# ---------------------------------------------------------------------------
# Figure 1: Main results (grouped bars) — one subplot per algorithm
# ---------------------------------------------------------------------------

def fig1_main_results(results, out_dir, algorithms):
    metrics = [
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown ($)"),
        ("var_95", "VaR 95% ($)"),
        ("total_pnl", "PnL ($)"),
    ]

    n_algos = len(algorithms)
    fig, axes = plt.subplots(n_algos, 4, figsize=(14, 3.5 * n_algos))
    if n_algos == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Performance by Algorithm, Reward Type, and Risk Gateway", fontsize=13, fontweight="bold")

    reward_types = ["std", "risk", "sharpe"]
    x = np.arange(len(reward_types))
    width = 0.35

    for row, algo in enumerate(algorithms):
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            seed_means = per_seed_means(results, metric_key)

            for i, gw in enumerate([False, True]):
                means, errs = [], []
                for rt in reward_types:
                    cond = _cond_name(algo, rt, gw)
                    vals = seed_means.get(cond, [0])
                    m, e = ci95(vals)
                    means.append(m)
                    errs.append(e)

                offset = (i - 0.5) * width
                ax.bar(x + offset, means, width, yerr=errs,
                      color=[COLORS[rt] for rt in reward_types],
                      alpha=0.9 if not gw else 0.6,
                      edgecolor="black", linewidth=0.5,
                      hatch=GW_HATCHES[gw], capsize=3,
                      label=GW_LABELS[gw] if col == 0 else "")

            ax.set_ylabel(metric_label if col == 0 or metric_label != axes[row, 0].get_ylabel() else "")
            ax.set_title(f"{ALGO_LABELS.get(algo, algo)} — {metric_label}" if row == 0 else metric_label)
            ax.set_xticks(x)
            ax.set_xticklabels([REWARD_LABELS[rt] for rt in reward_types], fontsize=7)
            if col == 0 and row == 0:
                ax.legend(loc="best", framealpha=0.8)

        # Add algorithm label on the left
        axes[row, 0].set_ylabel(f"{ALGO_LABELS.get(algo, algo)}\n{metrics[0][1]}", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_main_results")


# ---------------------------------------------------------------------------
# Figure 2: Safety metrics
# ---------------------------------------------------------------------------

def fig2_safety(results, out_dir, algorithms):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Safety Metrics Across Algorithms", fontsize=13, fontweight="bold")

    reward_types = ["std", "risk", "sharpe"]
    n_algos = len(algorithms)
    x = np.arange(len(reward_types))
    total_width = 0.8
    bar_width = total_width / n_algos

    # Panel 1: Safety violations (no gateway)
    for i, algo in enumerate(algorithms):
        means, errs = [], []
        for rt in reward_types:
            cond = _cond_name(algo, rt, False)
            sm = per_seed_means(results, "safety_violations")
            vals = sm.get(cond, [0])
            m, e = ci95(vals)
            means.append(m)
            errs.append(e)

        offset = (i - (n_algos - 1) / 2) * bar_width
        ax1.bar(x + offset, means, bar_width, yerr=errs,
                color=ALGO_COLORS[algo], alpha=0.8,
                edgecolor="black", linewidth=0.5, capsize=2,
                label=ALGO_LABELS[algo])

    ax1.set_ylabel("Safety Violations / Episode")
    ax1.set_xticks(x)
    ax1.set_xticklabels([REWARD_LABELS[rt] for rt in reward_types])
    ax1.set_title("Violations (No Gateway)")
    ax1.legend(fontsize=7)

    # Panel 2: Gateway rejection rates
    for i, algo in enumerate(algorithms):
        means, errs = [], []
        for rt in reward_types:
            cond = _cond_name(algo, rt, True)
            groups = defaultdict(list)
            for r in results:
                groups[r["condition"]].append(r)
            eps = groups.get(cond, [])
            seed_rej: dict[int, list[float]] = defaultdict(list)
            for e in eps:
                seed_rej[e["seed"]].append(e.get("n_rejected", 0) / max(e.get("n_orders", 1), 1))
            per_seed = [float(np.mean(v)) for v in seed_rej.values()]
            m, e_val = ci95(per_seed) if per_seed else (0, 0)
            means.append(m * 100)
            errs.append(e_val * 100)

        offset = (i - (n_algos - 1) / 2) * bar_width
        ax2.bar(x + offset, means, bar_width, yerr=errs,
                color=ALGO_COLORS[algo], alpha=0.8,
                edgecolor="black", linewidth=0.5, capsize=2,
                label=ALGO_LABELS[algo])

    ax2.set_ylabel("Rejection Rate (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([REWARD_LABELS[rt] for rt in reward_types])
    ax2.set_title("Gateway Rejections")
    ax2.legend(fontsize=7)

    # Panel 3: Mean inventory
    for i, algo in enumerate(algorithms):
        means = []
        for rt in reward_types:
            cond = _cond_name(algo, rt, False)
            sm = per_seed_means(results, "mean_abs_inventory")
            vals = sm.get(cond, [0])
            m, _ = ci95(vals)
            means.append(m)

        offset = (i - (n_algos - 1) / 2) * bar_width
        ax3.bar(x + offset, means, bar_width,
                color=ALGO_COLORS[algo], alpha=0.8,
                edgecolor="black", linewidth=0.5,
                label=ALGO_LABELS[algo])

    ax3.set_ylabel("Mean |Inventory| (shares)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([REWARD_LABELS[rt] for rt in reward_types])
    ax3.set_title("Inventory Utilisation")
    ax3.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "fig2_safety")


# ---------------------------------------------------------------------------
# Figure 3: Risk-return scatter (all algorithms)
# ---------------------------------------------------------------------------

def fig3_risk_return(results, out_dir, algorithms):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Risk-Return Trade-off (per-seed means, all algorithms)", fontweight="bold")

    seed_sharpe = per_seed_means(results, "sharpe")
    seed_dd = per_seed_means(results, "max_drawdown")

    for algo in algorithms:
        for rt in ["std", "risk", "sharpe"]:
            for gw in [False, True]:
                cond = _cond_name(algo, rt, gw)
                sh = seed_sharpe.get(cond, [])
                dd = seed_dd.get(cond, [])
                if not sh:
                    continue

                marker = ALGO_MARKERS[algo]
                color = COLORS[rt]
                facecolor = color if gw else "none"

                ax.scatter(dd, sh, marker=marker, s=30, alpha=0.3,
                          facecolors=facecolor, edgecolors=color, linewidth=1)
                ax.scatter([np.mean(dd)], [np.mean(sh)], marker=marker, s=150,
                          facecolors=facecolor, edgecolors=color, linewidth=2, zorder=10)

    # Heuristic
    sh_h = seed_sharpe.get("heuristic", [])
    dd_h = seed_dd.get("heuristic", [])
    if sh_h:
        ax.scatter(dd_h, sh_h, marker="v", s=30, alpha=0.4, color="gray")
        ax.scatter([np.mean(dd_h)], [np.mean(sh_h)], marker="v", s=150, color="gray", zorder=10)

    # Build legend
    legend_elements = []
    for rt in ["std", "risk", "sharpe"]:
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color='w',
                      markerfacecolor=COLORS[rt], markeredgecolor=COLORS[rt],
                      markersize=8, label=REWARD_LABELS[rt]))
    for algo in algorithms:
        legend_elements.append(
            plt.Line2D([0], [0], marker=ALGO_MARKERS[algo], color='w',
                      markerfacecolor='gray', markeredgecolor='gray',
                      markersize=8, label=ALGO_LABELS[algo]))
    legend_elements.append(
        plt.Line2D([0], [0], marker="o", color='w',
                  markerfacecolor='black', markersize=6, label="+ GW (filled)"))
    legend_elements.append(
        plt.Line2D([0], [0], marker="o", color='w',
                  markerfacecolor='none', markeredgecolor='black',
                  markersize=6, label="No GW (hollow)"))

    ax.legend(handles=legend_elements, loc="best", fontsize=7, ncol=2)
    ax.set_xlabel("Max Drawdown ($)")
    ax.set_ylabel("Sharpe Ratio")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    _save(fig, out_dir, "fig3_risk_return")


# ---------------------------------------------------------------------------
# Figure 4: Interaction plots (per algorithm)
# ---------------------------------------------------------------------------

def fig4_interaction(results, out_dir, algorithms):
    metrics = [
        ("max_drawdown", "Max Drawdown ($)"),
        ("sharpe", "Sharpe Ratio"),
        ("safety_violations", "Violations / Episode"),
    ]

    n_algos = len(algorithms)
    fig, axes = plt.subplots(n_algos, 3, figsize=(12, 3.5 * n_algos))
    if n_algos == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Factorial Interaction: Reward Shaping x Risk Gateway", fontsize=13, fontweight="bold")

    for row, algo in enumerate(algorithms):
        for col, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for rt in ["std", "risk", "sharpe"]:
                means, errs = [], []
                for gw in [False, True]:
                    cond = _cond_name(algo, rt, gw)
                    sm = per_seed_means(results, metric_key)
                    vals = sm.get(cond, [0])
                    m, e = ci95(vals)
                    means.append(m)
                    errs.append(e)

                ax.errorbar([0, 1], means, yerr=errs, marker="o",
                           label=REWARD_LABELS[rt] if row == 0 else "",
                           color=COLORS[rt], linewidth=2, capsize=4)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No Gateway", "With Gateway"])
            if col == 0:
                ax.set_ylabel(f"{ALGO_LABELS[algo]}\n{metric_label}")
            else:
                ax.set_ylabel(metric_label)
            if row == 0:
                ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "fig4_interaction")


# ---------------------------------------------------------------------------
# Figure 5: Boxplots per seed
# ---------------------------------------------------------------------------

def fig5_boxplots(results, out_dir, algorithms):
    fig, axes = plt.subplots(len(algorithms), 3, figsize=(14, 4 * len(algorithms)))
    if len(algorithms) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Per-Seed Distributions", fontsize=13, fontweight="bold")

    bp_metrics = [("sharpe", "Sharpe Ratio"), ("max_drawdown", "Max Drawdown ($)"), ("safety_violations", "Violations")]

    for row, algo in enumerate(algorithms):
        conditions = []
        short_labels = []
        colors_list = []
        for rt in ["std", "risk", "sharpe"]:
            for gw in [False, True]:
                conditions.append(_cond_name(algo, rt, gw))
                gw_str = "+GW" if gw else ""
                short_labels.append(f"{REWARD_LABELS[rt][:4]}{gw_str}")
                colors_list.append(COLORS[rt])

        for col, (metric_key, metric_label) in enumerate(bp_metrics):
            ax = axes[row, col]
            sm = per_seed_means(results, metric_key)
            data = [sm.get(c, [0]) for c in conditions]

            bp = ax.boxplot(data, patch_artist=True, tick_labels=short_labels, widths=0.6)
            for i, (patch, color, cond) in enumerate(zip(bp["boxes"], colors_list, conditions)):
                alpha = 0.6 if "riskgw" in cond else 0.9
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

            if col == 0:
                ax.set_ylabel(f"{ALGO_LABELS[algo]}\n{metric_label}")
            else:
                ax.set_ylabel(metric_label)
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save(fig, out_dir, "fig5_boxplots")


# ---------------------------------------------------------------------------
# Figure 6: Risk reduction waterfall
# ---------------------------------------------------------------------------

def fig6_reduction(results, out_dir, algorithms):
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5))
    if len(algorithms) == 1:
        axes = [axes]
    fig.suptitle("Risk Reduction vs. Standard RL (no Gateway)", fontweight="bold")

    risk_metrics = [("max_drawdown", "MaxDD"), ("var_95", "VaR-95"), ("es_95", "ES-95")]

    for ax, algo in zip(axes, algorithms):
        conditions = [
            (_cond_name(algo, "std", True), "Std+GW"),
            (_cond_name(algo, "risk", False), "Risk-Adj"),
            (_cond_name(algo, "risk", True), "Risk+GW"),
            (_cond_name(algo, "sharpe", False), "Sharpe"),
            (_cond_name(algo, "sharpe", True), "Sharpe+GW"),
        ]

        x = np.arange(len(conditions))
        width = 0.25

        for i, (metric_key, metric_label) in enumerate(risk_metrics):
            sm = per_seed_means(results, metric_key)
            baseline_cond = _cond_name(algo, "std", False)
            baseline = np.mean(sm.get(baseline_cond, [1]))
            if baseline < 1e-6:
                continue

            reductions = []
            for cond, _ in conditions:
                vals = sm.get(cond, [baseline])
                mean = np.mean(vals)
                reduction = (baseline - mean) / baseline * 100
                reductions.append(reduction)

            ax.bar(x + i * width, reductions, width, label=metric_label, alpha=0.8)

        ax.set_ylabel("Reduction vs. Std (%)")
        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_xticks(x + width)
        ax.set_xticklabels([label for _, label in conditions], rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    _save(fig, out_dir, "fig6_reduction")


# ---------------------------------------------------------------------------
# Figure 7: Cross-algorithm consistency (NEW)
# ---------------------------------------------------------------------------

def fig7_cross_algorithm(results, out_dir, algorithms):
    """Show that safety violation reduction is consistent across algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Cross-Algorithm Consistency: Reward Shaping Effects", fontsize=13, fontweight="bold")

    key_metrics = [
        ("safety_violations", "Safety Violations / Episode"),
        ("max_drawdown", "Max Drawdown ($)"),
        ("sharpe", "Sharpe Ratio"),
    ]
    reward_types = ["std", "risk", "sharpe"]
    x = np.arange(len(algorithms))
    width = 0.25

    for ax, (metric_key, metric_label) in zip(axes, key_metrics):
        sm = per_seed_means(results, metric_key)

        for i, rt in enumerate(reward_types):
            means, errs = [], []
            for algo in algorithms:
                cond = _cond_name(algo, rt, False)
                vals = sm.get(cond, [0])
                m, e = ci95(vals)
                means.append(m)
                errs.append(e)

            offset = (i - 1) * width
            ax.bar(x + offset, means, width, yerr=errs,
                  color=COLORS[rt], alpha=0.8,
                  edgecolor="black", linewidth=0.5, capsize=3,
                  label=REWARD_LABELS[rt])

        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algorithms])
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "fig7_cross_algorithm")


# ---------------------------------------------------------------------------
# Figure 8: Passivity analysis (NEW)
# ---------------------------------------------------------------------------

def fig8_passivity(results, out_dir, algorithms):
    """Prove risk-aware agents are actively trading, not just passive."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Passivity Analysis: Are Risk-Aware Agents Genuinely Trading?",
                 fontsize=13, fontweight="bold")

    reward_types = ["std", "risk", "sharpe"]
    x = np.arange(len(algorithms))
    width = 0.25

    passivity_metrics = [
        ("active_ratio", "Active Step Ratio", axes[0, 0]),
        ("mean_action_magnitude", "Mean |Action| Magnitude", axes[0, 1]),
        ("profit_per_trade", "Profit per Trade ($)", axes[1, 0]),
        ("direction_changes", "Direction Changes / Episode", axes[1, 1]),
    ]

    for metric_key, metric_label, ax in passivity_metrics:
        sm = per_seed_means(results, metric_key)

        for i, rt in enumerate(reward_types):
            means, errs = [], []
            for algo in algorithms:
                cond = _cond_name(algo, rt, False)
                vals = sm.get(cond, [0])
                m, e = ci95(vals)
                means.append(m)
                errs.append(e)

            offset = (i - 1) * width
            ax.bar(x + offset, means, width, yerr=errs,
                  color=COLORS[rt], alpha=0.8,
                  edgecolor="black", linewidth=0.5, capsize=3,
                  label=REWARD_LABELS[rt])

        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algorithms])
        ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "fig8_passivity")


# ---------------------------------------------------------------------------
# Figure 9: Lagrangian PPO comparison (NEW)
# ---------------------------------------------------------------------------

def fig9_lagrangian(results, out_dir, algorithms):
    """Compare Lagrangian PPO (formal constrained RL) vs reward shaping."""
    if "lag_ppo" not in algorithms:
        print("  fig9_lagrangian skipped (no lag_ppo data).")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Constrained RL (Lag-PPO) vs. Reward Shaping", fontsize=13, fontweight="bold")

    compare_metrics = [
        ("safety_violations", "Safety Violations"),
        ("max_drawdown", "Max Drawdown ($)"),
        ("sharpe", "Sharpe Ratio"),
        ("active_ratio", "Active Ratio"),
    ]

    # Compare: PPO-std, PPO-risk, LagPPO-std, LagPPO-risk
    compare_conds = [
        (_cond_name("ppo", "std", False), "PPO / Std", COLORS["std"]),
        (_cond_name("ppo", "risk", False), "PPO / Risk", COLORS["risk"]),
        (_cond_name("lag_ppo", "std", False), "Lag-PPO / Std", ALGO_COLORS["lag_ppo"]),
        (_cond_name("lag_ppo", "risk", False), "Lag-PPO / Risk", "#E78AC3"),
    ]

    x = np.arange(len(compare_conds))

    for ax, (metric_key, metric_label) in zip(axes, compare_metrics):
        sm = per_seed_means(results, metric_key)
        means, errs, colors = [], [], []

        for cond, label, color in compare_conds:
            vals = sm.get(cond, [0])
            m, e = ci95(vals)
            means.append(m)
            errs.append(e)
            colors.append(color)

        ax.bar(x, means, 0.6, yerr=errs, color=colors, alpha=0.8,
              edgecolor="black", linewidth=0.5, capsize=4)
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l, _ in compare_conds], rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    _save(fig, out_dir, "fig9_lagrangian")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fig10_ablation(ablation_path, out_dir):
    """Generate ablation sensitivity figure from ablation results."""
    abl_path = Path(ablation_path)
    if not abl_path.exists():
        print("  fig10_ablation skipped (no ablation data).")
        return

    abl_data = json.loads(abl_path.read_text())
    if not abl_data:
        print("  fig10_ablation skipped (empty ablation data).")
        return

    INV_LAMBDAS = [5e-4, 1e-3, 2e-3, 5e-3]
    DD_LAMBDAS = [1e-3, 5e-3, 1e-2, 5e-2]
    VAR_LAMBDAS = [5e-4, 1e-3, 5e-3, 1e-2]

    by_sweep = defaultdict(list)
    for r in abl_data:
        by_sweep[r["sweep_type"]].append(r)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Ablation: Reward Shaping Hyperparameter Sensitivity",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("mean_violations", "Safety Violations"),
        ("mean_sharpe", "Sharpe Ratio"),
        ("mean_max_drawdown", "Max Drawdown ($)"),
    ]
    sweep_configs = [
        ("inv", "inv_lambda", r"$\lambda_{\mathrm{inv}}$", INV_LAMBDAS),
        ("dd", "dd_lambda", r"$\lambda_{\mathrm{dd}}$", DD_LAMBDAS),
        ("var", "var_lambda", r"$\lambda_{\mathrm{VaR}}$", VAR_LAMBDAS),
    ]

    for row, (metric_key, metric_label) in enumerate(metrics):
        for col, (sweep_type, param_key, param_label, param_vals) in enumerate(sweep_configs):
            ax = axes[row, col]
            runs = by_sweep.get(sweep_type, [])
            if not runs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            by_val = defaultdict(list)
            for r in runs:
                by_val[r[param_key]].append(r[metric_key])

            x_vals = sorted(by_val.keys())
            means = [np.mean(by_val[v]) for v in x_vals]
            stds = [np.std(by_val[v]) / max(1, np.sqrt(len(by_val[v]))) for v in x_vals]

            ax.errorbar(range(len(x_vals)), means, yerr=stds,
                        marker="o", capsize=4, linewidth=2, color="#4C72B0")
            ax.fill_between(range(len(x_vals)),
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color="#4C72B0")
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels([f"{v:.0e}" for v in x_vals], rotation=45, fontsize=7)
            ax.set_xlabel(param_label)
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == 0:
                ax.set_title(f"Sweep: {param_label}")

            baseline_map = {"inv": 1e-3, "dd": 5e-3, "var": 1e-3}
            base = baseline_map[sweep_type]
            if base in x_vals:
                idx = x_vals.index(base)
                ax.axvline(idx, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, out_dir, "fig10_ablation")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="output/experiments/raw_results.json")
    p.add_argument("--outdir", default="output/experiments/figures")
    p.add_argument("--ablation", default="output/ablation/ablation_results.json",
                   help="Path to ablation results JSON")
    args = p.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} episodes.")

    algorithms = detect_algorithms(results)
    print(f"Detected algorithms: {', '.join(algorithms)}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_main_results(results, out_dir, algorithms)
    fig2_safety(results, out_dir, algorithms)
    fig3_risk_return(results, out_dir, algorithms)
    fig4_interaction(results, out_dir, algorithms)
    fig5_boxplots(results, out_dir, algorithms)
    fig6_reduction(results, out_dir, algorithms)
    fig7_cross_algorithm(results, out_dir, algorithms)
    fig8_passivity(results, out_dir, algorithms)
    fig9_lagrangian(results, out_dir, algorithms)
    fig10_ablation(args.ablation, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
