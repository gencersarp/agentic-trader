"""Post-hoc analysis for intrinsic vs. extrinsic safety experiments.

Extended to support multi-algorithm design (PPO, SAC, TD3, Lagrangian PPO)
and passivity analysis.

Generates:
  - Summary statistics with confidence intervals
  - Per-algorithm two-way ANOVA (reward_type x gateway)
  - Cross-algorithm consistency analysis
  - Pairwise significance tests with Holm-Bonferroni correction
  - Effect sizes (Cohen's d)
  - Passivity analysis (proving agents aren't just inactive)
  - Per-regime performance decomposition
  - Markdown report

Usage:
    python scripts/analyze_results.py [--results output/experiments/raw_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------

METRICS = [
    ("total_pnl", "PnL ($)", "higher"),
    ("sharpe", "Sharpe Ratio", "higher"),
    ("sortino", "Sortino Ratio", "higher"),
    ("max_drawdown", "Max Drawdown ($)", "lower"),
    ("var_95", "VaR 95% ($)", "lower"),
    ("es_95", "ES 95% ($)", "lower"),
    ("calmar", "Calmar Ratio", "higher"),
    ("safety_violations", "Safety Violations", "lower"),
    ("mean_abs_inventory", "Mean |Inventory|", "lower"),
]

PASSIVITY_METRICS = [
    ("mean_action_magnitude", "Mean |Action|", "higher"),
    ("active_ratio", "Active Ratio", "higher"),
    ("profit_per_trade", "Profit/Trade ($)", "higher"),
    ("direction_changes", "Direction Changes", "higher"),
    ("mean_hold_time", "Mean Hold Time (steps)", "neutral"),
]

ALGO_LABELS = {"ppo": "PPO", "sac": "SAC", "td3": "TD3", "lag_ppo": "Lag-PPO", "n/a": "—"}
REWARD_LABELS_SHORT = {"std": "Std", "risk": "Risk", "sharpe": "Sharpe", "n/a": "—"}


def detect_algorithms(results: list[dict]) -> list[str]:
    """Detect which algorithms are present in the results."""
    algos = sorted(set(r.get("algorithm", "ppo") for r in results if r.get("algorithm", "n/a") != "n/a"))
    return algos if algos else ["ppo"]


def detect_conditions(results: list[dict]) -> list[str]:
    """Detect all conditions in order."""
    seen = []
    for r in results:
        c = r["condition"]
        if c not in seen:
            seen.append(c)
    return seen


def condition_label(cond: str) -> str:
    """Generate human-readable label for a condition."""
    if cond in ("random", "heuristic"):
        return cond.capitalize()
    parts = cond.split("_")
    if len(parts) >= 3:
        # e.g. ppo_std_naked -> PPO / Std / No GW
        algo = ALGO_LABELS.get(parts[0], parts[0].upper())
        if parts[0] == "lag" and len(parts) >= 4:
            algo = "Lag-PPO"
            reward = REWARD_LABELS_SHORT.get(parts[2], parts[2])
            gw = "GW" if parts[3] == "riskgw" else "No GW"
        else:
            reward = REWARD_LABELS_SHORT.get(parts[1], parts[1])
            gw = "GW" if parts[2] == "riskgw" else "No GW"
        return f"{algo} / {reward} / {gw}"
    # Legacy format: rl_std_naked
    if cond.startswith("rl_"):
        rest = cond[3:]
        parts = rest.split("_")
        reward = REWARD_LABELS_SHORT.get(parts[0], parts[0])
        gw = "GW" if len(parts) > 1 and parts[1] == "riskgw" else "No GW"
        return f"PPO / {reward} / {gw}"
    return cond


def group_by_condition(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["condition"]].append(r)
    return groups


def per_seed_means(results: list[dict], metric: str) -> dict[str, list[float]]:
    """Compute per-seed mean of a metric for each condition."""
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


def welch_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    # Handle constant arrays
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        if abs(np.mean(a) - np.mean(b)) < 1e-12:
            return 0.0, 1.0
        return float('inf'), 0.0
    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    if np.isnan(t):
        return 0.0, 1.0
    return float(t), float(p)


def cohens_d(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    n1, n2 = len(a_arr), len(b_arr)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(a_arr, ddof=1) + (n2 - 1) * np.var(b_arr, ddof=1)) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    for rank in range(1, n):
        orig_idx = indexed[rank][0]
        prev_idx = indexed[rank - 1][0]
        adjusted[orig_idx] = max(adjusted[orig_idx], adjusted[prev_idx])
    return adjusted


def confidence_interval(vals: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    a = np.array(vals, dtype=float)
    n = len(a)
    mean = float(np.mean(a))
    if n < 2:
        return mean, mean, mean
    se = float(scipy_stats.sem(a))
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


# ---------------------------------------------------------------------------
# Two-way ANOVA (per algorithm)
# ---------------------------------------------------------------------------

def two_way_anova_manual(results: list[dict], metric: str, algorithm: str = None) -> dict:
    """Two-way ANOVA for reward_type x gateway, optionally filtered by algorithm."""
    rl_results = [r for r in results if r["condition"] not in ("random", "heuristic")]
    if algorithm:
        rl_results = [r for r in rl_results if r.get("algorithm") == algorithm]

    cells: dict[tuple[str, bool], list[float]] = defaultdict(list)
    by_seed: dict[tuple[str, bool, int], list[float]] = defaultdict(list)

    for r in rl_results:
        rt = r.get("reward_type", "std")
        gw = r.get("has_gateway", False)
        key = (rt, gw, r["seed"])
        val = r.get(metric, 0.0)
        if val is None:
            val = 0.0
        by_seed[key].append(float(val))

    for (rt, gw, seed), vals in by_seed.items():
        cells[(rt, gw)].append(float(np.mean(vals)))

    reward_types = sorted(set(k[0] for k in cells.keys()))
    if len(reward_types) < 2:
        return {"reward_F": 0, "reward_p": 1, "gateway_F": 0, "gateway_p": 1,
                "interaction_F": 0, "interaction_p": 1,
                "reward_eta2": 0, "gateway_eta2": 0, "interaction_eta2": 0}

    grand_mean = np.mean([v for vals in cells.values() for v in vals])

    row_means = {}
    for rt in reward_types:
        vals = []
        for gw in [False, True]:
            vals.extend(cells.get((rt, gw), []))
        row_means[rt] = np.mean(vals) if vals else grand_mean

    col_means = {}
    for gw in [False, True]:
        vals = []
        for rt in reward_types:
            vals.extend(cells.get((rt, gw), []))
        col_means[gw] = np.mean(vals) if vals else grand_mean

    n_per_cell = len(cells.get((reward_types[0], False), []))
    if n_per_cell == 0:
        return {"reward_F": 0, "reward_p": 1, "gateway_F": 0, "gateway_p": 1,
                "interaction_F": 0, "interaction_p": 1,
                "reward_eta2": 0, "gateway_eta2": 0, "interaction_eta2": 0}

    n_gw = 2
    n_rt = len(reward_types)

    ss_reward = n_gw * n_per_cell * sum((row_means[rt] - grand_mean) ** 2 for rt in reward_types)
    ss_gateway = n_rt * n_per_cell * sum((col_means[gw] - grand_mean) ** 2 for gw in [False, True])

    ss_interaction = 0
    for rt in reward_types:
        for gw in [False, True]:
            cell_vals = cells.get((rt, gw), [])
            if cell_vals:
                cell_mean = np.mean(cell_vals)
                expected = row_means[rt] + col_means[gw] - grand_mean
                ss_interaction += n_per_cell * (cell_mean - expected) ** 2

    ss_within = sum(
        sum((v - np.mean(vals)) ** 2 for v in vals)
        for vals in cells.values()
    )

    df_reward = n_rt - 1
    df_gateway = 1
    df_interaction = df_reward * df_gateway
    df_within = sum(len(vals) - 1 for vals in cells.values())

    if df_within <= 0:
        return {"reward_F": 0, "reward_p": 1, "gateway_F": 0, "gateway_p": 1,
                "interaction_F": 0, "interaction_p": 1,
                "reward_eta2": 0, "gateway_eta2": 0, "interaction_eta2": 0}

    ms_reward = ss_reward / max(df_reward, 1)
    ms_gateway = ss_gateway / max(df_gateway, 1)
    ms_interaction = ss_interaction / max(df_interaction, 1)
    ms_within = ss_within / df_within

    if ms_within < 1e-12:
        return {"reward_F": 0, "reward_p": 1, "gateway_F": 0, "gateway_p": 1,
                "interaction_F": 0, "interaction_p": 1,
                "reward_eta2": 0, "gateway_eta2": 0, "interaction_eta2": 0}

    f_reward = ms_reward / ms_within
    f_gateway = ms_gateway / ms_within
    f_interaction = ms_interaction / ms_within

    p_reward = 1 - scipy_stats.f.cdf(f_reward, df_reward, df_within)
    p_gateway = 1 - scipy_stats.f.cdf(f_gateway, df_gateway, df_within)
    p_interaction = 1 - scipy_stats.f.cdf(f_interaction, df_interaction, df_within)

    ss_total = ss_reward + ss_gateway + ss_interaction + ss_within
    eta2_reward = ss_reward / ss_total if ss_total > 0 else 0
    eta2_gateway = ss_gateway / ss_total if ss_total > 0 else 0
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    return {
        "reward_F": float(f_reward), "reward_p": float(p_reward), "reward_eta2": float(eta2_reward),
        "gateway_F": float(f_gateway), "gateway_p": float(p_gateway), "gateway_eta2": float(eta2_gateway),
        "interaction_F": float(f_interaction), "interaction_p": float(p_interaction), "interaction_eta2": float(eta2_interaction),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _sig_str(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def generate_report(results: list[dict]) -> str:
    lines = []
    lines.append("# Intrinsic vs. Extrinsic Safety — Extended Statistical Analysis\n")

    algorithms = detect_algorithms(results)
    all_conditions = detect_conditions(results)
    groups = group_by_condition(results)
    n_episodes = {c: len(eps) for c, eps in groups.items()}
    seeds_per_cond = {c: len(set(r["seed"] for r in eps)) for c, eps in groups.items()}

    # --- Overview ---
    lines.append("## Overview\n")
    lines.append(f"- **Algorithms**: {', '.join(ALGO_LABELS.get(a, a) for a in algorithms)}")
    lines.append(f"- **Total conditions**: {len(all_conditions)}")
    total_eps = sum(n_episodes.values())
    lines.append(f"- **Total episodes**: {total_eps:,}")
    lines.append("")
    for c in all_conditions:
        if c in n_episodes:
            lines.append(f"- **{condition_label(c)}**: {n_episodes[c]} episodes, {seeds_per_cond[c]} seeds")
    lines.append("")

    # --- Main results table (per algorithm) ---
    for algo in algorithms:
        lines.append(f"## Main Results — {ALGO_LABELS.get(algo, algo)} (per-seed means, 95% CI)\n")
        algo_conds = [c for c in all_conditions if c.startswith(f"{algo}_") or
                      (algo == "lag_ppo" and c.startswith("lag_ppo_"))]
        if not algo_conds:
            # Legacy format
            algo_conds = [c for c in all_conditions if c.startswith("rl_")]
        display_conds = ["random", "heuristic"] + algo_conds

        key_metrics = [("total_pnl", "PnL ($)"), ("sharpe", "Sharpe"), ("max_drawdown", "MaxDD ($)"),
                       ("var_95", "VaR95 ($)"), ("safety_violations", "Violations"),
                       ("mean_abs_inventory", "Mean|Inv|"), ("active_ratio", "Active%")]
        lines.append("| Condition | " + " | ".join(label for _, label in key_metrics) + " | Rej% |")
        lines.append("|" + "|".join(["---"] * (len(key_metrics) + 2)) + "|")

        for cond in display_conds:
            if cond not in groups:
                continue
            row = [f"**{condition_label(cond)}**"]
            for metric_key, _ in key_metrics:
                seed_means_dict = per_seed_means(results, metric_key)
                vals = seed_means_dict.get(cond, [])
                if not vals:
                    row.append("—")
                    continue
                mean, ci_lo, ci_hi = confidence_interval(vals)
                if "pnl" in metric_key or "drawdown" in metric_key or "var" in metric_key:
                    row.append(f"{mean:+,.0f} [{ci_lo:+,.0f}, {ci_hi:+,.0f}]")
                elif "violation" in metric_key or "inventory" in metric_key:
                    row.append(f"{mean:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]")
                elif "ratio" in metric_key:
                    row.append(f"{mean:.1%}")
                else:
                    row.append(f"{mean:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]")

            # Rejection rate
            rej_seed_means = per_seed_means(results, "n_rejected")
            ord_seed_means = per_seed_means(results, "n_orders")
            rej_vals = rej_seed_means.get(cond, [])
            ord_vals = ord_seed_means.get(cond, [])
            if rej_vals and ord_vals:
                rej_rate = [r / max(o, 1) for r, o in zip(rej_vals, ord_vals)]
                mean_rr, _, _ = confidence_interval(rej_rate)
                row.append(f"{mean_rr:.1%}")
            else:
                row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- Passivity Analysis ---
    lines.append("## Passivity Analysis\n")
    lines.append("Proving that risk-aware agents actively trade (not just passive).\n")
    lines.append("| Condition | Active% | Mean|Action| | Profit/Trade ($) | Dir.Changes | Mean Hold (steps) | Orders/Ep |")
    lines.append("|---|---|---|---|---|---|---|")

    for cond in all_conditions:
        if cond in ("random", "heuristic"):
            continue
        if cond not in groups:
            continue

        sm_active = per_seed_means(results, "active_ratio")
        sm_action = per_seed_means(results, "mean_action_magnitude")
        sm_ppt = per_seed_means(results, "profit_per_trade")
        sm_dc = per_seed_means(results, "direction_changes")
        sm_ht = per_seed_means(results, "mean_hold_time")
        sm_orders = per_seed_means(results, "n_orders")

        def _safe_mean(sm, c):
            vals = sm.get(c, [])
            return float(np.mean(vals)) if vals else 0.0

        lines.append(
            f"| {condition_label(cond)} | "
            f"{_safe_mean(sm_active, cond):.1%} | "
            f"{_safe_mean(sm_action, cond):.3f} | "
            f"{_safe_mean(sm_ppt, cond):+.2f} | "
            f"{_safe_mean(sm_dc, cond):.0f} | "
            f"{_safe_mean(sm_ht, cond):.1f} | "
            f"{_safe_mean(sm_orders, cond):.0f} |"
        )
    lines.append("")

    # --- Per-algorithm ANOVA ---
    lines.append("## Two-Way ANOVA (Reward Type × Risk Gateway)\n")
    for algo in algorithms:
        lines.append(f"### {ALGO_LABELS.get(algo, algo)}\n")
        lines.append("| Metric | Factor | F-stat | p-value | η² | Sig? |")
        lines.append("|---|---|---|---|---|---|")

        for metric_key, metric_label, _ in METRICS:
            anova = two_way_anova_manual(results, metric_key, algorithm=algo)
            for factor, f_key, p_key, eta_key in [
                ("Reward Type", "reward_F", "reward_p", "reward_eta2"),
                ("Risk Gateway", "gateway_F", "gateway_p", "gateway_eta2"),
                ("Interaction", "interaction_F", "interaction_p", "interaction_eta2"),
            ]:
                f_val = anova[f_key]
                p_val = anova[p_key]
                eta2 = anova.get(eta_key, 0)
                sig = _sig_str(p_val)
                lines.append(f"| {metric_label} | {factor} | {f_val:.2f} | {p_val:.4f} | {eta2:.3f} | {sig} |")
        lines.append("")

    # --- Cross-algorithm consistency ---
    lines.append("## Cross-Algorithm Consistency\n")
    lines.append("Testing whether Risk-Adjusted reward reduces violations across all algorithms.\n")
    lines.append("| Algorithm | Std Violations | Risk Violations | Reduction | Cohen's d | p-value | Sig? |")
    lines.append("|---|---|---|---|---|---|---|")

    for algo in algorithms:
        std_cond = f"{algo}_std_naked"
        risk_cond = f"{algo}_risk_naked"
        # Legacy fallback
        if std_cond not in groups and "rl_std_naked" in groups:
            std_cond = "rl_std_naked"
            risk_cond = "rl_risk_naked"

        sm_viols = per_seed_means(results, "safety_violations")
        std_vals = sm_viols.get(std_cond, [])
        risk_vals = sm_viols.get(risk_cond, [])

        if not std_vals or not risk_vals:
            continue

        std_mean = float(np.mean(std_vals))
        risk_mean = float(np.mean(risk_vals))
        reduction = (std_mean - risk_mean) / max(std_mean, 1e-6) * 100
        d = cohens_d(std_vals, risk_vals)
        t, p = welch_ttest(std_vals, risk_vals)
        sig = _sig_str(p)

        lines.append(
            f"| {ALGO_LABELS.get(algo, algo)} | {std_mean:.1f} | {risk_mean:.1f} | "
            f"{reduction:.0f}% | {d:+.2f} | {p:.4f} | {sig} |"
        )
    lines.append("")

    # --- Key pairwise comparisons (per algorithm) ---
    lines.append("## Key Pairwise Comparisons (Holm-Bonferroni corrected)\n")

    for algo in algorithms:
        lines.append(f"### {ALGO_LABELS.get(algo, algo)}\n")

        # Build comparisons for this algorithm
        prefix = f"{algo}_"
        if not any(c.startswith(prefix) for c in all_conditions):
            prefix = "rl_"  # Legacy fallback

        key_comparisons = [
            (f"{prefix}std_naked", f"{prefix}risk_naked", "Std vs Risk-Adj (no GW)"),
            (f"{prefix}std_naked", f"{prefix}sharpe_naked", "Std vs Sharpe (no GW)"),
            (f"{prefix}risk_naked", f"{prefix}sharpe_naked", "Risk-Adj vs Sharpe (no GW)"),
            (f"{prefix}std_naked", f"{prefix}std_riskgw", "Effect of GW (Std)"),
            (f"{prefix}risk_naked", f"{prefix}risk_riskgw", "Effect of GW (Risk-Adj)"),
            (f"{prefix}std_riskgw", f"{prefix}risk_riskgw", "Std+GW vs Risk-Adj+GW"),
            ("heuristic", f"{prefix}risk_riskgw", "Heuristic vs Risk-Adj+GW"),
        ]

        for metric_key, metric_label, direction in METRICS:
            lines.append(f"#### {metric_label}\n")
            lines.append("| Comparison | Mean A | Mean B | Cohen's d | t-stat | p (raw) | p (adj) | Sig? |")
            lines.append("|---|---|---|---|---|---|---|---|")

            seed_means_dict = per_seed_means(results, metric_key)
            raw_ps = []
            comp_data = []

            for cond_a, cond_b, desc in key_comparisons:
                vals_a = seed_means_dict.get(cond_a, [])
                vals_b = seed_means_dict.get(cond_b, [])
                if not vals_a or not vals_b:
                    continue
                mean_a = float(np.mean(vals_a))
                mean_b = float(np.mean(vals_b))
                t, p = welch_ttest(vals_a, vals_b)
                d = cohens_d(vals_a, vals_b)
                raw_ps.append(p)
                comp_data.append((desc, mean_a, mean_b, d, t, p))

            adj_ps = holm_bonferroni(raw_ps)

            for i, (desc, mean_a, mean_b, d, t, p_raw) in enumerate(comp_data):
                p_adj = adj_ps[i] if i < len(adj_ps) else 1.0
                sig = _sig_str(p_adj)

                if "pnl" in metric_key or "drawdown" in metric_key or "var" in metric_key or "es" in metric_key:
                    lines.append(f"| {desc} | {mean_a:+,.0f} | {mean_b:+,.0f} | {d:+.2f} | {t:.2f} | {p_raw:.4f} | {p_adj:.4f} | {sig} |")
                elif "violation" in metric_key or "inventory" in metric_key:
                    lines.append(f"| {desc} | {mean_a:.1f} | {mean_b:.1f} | {d:+.2f} | {t:.2f} | {p_raw:.4f} | {p_adj:.4f} | {sig} |")
                else:
                    lines.append(f"| {desc} | {mean_a:+.2f} | {mean_b:+.2f} | {d:+.2f} | {t:.2f} | {p_raw:.4f} | {p_adj:.4f} | {sig} |")
            lines.append("")
        lines.append("")

    # --- Lagrangian PPO comparison ---
    if "lag_ppo" in algorithms:
        lines.append("## Lagrangian PPO vs. Reward Shaping\n")
        lines.append("Comparing formal constrained RL (Lag-PPO with std reward) against reward shaping.\n")
        lines.append("| Condition | PnL ($) | Sharpe | MaxDD ($) | Violations | Active% |")
        lines.append("|---|---|---|---|---|---|")

        compare_conds = []
        for algo in algorithms:
            for rt in ["std", "risk"]:
                cond = f"{algo}_{rt}_naked"
                if cond in groups:
                    compare_conds.append(cond)

        for cond in compare_conds:
            sm_pnl = per_seed_means(results, "total_pnl")
            sm_sh = per_seed_means(results, "sharpe")
            sm_dd = per_seed_means(results, "max_drawdown")
            sm_vi = per_seed_means(results, "safety_violations")
            sm_ar = per_seed_means(results, "active_ratio")

            def _m(sm, c):
                vals = sm.get(c, [])
                return float(np.mean(vals)) if vals else 0.0

            lines.append(
                f"| {condition_label(cond)} | {_m(sm_pnl, cond):+,.0f} | "
                f"{_m(sm_sh, cond):+.2f} | {_m(sm_dd, cond):,.0f} | "
                f"{_m(sm_vi, cond):.1f} | {_m(sm_ar, cond):.1%} |"
            )
        lines.append("")

    # --- Safety analysis ---
    lines.append("## Safety Analysis\n")
    lines.append("### Rejection Rates by Condition\n")
    lines.append("| Condition | Orders/Ep | Rejected/Ep | Rej% | Safety Violations/Ep |")
    lines.append("|---|---|---|---|---|")

    for cond in all_conditions:
        if cond not in groups:
            continue
        eps = groups[cond]
        orders = [e.get("n_orders", e.get("n_orders", 0)) for e in eps]
        rejected = [e.get("n_rejected", 0) for e in eps]
        violations = [e.get("safety_violations", 0) for e in eps]
        rej_rates = [r / max(o, 1) for r, o in zip(rejected, orders)]
        lines.append(
            f"| {condition_label(cond)} | {np.mean(orders):.0f} | {np.mean(rejected):.0f} | "
            f"{np.mean(rej_rates):.1%} | {np.mean(violations):.1f} |"
        )
    lines.append("")

    # --- Per-regime breakdown ---
    lines.append("## Per-Regime Breakdown\n")
    # Show for each algorithm's risk condition
    regime_conds = []
    for algo in algorithms:
        for rt in ["std", "risk"]:
            cond = f"{algo}_{rt}_riskgw"
            if cond in groups:
                regime_conds.append(cond)
    # Legacy fallback
    if not regime_conds:
        regime_conds = [c for c in all_conditions if c.endswith("_riskgw")]

    for cond in regime_conds:
        if cond not in groups:
            continue
        eps = groups[cond]
        calm = [e for e in eps if e.get("pct_calm", 0) > 0.5]
        vol = [e for e in eps if e.get("pct_highvol", 0) > 0.4]
        crisis = [e for e in eps if e.get("pct_crisis", 0) > 0.2]

        lines.append(f"### {condition_label(cond)}\n")
        lines.append("| Regime | N | PnL ($) | Sharpe | MaxDD ($) | Violations |")
        lines.append("|---|---|---|---|---|---|")

        for label, group in [("CALM-dominant", calm), ("HIGHVOL-dominant", vol), ("CRISIS-heavy", crisis)]:
            if not group:
                lines.append(f"| {label} | 0 | — | — | — | — |")
                continue
            pnls = [e["total_pnl"] for e in group]
            sharpes = [e["sharpe"] for e in group]
            dds = [e["max_drawdown"] for e in group]
            viols = [e.get("safety_violations", 0) for e in group]
            lines.append(
                f"| {label} | {len(group)} | {np.mean(pnls):+,.0f} | "
                f"{np.mean(sharpes):+.2f} | {np.mean(dds):,.0f} | {np.mean(viols):.1f} |"
            )
        lines.append("")

    # --- Key findings ---
    lines.append("## Key Findings\n")
    findings = []

    # Check for significant results across algorithms
    for algo in algorithms:
        sm_viols = per_seed_means(results, "safety_violations")
        std_cond = f"{algo}_std_naked"
        risk_cond = f"{algo}_risk_naked"
        if std_cond not in sm_viols:
            std_cond = "rl_std_naked"
            risk_cond = "rl_risk_naked"
        std_vals = sm_viols.get(std_cond, [])
        risk_vals = sm_viols.get(risk_cond, [])
        if std_vals and risk_vals:
            t, p = welch_ttest(std_vals, risk_vals)
            d = cohens_d(std_vals, risk_vals)
            algo_label = ALGO_LABELS.get(algo, algo)
            findings.append(
                f"- **{algo_label}**: Risk-adj reduces violations from {np.mean(std_vals):.1f} to "
                f"{np.mean(risk_vals):.1f} (d={d:+.2f}, p={p:.4f}{' *' if p < 0.05 else ''})"
            )

    # MaxDD comparison
    sm_dd = per_seed_means(results, "max_drawdown")
    for algo in algorithms:
        std_cond = f"{algo}_std_naked"
        risk_cond = f"{algo}_risk_naked"
        if std_cond not in sm_dd:
            std_cond = "rl_std_naked"
            risk_cond = "rl_risk_naked"
        std_dd = sm_dd.get(std_cond, [])
        risk_dd = sm_dd.get(risk_cond, [])
        if std_dd and risk_dd:
            reduction = (np.mean(std_dd) - np.mean(risk_dd)) / np.mean(std_dd) * 100
            t, p = welch_ttest(std_dd, risk_dd)
            findings.append(
                f"- **{ALGO_LABELS.get(algo, algo)}**: Risk-adj reduces MaxDD by {reduction:.0f}% (p={p:.4f}{' *' if p < 0.05 else ''})"
            )

    for f in findings:
        lines.append(f)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="output/experiments/raw_results.json")
    p.add_argument("--output", default="output/experiments/analysis_report.md")
    return p.parse_args()


def main():
    args = parse_args()
    results = load_results(args.results)
    print(f"Loaded {len(results)} episode results.")

    report = generate_report(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report written to {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
