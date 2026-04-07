"""Ablation study: sensitivity of reward shaping hyperparameters.

Sweeps over key lambda coefficients to demonstrate robustness (or lack thereof)
of the risk-adjusted reward function.  Addresses reviewer concern about whether
the specific hyperparameter choices are "tuned to win".

Sweep dimensions:
    - inv_penalty_lambda:       [5e-4, 1e-3, 2e-3, 5e-3]
    - drawdown_penalty_lambda:  [1e-3, 5e-3, 1e-2, 5e-2]
    - var_penalty_lambda:       [5e-4, 1e-3, 5e-3, 1e-2]

Each combination is trained with PPO (fastest) for a reduced step budget and
evaluated over multiple seeds.

Usage:
    python scripts/run_ablation.py [--seeds 10] [--train-steps 500000] \
        [--eval-episodes 20] [--algorithm ppo]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agentic_trader.config.settings import EnvConfig, GlobalRiskLimits, LocalRiskLimits
from agentic_trader.env.abides_env import TradingEnv
from agentic_trader.risk.risk_gateway import RiskGateway, ValidationResult
from agentic_trader.risk.risk_metrics import PortfolioState, RiskEngine

# Reuse evaluation infrastructure from run_experiments
# Import works both as `python scripts/run_ablation.py` and from notebook
try:
    from scripts.run_experiments import (
        EpisodeResult, ExperimentConfig, PassthroughGateway,
        SAFETY_VIOLATION_THRESHOLD, run_episode, train_agent,
    )
except ImportError:
    from run_experiments import (
        EpisodeResult, ExperimentConfig, PassthroughGateway,
        SAFETY_VIOLATION_THRESHOLD, run_episode, train_agent,
    )

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("ablation")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    n_seeds: int = 10
    train_steps: int = 500_000
    eval_episodes: int = 20
    episode_length: int = 390
    output_dir: str = "output/ablation"
    n_envs: int = 4
    algorithm: str = "ppo"
    device: str = "auto"
    checkpoint: bool = True


# Default sweep ranges
INV_LAMBDAS = [5e-4, 1e-3, 2e-3, 5e-3]
DD_LAMBDAS = [1e-3, 5e-3, 1e-2, 5e-2]
VAR_LAMBDAS = [5e-4, 1e-3, 5e-3, 1e-2]


@dataclass
class AblationResult:
    seed: int
    inv_lambda: float
    dd_lambda: float
    var_lambda: float
    sweep_type: str            # "inv", "dd", "var", or "joint"
    mean_pnl: float = 0.0
    mean_sharpe: float = 0.0
    mean_max_drawdown: float = 0.0
    mean_var_95: float = 0.0
    mean_violations: float = 0.0
    mean_active_ratio: float = 0.0
    mean_mean_inv: float = 0.0


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_ablation_point(
    seed: int,
    inv_lambda: float,
    dd_lambda: float,
    var_lambda: float,
    acfg: AblationConfig,
) -> list[EpisodeResult]:
    """Train one model with specific reward lambdas and evaluate."""
    from agentic_trader.agents.execution_agent import ExecutionAgent

    env_config = EnvConfig(
        regime_switching=True,
        episode_length=acfg.episode_length,
        reward_type="risk_adjusted",
        inv_penalty_lambda=inv_lambda,
        drawdown_penalty_lambda=dd_lambda,
        drawdown_threshold=5_000.0,
        var_penalty_lambda=var_lambda,
    )

    # Train
    train_seed = seed + 1_000_000
    exp_cfg = ExperimentConfig(
        train_steps=acfg.train_steps,
        n_envs=acfg.n_envs,
        episode_length=acfg.episode_length,
        device=acfg.device,
    )
    model = train_agent(train_seed, "risk", acfg.algorithm, exp_cfg)

    # Evaluate on standard env (no gateway)
    eval_env_config = EnvConfig(
        regime_switching=True,
        episode_length=acfg.episode_length,
        reward_type="pnl",
        inv_penalty_lambda=1e-3,
    )
    env = TradingEnv(config=eval_env_config, seed=seed)
    local_limits = LocalRiskLimits()
    global_limits = GlobalRiskLimits()
    risk_engine = RiskEngine()
    gateway = PassthroughGateway()

    agent = ExecutionAgent(
        policy=model, symbol="AAPL",
        local_limits=local_limits, agent_id="ablation",
    )

    results = []
    for ep in range(acfg.eval_episodes):
        r = run_episode(
            env=env, agent=agent, gateway=gateway,
            risk_engine=risk_engine, global_limits=global_limits,
            seed=seed, condition="ablation",
            algorithm=acfg.algorithm, reward_type="risk",
            has_gateway=False, episode_idx=ep,
        )
        results.append(r)

    return results


def aggregate_ablation(
    episodes: list[EpisodeResult],
    seed: int,
    inv_lambda: float,
    dd_lambda: float,
    var_lambda: float,
    sweep_type: str,
) -> AblationResult:
    return AblationResult(
        seed=seed,
        inv_lambda=inv_lambda,
        dd_lambda=dd_lambda,
        var_lambda=var_lambda,
        sweep_type=sweep_type,
        mean_pnl=float(np.mean([e.total_pnl for e in episodes])),
        mean_sharpe=float(np.mean([e.sharpe for e in episodes])),
        mean_max_drawdown=float(np.mean([e.max_drawdown for e in episodes])),
        mean_var_95=float(np.mean([e.var_95 for e in episodes])),
        mean_violations=float(np.mean([e.safety_violations for e in episodes])),
        mean_active_ratio=float(np.mean([e.active_ratio for e in episodes])),
        mean_mean_inv=float(np.mean([e.mean_abs_inventory for e in episodes])),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Ablation study on reward shaping hyperparameters.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--train-steps", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--algorithm", default="ppo", choices=["ppo", "sac", "td3"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--sweep", default="marginal", choices=["marginal", "joint"],
                   help="'marginal' sweeps one param at a time; 'joint' does full grid")
    args = p.parse_args()

    acfg = AblationConfig(
        n_seeds=args.seeds,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        algorithm=args.algorithm,
        device=args.device,
    )

    out_dir = Path(acfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build sweep points
    # Baseline values: inv=1e-3, dd=5e-3, var=1e-3
    base_inv, base_dd, base_var = 1e-3, 5e-3, 1e-3

    sweep_points: list[tuple[float, float, float, str]] = []

    if args.sweep == "marginal":
        # Sweep inventory penalty (hold others at baseline)
        for lam in INV_LAMBDAS:
            sweep_points.append((lam, base_dd, base_var, "inv"))
        # Sweep drawdown penalty
        for lam in DD_LAMBDAS:
            sweep_points.append((base_inv, lam, base_var, "dd"))
        # Sweep VaR penalty
        for lam in VAR_LAMBDAS:
            sweep_points.append((base_inv, base_dd, lam, "var"))
    else:
        # Full joint grid (expensive)
        for inv_l, dd_l, var_l in product(INV_LAMBDAS, DD_LAMBDAS, VAR_LAMBDAS):
            sweep_points.append((inv_l, dd_l, var_l, "joint"))

    # Deduplicate
    sweep_points = list(dict.fromkeys(sweep_points))

    total_runs = len(sweep_points) * acfg.n_seeds
    logger.info("Ablation study: %d sweep points x %d seeds = %d training runs",
                len(sweep_points), acfg.n_seeds, total_runs)

    all_results: list[dict] = []

    # Load checkpoint if exists
    ckpt_path = out_dir / "ablation_checkpoint.json"
    completed: set[str] = set()
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        all_results = ckpt.get("results", [])
        completed = set(ckpt.get("completed", []))
        logger.info("Resuming: %d completed runs found", len(completed))

    t_start = time.time()
    run_idx = 0

    for inv_l, dd_l, var_l, sweep_type in sweep_points:
        for seed_idx in range(acfg.n_seeds):
            run_key = f"{inv_l}_{dd_l}_{var_l}_{seed_idx}"
            if run_key in completed:
                run_idx += 1
                continue

            seed = seed_idx * 7 + 42
            run_idx += 1
            logger.info("[%d/%d] λ_inv=%.4f, λ_dd=%.4f, λ_var=%.4f, seed=%d",
                        run_idx, total_runs, inv_l, dd_l, var_l, seed)

            episodes = run_ablation_point(seed, inv_l, dd_l, var_l, acfg)
            agg = aggregate_ablation(episodes, seed, inv_l, dd_l, var_l, sweep_type)
            all_results.append(asdict(agg))
            completed.add(run_key)

            if acfg.checkpoint:
                ckpt_path.write_text(json.dumps({
                    "completed": sorted(completed),
                    "results": all_results,
                }))

    elapsed = time.time() - t_start
    logger.info("Ablation complete in %.1fs (%d results)", elapsed, len(all_results))

    # Save final results
    raw_path = out_dir / "ablation_results.json"
    raw_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", raw_path)

    # Clean checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    # --- Summary table ---
    print("\n" + "=" * 120)
    print("ABLATION RESULTS")
    print("=" * 120)
    header = (f"{'Sweep':>5} | {'λ_inv':>8} | {'λ_dd':>8} | {'λ_var':>8} | "
              f"{'PnL ($)':>10} | {'Sharpe':>8} | {'MaxDD ($)':>10} | "
              f"{'VaR95':>8} | {'Violations':>10} | {'Active%':>8}")
    print(header)
    print("-" * len(header))

    # Group by sweep point
    from collections import defaultdict
    by_point: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["inv_lambda"], r["dd_lambda"], r["var_lambda"], r["sweep_type"])
        by_point[key].append(r)

    for (inv_l, dd_l, var_l, stype), runs in sorted(by_point.items()):
        mean_pnl = np.mean([r["mean_pnl"] for r in runs])
        mean_sh = np.mean([r["mean_sharpe"] for r in runs])
        mean_dd = np.mean([r["mean_max_drawdown"] for r in runs])
        mean_var = np.mean([r["mean_var_95"] for r in runs])
        mean_vi = np.mean([r["mean_violations"] for r in runs])
        mean_ar = np.mean([r["mean_active_ratio"] for r in runs])
        print(
            f"{stype:>5} | {inv_l:>8.4f} | {dd_l:>8.4f} | {var_l:>8.4f} | "
            f"{mean_pnl:>+10,.0f} | {mean_sh:>+8.2f} | {mean_dd:>10,.0f} | "
            f"{mean_var:>8,.0f} | {mean_vi:>10.1f} | {mean_ar:>7.1%}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
