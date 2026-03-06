"""End-to-end demo: trained policy → Orchestrator → single episode → stats.

Loads a pre-trained ExecutionAgent (or falls back to a heuristic policy if no
trained model is found), wires up all system components, runs one episode
through the full Orchestrator pipeline, and prints a rich summary.

Usage:
    python scripts/demo_end_to_end.py [--episodes N] [--render]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_trader.agents.execution_agent import (
    ExecutionAgent,
    MeanReversionHeuristicPolicy,
    RandomPolicy,
)
from agentic_trader.agents.regime_agent import RegimeAgent
from agentic_trader.agents.strategy_governance_agent import StrategyGovernanceAgent
from agentic_trader.config.settings import Settings
from agentic_trader.env.abides_env import TradingEnv
from agentic_trader.observability.tracing import get_tracer
from agentic_trader.orchestrator.orchestrator import Orchestrator
from agentic_trader.risk.risk_gateway import RiskGateway
from agentic_trader.risk.risk_metrics import RiskEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end demo of the agentic trader.")
    p.add_argument("--episodes", type=int, default=3, help="Number of demo episodes.")
    p.add_argument("--render", action="store_true", help="Print env state each step.")
    p.add_argument(
        "--policy",
        choices=["trained", "heuristic", "random"],
        default="heuristic",
        help="Which execution policy to use.",
    )
    return p.parse_args()


def build_execution_agent(policy_type: str, settings: Settings) -> ExecutionAgent:
    """Load or construct an ExecutionAgent based on the requested policy type."""
    local_limits = settings.local_limits
    symbol = settings.env.symbol

    if policy_type == "trained":
        model_path = Path(settings.training.policy_save_path)
        if model_path.with_suffix(".zip").exists():
            console.print(f"[green]Loading trained policy from {model_path}.zip[/green]")
            return ExecutionAgent.load(model_path, symbol=symbol, local_limits=local_limits)
        console.print(
            "[yellow]Trained policy not found — falling back to heuristic.[/yellow]\n"
            f"  Expected: {model_path}.zip\n"
            "  Run: python scripts/train_execution_agent.py"
        )

    if policy_type in ("heuristic", "trained"):  # fallback
        console.print("[cyan]Using MeanReversionHeuristicPolicy.[/cyan]")
        return ExecutionAgent(
            policy=MeanReversionHeuristicPolicy(),
            symbol=symbol,
            local_limits=local_limits,
            agent_id="heuristic_agent",
        )

    # random
    console.print("[yellow]Using RandomPolicy.[/yellow]")
    return ExecutionAgent(
        policy=RandomPolicy(),
        symbol=symbol,
        local_limits=local_limits,
        agent_id="random_agent",
    )


def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()

    console.rule("[bold magenta]Agentic Trader — End-to-End Demo[/bold magenta]")
    console.print(f"Policy: [bold]{args.policy}[/bold]  |  Episodes: {args.episodes}")
    console.print()

    # ── Build components ──────────────────────────────────────────────────────
    env = TradingEnv(config=settings.env, seed=0)
    exec_agent = build_execution_agent(args.policy, settings)

    risk_engine = RiskEngine()
    risk_gateway = RiskGateway(global_limits=settings.global_limits, risk_engine=risk_engine)
    regime_agent = RegimeAgent()
    governance_agent = StrategyGovernanceAgent()
    tracer = get_tracer(settings.observability.log_dir)

    orchestrator = Orchestrator(
        env=env,
        execution_agents={"primary": exec_agent},
        risk_gateway=risk_gateway,
        regime_agent=regime_agent,
        strategy_agent=governance_agent,
        settings=settings,
        tracer=tracer,
    )

    # ── Run episodes ──────────────────────────────────────────────────────────
    console.print("[bold]Running episodes...[/bold]")
    history = orchestrator.training_loop(n_episodes=args.episodes)

    # ── Print per-episode table ───────────────────────────────────────────────
    ep_table = Table(title="Episode Summary", show_lines=False)
    ep_table.add_column("Ep", style="bold", justify="right")
    ep_table.add_column("PnL ($)", justify="right")
    ep_table.add_column("Reward", justify="right")
    ep_table.add_column("Orders", justify="right")
    ep_table.add_column("Rejected", justify="right")
    ep_table.add_column("Dom. Regime")
    ep_table.add_column("Sharpe", justify="right")

    for ep_stat in history:
        dom_regime = max(ep_stat.regime_counts, key=ep_stat.regime_counts.get, default="N/A")
        sharpe = (
            f"{ep_stat.performance.sharpe_ratio:.2f}"
            if ep_stat.performance else "N/A"
        )
        pnl_colour = "green" if ep_stat.total_pnl >= 0 else "red"
        ep_table.add_row(
            str(ep_stat.episode),
            f"[{pnl_colour}]{ep_stat.total_pnl:+,.0f}[/{pnl_colour}]",
            f"{ep_stat.total_reward:.2f}",
            str(ep_stat.n_orders_submitted),
            str(ep_stat.n_orders_rejected),
            dom_regime,
            sharpe,
        )

    console.print(ep_table)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    agg = orchestrator.summary_stats()
    agg_table = Table(title="Aggregate Statistics", show_header=True)
    agg_table.add_column("Metric", style="cyan")
    agg_table.add_column("Value", justify="right")
    for k, v in agg.items():
        agg_table.add_row(
            k,
            f"{v:.4f}" if isinstance(v, float) else str(v),
        )
    console.print(agg_table)

    # ── Risk gateway rejection breakdown ─────────────────────────────────────
    rejection_counts = risk_gateway.rejection_counts
    if rejection_counts:
        console.print(
            Panel(
                "\n".join(f"  {k}: {v}" for k, v in rejection_counts.items()),
                title="[red]Risk Gateway Rejections[/red]",
            )
        )
    else:
        console.print("[green]No orders were rejected by the Risk Gateway.[/green]")

    # ── Save aggregate stats ──────────────────────────────────────────────────
    out_path = settings.output_path("demo_stats.json")
    out_path.write_text(json.dumps(agg, indent=2))
    console.print(f"\nAggregate stats saved to {out_path}")

    console.rule("[bold magenta]Demo complete[/bold magenta]")


if __name__ == "__main__":
    run(parse_args())
