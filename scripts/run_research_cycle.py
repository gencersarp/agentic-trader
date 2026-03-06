"""Run one complete R-Quant research cycle and write a report.

Pipeline:
    DataCurator → AlphaDiscovery → RiskAnalyst → Compliance → Reflection

Usage:
    python scripts/run_research_cycle.py [--strategy "description"] [--symbol SYM]
    python scripts/run_research_cycle.py --strategy "momentum on daily closes" --symbol AAPL

In mock mode (USE_MOCK_LLM=true, default), no API key is needed.
Set USE_MOCK_LLM=false in .env and provide OPENAI_API_KEY for real LLM calls.
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

from agentic_trader.agents.rquants.alpha_discovery_agent import AlphaDiscoveryAgent
from agentic_trader.agents.rquants.compliance_agent import ComplianceAgent
from agentic_trader.agents.rquants.data_curator_agent import DataCuratorAgent
from agentic_trader.agents.rquants.reflection_agent import ReflectionAgent
from agentic_trader.agents.rquants.risk_analyst_agent import RiskAnalystAgent
from agentic_trader.config.settings import Settings
from agentic_trader.observability.tracing import get_tracer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("run_research_cycle")
console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an R-Quant research cycle.")
    p.add_argument(
        "--strategy",
        default="mean-reversion: fade z-score of 20-day return window",
        help="Free-text strategy description.",
    )
    p.add_argument("--symbol", default="SYNTHETIC", help="Symbol to research.")
    p.add_argument("--start", default="2020-01-01", help="Backtest start date.")
    p.add_argument("--end", default="2023-12-31", help="Backtest end date.")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    settings = Settings.from_env()
    tracer = get_tracer(settings.observability.log_dir)
    trace = tracer.start_trace("research_cycle_standalone", {"strategy": args.strategy})

    console.rule("[bold cyan]R-Quant Research Cycle")
    console.print(f"[bold]Strategy:[/bold] {args.strategy}")
    console.print(f"[bold]Symbol:[/bold]   {args.symbol}  ({args.start} – {args.end})")
    console.print(f"[bold]LLM mode:[/bold] {'MOCK' if settings.llm.use_mock else 'REAL — ' + settings.llm.model}")
    console.print()

    # ── 1. DataCurator ────────────────────────────────────────────────────────
    console.print("[bold yellow]Step 1/5 — Data Curation[/bold yellow]")
    curator = DataCuratorAgent()
    data = curator.prepare(args.symbol, args.start, args.end)
    console.print(f"  {curator.summary(data)}")
    tracer.log_event(trace, "data_curation_done", {"n_bars": data.n_bars})

    # ── 2. AlphaDiscovery ─────────────────────────────────────────────────────
    console.print("\n[bold yellow]Step 2/5 — Alpha Discovery + Backtest[/bold yellow]")
    alpha_agent = AlphaDiscoveryAgent(
        llm_config=settings.llm, backtest_config=settings.backtest
    )
    proposal = alpha_agent.discover(args.strategy, data)
    bt = proposal.backtest_result
    tracer.log_event(trace, "alpha_discovery_done", bt)

    bt_table = Table(title="Backtest Results", show_header=True)
    bt_table.add_column("Metric", style="cyan")
    bt_table.add_column("Value", justify="right")
    for k, v in bt.items():
        if k not in ("approved", "summary") and not isinstance(v, str):
            bt_table.add_row(k, f"{v:,.4g}" if isinstance(v, float) else str(v))
    console.print(bt_table)
    console.print(f"  Summary: {bt.get('summary', 'N/A')}")

    # ── 3. RiskAnalyst ────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Step 3/5 — Risk Analysis[/bold yellow]")
    risk_agent = RiskAnalystAgent(llm_config=settings.llm)
    risk_verdict = risk_agent.evaluate(bt, args.strategy)
    status_colour = "green" if risk_verdict.passed else "red"
    console.print(
        Panel(
            risk_verdict.narrative,
            title=f"[{status_colour}]Risk: {'PASS' if risk_verdict.passed else 'FAIL'}[/{status_colour}]",
        )
    )
    tracer.log_event(trace, "risk_analysis_done", {"passed": risk_verdict.passed})

    # ── 4. Compliance ─────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Step 4/5 — Compliance Check[/bold yellow]")
    compliance = ComplianceAgent(llm_config=settings.llm)
    comp_result = compliance.check(proposal.description, proposal.strategy_code)
    status_colour = "green" if comp_result.passed else "red"
    console.print(
        Panel(
            comp_result.narrative,
            title=f"[{status_colour}]Compliance: {'PASS' if comp_result.passed else 'FAIL'}[/{status_colour}]",
        )
    )
    tracer.log_event(trace, "compliance_done", {"passed": comp_result.passed})

    # ── 5. Reflection ─────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Step 5/5 — Reflection[/bold yellow]")
    reflection = ReflectionAgent(
        store_path=settings.output_path("reflections.json"),
        llm_config=settings.llm,
    )
    overall_pass = risk_verdict.passed and comp_result.passed
    if not overall_pass:
        lesson = reflection.reflect(
            proposal.description,
            bt,
            extra_context=f"risk={risk_verdict.passed}, compliance={comp_result.passed}",
        )
        console.print(f"  Lesson stored (id={lesson.id}): {lesson.lesson_text[:120]}...")
    else:
        console.print("  Strategy approved — no lesson needed.")
    tracer.log_event(trace, "reflection_done", {"overall_pass": overall_pass})

    # ── Write report ──────────────────────────────────────────────────────────
    outcome = {
        "strategy_description": proposal.description,
        "source": proposal.source,
        "backtest": bt,
        "risk_passed": risk_verdict.passed,
        "risk_narrative": risk_verdict.narrative,
        "compliance_passed": comp_result.passed,
        "compliance_narrative": comp_result.narrative,
        "overall_approved": overall_pass,
        "recent_lessons": [
            {"id": l.id, "text": l.lesson_text}
            for l in reflection.retrieve(n=3)
        ],
    }

    json_path = settings.output_path("research_report.json")
    json_path.write_text(json.dumps(outcome, indent=2))

    md_path = settings.output_path("research_report.md")
    md_path.write_text(_to_markdown(outcome))

    console.print()
    overall_colour = "bold green" if overall_pass else "bold red"
    console.rule(f"[{overall_colour}]OUTCOME: {'APPROVED' if overall_pass else 'REJECTED'}[/{overall_colour}]")
    console.print(f"  Report → {json_path}")
    console.print(f"  Report → {md_path}")

    tracer.end_trace(trace, status="ok" if overall_pass else "rejected")
    return outcome


def _to_markdown(outcome: dict) -> str:
    bt = outcome["backtest"]
    lines = [
        "# Research Report\n",
        f"**Strategy:** {outcome['strategy_description']}\n",
        f"**Overall:** {'✅ APPROVED' if outcome['overall_approved'] else '❌ REJECTED'}\n",
        "\n## Backtest Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in bt.items():
        if k not in ("approved", "summary", "prices_json") and not isinstance(v, str):
            lines.append(f"| {k} | {v:,.4g}" if isinstance(v, float) else f"| {k} | {v} |")
    lines += [
        f"\n**Summary:** {bt.get('summary', 'N/A')}\n",
        f"\n## Risk Assessment\n{outcome['risk_narrative']}\n",
        f"\n## Compliance\n{outcome['compliance_narrative']}\n",
    ]
    if outcome.get("recent_lessons"):
        lines.append("\n## Recent Lessons\n")
        for l in outcome["recent_lessons"]:
            lines.append(f"- **{l['id']}**: {l['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    run(parse_args())
