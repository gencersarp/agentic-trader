# Agentic Trader — PoC Multi-Agent Trading System

A hierarchical, risk-aware multi-agent trading system that couples MARL execution agents
with LLM-based Research Quants (R-Quants) inside a realistic LOB simulation.

## Architecture overview

```
Layer 3  R-Quant LLM agents (AutoGen)       research cycle, alpha discovery
           DataCurator → AlphaDiscovery → RiskAnalyst → Compliance → Reflection
Layer 2  RegimeAgent + StrategyGovernanceAgent     mid-frequency strategy selection
Layer 1  ExecutionAgent(s) (SB3/PPO)               per-step LOB order placement
Layer 0  TradingEnv (synthetic LOB / ABIDES-MARL)  market simulation
           ↑ every order passes through RiskGateway (hard deterministic constraints)
```

## Installation

```bash
# 1. Clone and enter project
git clone <repo> && cd agentic-trader

# 2. Create a virtual environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate

# 3. Install package + dependencies
pip install -e ".[dev]"

# 4. (Optional) set your LLM API key for real R-Quant conversations
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

`.env.example`:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
USE_MOCK_LLM=true          # set false to use real API calls
```

## Running

### 1. Train the execution agent (RL/PPO in synthetic LOB)
```bash
python scripts/train_execution_agent.py
# Saves policy to models/execution_agent_policy.zip
```

### 2. Run a research cycle (LLM R-Quants propose + backtest a strategy)
```bash
python scripts/run_research_cycle.py
# Writes output/research_report.json and output/research_report.md
# Uses mock LLM by default; set USE_MOCK_LLM=false in .env to use real API
```

### 3. End-to-end demo (trained agent + regime logic + risk gateway)
```bash
python scripts/demo_end_to_end.py
# Runs one episode through the full Orchestrator and prints summary stats
```

## Project layout

```
agentic_trader/
  config/settings.py          — dataclass-based configuration
  env/abides_env.py           — Gym-compatible LOB environment (synthetic; ABIDES-pluggable)
  risk/risk_metrics.py        — VaR, Sharpe, Sortino, drawdown
  risk/risk_gateway.py        — deterministic pre-trade risk filter
  agents/
    execution_agent.py        — RL policy wrapper with local constraints
    regime_agent.py           — rule-based market regime classifier
    strategy_governance_agent.py — policy registry + regime-aware selection
    rquants/
      data_curator_agent.py   — data preparation agent
      alpha_discovery_agent.py — LLM strategy proposal + backtest execution
      risk_analyst_agent.py   — risk metric interpretation agent
      compliance_agent.py     — keyword-based compliance check
      reflection_agent.py     — lesson storage + retrieval (JSON store)
  orchestrator/orchestrator.py — training loop + research cycle coordinator
  observability/tracing.py    — structured logging (OpenTelemetry-ready stubs)
scripts/
  train_execution_agent.py
  run_research_cycle.py
  demo_end_to_end.py
```

## Swapping in ABIDES-MARL

1. Install ABIDES-MARL (see `pyproject.toml` optional deps).
2. In `agentic_trader/env/abides_env.py`, replace `_evolve_mid_price` and
   `_execute_market_order` with calls to the real ABIDES exchange agent.
   The Gym interface (`reset`, `step`, observation/action spaces) stays unchanged.

## Extending R-Quants

Set `USE_MOCK_LLM=false` and provide an `OPENAI_API_KEY`. The AutoGen group-chat in
`run_research_cycle.py` will then use real GPT-4o (or any OpenAI-compatible model)
to write and critique strategies. Switch the model in `config/settings.py`.
