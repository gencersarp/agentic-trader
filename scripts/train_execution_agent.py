"""Train an RL execution agent (SB3 PPO) in the synthetic LOB environment.

Usage:
    python scripts/train_execution_agent.py [--timesteps N] [--algo PPO|TD3]

Saves the trained policy to models/execution_agent_policy.zip.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure package root is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_trader.config.settings import Settings
from agentic_trader.env.abides_env import TradingEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("train_execution_agent")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RL execution agent.")
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    p.add_argument("--algo", choices=["PPO", "TD3"], default=None, help="RL algorithm.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", action="store_true", help="Run quick evaluation after training.")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO, TD3                    # type: ignore[import-untyped]
        from stable_baselines3.common.env_util import make_vec_env # type: ignore[import-untyped]
        from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "stable-baselines3 is not installed. Run: pip install stable-baselines3"
        )
        sys.exit(1)

    settings = Settings.from_env()
    cfg = settings.training
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.algo:
        cfg.algorithm = args.algo

    logger.info("Training config: algo=%s, timesteps=%d", cfg.algorithm, cfg.total_timesteps)

    # --- build environment ---------------------------------------------------
    def make_env():
        return TradingEnv(config=settings.env, seed=args.seed)

    vec_env = make_vec_env(make_env, n_envs=cfg.n_envs, seed=args.seed)

    # --- instantiate algorithm -----------------------------------------------
    AlgoCls = {"PPO": PPO, "TD3": TD3}[cfg.algorithm]

    # TD3 requires a continuous action space but also needs a replay buffer.
    # For simplicity, use PPO-specific kwargs only when algo=PPO.
    common_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=cfg.learning_rate,
        verbose=1,
        seed=args.seed,
    )
    if cfg.algorithm == "PPO":
        model = AlgoCls(n_steps=cfg.n_steps, batch_size=cfg.batch_size, **common_kwargs)
    else:
        model = AlgoCls(batch_size=cfg.batch_size, **common_kwargs)

    logger.info("Starting training (%d timesteps)...", cfg.total_timesteps)
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)

    # --- save ----------------------------------------------------------------
    save_path = Path(cfg.policy_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("Policy saved to %s.zip", save_path)

    # --- optional evaluation -------------------------------------------------
    if args.eval:
        logger.info("Evaluating policy over %d episodes...", cfg.eval_episodes)
        eval_env = TradingEnv(config=settings.env, seed=args.seed + 999)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=cfg.eval_episodes, deterministic=True
        )
        logger.info(
            "Eval result: mean_reward=%.4f ± %.4f", mean_reward, std_reward
        )

    vec_env.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    train(parse_args())
