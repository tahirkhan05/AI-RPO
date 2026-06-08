"""
PPO training script — two-stage curriculum.

Stage 1: no randomization, nominal rocket, light wind  (0 -> stage1_steps)
Stage 2: full domain randomization                     (stage1_steps -> total_steps)

Usage:
    python -m training.train
    python -m training.train --resume models/ppo_v3_stage1
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env import RocketEnv
from training.config import DEFAULT_TRAIN, TrainConfig
from training.callbacks import CheckpointCallback, ProgressCallback


def _make_env(randomize: bool, seed: int):
    def _init():
        return RocketEnv(randomize=randomize, seed=seed)
    return _init


def _linear_schedule(initial: float, final: float = 1e-5):
    """Linear LR decay from initial (progress=1.0) to final (progress=0.0)."""
    def schedule(progress: float) -> float:
        return final + (initial - final) * progress
    return schedule


def build_model(env, cfg: TrainConfig, device: str) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=_linear_schedule(cfg.learning_rate, cfg.lr_final),
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        policy_kwargs=dict(net_arch=list(cfg.net_arch)),
        verbose=0,
        device=device,
        tensorboard_log="logs/tensorboard",
    )


def train(cfg: TrainConfig = DEFAULT_TRAIN, resume_path: str | None = None) -> None:
    # MLP policies train faster on CPU — see notes/00_problems_and_decisions_log DECISION-004
    device = "cpu"
    print(f"Device:  {device}")
    print(f"Stage 1: {cfg.stage1_steps:,} steps  (no randomization)")
    print(f"Stage 2: {cfg.stage2_steps:,} steps  (full randomization)")
    print(f"LR:      {cfg.learning_rate} -> {cfg.lr_final} (linear decay)")
    print()

    os.makedirs("models", exist_ok=True)

    callbacks = [
        CheckpointCallback(cfg.save_interval, "models", verbose=1),
        ProgressCallback(cfg.log_interval, cfg.total_steps),
    ]

    # ── Stage 1 ──────────────────────────────────────────────────────────
    print("=== Stage 1: curriculum warmup ===")
    env1 = make_vec_env(_make_env(randomize=False, seed=0), n_envs=4)
    env1 = VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env1, device=device)
    else:
        model = build_model(env1, cfg, device)

    model.learn(
        total_timesteps=cfg.stage1_steps,
        callback=callbacks,
        reset_num_timesteps=resume_path is None,
        tb_log_name="stage1_v3",
    )
    model.save("models/ppo_v3_stage1")
    env1.save("models/vecnorm_v3_stage1.pkl")
    print("Stage 1 saved -> models/ppo_v3_stage1.zip\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    print("=== Stage 2: full domain randomization ===")
    env2 = make_vec_env(_make_env(randomize=True, seed=42), n_envs=4)
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model.set_env(env2)
    model.learn(
        total_timesteps=cfg.stage2_steps,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name="stage2_v3",
    )
    model.save("models/ppo_v3_final")
    env2.save("models/vecnorm_v3_final.pkl")
    print("Final model saved -> models/ppo_v3_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint zip to resume from")
    args = parser.parse_args()
    train(resume_path=args.resume)
