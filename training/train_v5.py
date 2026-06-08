"""
PPO v5 training — PINN-guided + EKF state estimation.

Extends v4 by replacing ground-truth state observations with EKF-estimated
state from simulated IMU, barometer, and GPS sensors. This tests whether the
agent can learn to fly using only realistic sensor measurements.

Key difference from v4:
  v4: agent sees [x, y, vx, vy, ...] — ground truth (cheating)
  v5: agent sees [x̂, ŷ, v̂x, v̂y, ...] — EKF estimate from noisy sensors

Reward is still computed from true state (for training stability).
In real deployment: only EKF state is available.

See notes/08_phase7_ekf.md for full design rationale and math.

Usage:
    python -m training.train_v5
    python -m training.train_v5 --resume models/ppo_v5_stage1
"""

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env import RocketEnv
from training.config import DEFAULT_TRAIN, TrainConfig
from training.callbacks import CheckpointCallback, ProgressCallback


def _make_env(randomize: bool, seed: int,
              physics_guided: bool = True, use_ekf: bool = True):
    def _init():
        return RocketEnv(randomize=randomize, seed=seed,
                         physics_guided=physics_guided, use_ekf=use_ekf)
    return _init


def _linear_schedule(initial: float, final: float = 1e-5):
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
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    device = "cpu"
    print(f"Device:         {device}")
    print(f"Mode:           PINN-guided + EKF state estimation (v5)")
    print(f"PINN model:     models/pinn_param_v1.pt")
    print(f"Sensors:        barometer (s=5m), GPS pos (s=3m), GPS vel (s=0.1m/s)")
    print(f"Stage 1:        {cfg.stage1_steps:,} steps  (no randomization)")
    print(f"Stage 2:        {cfg.stage2_steps:,} steps  (full randomization)")
    print(f"LR:             {cfg.learning_rate} -> {cfg.lr_final} (linear decay)")
    print()

    os.makedirs("models", exist_ok=True)

    callbacks = [
        CheckpointCallback(cfg.save_interval, "models", verbose=1),
        ProgressCallback(cfg.log_interval, cfg.total_steps),
    ]

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    # Nominal config, PINN reference, EKF active.
    # Agent must learn to track through noisy observations from the start.
    print("=== Stage 1: curriculum warmup (nominal, PINN + EKF) ===")
    env1 = make_vec_env(
        _make_env(randomize=False, seed=0, physics_guided=True, use_ekf=True),
        n_envs=4
    )
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
        tb_log_name="stage1_v5",
    )
    model.save("models/ppo_v5_stage1")
    env1.save("models/vecnorm_v5_stage1.pkl")
    print("Stage 1 saved -> models/ppo_v5_stage1.zip\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # Full domain randomisation + EKF. Each episode has a different rocket AND
    # EKF initialises fresh with that rocket's config. The agent must generalise
    # over both config space and sensor noise simultaneously.
    print("=== Stage 2: full domain randomization (PINN + EKF) ===")
    env2 = make_vec_env(
        _make_env(randomize=True, seed=42, physics_guided=True, use_ekf=True),
        n_envs=4
    )
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model.set_env(env2)
    model.learn(
        total_timesteps=cfg.stage2_steps,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name="stage2_v5",
    )
    model.save("models/ppo_v5_final")
    env2.save("models/vecnorm_v5_final.pkl")
    print("Final model saved -> models/ppo_v5_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(resume_path=args.resume)
