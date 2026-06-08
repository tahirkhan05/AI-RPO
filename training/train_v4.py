"""
PPO v4 training — PINN-guided physics-consistent curriculum.

Key difference from v3: RocketEnv is initialised with physics_guided=True.
This means:
  1. The reward reference comes from the parameterised PINN (pinn_param_v1.pt)
     queried per episode for the actual episode rocket config — not the fixed
     nominal RK4 trajectory used in v3.
  2. A quadratic physics consistency penalty is added to the reward, penalising
     states that deviate from the PINN's prediction for (t, p_episode).

The two-stage curriculum is preserved:
  Stage 1: no domain randomization, nominal rocket, PINN reference ≈ RK4
  Stage 2: full domain randomization, per-episode PINN reference

Models saved as ppo_v4_* to avoid overwriting v3 checkpoints.

See notes/07_phase6_integration.md for full design rationale.

Usage:
    python -m training.train_v4
    python -m training.train_v4 --resume models/ppo_v4_stage1
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env import RocketEnv
from training.config import DEFAULT_TRAIN, TrainConfig
from training.callbacks import CheckpointCallback, ProgressCallback


def _make_env(randomize: bool, seed: int, physics_guided: bool = True):
    def _init():
        return RocketEnv(randomize=randomize, seed=seed,
                         physics_guided=physics_guided)
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
    device = "cpu"   # MLP policy — CPU faster than GPU for small nets
    print(f"Device:         {device}")
    print(f"Mode:           PINN-guided physics-consistent (v4)")
    print(f"PINN model:     models/pinn_param_v1.pt")
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
    # No randomization — PINN reference ≈ RK4 nominal (same config).
    # This warmup lets the agent learn basic trajectory tracking before physics
    # guidance kicks in during Stage 2 randomisation.
    print("=== Stage 1: curriculum warmup (nominal config, PINN reference) ===")
    env1 = make_vec_env(_make_env(randomize=False, seed=0, physics_guided=True), n_envs=4)
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
        tb_log_name="stage1_v4",
    )
    model.save("models/ppo_v4_stage1")
    env1.save("models/vecnorm_v4_stage1.pkl")
    print("Stage 1 saved -> models/ppo_v4_stage1.zip\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # Full domain randomization — PINN reference adapts to each episode's config.
    # Each reset() queries the PINN for the sampled rocket, giving a physically
    # correct per-episode reference that v3 never had.
    print("=== Stage 2: full domain randomization (per-episode PINN reference) ===")
    env2 = make_vec_env(_make_env(randomize=True, seed=42, physics_guided=True), n_envs=4)
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model.set_env(env2)
    model.learn(
        total_timesteps=cfg.stage2_steps,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name="stage2_v4",
    )
    model.save("models/ppo_v4_final")
    env2.save("models/vecnorm_v4_final.pkl")
    print("Final model saved -> models/ppo_v4_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(resume_path=args.resume)
