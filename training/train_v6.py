"""
PPO v6 — 3D PINN-guided + EKF guidance.

Trains in the 3D environment (RocketEnv3D) with:
  - 2D action space (pitch delta, yaw delta)
  - 15D observation
  - Per-episode 3D PINN reference (if pinn3d_param_v1.pt exists)
  - EKF3D state estimation

Same two-stage curriculum as v4/v5.

Usage:
    python -m training.train_v6
    python -m training.train_v6 --resume models/ppo_v6_stage1
"""

import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env3d import RocketEnv3D
from training.config import DEFAULT_TRAIN, TrainConfig
from training.callbacks import CheckpointCallback, ProgressCallback


def _make_env(randomize, seed, physics_guided=True, use_ekf=True):
    def _init():
        return RocketEnv3D(randomize=randomize, seed=seed,
                           physics_guided=physics_guided, use_ekf=use_ekf)
    return _init


def _linear_schedule(initial, final=1e-5):
    def schedule(progress):
        return final + (initial - final) * progress
    return schedule


def train(cfg: TrainConfig = DEFAULT_TRAIN, resume_path=None):
    device = "cpu"
    print(f"Device:     {device}")
    print(f"Mode:       3D PINN-guided + EKF (v6)")
    print(f"Obs space:  15D  |  Action: 2D (pitch, yaw)")
    print(f"Stage 1:    {cfg.stage1_steps:,} steps  (no randomization)")
    print(f"Stage 2:    {cfg.stage2_steps:,} steps  (full randomization)")
    print()

    os.makedirs("models", exist_ok=True)

    cbs = [
        CheckpointCallback(cfg.save_interval, "models", verbose=1),
        ProgressCallback(cfg.log_interval, cfg.total_steps),
    ]

    print("=== Stage 1: 3D curriculum warmup ===")
    env1 = make_vec_env(_make_env(False, 0, True, True), n_envs=4)
    env1 = VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if resume_path:
        model = PPO.load(resume_path, env=env1, device=device)
    else:
        model = PPO(
            "MlpPolicy", env1,
            learning_rate=_linear_schedule(cfg.learning_rate, cfg.lr_final),
            n_steps=cfg.n_steps, batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs, gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
            policy_kwargs=dict(net_arch=list(cfg.net_arch)),
            verbose=0, device=device, tensorboard_log="logs/tensorboard",
        )

    model.learn(cfg.stage1_steps, callback=cbs,
                reset_num_timesteps=resume_path is None,
                tb_log_name="stage1_v6")
    model.save("models/ppo_v6_stage1")
    env1.save("models/vecnorm_v6_stage1.pkl")
    print("Stage 1 saved -> models/ppo_v6_stage1.zip\n")

    print("=== Stage 2: 3D full domain randomization ===")
    env2 = make_vec_env(_make_env(True, 42, True, True), n_envs=4)
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model.set_env(env2)
    model.learn(cfg.stage2_steps, callback=cbs,
                reset_num_timesteps=False, tb_log_name="stage2_v6")
    model.save("models/ppo_v6_final")
    env2.save("models/vecnorm_v6_final.pkl")
    print("Final model saved -> models/ppo_v6_final.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(resume_path=args.resume)
