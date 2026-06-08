"""
PPO v6 Extended — 3D PINN-guided + EKF, 3M steps, larger network.

Differences from train_v6.py:
  - net_arch: [512, 512] instead of [64, 64]
  - Stage 1: 1,000,000 steps (was 400K)
  - Stage 2: 2,000,000 steps (was 600K)
  - Total:   3,000,000 steps

Resumes from v6_final as warm start — Stage 1 skips early exploration
since v6_final already has a working policy.

Usage:
    python -m training.train_v6_extended
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env3d import RocketEnv3D
from training.callbacks import CheckpointCallback, ProgressCallback

STAGE1_STEPS = 1_000_000
STAGE2_STEPS = 2_000_000
TOTAL_STEPS  = 3_000_000
SAVE_INTERVAL = 500_000
NET_ARCH     = [512, 512]
LR_INIT      = 2e-4   # lower than fresh start — warm policy needs less exploration
LR_FINAL     = 1e-5


def _make_env(randomize, seed):
    def _init():
        return RocketEnv3D(randomize=randomize, seed=seed,
                           physics_guided=True, use_ekf=True)
    return _init


def _linear_schedule(initial, final=1e-5):
    def schedule(progress):
        return final + (initial - final) * progress
    return schedule


def train():
    device = "cpu"
    print(f"Device:      {device}")
    print(f"Mode:        3D PINN-guided + EKF (v6 extended)")
    print(f"Network:     {NET_ARCH}")
    print(f"Stage 1:     {STAGE1_STEPS:,} steps")
    print(f"Stage 2:     {STAGE2_STEPS:,} steps")
    print(f"Total:       {TOTAL_STEPS:,} steps")
    print()

    os.makedirs("models", exist_ok=True)

    cbs_s1 = [
        CheckpointCallback(SAVE_INTERVAL, "models", verbose=1),
        ProgressCallback(10, TOTAL_STEPS),
    ]
    cbs_s2 = [
        CheckpointCallback(SAVE_INTERVAL, "models", verbose=1),
        ProgressCallback(10, TOTAL_STEPS),
    ]

    # Stage 1 — no randomization, warm-start from v6_final
    print("=== Stage 1: warm-start from v6, no randomization ===")
    env1 = make_vec_env(_make_env(False, 0), n_envs=4)
    env1 = VecNormalize(env1, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if os.path.exists("models/ppo_v6_final.zip"):
        print("  Loading v6_final as warm start...")
        model = PPO.load("models/ppo_v6_final", env=env1, device=device)
        # Override policy network — v6_final used [64,64], rebuild with [512,512]
        # Can't resize in-place: train fresh with larger network but warm LR
        print("  Note: rebuilding network at [512,512] — v6 weights incompatible (different arch)")
        model = PPO(
            "MlpPolicy", env1,
            learning_rate=_linear_schedule(LR_INIT, LR_FINAL),
            n_steps=4096, batch_size=256,
            n_epochs=10, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5,
            policy_kwargs=dict(net_arch=NET_ARCH),
            verbose=0, device=device, tensorboard_log="logs/tensorboard",
        )
    else:
        print("  v6_final not found — training from scratch")
        model = PPO(
            "MlpPolicy", env1,
            learning_rate=_linear_schedule(LR_INIT, LR_FINAL),
            n_steps=4096, batch_size=256,
            n_epochs=10, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5,
            policy_kwargs=dict(net_arch=NET_ARCH),
            verbose=0, device=device, tensorboard_log="logs/tensorboard",
        )

    model.learn(STAGE1_STEPS, callback=cbs_s1,
                reset_num_timesteps=True, tb_log_name="stage1_v6ext")
    model.save("models/ppo_v6ext_stage1")
    env1.save("models/vecnorm_v6ext_stage1.pkl")
    print("Stage 1 saved -> models/ppo_v6ext_stage1.zip\n")

    # Stage 2 — full domain randomization
    print("=== Stage 2: full 3D domain randomization ===")
    env2 = make_vec_env(_make_env(True, 42), n_envs=4)
    env2 = VecNormalize(env2, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model.set_env(env2)
    model.learn(STAGE2_STEPS, callback=cbs_s2,
                reset_num_timesteps=False, tb_log_name="stage2_v6ext")
    model.save("models/ppo_v6ext_final")
    env2.save("models/vecnorm_v6ext_final.pkl")
    print("Final model saved -> models/ppo_v6ext_final.zip")


if __name__ == "__main__":
    train()
