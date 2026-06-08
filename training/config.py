"""
Training hyperparameters. All tunable values in one place.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # PPO hyperparameters
    learning_rate: float = 3e-4   # initial LR; decays linearly to lr_final
    lr_final: float = 1e-5        # LR at end of training
    n_steps: int = 4096           # timesteps per update (was 2048 — see BUG-011)
    batch_size: int = 128         # scaled with n_steps
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005       # reduced: let policy commit faster
    vf_coef: float = 0.5

    # Network
    net_arch: tuple = (64, 64)

    # Curriculum schedule
    stage1_steps: int = 400_000   # no randomization (was 200K)
    stage2_steps: int = 600_000   # full domain randomization (was 300K)
    total_steps: int = 1_000_000

    # Logging
    log_interval: int = 10
    save_interval: int = 200_000  # scaled with total steps


DEFAULT_TRAIN = TrainConfig()
