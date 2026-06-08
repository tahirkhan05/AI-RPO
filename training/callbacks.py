"""
Custom SB3 callbacks for checkpointing and progress logging.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """Save model every `save_freq` steps."""

    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            path = os.path.join(self._save_dir, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  [checkpoint] saved: {path}.zip")
        return True


class ProgressCallback(BaseCallback):
    """
    Print a one-line progress summary every `log_freq` PPO updates.

    Reads episode stats from model.ep_info_buffer (SB3's internal deque),
    which is populated correctly regardless of when _on_rollout_end fires.
    This fixes the "no ep yet" problem caused by reading logger.name_to_value
    before SB3 flushes it (BUG-013 in notes/00_problems_and_decisions_log.md).
    """

    def __init__(self, log_freq: int = 10, total_steps: int = 500_000) -> None:
        super().__init__()
        self._log_freq = log_freq
        self._total = total_steps
        self._update = 0

    def _on_rollout_end(self) -> None:
        self._update += 1
        if self._update % self._log_freq != 0:
            return

        # ep_info_buffer is a deque of {"r": ep_reward, "l": ep_length, "t": time}
        # populated by SB3's Monitor wrapper after every completed episode.
        # This is always up-to-date — no flush timing issues.
        buf = self.model.ep_info_buffer
        if buf:
            rew  = float(np.mean([ep["r"] for ep in buf]))
            elen = float(np.mean([ep["l"] for ep in buf]))
            rew_str  = f"{rew:+8.2f}"
            elen_str = f"{elen:6.0f}"
        else:
            rew_str  = "  (no ep yet)"
            elen_str = "   ---"

        kl  = self.logger.name_to_value.get("train/approx_kl", float("nan"))
        pct = 100 * self.num_timesteps / self._total

        print(
            f"  step {self.num_timesteps:>7,} ({pct:4.1f}%) | "
            f"rew {rew_str} | "
            f"ep_len {elen_str} | "
            f"kl {kl:.4f}"
        )

    def _on_step(self) -> bool:
        return True
