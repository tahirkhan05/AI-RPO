"""
Wind model: altitude-varying mean profile + Ornstein-Uhlenbeck turbulent gusts.
Returns a horizontal wind force (N) to add to the x-axis force at each step.
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class WindConfig:
    v_ref: float = 10.0       # m/s — mean wind speed at h_ref
    h_ref: float = 10.0       # m   — reference height (standard met height)
    alpha: float = 0.14       # shear exponent (1/7 law for open terrain)
    h_cap: float = 30_000.0   # m   — altitude above which wind is zero
    gust_theta: float = 0.5   # OU mean-reversion rate
    gust_sigma: float = 1.5   # OU volatility (m/s per sqrt(s))


class WindModel:
    """
    Stateful wind model. Call step() each simulation timestep to get
    the current horizontal wind velocity (m/s).
    Wind velocity is converted to force by the caller (needs air density + area).
    """

    def __init__(self, cfg: WindConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng
        self._gust: float = 0.0  # current OU gust velocity (m/s)

    def step(self, altitude: float, dt: float) -> float:
        """Return total horizontal wind velocity (m/s) at this altitude and time."""
        mean = self._mean_wind(altitude)
        self._gust = self._ou_step(self._gust, dt)
        return mean + self._gust

    def reset(self) -> None:
        self._gust = 0.0

    # --- private ---

    def _mean_wind(self, altitude: float) -> float:
        if altitude <= 0.0 or altitude >= self._cfg.h_cap:
            return 0.0
        return self._cfg.v_ref * (altitude / self._cfg.h_ref) ** self._cfg.alpha

    def _ou_step(self, w: float, dt: float) -> float:
        cfg = self._cfg
        drift = -cfg.gust_theta * w * dt
        diffusion = cfg.gust_sigma * math.sqrt(dt) * self._rng.standard_normal()
        return w + drift + diffusion
