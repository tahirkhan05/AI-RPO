"""
3D wind model — extends 2D WindModel with an independent z-component.

Both x and z wind components use separate Ornstein-Uhlenbeck processes.
The altitude-dependent mean wind profile applies to both components
independently, scaled by their respective reference velocities.

See simulation/wind.py for the 2D version this extends.
"""

import math
import numpy as np
from dataclasses import dataclass
from simulation.wind import WindConfig


@dataclass
class WindConfig3D:
    v_ref_x:   float = 10.0   # m/s — mean north wind at h_ref
    v_ref_z:   float = 0.0    # m/s — mean east wind at h_ref (crosswind)
    h_ref:     float = 1000.0 # m   — reference altitude
    h_cap:     float = 30000.0# m   — wind caps above this
    alpha:     float = 0.14   # power-law exponent (neutral atmosphere)
    gust_sigma:float = 2.0    # m/s — gust standard deviation
    tau:       float = 10.0   # s   — OU correlation time


class WindModel3D:
    """
    3D wind: two independent OU processes for north (x) and east (z) components.
    Returns (wind_vx, wind_vz) per step.
    """

    def __init__(self, cfg: WindConfig3D = WindConfig3D(),
                 rng: np.random.Generator | None = None) -> None:
        self._cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng()
        self._gust_x: float = 0.0
        self._gust_z: float = 0.0

    def reset(self) -> None:
        self._gust_x = 0.0
        self._gust_z = 0.0

    def step(self, altitude: float, dt: float) -> tuple[float, float]:
        """Return (wind_vx, wind_vz) at this altitude and timestep."""
        cfg = self._cfg
        if altitude <= 0.0 or altitude >= cfg.h_cap:
            return 0.0, 0.0

        mean_scale = (altitude / cfg.h_ref) ** cfg.alpha

        # OU update for both components
        theta = dt / cfg.tau
        diffusion = cfg.gust_sigma * math.sqrt(dt) * self._rng.standard_normal(2)
        self._gust_x = self._gust_x * (1 - theta) + diffusion[0]
        self._gust_z = self._gust_z * (1 - theta) + diffusion[1]

        wind_vx = cfg.v_ref_x * mean_scale + self._gust_x
        wind_vz = cfg.v_ref_z * mean_scale + self._gust_z
        return float(wind_vx), float(wind_vz)


def sample_wind3d(rng: np.random.Generator) -> WindConfig3D:
    """Domain-randomised 3D wind config."""
    return WindConfig3D(
        v_ref_x=float(rng.uniform(-15.0, 15.0)),
        v_ref_z=float(rng.uniform(-10.0, 10.0)),
    )
