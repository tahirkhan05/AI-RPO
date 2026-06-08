"""
Target trajectory providers for the RL environment reward.

Two implementations:
  TargetTrajectory            — fixed nominal RK4 reference (used in v3 training)
  PhysicsGuidedTargetTrajectory — per-episode PINN reference (used in v4 training)

The PINN-based version queries the parameterised PINN surrogate (pinn_param_v1.pt)
at episode reset to get a physically consistent reference for the *actual* rocket
config being flown that episode. This is the key Phase 6 innovation.

See notes/07_phase6_integration.md for full math and design rationale.
"""

import os
import numpy as np
from scipy.interpolate import interp1d

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM, RocketConfig
from simulation.trajectory import run, TrajectoryResult


# ── Default model path — can be overridden via env var for testing ─────────────
_PINN_PARAM_PATH = os.environ.get(
    "PINN_PARAM_MODEL", "models/pinn_param_v1.pt"
)

# Precompute resolution: dt=0.1s gives 1850 points across 185s flight.
# Fine enough for interpolation (physics dt=0.01s), coarse enough to be fast.
_CACHE_DT = 0.1


def _build_baseline() -> TrajectoryResult:
    return run(DEFAULT_ROCKET, DEFAULT_SIM, wind=None)


class TargetTrajectory:
    """
    Fixed nominal RK4 reference trajectory (DEFAULT_ROCKET, no wind).
    Used by the v3 PPO agent. Kept for backward compatibility and benchmarking.
    """

    def __init__(self) -> None:
        baseline = _build_baseline()
        t = baseline.time
        self.t_max = float(t[-1])
        self._y  = interp1d(t, baseline.y,  bounds_error=False, fill_value=(baseline.y[0],  baseline.y[-1]))
        self._vy = interp1d(t, baseline.vy, bounds_error=False, fill_value=(baseline.vy[0], baseline.vy[-1]))
        self._vx = interp1d(t, baseline.vx, bounds_error=False, fill_value=(baseline.vx[0], baseline.vx[-1]))

    def query(self, t: float) -> tuple[float, float, float]:
        """Return (target_y, target_vy, target_vx) at time t."""
        return float(self._y(t)), float(self._vy(t)), float(self._vx(t))


class PhysicsGuidedTargetTrajectory:
    """
    Per-episode PINN reference trajectory for v4 PPO training.

    At episode reset, call set_episode(rocket, wind_v_ref) to precompute
    a full trajectory from the PINN parameterised surrogate for this episode's
    specific rocket configuration and wind. The result is cached and interpolated
    at each step — no PINN forward pass during the episode rollout itself.

    Falls back to TargetTrajectory (RK4 nominal) if the PINN model is missing.

    Math: see notes/07_phase6_integration.md Section 2.
    """

    def __init__(self, model_path: str = _PINN_PARAM_PATH) -> None:
        self._model = None
        self._t_max_model = DEFAULT_SIM.max_time
        self._fallback = TargetTrajectory()
        self.t_max = self._fallback.t_max

        # Lazy import torch — only needed when PINN model is used
        self._model_path = model_path
        self._loaded = False
        self._using_pinn = False

        # Interpolators — set by set_episode()
        self._y_interp  = None
        self._vy_interp = None
        self._vx_interp = None

    def _ensure_loaded(self) -> bool:
        """Load the PINN model on first use. Returns True if PINN available."""
        if self._loaded:
            return self._using_pinn

        self._loaded = True
        if not os.path.exists(self._model_path):
            print(f"[PhysicsGuidedTargetTrajectory] PINN not found at {self._model_path}, "
                  f"falling back to RK4 nominal reference.")
            self._using_pinn = False
            return False

        try:
            import torch
            from simulation.pinn_param import RocketPINNParam
            ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
            hidden = ckpt.get("hidden", 256)
            layers = ckpt.get("layers", 6)
            self._model = RocketPINNParam(hidden=hidden, layers=layers)
            self._model.load_state_dict(ckpt["model_state"])
            self._model.eval()
            self._t_max_model = ckpt["t_max"]
            self._using_pinn = True
            return True
        except Exception as e:
            print(f"[PhysicsGuidedTargetTrajectory] Failed to load PINN: {e}. "
                  f"Falling back to RK4.")
            self._using_pinn = False
            return False

    def set_episode(self, rocket: RocketConfig, wind_v_ref: float = 0.0) -> None:
        """
        Precompute and cache the PINN reference trajectory for this episode.
        Must be called at episode reset() before any query().

        rocket:      the sampled RocketConfig for this episode
        wind_v_ref:  the episode's reference wind speed (m/s), used as p[5]
                     — should be the wind model's v_ref, not the noisy per-step wind
        """
        if not self._ensure_loaded():
            # PINN unavailable — fallback uses same interpolators regardless of config
            return

        import torch
        from simulation.pinn_param import PARAM_RANGES, normalise_params

        # Clamp parameters to PINN training range (clip, don't extrapolate)
        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        p_raw = np.array([
            _clamp(rocket.mass_wet,   *PARAM_RANGES["mass_wet"][:2]),
            _clamp(rocket.mass_dry,   *PARAM_RANGES["mass_dry"][:2]),
            _clamp(rocket.thrust,     *PARAM_RANGES["thrust"][:2]),
            _clamp(rocket.burn_rate,  *PARAM_RANGES["burn_rate"][:2]),
            _clamp(rocket.drag_coeff, *PARAM_RANGES["drag_coeff"][:2]),
            _clamp(wind_v_ref,        *PARAM_RANGES["wind_vx"][:2]),
        ], dtype=np.float32)

        # Build time grid: 0 to t_max in steps of _CACHE_DT
        t_grid = np.arange(0.0, self._t_max_model + _CACHE_DT, _CACHE_DT,
                           dtype=np.float32)
        n = len(t_grid)

        # PINN forward pass — single batched inference (no grad needed)
        t_tensor = torch.tensor(t_grid).unsqueeze(-1)          # (N, 1)
        p_tensor = torch.tensor(p_raw).unsqueeze(0).expand(n, -1)  # (N, 6)

        with torch.no_grad():
            pred = self._model.predict(t_tensor, p_tensor).numpy()  # (N, 5)

        y_arr  = pred[:, 1]   # altitude
        vy_arr = pred[:, 3]   # vertical velocity
        vx_arr = pred[:, 2]   # horizontal velocity

        fill_y  = (float(y_arr[0]),  float(y_arr[-1]))
        fill_vy = (float(vy_arr[0]), float(vy_arr[-1]))
        fill_vx = (float(vx_arr[0]), float(vx_arr[-1]))

        self._y_interp  = interp1d(t_grid, y_arr,  bounds_error=False, fill_value=fill_y)
        self._vy_interp = interp1d(t_grid, vy_arr, bounds_error=False, fill_value=fill_vy)
        self._vx_interp = interp1d(t_grid, vx_arr, bounds_error=False, fill_value=fill_vx)
        self.t_max = float(t_grid[-1])

    def query(self, t: float) -> tuple[float, float, float]:
        """Return (target_y, target_vy, target_vx) at time t."""
        if not self._using_pinn or self._y_interp is None:
            return self._fallback.query(t)
        return (float(self._y_interp(t)),
                float(self._vy_interp(t)),
                float(self._vx_interp(t)))
