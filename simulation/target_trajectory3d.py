"""
3D target trajectory providers.

TargetTrajectory3D        — fixed RK4 nominal reference (no control, gravity turn)
PhysicsGuidedTargetTrajectory3D — per-episode 3D PINN reference

query3d(t) returns (target_y, target_vy, target_vx, target_vz)
"""

import os
import numpy as np
from scipy.interpolate import interp1d

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM, RocketConfig
from simulation.physics3d import rk4_step3d

_PINN3D_PATH = os.environ.get("PINN3D_MODEL", "models/pinn3d_param_v1.pt")
_CACHE_DT = 0.1
_LAUNCH_PITCH = np.radians(85.0)


def _run_nominal_3d():
    """RK4 gravity-turn trajectory for nominal rocket, no wind, no yaw."""
    rocket = DEFAULT_ROCKET
    sim = DEFAULT_SIM
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rocket.mass_wet])
    pitch = _LAUNCH_PITCH
    t = 0.0
    times, states = [t], [state.copy()]

    while t < sim.max_time and state[1] >= 0.0:
        state = rk4_step3d(state, pitch, 0.0, sim.dt, rocket, 0.0, 0.0)
        t += sim.dt
        times.append(t)
        states.append(state.copy())

    times  = np.array(times)
    states = np.array(states)
    return times, states


class TargetTrajectory3D:
    """Fixed nominal 3D RK4 reference — backward compatible default."""

    def __init__(self) -> None:
        times, states = _run_nominal_3d()
        fv = lambda arr: (float(arr[0]), float(arr[-1]))
        self._y  = interp1d(times, states[:,1], bounds_error=False, fill_value=fv(states[:,1]))
        self._vy = interp1d(times, states[:,4], bounds_error=False, fill_value=fv(states[:,4]))
        self._vx = interp1d(times, states[:,3], bounds_error=False, fill_value=fv(states[:,3]))
        self._vz = interp1d(times, states[:,5], bounds_error=False, fill_value=fv(states[:,5]))
        self.t_max = float(times[-1])

    def query3d(self, t: float) -> tuple[float, float, float, float]:
        """Return (target_y, target_vy, target_vx, target_vz)."""
        return (float(self._y(t)), float(self._vy(t)),
                float(self._vx(t)), float(self._vz(t)))


class PhysicsGuidedTargetTrajectory3D:
    """Per-episode 3D PINN reference. Falls back to RK4 if model missing."""

    def __init__(self, model_path: str = _PINN3D_PATH) -> None:
        self._model = None
        self._model_path = model_path
        self._loaded = False
        self._using_pinn = False
        self._fallback = TargetTrajectory3D()
        self.t_max = self._fallback.t_max
        self._t_max_model = DEFAULT_SIM.max_time
        self._y_interp = self._vy_interp = self._vx_interp = self._vz_interp = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._using_pinn
        self._loaded = True
        if not os.path.exists(self._model_path):
            self._using_pinn = False
            return False
        try:
            import torch
            from simulation.pinn3d import RocketPINN3D
            ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
            hidden = ckpt.get("hidden", 256)
            layers = ckpt.get("layers", 6)
            self._model = RocketPINN3D(hidden=hidden, layers=layers)
            self._model.load_state_dict(ckpt["model_state"])
            self._model.eval()
            self._t_max_model = ckpt.get("t_max", 185.0)
            self._using_pinn = True
            return True
        except Exception as e:
            print(f"[PhysicsGuidedTargetTrajectory3D] Failed: {e}. Using RK4.")
            self._using_pinn = False
            return False

    def set_episode(self, rocket: RocketConfig,
                    wind_vx_ref: float = 0.0, wind_vz_ref: float = 0.0) -> None:
        if not self._ensure_loaded():
            return

        import torch
        from simulation.pinn3d import PARAM_RANGES_3D, PARAM_KEYS_3D

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        p_raw = np.array([
            _clamp(rocket.mass_wet,   *PARAM_RANGES_3D["mass_wet"][:2]),
            _clamp(rocket.mass_dry,   *PARAM_RANGES_3D["mass_dry"][:2]),
            _clamp(rocket.thrust,     *PARAM_RANGES_3D["thrust"][:2]),
            _clamp(rocket.burn_rate,  *PARAM_RANGES_3D["burn_rate"][:2]),
            _clamp(rocket.drag_coeff, *PARAM_RANGES_3D["drag_coeff"][:2]),
            _clamp(wind_vx_ref,       *PARAM_RANGES_3D["wind_vx"][:2]),
            _clamp(wind_vz_ref,       *PARAM_RANGES_3D["wind_vz"][:2]),
        ], dtype=np.float32)

        t_grid = np.arange(0.0, self._t_max_model + _CACHE_DT,
                           _CACHE_DT, dtype=np.float32)
        n = len(t_grid)
        t_tensor = torch.tensor(t_grid).unsqueeze(-1)
        p_tensor = torch.tensor(p_raw).unsqueeze(0).expand(n, -1)

        with torch.no_grad():
            pred = self._model.predict(t_tensor, p_tensor).numpy()  # (N, 7)

        fv = lambda arr: (float(arr[0]), float(arr[-1]))
        self._y_interp  = interp1d(t_grid, pred[:,1], bounds_error=False, fill_value=fv(pred[:,1]))
        self._vy_interp = interp1d(t_grid, pred[:,4], bounds_error=False, fill_value=fv(pred[:,4]))
        self._vx_interp = interp1d(t_grid, pred[:,3], bounds_error=False, fill_value=fv(pred[:,3]))
        self._vz_interp = interp1d(t_grid, pred[:,5], bounds_error=False, fill_value=fv(pred[:,5]))
        self.t_max = float(t_grid[-1])

    def query3d(self, t: float) -> tuple[float, float, float, float]:
        if not self._using_pinn or self._y_interp is None:
            return self._fallback.query3d(t)
        return (float(self._y_interp(t)),  float(self._vy_interp(t)),
                float(self._vx_interp(t)), float(self._vz_interp(t)))
