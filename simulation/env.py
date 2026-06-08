"""
RocketEnv — Gymnasium environment wrapping the 2D rocket physics simulation.

Observation space : 10-dimensional normalised vector (see notes/04_*)
  [x, y, vx, vy, mass_fraction, wind_est, dy_to_target, dvx_to_target,
   dvy_to_target, angle_deg]
Action space      : 1-dimensional continuous, normalised to [-1, 1]
                    maps to thrust angle delta per step

Flags:
  physics_guided : bool — use per-episode PINN reference (Phase 6)
  use_ekf        : bool — replace true state with EKF estimate in observation (Phase 7)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.config import RocketConfig, SimConfig, TRAINING_SIM
from simulation.physics import rk4_step
from simulation.wind import WindModel, WindConfig
from simulation.domain_randomization import sample_rocket, sample_wind
from simulation.target_trajectory import TargetTrajectory, PhysicsGuidedTargetTrajectory
from simulation.sensors import SensorSuite, SensorConfig
from simulation.ekf import EKF

# Observation normalisation scales (rough max expected values)
_OBS_SCALE = np.array([
    20_000.0,   # x              (m)
    35_000.0,   # y              (m)
    600.0,      # vx             (m/s)
    900.0,      # vy             (m/s)
    1.0,        # mass_fraction  (already 0-1)
    30.0,       # wind_vx_est    (m/s)
    35_000.0,   # dy_to_target
    600.0,      # dvx_to_target
    900.0,      # dvy_to_target
    90.0,       # angle_deg      (degrees) — agent sees its own thrust angle
], dtype=np.float32)

# Reward weights — tracking must dominate; fuel is a tie-breaker only.
# See notes/00_problems_and_decisions_log BUG-010 for why these ratios matter.
_W_ALT    = 2.0   # altitude tracking (primary signal)
_W_VEL    = 1.0   # velocity tracking
_W_FUEL   = 0.05  # fuel efficiency (tiny — must not be passively maximisable)
_W_SMOOTH = 0.1   # control smoothness
_W_PINN   = 0.5   # physics consistency penalty (Phase 6 only, when physics_guided=True)

# Action scaling: ±1 maps to ±max_delta_rad per step
_MAX_DELTA_DEG = 5.0
_MAX_DELTA_RAD = math.radians(_MAX_DELTA_DEG)

# Angle limits (from horizontal)
_MIN_ANGLE_RAD = math.radians(10.0)
_MAX_ANGLE_RAD = math.radians(90.0)

# Landing safety: max vertical speed at touchdown
_MAX_LANDING_VY = -50.0  # m/s — faster than this = crash


class RocketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim: SimConfig = TRAINING_SIM,
                 randomize: bool = True, seed: int | None = None,
                 physics_guided: bool = False,
                 use_ekf: bool = False) -> None:
        super().__init__()
        self._sim = sim
        self._randomize = randomize
        self._physics_guided = physics_guided
        self._use_ekf = use_ekf
        self._rng = np.random.default_rng(seed)
        self._target = (PhysicsGuidedTargetTrajectory()
                        if physics_guided else TargetTrajectory())

        # EKF and sensor objects — initialised at first reset()
        self._sensors: SensorSuite | None = (
            SensorSuite(SensorConfig(dt=sim.dt), rng=self._rng) if use_ekf else None
        )
        self._ekf: EKF | None = None  # built at reset() when rocket config is known

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # episode state — initialised in reset() before any step() call
        self._state: np.ndarray = np.zeros(5)
        self._angle_rad: float = 0.0
        self._t: float = 0.0
        self._fuel_total: float = 0.0

        # guaranteed non-None after reset(); placeholder avoids Optional overhead
        from simulation.config import DEFAULT_ROCKET
        self._rocket: RocketConfig = DEFAULT_ROCKET
        self._wind: WindModel = WindModel(WindConfig(), self._rng)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)  # required: sets self._np_random for Gymnasium
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._randomize:
            self._rocket = sample_rocket(self._rng)
            wind_cfg = sample_wind(self._rng)
        else:
            from simulation.config import DEFAULT_ROCKET
            self._rocket = DEFAULT_ROCKET
            wind_cfg = WindConfig()

        self._wind = WindModel(wind_cfg, self._rng)
        self._state = np.array([0.0, 0.0, 0.0, 0.0, self._rocket.mass_wet])
        self._angle_rad = math.radians(self._sim.launch_angle_deg)
        self._t = 0.0
        self._fuel_total = self._rocket.mass_wet - self._rocket.mass_dry

        # Update PINN reference for this episode's specific rocket config
        if self._physics_guided:
            self._target.set_episode(self._rocket, wind_v_ref=wind_cfg.v_ref)

        # Reset EKF and sensors with the new rocket config and known initial state
        if self._use_ekf:
            self._ekf = EKF(self._rocket, self._sim.dt)
            self._ekf.reset(self._state)
            self._sensors.reset()

        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Apply action — update thrust angle
        delta = float(np.clip(action[0], -1.0, 1.0)) * _MAX_DELTA_RAD
        self._angle_rad = float(np.clip(
            self._angle_rad + delta,
            _MIN_ANGLE_RAD, _MAX_ANGLE_RAD
        ))

        # 2. Wind at current altitude
        _, y, _, _, _ = self._state
        wind_vx = self._wind.step(y, self._sim.dt)

        # 3. Advance physics
        self._state = rk4_step(
            self._state, self._angle_rad, self._sim.dt,
            self._rocket, wind_vx=wind_vx
        )
        self._t += self._sim.dt

        # 3b. EKF predict + update (when enabled)
        if self._use_ekf and self._ekf is not None:
            self._ekf.predict(self._angle_rad, wind_vx)
            meas = self._sensors.measure(self._state)
            self._ekf.update_baro(meas["baro"])
            if meas["gps_pos"] is not None:
                self._ekf.update_gps(meas["gps_pos"], meas["gps_vel"])

        # 4. Reward — always uses TRUE state for training stability
        reward, info = self._compute_reward(delta)

        # 5. Termination
        _, y, _, vy, _ = self._state
        landed = bool(y <= 0.0 and self._t > 0.5)
        timeout = bool(self._t >= self._sim.max_time)

        if landed:
            if vy >= _MAX_LANDING_VY:
                reward += 100.0
                info["outcome"] = "landed_safe"
            else:
                reward -= 100.0
                info["outcome"] = "crashed"

        return self._observe(), float(reward), landed, timeout, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        # When EKF is active, the agent sees estimated state, not ground truth.
        # Reward is still computed from true state — only the observation changes.
        if self._use_ekf and self._ekf is not None:
            obs_state = self._ekf.state
        else:
            obs_state = self._state
        x, y, vx, vy, mass = obs_state
        fuel_remaining = max(0.0, mass - self._rocket.mass_dry)
        mass_fraction = fuel_remaining / self._fuel_total if self._fuel_total > 0 else 0.0

        # noisy wind estimate (true wind + 10% gaussian noise)
        true_wind = self._wind._gust + self._wind._mean_wind(y)
        wind_est = true_wind * self._rng.normal(1.0, 0.10)

        target_y, target_vy, target_vx = self._target.query(self._t)

        raw = np.array([
            x,
            y,
            vx,
            vy,
            mass_fraction,
            wind_est,
            target_y - y,
            target_vx - vx,
            target_vy - vy,
            math.degrees(self._angle_rad),   # agent sees its own thrust angle
        ], dtype=np.float32)

        return np.clip(raw / _OBS_SCALE, -3.0, 3.0)

    def _compute_reward(self, delta_rad: float) -> tuple[float, dict]:
        _, y, vx, vy, mass = self._state
        target_y, target_vy, target_vx = self._target.query(self._t)

        fuel_remaining = max(0.0, mass - self._rocket.mass_dry)
        mass_fraction = fuel_remaining / self._fuel_total if self._fuel_total > 0 else 0.0

        r_alt    = -_W_ALT   * abs(target_y  - y)  / _OBS_SCALE[6]
        r_vel    = -_W_VEL   * abs(target_vy - vy) / _OBS_SCALE[8]
        r_fuel   =  _W_FUEL  * mass_fraction
        r_smooth = -_W_SMOOTH * (delta_rad / _MAX_DELTA_RAD) ** 2

        # Physics consistency penalty (only when physics_guided=True).
        # Quadratic: small deviations negligible, large unphysical deviations penalised.
        # See notes/07_phase6_integration.md Section 3 for derivation and weight choice.
        r_physics = 0.0
        if self._physics_guided:
            r_physics = -_W_PINN * (
                ((target_y  - y)  / _OBS_SCALE[6]) ** 2 +
                ((target_vy - vy) / _OBS_SCALE[8]) ** 2
            )

        total = r_alt + r_vel + r_fuel + r_smooth + r_physics

        info = {
            "r_alt": r_alt, "r_vel": r_vel,
            "r_fuel": r_fuel, "r_smooth": r_smooth,
            "r_physics": r_physics,
            "t": self._t, "y": y, "mass_fraction": mass_fraction,
        }
        return total, info
