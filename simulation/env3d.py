"""
RocketEnv3D — 3D Gymnasium environment.

State:       [x, y, z, vx, vy, vz, mass]   (7D)
Observation: [x, y, z, vx, vy, vz,
              mass_fraction,
              wind_vx_est, wind_vz_est,
              dy_to_target, dvx_to_target, dvy_to_target, dvz_to_target,
              pitch_deg, yaw_deg]            (15D)
Action:      [delta_pitch, delta_yaw]        (2D, each in [-1, 1])

Flags:
  physics_guided : bool — per-episode 3D PINN reference
  use_ekf        : bool — EKF3D estimated state in observation

See notes/09_phase8_3d.md for full design.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.config import RocketConfig, SimConfig, TRAINING_SIM
from simulation.physics3d import rk4_step3d
from simulation.wind3d import WindModel3D, WindConfig3D, sample_wind3d
from simulation.domain_randomization import sample_rocket
from simulation.sensors import SensorSuite, SensorConfig
from simulation.ekf3d import EKF3D

# Observation normalisation scales (15D)
_OBS_SCALE = np.array([
    20_000.0,   # x       (m)
    35_000.0,   # y       (m)
    20_000.0,   # z       (m)
    600.0,      # vx      (m/s)
    900.0,      # vy      (m/s)
    600.0,      # vz      (m/s)
    1.0,        # mass_fraction
    30.0,       # wind_vx_est
    30.0,       # wind_vz_est
    35_000.0,   # dy_to_target
    600.0,      # dvx_to_target
    900.0,      # dvy_to_target
    600.0,      # dvz_to_target
    90.0,       # pitch_deg
    45.0,       # yaw_deg
], dtype=np.float32)

_W_ALT    = 2.0
_W_VEL    = 1.0
_W_FUEL   = 0.05
_W_SMOOTH = 0.1
_W_PINN   = 0.5

_MAX_DELTA_DEG = 5.0
_MAX_DELTA_RAD = math.radians(_MAX_DELTA_DEG)

_MIN_PITCH_RAD = math.radians(10.0)
_MAX_PITCH_RAD = math.radians(90.0)
_MAX_YAW_RAD   = math.radians(45.0)

_MAX_LANDING_VY = -50.0


class RocketEnv3D(gym.Env):
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

        # Target trajectory — lazy import to avoid circular deps
        self._target = None
        self._target_ready = False

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Episode state
        self._state: np.ndarray = np.zeros(7)
        self._pitch_rad: float = 0.0
        self._yaw_rad:   float = 0.0
        self._t:         float = 0.0
        self._fuel_total: float = 0.0

        from simulation.config import DEFAULT_ROCKET
        self._rocket: RocketConfig = DEFAULT_ROCKET
        self._wind = WindModel3D(WindConfig3D(), self._rng)

        self._sensors: SensorSuite | None = (
            SensorSuite(SensorConfig(dt=sim.dt), rng=self._rng) if use_ekf else None
        )
        self._ekf: EKF3D | None = None

    def _get_target(self):
        """Lazy-load target trajectory to avoid import-time overhead."""
        if self._target is None:
            if self._physics_guided:
                from simulation.target_trajectory3d import PhysicsGuidedTargetTrajectory3D
                self._target = PhysicsGuidedTargetTrajectory3D()
            else:
                from simulation.target_trajectory3d import TargetTrajectory3D
                self._target = TargetTrajectory3D()
        return self._target

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._randomize:
            self._rocket = sample_rocket(self._rng)
            wind_cfg = WindConfig3D(
                v_ref_x=float(self._rng.uniform(-15.0, 15.0)),
                v_ref_z=float(self._rng.uniform(-10.0, 10.0)),
            )
        else:
            from simulation.config import DEFAULT_ROCKET
            self._rocket = DEFAULT_ROCKET
            wind_cfg = WindConfig3D()

        self._wind = WindModel3D(wind_cfg, self._rng)
        self._state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 self._rocket.mass_wet])
        self._pitch_rad = math.radians(self._sim.launch_angle_deg)
        self._yaw_rad   = 0.0
        self._t = 0.0
        self._fuel_total = self._rocket.mass_wet - self._rocket.mass_dry

        target = self._get_target()
        if self._physics_guided:
            target.set_episode(self._rocket,
                               wind_vx_ref=wind_cfg.v_ref_x,
                               wind_vz_ref=wind_cfg.v_ref_z)

        if self._use_ekf:
            self._ekf = EKF3D(self._rocket, self._sim.dt)
            self._ekf.reset(self._state)
            self._sensors.reset()

        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        d_pitch = float(np.clip(action[0], -1.0, 1.0)) * _MAX_DELTA_RAD
        d_yaw   = float(np.clip(action[1], -1.0, 1.0)) * _MAX_DELTA_RAD

        self._pitch_rad = float(np.clip(
            self._pitch_rad + d_pitch, _MIN_PITCH_RAD, _MAX_PITCH_RAD))
        self._yaw_rad = float(np.clip(
            self._yaw_rad + d_yaw, -_MAX_YAW_RAD, _MAX_YAW_RAD))

        wind_vx, wind_vz = self._wind.step(self._state[1], self._sim.dt)

        self._state = rk4_step3d(
            self._state, self._pitch_rad, self._yaw_rad,
            self._sim.dt, self._rocket, wind_vx, wind_vz
        )
        self._t += self._sim.dt

        if self._use_ekf and self._ekf is not None:
            self._ekf.predict(self._pitch_rad, self._yaw_rad, wind_vx, wind_vz)
            # Build 3D sensor measurements from true 3D state
            # Reuse 2D SensorSuite for baro+GPS (it only knows x,y,vx,vy)
            # Feed it the 2D slice and augment with z
            state2d_approx = np.array([
                self._state[0], self._state[1],
                self._state[3], self._state[4], self._state[6]
            ])
            meas = self._sensors.measure(state2d_approx)
            self._ekf.update_baro(meas["baro"])
            if meas["gps_pos"] is not None:
                z_pos = np.array([
                    meas["gps_pos"][0],
                    meas["gps_pos"][1],
                    self._state[2] + self._rng.normal(0.0, 3.0),
                ])
                z_vel = np.array([
                    meas["gps_vel"][0],
                    meas["gps_vel"][1],
                    self._state[5] + self._rng.normal(0.0, 0.1),
                ])
                self._ekf.update_gps(z_pos, z_vel)

        delta_rad = math.sqrt(d_pitch**2 + d_yaw**2)
        reward, info = self._compute_reward(delta_rad)

        y = self._state[1]; vy = self._state[4]
        landed  = bool(y <= 0.0 and self._t > 0.5)
        timeout = bool(self._t >= self._sim.max_time)

        if landed:
            if vy >= _MAX_LANDING_VY:
                reward += 100.0; info["outcome"] = "landed_safe"
            else:
                reward -= 100.0; info["outcome"] = "crashed"

        return self._observe(), float(reward), landed, timeout, info

    def _observe(self) -> np.ndarray:
        if self._use_ekf and self._ekf is not None:
            s = self._ekf.state
            x,y,z,vx,vy,vz,mass = s
        else:
            x,y,z,vx,vy,vz,mass = self._state

        fuel_remaining = max(0.0, mass - self._rocket.mass_dry)
        mass_fraction = fuel_remaining / self._fuel_total if self._fuel_total > 0 else 0.0

        # Noisy wind estimate
        true_wx, true_wz = self._wind._gust_x, self._wind._gust_z
        wind_vx_est = true_wx * self._rng.normal(1.0, 0.10)
        wind_vz_est = true_wz * self._rng.normal(1.0, 0.10)

        target = self._get_target()
        ty, tvy, tvx, tvz = target.query3d(self._t)

        raw = np.array([
            x, y, z, vx, vy, vz,
            mass_fraction,
            wind_vx_est, wind_vz_est,
            ty - y, tvx - vx, tvy - vy, tvz - vz,
            math.degrees(self._pitch_rad),
            math.degrees(self._yaw_rad),
        ], dtype=np.float32)

        return np.clip(raw / _OBS_SCALE, -3.0, 3.0)

    def _compute_reward(self, delta_rad: float) -> tuple[float, dict]:
        x,y,z,vx,vy,vz,mass = self._state
        target = self._get_target()
        ty, tvy, tvx, tvz = target.query3d(self._t)

        fuel_remaining = max(0.0, mass - self._rocket.mass_dry)
        mass_fraction = fuel_remaining / self._fuel_total if self._fuel_total > 0 else 0.0

        r_alt    = -_W_ALT   * abs(ty  - y)  / _OBS_SCALE[9]
        r_vel    = -_W_VEL   * abs(tvy - vy) / _OBS_SCALE[11]
        r_fuel   =  _W_FUEL  * mass_fraction
        r_smooth = -_W_SMOOTH * (delta_rad / _MAX_DELTA_RAD) ** 2

        r_physics = 0.0
        if self._physics_guided:
            r_physics = -_W_PINN * (
                ((ty  - y)  / _OBS_SCALE[9])  ** 2 +
                ((tvy - vy) / _OBS_SCALE[11]) ** 2 +
                ((tvz - vz) / _OBS_SCALE[12]) ** 2
            )

        total = r_alt + r_vel + r_fuel + r_smooth + r_physics
        info  = {"r_alt": r_alt, "r_vel": r_vel, "r_fuel": r_fuel,
                 "r_smooth": r_smooth, "r_physics": r_physics,
                 "t": self._t, "y": y, "z": z}
        return total, info
