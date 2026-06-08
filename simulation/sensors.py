"""
Simulated sensor suite for the 2D rocket.

Each sensor takes the true state and returns a noisy measurement. The noise
model matches real sensor specs so EKF tuning is physically grounded.

Sensors modelled:
  Barometer  — altitude y, σ=5m, 100 Hz (fires every step at dt=0.01s)
  IMU        — acceleration (ax, ay), σ=0.3 m/s², 100 Hz
  GPS        — position (x,y) σ=3m, velocity (vx,vy) σ=0.1m/s, 10 Hz

GPS fires at 10 Hz → every 10th step (every 0.1s when dt=0.01s).
Barometer and IMU fire every step.

See notes/08_phase7_ekf.md Section 5 for the sensor noise table.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SensorConfig:
    baro_sigma:     float = 5.0    # m      — barometer altitude noise
    imu_sigma:      float = 0.3    # m/s²   — accelerometer noise
    gps_pos_sigma:  float = 3.0    # m      — GPS position noise
    gps_vel_sigma:  float = 0.1    # m/s    — GPS velocity noise
    gps_hz:         float = 10.0   # Hz     — GPS update rate
    dt:             float = 0.01   # s      — simulation timestep


DEFAULT_SENSORS = SensorConfig()


class SensorSuite:
    """
    Generates noisy sensor measurements from the true rocket state.

    Measurement vectors returned per step:
      baro:     [y_meas]            shape (1,)
      gps_pos:  [x_meas, y_meas]   shape (2,) — None when GPS not firing
      gps_vel:  [vx_meas, vy_meas] shape (2,) — None when GPS not firing

    IMU is used inside EKF.predict() (acceleration comparison), not as a
    standalone measurement update — the physics model IS the IMU model.
    """

    def __init__(self, cfg: SensorConfig = DEFAULT_SENSORS,
                 rng: np.random.Generator | None = None) -> None:
        self._cfg = cfg
        self._rng = rng if rng is not None else np.random.default_rng()
        self._step_count: int = 0
        # How many simulation steps between GPS updates
        self._gps_interval: int = max(1, round(1.0 / (cfg.gps_hz * cfg.dt)))

    def reset(self) -> None:
        self._step_count = 0

    def measure(self, true_state: np.ndarray) -> dict:
        """
        Return sensor readings for one timestep.

        true_state: [x, y, vx, vy, mass]

        Returns dict with keys:
          'baro'    : np.ndarray shape (1,)  — always present
          'gps_pos' : np.ndarray shape (2,) or None
          'gps_vel' : np.ndarray shape (2,) or None
        """
        x, y, vx, vy, _ = true_state
        cfg = self._cfg
        rng = self._rng

        # Barometer — fires every step
        baro = np.array([y + rng.normal(0.0, cfg.baro_sigma)])

        # GPS — fires every gps_interval steps
        gps_fires = (self._step_count % self._gps_interval == 0)
        if gps_fires:
            gps_pos = np.array([
                x + rng.normal(0.0, cfg.gps_pos_sigma),
                y + rng.normal(0.0, cfg.gps_pos_sigma),
            ])
            gps_vel = np.array([
                vx + rng.normal(0.0, cfg.gps_vel_sigma),
                vy + rng.normal(0.0, cfg.gps_vel_sigma),
            ])
        else:
            gps_pos = None
            gps_vel = None

        self._step_count += 1
        return {"baro": baro, "gps_pos": gps_pos, "gps_vel": gps_vel}
