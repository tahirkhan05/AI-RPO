"""
Extended Kalman Filter for 3D rocket state estimation.

State vector: [x, y, z, vx, vy, vz, mass]  (7 dimensions)

Direct extension of ekf.py to 3D. The Jacobian grows from 5x5 to 7x7;
the new rows/columns follow the same analytic structure — z is symmetric
to x, with wind_vz replacing wind_vx for the z-axis.

Sensors:
  Barometer   — altitude y only, sigma=5m, every step
  GPS pos     — (x, y, z),  sigma=3m,   every 10th step
  GPS vel     — (vx,vy,vz), sigma=0.1m/s, every 10th step

See notes/09_phase8_3d.md Section 6 for design.
"""

import math
import numpy as np
from simulation.config import RocketConfig
from simulation.physics3d import rk4_step3d

_X, _Y, _Z, _VX, _VY, _VZ, _MASS = 0, 1, 2, 3, 4, 5, 6

_G       = 9.81
_RHO_SEA = 1.225
_H_SCALE = 8500.0

# Process noise — same philosophy as 2D, z added symmetrically
_Q_DIAG = np.array([0.01, 0.01, 0.01,   # position (m²)
                    0.1,  0.1,  0.1,     # velocity (m²/s²)
                    1e-6])               # mass (kg²)

# Measurement noise
_R_BARO    = np.array([[25.0]])
_R_GPS_POS = np.diag([9.0, 9.0, 9.0])
_R_GPS_VEL = np.diag([0.01, 0.01, 0.01])

# H matrices
_H_BARO    = np.array([[0,1,0,0,0,0,0]], dtype=float)
_H_GPS_POS = np.array([[1,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,0,1,0,0,0,0]], dtype=float)
_H_GPS_VEL = np.array([[0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,0,1,0]], dtype=float)


class EKF3D:
    """EKF for 7D rocket state [x,y,z,vx,vy,vz,mass]."""

    def __init__(self, rocket: RocketConfig, dt: float) -> None:
        self._rocket = rocket
        self._dt = dt
        self._x = np.zeros(7)
        self._P = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01])
        self._Q = np.diag(_Q_DIAG)
        self._I = np.eye(7)

    def reset(self, initial_state: np.ndarray) -> None:
        self._x = initial_state.copy().astype(float)
        self._P = np.diag([0.01]*7)

    @property
    def state(self) -> np.ndarray:
        return self._x.astype(np.float32)

    def predict(self, pitch_rad: float, yaw_rad: float,
                wind_vx: float, wind_vz: float) -> None:
        self._x = rk4_step3d(
            self._x.astype(np.float32), pitch_rad, yaw_rad,
            self._dt, self._rocket, wind_vx, wind_vz
        ).astype(float)
        F = self._jacobian(self._x, pitch_rad, yaw_rad, wind_vx, wind_vz)
        self._P = F @ self._P @ F.T + self._Q

    def update_baro(self, z_baro: np.ndarray) -> None:
        self._update(z_baro, _H_BARO, _R_BARO)

    def update_gps(self, z_pos: np.ndarray, z_vel: np.ndarray) -> None:
        self._update(z_pos, _H_GPS_POS, _R_GPS_POS)
        self._update(z_vel, _H_GPS_VEL, _R_GPS_VEL)

    def _update(self, z, H, R):
        y_innov = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y_innov
        I_KH = self._I - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    def _jacobian(self, state, pitch_rad, yaw_rad, wind_vx, wind_vz):
        x, y, z, vx, vy, vz, mass = state
        cfg = self._rocket
        rho = _RHO_SEA * math.exp(-max(y, 0.0) / _H_SCALE)

        rel_vx = vx - wind_vx
        rel_vy = vy
        rel_vz = vz - wind_vz
        airspeed = math.sqrt(rel_vx**2 + rel_vy**2 + rel_vz**2)

        is_burning = mass > cfg.mass_dry
        T = cfg.thrust if is_burning else 0.0

        Jc = np.zeros((7, 7))

        # Position rows — trivial
        Jc[_X,  _VX] = 1.0
        Jc[_Y,  _VY] = 1.0
        Jc[_Z,  _VZ] = 1.0

        if airspeed > 1e-6:
            dc = 0.5 * cfg.drag_coeff * cfg.cross_section

            # Thrust components
            cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
            cos_ya, sin_ya = math.cos(yaw_rad), math.sin(yaw_rad)
            net_fx = T * cos_p * cos_ya - rho * dc * airspeed * rel_vx
            net_fy = T * sin_p           - rho * dc * airspeed * rel_vy
            net_fz = T * cos_p * sin_ya  - rho * dc * airspeed * rel_vz

            # ∂ax/∂vx (drag self-damping)
            Jc[_VX, _VX] = -rho * dc * (airspeed + rel_vx**2/airspeed) / mass
            Jc[_VY, _VY] = -rho * dc * (airspeed + rel_vy**2/airspeed) / mass
            Jc[_VZ, _VZ] = -rho * dc * (airspeed + rel_vz**2/airspeed) / mass

            # Cross-axis drag coupling
            Jc[_VX, _VY] = -rho * dc * rel_vx * rel_vy / (airspeed * mass)
            Jc[_VX, _VZ] = -rho * dc * rel_vx * rel_vz / (airspeed * mass)
            Jc[_VY, _VX] = Jc[_VX, _VY]
            Jc[_VY, _VZ] = -rho * dc * rel_vy * rel_vz / (airspeed * mass)
            Jc[_VZ, _VX] = Jc[_VX, _VZ]
            Jc[_VZ, _VY] = Jc[_VY, _VZ]

            # ∂accel/∂y — rho decreases with altitude, drag decreases
            Jc[_VX, _Y] = net_fx / (mass * _H_SCALE)
            Jc[_VY, _Y] = net_fy / (mass * _H_SCALE)
            Jc[_VZ, _Y] = net_fz / (mass * _H_SCALE)

            # ∂accel/∂mass — F/m → ∂/∂m = -F/m²
            Jc[_VX, _MASS] = -net_fx / mass**2
            Jc[_VY, _MASS] = -net_fy / mass**2
            Jc[_VZ, _MASS] = -net_fz / mass**2

        # Discrete approximation: F ≈ I + Jc * dt
        return self._I + Jc * self._dt
