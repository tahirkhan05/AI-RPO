"""
Extended Kalman Filter (EKF) for 2D rocket state estimation.

State vector: [x, y, vx, vy, mass]  (5 dimensions — same as physics)

The EKF combines:
  - Predict step: propagate state and uncertainty through nonlinear physics
  - Update step:  correct estimate using noisy sensor measurements

The nonlinear physics is linearised via the analytic Jacobian of our
`derivatives()` function with respect to the state. See notes/08_phase7_ekf.md
Sections 3-4 for the full derivation with real numbers.

Usage:
    ekf = EKF(rocket_cfg, sim_cfg.dt)
    ekf.reset(initial_state)
    ...
    ekf.predict(angle_rad, wind_vx)       # one physics step
    ekf.update_baro(baro_measurement)     # incorporate barometer
    ekf.update_gps(gps_pos, gps_vel)      # incorporate GPS (when available)
    x_est = ekf.state                     # current estimate
"""

import math
import numpy as np
from simulation.config import RocketConfig
from simulation.physics import rk4_step

# State indices for readability
_X, _Y, _VX, _VY, _MASS = 0, 1, 2, 3, 4

# Process noise Q — how much does our physics model drift per step?
# Based on known model imperfections:
#   x, y: 0.01 m²  — wind not perfectly modelled
#   vx, vy: 0.1 m²/s² — drag coefficient uncertainty ~10%
#   mass: 1e-6 kg²  — burn rate is well-known
_Q_DIAG = np.array([0.01, 0.01, 0.1, 0.1, 1e-6])

# Measurement noise R — from sensor specifications
# Barometer: σ=5m → R=25 m²
# GPS position: σ=3m → R=9 m²
# GPS velocity: σ=0.1m/s → R=0.01 m²/s²
_R_BARO    = np.array([[25.0]])
_R_GPS_POS = np.diag([9.0, 9.0])
_R_GPS_VEL = np.diag([0.01, 0.01])

# H matrices: which states does each sensor observe?
# Barometer measures y only
_H_BARO    = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
# GPS position measures x and y
_H_GPS_POS = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0]])
# GPS velocity measures vx and vy
_H_GPS_VEL = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0]])

# Physical constants (duplicated from physics.py to avoid circular import)
_G        = 9.81
_RHO_SEA  = 1.225
_H_SCALE  = 8500.0


class EKF:
    """
    Extended Kalman Filter for 2D rocket state estimation.

    Maintains a Gaussian belief over the state x = [x, y, vx, vy, mass]
    represented as (mean x̂, covariance P).
    """

    def __init__(self, rocket: RocketConfig, dt: float) -> None:
        self._rocket = rocket
        self._dt = dt

        # State estimate — set properly by reset()
        self._x = np.zeros(5)

        # Initial covariance: generous uncertainty at launch
        # Position known (launch pad), velocity known (zero), mass known.
        self._P = np.diag([1.0, 1.0, 0.1, 0.1, 0.01])

        # Fixed matrices
        self._Q = np.diag(_Q_DIAG)
        self._I = np.eye(5)

    def reset(self, initial_state: np.ndarray) -> None:
        """
        Initialise filter at episode start with known launch conditions.
        At t=0 position and velocity are precisely known (launch pad).
        """
        self._x = initial_state.copy().astype(float)
        # Very small initial uncertainty — rocket is on the pad
        self._P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])

    @property
    def state(self) -> np.ndarray:
        """Current state estimate as float32 array (matches physics convention)."""
        return self._x.astype(np.float32)

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------

    def predict(self, angle_rad: float, wind_vx: float) -> None:
        """
        Propagate state and covariance one timestep forward through physics.

        angle_rad: current thrust angle (control input, known exactly)
        wind_vx:   current wind velocity estimate (m/s)
        """
        # 1. State prediction: propagate through nonlinear physics
        self._x = rk4_step(
            self._x.astype(np.float32), angle_rad, self._dt,
            self._rocket, wind_vx
        ).astype(float)

        # 2. Covariance prediction: propagate through linearised physics
        F = self._jacobian(self._x, angle_rad, wind_vx)
        self._P = F @ self._P @ F.T + self._Q

    # ------------------------------------------------------------------
    # Update steps — one per sensor type
    # ------------------------------------------------------------------

    def update_baro(self, z_baro: np.ndarray) -> None:
        """
        Incorporate barometer altitude measurement.
        z_baro: shape (1,) — measured altitude in metres
        """
        self._update(z_baro, _H_BARO, _R_BARO)

    def update_gps(self, z_pos: np.ndarray, z_vel: np.ndarray) -> None:
        """
        Incorporate GPS position and velocity measurements.
        z_pos: shape (2,) — [x_meas, y_meas] in metres
        z_vel: shape (2,) — [vx_meas, vy_meas] in m/s
        """
        self._update(z_pos, _H_GPS_POS, _R_GPS_POS)
        self._update(z_vel, _H_GPS_VEL, _R_GPS_VEL)

    def _update(self, z: np.ndarray,
                H: np.ndarray, R: np.ndarray) -> None:
        """
        Generic EKF measurement update.

        z: measurement vector
        H: observation matrix (maps state to measurement space)
        R: measurement noise covariance
        """
        # Innovation: difference between measurement and prediction
        y_innov = z - H @ self._x

        # Innovation covariance
        S = H @ self._P @ H.T + R

        # Kalman gain: how much to trust the measurement
        K = self._P @ H.T @ np.linalg.inv(S)

        # State update
        self._x = self._x + K @ y_innov

        # Covariance update (Joseph form for numerical stability)
        I_KH = self._I - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    # ------------------------------------------------------------------
    # Jacobian of physics — analytic linearisation
    # ------------------------------------------------------------------

    def _jacobian(self, state: np.ndarray,
                  angle_rad: float, wind_vx: float) -> np.ndarray:
        """
        Analytic Jacobian F = ∂f/∂x of the rocket dynamics.

        Derived from derivatives() in physics.py. See notes/08_phase7_ekf.md
        Section 8 for the full derivation with real numbers.

        Returns 5×5 matrix.
        """
        x, y, vx, vy, mass = state

        cfg = self._rocket
        rho = _RHO_SEA * math.exp(-max(y, 0.0) / _H_SCALE)

        rel_vx = vx - wind_vx
        rel_vy = vy
        airspeed = math.sqrt(rel_vx**2 + rel_vy**2)

        is_burning = mass > cfg.mass_dry
        T = cfg.thrust if is_burning else 0.0

        # F is initialised to identity * dt_factor — we build the continuous
        # Jacobian (∂ẋ/∂x) and the discrete approximation is F ≈ I + Jc * dt.
        # For small dt=0.01s this is accurate enough.
        Jc = np.zeros((5, 5))

        # ẋ = vx, ẏ = vy — trivial
        Jc[_X, _VX] = 1.0
        Jc[_Y, _VY] = 1.0

        if airspeed > 1e-6:
            # Drag force components:
            #   Dx = -0.5 * rho * Cd * A * airspeed * rel_vx
            #   Dy = -0.5 * rho * Cd * A * airspeed * rel_vy
            drag_coeff_eff = 0.5 * cfg.drag_coeff * cfg.cross_section

            # ∂(Dx)/∂(vx): drag_x = -rho*Cd_eff * airspeed * rel_vx
            #   = -rho*Cd_eff * (rel_vx² + rel_vy²)^0.5 * rel_vx
            # Using product rule + chain rule (symmetric in vx for aligned flow):
            #   ≈ -rho * Cd_eff * (airspeed + rel_vx²/airspeed)
            ddx_dvx = -rho * drag_coeff_eff * (airspeed + rel_vx**2 / airspeed) / mass
            ddy_dvy = -rho * drag_coeff_eff * (airspeed + rel_vy**2 / airspeed) / mass

            # Cross terms (∂Dx/∂vy) — small but included for correctness
            ddx_dvy = -rho * drag_coeff_eff * (rel_vx * rel_vy / airspeed) / mass
            ddy_dvx = ddx_dvy  # symmetric

            # ∂acceleration/∂y: rho decreases with altitude → drag decreases
            # ∂rho/∂y = -rho / H_scale
            # ∂Dx/∂y = (-∂rho/∂y) * Cd_eff * airspeed * rel_vx / mass (sign: drag opposes)
            drag_x = -rho * drag_coeff_eff * airspeed * rel_vx
            drag_y_comp = -rho * drag_coeff_eff * airspeed * rel_vy
            ddx_dy = drag_x / (mass * _H_SCALE)   # positive: less drag at higher alt
            ddy_dy = drag_y_comp / (mass * _H_SCALE)

            # ∂acceleration/∂mass: a = F/m → ∂a/∂m = -F/m²
            net_fx = T * math.cos(angle_rad) - rho * drag_coeff_eff * airspeed * rel_vx
            net_fy = T * math.sin(angle_rad) - rho * drag_coeff_eff * airspeed * rel_vy
            ddx_dmass = -net_fx / mass**2
            ddy_dmass = -net_fy / mass**2

            Jc[_VX, _Y]    = ddx_dy
            Jc[_VX, _VX]   = ddx_dvx
            Jc[_VX, _VY]   = ddx_dvy
            Jc[_VX, _MASS] = ddx_dmass

            Jc[_VY, _Y]    = ddy_dy
            Jc[_VY, _VX]   = ddy_dvx
            Jc[_VY, _VY]   = ddy_dvy
            Jc[_VY, _MASS] = ddy_dmass

        # Row 4 (mass): dm/dt = constant → all zeros ✓

        # Discrete Jacobian: F ≈ I + Jc * dt  (first-order ZOH approximation)
        F = self._I + Jc * self._dt
        return F
