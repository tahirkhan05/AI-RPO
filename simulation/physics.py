"""
2D rocket physics engine.

State vector: [x, y, vx, vy, mass]
  x, y   — position (m). x = downrange, y = altitude.
  vx, vy — velocity (m/s)
  mass   — current wet mass (kg)

All forces in Newtons. Angles in radians internally.
"""

import math
import numpy as np
from simulation.config import RocketConfig

# Constants
G = 9.81          # m/s² — gravitational acceleration (constant, valid < 50 km)
RHO_SEA = 1.225   # kg/m³ — air density at sea level


def air_density(altitude: float) -> float:
    """Exponential atmosphere model. More realistic than constant rho."""
    scale_height = 8500.0  # m — Earth's atmospheric scale height
    return RHO_SEA * math.exp(-altitude / scale_height)


def thrust_components(thrust: float, angle_rad: float) -> tuple[float, float]:
    """Resolve thrust vector into x, y components. Angle from horizontal."""
    return thrust * math.cos(angle_rad), thrust * math.sin(angle_rad)


def drag_components(speed: float, vx: float, vy: float,
                    altitude: float, cfg: RocketConfig) -> tuple[float, float]:
    """Drag magnitude scaled by velocity direction unit vector."""
    if speed < 1e-6:
        return 0.0, 0.0
    rho = air_density(altitude)
    drag_mag = 0.5 * rho * speed**2 * cfg.drag_coeff * cfg.cross_section
    return -drag_mag * (vx / speed), -drag_mag * (vy / speed)


def derivatives(state: np.ndarray, thrust_angle_rad: float,
                cfg: RocketConfig, wind_vx: float = 0.0) -> np.ndarray:
    """
    Compute d(state)/dt given current state and control input.
    wind_vx: horizontal wind velocity (m/s). Wind adds to apparent airspeed
             for drag but does NOT directly push the rocket — drag acts on
             the relative velocity between rocket and air.
    """
    x, y, vx, vy, mass = state

    # Velocity relative to the air mass (wind shifts the reference frame)
    rel_vx = vx - wind_vx
    rel_vy = vy
    airspeed = math.sqrt(rel_vx**2 + rel_vy**2)

    is_burning = mass > cfg.mass_dry
    raw_thrust = cfg.thrust if is_burning else 0.0
    mass_dot = -cfg.burn_rate if is_burning else 0.0

    tx, ty = thrust_components(raw_thrust, thrust_angle_rad)
    dx_drag, dy_drag = drag_components(airspeed, rel_vx, rel_vy, y, cfg)

    ax = (tx + dx_drag) / mass
    ay = (ty + dy_drag) / mass - G

    return np.array([vx, vy, ax, ay, mass_dot])


def rk4_step(state: np.ndarray, thrust_angle_rad: float,
             dt: float, cfg: RocketConfig, wind_vx: float = 0.0) -> np.ndarray:
    """Single RK4 integration step. Returns next state."""
    k1 = derivatives(state, thrust_angle_rad, cfg, wind_vx)
    k2 = derivatives(state + 0.5 * dt * k1, thrust_angle_rad, cfg, wind_vx)
    k3 = derivatives(state + 0.5 * dt * k2, thrust_angle_rad, cfg, wind_vx)
    k4 = derivatives(state + dt * k3, thrust_angle_rad, cfg, wind_vx)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
