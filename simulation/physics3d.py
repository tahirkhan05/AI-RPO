"""
3D rocket physics engine.

State vector: [x, y, z, vx, vy, vz, mass]   (7 dimensions)
  x, y, z   — position (m). x=north, y=altitude, z=east
  vx, vy, vz — velocity (m/s)
  mass       — current wet mass (kg)

Control inputs:
  pitch_rad  — elevation angle from horizontal (radians)
  yaw_rad    — azimuth angle from north (radians)

Wind: (wind_vx, wind_vz) — horizontal 2D wind vector (m/s)
  wind affects drag via relative velocity, same as 2D

See notes/09_phase8_3d.md Section 2 for full derivation with real numbers.
Symmetry guarantee: with yaw=0, wind_vz=0, vz=0 → az=0 exactly.
"""

import math
import numpy as np
from simulation.config import RocketConfig

G        = 9.81
RHO_SEA  = 1.225
H_SCALE  = 8500.0


def air_density(altitude: float) -> float:
    return RHO_SEA * math.exp(-max(altitude, 0.0) / H_SCALE)


def derivatives3d(state: np.ndarray, pitch_rad: float, yaw_rad: float,
                  cfg: RocketConfig,
                  wind_vx: float = 0.0, wind_vz: float = 0.0) -> np.ndarray:
    """
    Compute d(state)/dt for the 3D rocket.

    state:     [x, y, z, vx, vy, vz, mass]
    pitch_rad: elevation angle from horizontal
    yaw_rad:   azimuth angle from north
    wind_vx:   north wind component (m/s)
    wind_vz:   east wind component  (m/s)
    """
    x, y, z, vx, vy, vz, mass = state

    rho = air_density(y)

    # Velocity relative to air (wind shifts reference frame)
    rel_vx = vx - wind_vx
    rel_vy = vy
    rel_vz = vz - wind_vz
    airspeed = math.sqrt(rel_vx**2 + rel_vy**2 + rel_vz**2)

    is_burning = mass > cfg.mass_dry
    T        = cfg.thrust    if is_burning else 0.0
    mass_dot = -cfg.burn_rate if is_burning else 0.0

    # Thrust components — pitch rotates in y-x plane, yaw rotates in x-z plane
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_y, sin_y = math.cos(yaw_rad),   math.sin(yaw_rad)
    Tx = T * cos_p * cos_y
    Ty = T * sin_p
    Tz = T * cos_p * sin_y

    # Drag components — opposes relative velocity
    if airspeed > 1e-6:
        drag_mag = 0.5 * rho * airspeed**2 * cfg.drag_coeff * cfg.cross_section
        Dx = -drag_mag * rel_vx / airspeed
        Dy = -drag_mag * rel_vy / airspeed
        Dz = -drag_mag * rel_vz / airspeed
    else:
        Dx = Dy = Dz = 0.0

    ax = (Tx + Dx) / mass
    ay = (Ty + Dy) / mass - G
    az = (Tz + Dz) / mass

    return np.array([vx, vy, vz, ax, ay, az, mass_dot])


def rk4_step3d(state: np.ndarray, pitch_rad: float, yaw_rad: float,
               dt: float, cfg: RocketConfig,
               wind_vx: float = 0.0, wind_vz: float = 0.0) -> np.ndarray:
    """Single RK4 step for 3D state. Returns next 7D state."""
    def f(s):
        return derivatives3d(s, pitch_rad, yaw_rad, cfg, wind_vx, wind_vz)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
