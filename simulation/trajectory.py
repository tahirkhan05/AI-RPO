"""
Trajectory runner — steps the physics engine forward in time and records history.
Returns a structured result; does not plot or print.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from simulation.config import RocketConfig, SimConfig
from simulation.physics import rk4_step
from simulation.wind import WindModel


@dataclass
class TrajectoryResult:
    time: np.ndarray        # (N,)
    x: np.ndarray           # (N,) downrange position, m
    y: np.ndarray           # (N,) altitude, m
    vx: np.ndarray          # (N,) m/s
    vy: np.ndarray          # (N,) m/s
    mass: np.ndarray        # (N,) kg
    speed: np.ndarray       # (N,) m/s
    wind_vx: np.ndarray     # (N,) horizontal wind velocity at each step, m/s
    apogee: float           # m — max altitude reached
    max_speed: float        # m/s
    burnout_time: float     # s — when fuel ran out


def run(rocket: RocketConfig, sim: SimConfig,
        thrust_angle_deg: float | None = None,
        wind: WindModel | None = None) -> TrajectoryResult:
    """
    Run a fixed-angle trajectory simulation.

    thrust_angle_deg: constant angle from horizontal. Defaults to sim.launch_angle_deg.
    wind: optional WindModel instance. Pass None for no wind (clean baseline).
    """
    angle_rad = math.radians(
        thrust_angle_deg if thrust_angle_deg is not None else sim.launch_angle_deg
    )

    state = np.array([0.0, 0.0, 0.0, 0.0, rocket.mass_wet])
    t = 0.0
    burnout_time = 0.0

    times, xs, ys, vxs, vys, masses, winds = [], [], [], [], [], [], []

    while t <= sim.max_time:
        x, y, vx, vy, mass = state

        if y < 0.0 and t > 0.1:
            break

        w = wind.step(y, sim.dt) if wind is not None else 0.0

        times.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        masses.append(mass)
        winds.append(w)

        state = rk4_step(state, angle_rad, sim.dt, rocket, wind_vx=w)

        if burnout_time == 0.0 and state[4] <= rocket.mass_dry:
            burnout_time = t + sim.dt
        t += sim.dt

    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)
    vxs = np.array(vxs)
    vys = np.array(vys)
    masses = np.array(masses)
    winds = np.array(winds)
    speeds = np.sqrt(vxs**2 + vys**2)

    return TrajectoryResult(
        time=times,
        x=xs,
        y=ys,
        vx=vxs,
        vy=vys,
        mass=masses,
        speed=speeds,
        wind_vx=winds,
        apogee=float(ys.max()),
        max_speed=float(speeds.max()),
        burnout_time=burnout_time,
    )
