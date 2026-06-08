from dataclasses import dataclass


@dataclass(frozen=True)
class RocketConfig:
    mass_wet: float       # kg — rocket + full fuel
    mass_dry: float       # kg — rocket body only, no fuel
    thrust: float         # N
    burn_rate: float      # kg/s — fuel consumed per second
    exhaust_velocity: float  # m/s
    drag_coeff: float     # dimensionless
    cross_section: float  # m² — frontal area


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.01          # s — timestep
    max_time: float = 300.0   # s — simulation cutoff (apogee + descent ~240s)
    launch_angle_deg: float = 85.0  # degrees from horizontal


# Shorter episode for RL training — fits multiple full episodes per rollout buffer.
# Full flight is ~185s but the agent only needs to track the ascent phase well.
# Descent tracking is less critical and wastes rollout steps.
TRAINING_SIM = SimConfig(dt=0.01, max_time=120.0, launch_angle_deg=85.0)

DEFAULT_ROCKET = RocketConfig(
    mass_wet=100.0,
    mass_dry=50.0,
    thrust=5000.0,
    burn_rate=2.0,
    exhaust_velocity=2500.0,
    drag_coeff=0.4,
    cross_section=0.05,
)

DEFAULT_SIM = SimConfig()
