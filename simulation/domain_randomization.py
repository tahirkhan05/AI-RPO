"""
Domain randomization — samples perturbed rocket and wind configs for each episode.
All parameters drawn from Gaussian distributions centred on nominal values.
The 'pct' values are 1-sigma fractions (99.7% of samples stay within ±3*pct).
"""

import numpy as np
from simulation.config import RocketConfig, DEFAULT_ROCKET
from simulation.wind import WindConfig


# Gaussian perturbation fractions (1-sigma) per parameter
_ROCKET_SIGMAS = {
    "mass_wet":   0.033,   # ±10% at 3-sigma
    "mass_dry":   0.017,   # ±5%
    "thrust":     0.050,   # ±15%
    "burn_rate":  0.033,   # ±10%
    "drag_coeff": 0.067,   # ±20%
    "cross_section": 0.033,
    "exhaust_velocity": 0.017,
}

_WIND_SIGMAS = {
    "v_ref":      0.167,   # ±50%
    "gust_sigma": 0.133,   # ±40%
}


def _perturb(nominal: float, sigma_frac: float, rng: np.random.Generator,
             lo: float = 0.01) -> float:
    """Multiply nominal by a Gaussian factor. Clamp to lo to avoid negatives."""
    factor = rng.normal(1.0, sigma_frac)
    return max(lo, nominal * factor)


def sample_rocket(rng: np.random.Generator,
                  base: RocketConfig = DEFAULT_ROCKET) -> RocketConfig:
    """Return a randomly perturbed RocketConfig."""
    mass_wet = _perturb(base.mass_wet, _ROCKET_SIGMAS["mass_wet"], rng)
    mass_dry = _perturb(base.mass_dry, _ROCKET_SIGMAS["mass_dry"], rng)

    # dry mass must stay below wet mass
    mass_dry = min(mass_dry, mass_wet * 0.85)

    return RocketConfig(
        mass_wet=mass_wet,
        mass_dry=mass_dry,
        thrust=_perturb(base.thrust, _ROCKET_SIGMAS["thrust"], rng),
        burn_rate=_perturb(base.burn_rate, _ROCKET_SIGMAS["burn_rate"], rng),
        exhaust_velocity=_perturb(base.exhaust_velocity,
                                  _ROCKET_SIGMAS["exhaust_velocity"], rng),
        drag_coeff=_perturb(base.drag_coeff, _ROCKET_SIGMAS["drag_coeff"], rng),
        cross_section=_perturb(base.cross_section,
                               _ROCKET_SIGMAS["cross_section"], rng),
    )


def sample_wind(rng: np.random.Generator,
                base: WindConfig | None = None) -> WindConfig:
    """Return a randomly perturbed WindConfig."""
    b = base or WindConfig()
    return WindConfig(
        v_ref=_perturb(b.v_ref, _WIND_SIGMAS["v_ref"], rng, lo=0.0),
        h_ref=b.h_ref,
        alpha=b.alpha,
        h_cap=b.h_cap,
        gust_theta=b.gust_theta,
        gust_sigma=_perturb(b.gust_sigma, _WIND_SIGMAS["gust_sigma"], rng, lo=0.0),
    )
