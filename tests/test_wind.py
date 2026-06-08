"""Tests for wind model and domain randomization."""

import numpy as np
import pytest

from simulation.wind import WindModel, WindConfig
from simulation.domain_randomization import sample_rocket, sample_wind
from simulation.config import DEFAULT_ROCKET
from simulation.trajectory import run
from simulation.config import DEFAULT_SIM


RNG = np.random.default_rng(42)


def make_wind(v_ref=10.0) -> WindModel:
    return WindModel(WindConfig(v_ref=v_ref), np.random.default_rng(0))


def test_mean_wind_zero_at_ground():
    # Mean profile is zero at altitude=0; only gust contributes.
    # We test the private helper directly.
    w = make_wind()
    assert w._mean_wind(0.0) == 0.0


def test_mean_wind_zero_above_cap():
    cfg = WindConfig(h_cap=30_000.0)
    w = WindModel(cfg, np.random.default_rng(0))
    assert w._mean_wind(35_000.0) == 0.0


def test_wind_increases_with_altitude():
    # mean wind (no gust) grows with altitude
    # use very small gust_sigma to isolate the mean profile
    cfg = WindConfig(v_ref=10.0, gust_sigma=1e-9, gust_theta=1e9)
    w = WindModel(cfg, np.random.default_rng(0))
    low = w.step(100.0, 0.01)
    w.reset()
    high = w.step(5000.0, 0.01)
    assert high > low


def test_domain_randomization_rocket_valid():
    rng = np.random.default_rng(7)
    for _ in range(50):
        r = sample_rocket(rng)
        assert r.mass_dry < r.mass_wet
        assert r.thrust > 0
        assert r.burn_rate > 0
        assert r.drag_coeff > 0


def test_domain_randomization_wind_valid():
    rng = np.random.default_rng(7)
    for _ in range(50):
        w = sample_wind(rng)
        assert w.v_ref >= 0
        assert w.gust_sigma >= 0


def test_trajectory_with_wind_completes():
    rng = np.random.default_rng(99)
    wind = WindModel(WindConfig(), rng)
    result = run(DEFAULT_ROCKET, DEFAULT_SIM, wind=wind)
    assert result.apogee > 0
    assert len(result.wind_vx) == len(result.time)


def test_wind_causes_downrange_deviation():
    """Rocket with strong headwind should land further or shorter than no-wind."""
    rng = np.random.default_rng(1)
    no_wind = run(DEFAULT_ROCKET, DEFAULT_SIM, wind=None)

    # strong rightward wind
    cfg = WindConfig(v_ref=30.0, gust_sigma=0.0)
    wind = WindModel(cfg, rng)
    with_wind = run(DEFAULT_ROCKET, DEFAULT_SIM, wind=wind)

    # trajectories should differ
    assert abs(with_wind.x[-1] - no_wind.x[-1]) > 10.0
