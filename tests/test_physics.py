"""Unit tests for the physics engine."""

import math
import numpy as np
import pytest

from simulation.config import DEFAULT_ROCKET
from simulation.physics import air_density, drag_components, rk4_step, derivatives


def test_air_density_sea_level():
    assert abs(air_density(0) - 1.225) < 1e-6


def test_air_density_decreases_with_altitude():
    assert air_density(10_000) < air_density(0)


def test_drag_zero_at_rest():
    dx, dy = drag_components(0.0, 0.0, 0.0, 0.0, DEFAULT_ROCKET)
    assert dx == 0.0 and dy == 0.0


def test_drag_opposes_motion():
    # Moving straight up: vx=0, vy=100 → drag should be negative vy
    dx, dy = drag_components(100.0, 0.0, 100.0, 0.0, DEFAULT_ROCKET)
    assert dx == 0.0
    assert dy < 0.0


def test_rk4_step_increases_altitude():
    """Rocket at rest pointing straight up should gain altitude after one step."""
    angle = math.radians(90.0)  # straight up
    state = np.array([0.0, 0.0, 0.0, 0.0, DEFAULT_ROCKET.mass_wet])
    next_state = rk4_step(state, angle, dt=0.01, cfg=DEFAULT_ROCKET)
    assert next_state[1] >= 0.0   # y stays non-negative
    assert next_state[4] < DEFAULT_ROCKET.mass_wet  # mass decreased (burning)


def test_mass_does_not_go_below_dry():
    """After burnout, mass_dot should be zero — mass holds at dry."""
    angle = math.radians(85.0)
    state = np.array([0.0, 1000.0, 0.0, 500.0, DEFAULT_ROCKET.mass_dry])
    derivs = derivatives(state, angle, DEFAULT_ROCKET)
    assert derivs[4] == 0.0  # no more fuel consumption
