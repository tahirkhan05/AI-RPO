"""
Unit tests for EKF and sensor suite.

Tests verify:
  1. EKF estimation error < raw sensor noise (EKF adds value)
  2. EKF covariance P decreases after measurements (filter is converging)
  3. SensorSuite GPS fires at correct rate (10 Hz = every 10 steps)
  4. EKF handles burnout discontinuity (mass hits mass_dry — no crash)
  5. RocketEnv with use_ekf=True runs a full episode without errors
"""

import math
import numpy as np
import pytest

from simulation.config import DEFAULT_ROCKET, TRAINING_SIM
from simulation.physics import rk4_step
from simulation.sensors import SensorSuite, SensorConfig
from simulation.ekf import EKF
from simulation.env import RocketEnv


@pytest.fixture
def nominal_run():
    """200-step trajectory with EKF and sensors running."""
    rng = np.random.default_rng(7)
    rocket = DEFAULT_ROCKET
    dt = TRAINING_SIM.dt
    angle_rad = math.radians(85.0)

    sensors = SensorSuite(SensorConfig(dt=dt), rng=rng)
    ekf = EKF(rocket, dt)
    state = np.array([0.0, 0.0, 0.0, 0.0, rocket.mass_wet])
    ekf.reset(state)
    sensors.reset()

    true_ys, est_ys = [], []
    p_traces = []

    for _ in range(200):
        state = rk4_step(state, angle_rad, dt, rocket, wind_vx=0.0)
        meas = sensors.measure(state)
        ekf.predict(angle_rad, 0.0)
        ekf.update_baro(meas["baro"])
        if meas["gps_pos"] is not None:
            ekf.update_gps(meas["gps_pos"], meas["gps_vel"])
        true_ys.append(state[1])
        est_ys.append(ekf.state[1])
        p_traces.append(ekf._P[1, 1])  # P[y,y]

    return {
        "true_y": np.array(true_ys),
        "est_y":  np.array(est_ys),
        "p_yy":   np.array(p_traces),
    }


def test_ekf_beats_raw_sensor(nominal_run):
    """EKF altitude RMSE must be less than barometer sigma (5m)."""
    errors = np.abs(nominal_run["true_y"] - nominal_run["est_y"])
    rmse = float(np.sqrt(np.mean(errors**2)))
    assert rmse < 5.0, f"EKF RMSE {rmse:.3f}m >= baro sigma 5m — filter not improving on sensor"


def test_ekf_covariance_bounded(nominal_run):
    """P[y,y] must remain finite and small after 200 steps."""
    p_final = nominal_run["p_yy"][-1]
    assert np.isfinite(p_final), "P[y,y] diverged to inf/nan"
    assert p_final < 100.0, f"P[y,y]={p_final:.2f} too large — filter not converging"


def test_gps_fires_at_10hz():
    """GPS should fire every 10 steps (10 Hz at dt=0.01s)."""
    rng = np.random.default_rng(0)
    cfg = SensorConfig(dt=0.01, gps_hz=10.0)
    sensors = SensorSuite(cfg, rng=rng)
    sensors.reset()

    state = np.array([0.0, 100.0, 10.0, 50.0, 90.0])
    gps_steps = []
    for i in range(50):
        meas = sensors.measure(state)
        if meas["gps_pos"] is not None:
            gps_steps.append(i)

    # Should fire at steps 0, 10, 20, 30, 40
    assert gps_steps == [0, 10, 20, 30, 40], f"GPS fired at wrong steps: {gps_steps}"


def test_ekf_survives_burnout():
    """EKF must not crash when mass hits mass_dry (thrust drops to zero)."""
    rng = np.random.default_rng(3)
    rocket = DEFAULT_ROCKET
    dt = TRAINING_SIM.dt
    angle_rad = math.radians(85.0)

    sensors = SensorSuite(SensorConfig(dt=dt), rng=rng)
    ekf = EKF(rocket, dt)
    state = np.array([0.0, 0.0, 0.0, 0.0, rocket.mass_wet])
    ekf.reset(state)
    sensors.reset()

    # Run past burnout (burn_time = (mass_wet - mass_dry) / burn_rate = 25s)
    burnout_steps = int(26.0 / dt)  # 26 seconds, past the 25s burnout
    for _ in range(burnout_steps):
        state = rk4_step(state, angle_rad, dt, rocket, wind_vx=0.0)
        meas = sensors.measure(state)
        ekf.predict(angle_rad, 0.0)
        ekf.update_baro(meas["baro"])
        if meas["gps_pos"] is not None:
            ekf.update_gps(meas["gps_pos"], meas["gps_vel"])

    est = ekf.state
    assert np.all(np.isfinite(est)), f"EKF state has nan/inf after burnout: {est}"
    assert est[4] >= rocket.mass_dry - 1.0, "EKF mass estimate went below mass_dry"


def test_env_ekf_episode_completes():
    """Full episode with use_ekf=True must complete without errors."""
    env = RocketEnv(randomize=False, seed=0, physics_guided=False, use_ekf=True)
    obs, _ = env.reset()
    assert obs.shape == (10,)
    assert np.all(np.isfinite(obs))

    done = False
    steps = 0
    while not done and steps < 5000:
        action = np.array([0.0])
        obs, reward, term, trunc, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"Non-finite obs at step {steps}"
        assert np.isfinite(reward), f"Non-finite reward at step {steps}"
        done = term or trunc
        steps += 1

    assert steps > 0, "Episode ended immediately"
