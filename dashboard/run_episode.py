"""
Run one episode with a trained agent and collect all data for dashboard plotting.

Returns a dict with everything the dashboard needs:
  t, y, vy, x, z, mass_frac,
  target_y, target_vy,
  ekf_y (if EKF enabled),
  pitch, yaw,
  reward_total, r_alt, r_vel, r_fuel, r_smooth, r_physics,
  lstm_forecast, lstm_warned,
  outcome (landed_safe / crashed / timeout)
"""

import numpy as np
import torch


def run_episode(model_path: str, vecnorm_path: str,
                env_cls, env_kwargs: dict,
                lstm_path: str = None,
                seed: int = 0) -> dict:
    """
    Run a single episode and return trajectory + diagnostic data.

    Args:
        model_path:   PPO model path (no .zip)
        vecnorm_path: VecNormalize pkl path
        env_cls:      Environment class (RocketEnv or RocketEnv3D)
        env_kwargs:   kwargs passed to env_cls (randomize, physics_guided, use_ekf)
        lstm_path:    optional LSTM model path for deviation forecast overlay
        seed:         episode seed
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    from simulation.lstm_dataset import _extract_windows, WINDOW_LEN, N_FEATURES
    from simulation.lstm_forecaster import LSTMForecaster

    # Load PPO
    model = PPO.load(model_path, device="cpu")

    # Load LSTM if requested
    lstm_model = None
    if lstm_path:
        try:
            ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
            lstm_model = LSTMForecaster(hidden_size=128, num_layers=2)
            lstm_model.load_state_dict(ckpt["model_state"])
            lstm_model.eval()
        except Exception as e:
            print(f"[dashboard] LSTM load failed: {e}")

    # Raw env for recording
    raw_env = env_cls(seed=seed, **env_kwargs)
    obs_raw, _ = raw_env.reset(seed=seed)

    # Vec env for normalisation
    _s = seed
    _kw = dict(env_kwargs)
    vec_env = make_vec_env(lambda: env_cls(seed=_s, **_kw), n_envs=1)
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.reset()

    # Storage
    ts, ys, vys, xs, zs, mass_fracs = [], [], [], [], [], []
    target_ys, target_vys = [], []
    ekf_ys = []
    pitches, yaws = [], []
    r_tots, r_alts, r_vels, r_fuels, r_smooths, r_phys = [], [], [], [], [], []
    lstm_forecasts, lstm_warned = [], []

    # LSTM rolling window buffer
    lstm_window = []   # list of 5-feature vectors

    done = False
    outcome = "timeout"

    while not done:
        obs_norm = vec_env.normalize_obs(obs_raw[np.newaxis])[0]
        action, _ = model.predict(obs_norm, deterministic=True)

        state = raw_env._state
        t_now = raw_env._t

        # Target
        if hasattr(raw_env, '_get_target'):
            target = raw_env._get_target()
        else:
            target = raw_env._target

        if hasattr(target, 'query3d'):
            ty, tvy, _, _ = target.query3d(t_now)
        else:
            result = target.query(t_now)
            ty, tvy = float(result[0]), float(result[1])

        # EKF estimate
        if hasattr(raw_env, '_ekf') and raw_env._ekf is not None:
            ekf_ys.append(float(raw_env._ekf.state[1]))
        else:
            ekf_ys.append(float(state[1]))

        # Mass fraction
        mass = float(state[-1]) if len(state) == 7 else float(state[4])
        rocket = raw_env._rocket
        fuel_total = rocket.mass_wet - rocket.mass_dry
        fuel_rem = max(0.0, mass - rocket.mass_dry)
        mass_frac = fuel_rem / fuel_total if fuel_total > 0 else 0.0

        # Record
        ts.append(t_now)
        ys.append(float(state[1]))
        vys.append(float(state[4]))
        xs.append(float(state[0]))
        zs.append(float(state[2]) if len(state) == 7 else 0.0)
        mass_fracs.append(mass_frac)
        target_ys.append(float(ty))
        target_vys.append(float(tvy))

        if hasattr(raw_env, '_pitch_rad'):
            pitches.append(float(np.degrees(raw_env._pitch_rad)))
            yaws.append(float(np.degrees(raw_env._yaw_rad))
                        if hasattr(raw_env, '_yaw_rad') else 0.0)
        else:
            pitches.append(float(np.degrees(raw_env._angle_rad)))
            yaws.append(0.0)

        # LSTM forecast
        if lstm_model is not None:
            y_val  = float(state[1])
            vy_val = float(state[4])
            err    = abs(y_val - float(ty))
            feat = [y_val, vy_val, float(ty), err, mass_frac]
            lstm_window.append(feat)
            if len(lstm_window) >= WINDOW_LEN:
                win = np.array(lstm_window[-WINDOW_LEN:], dtype=np.float32)
                win_t = torch.tensor(win).unsqueeze(0)
                from simulation.lstm_dataset import _FEAT_SCALE
                win_norm = win_t / torch.tensor(_FEAT_SCALE)
                with torch.no_grad():
                    forecast_m = float(lstm_model(win_norm).item()) * 1000.0
                lstm_forecasts.append(forecast_m)
                lstm_warned.append(forecast_m > 2000.0)
            else:
                lstm_forecasts.append(0.0)
                lstm_warned.append(False)
        else:
            lstm_forecasts.append(0.0)
            lstm_warned.append(False)

        # Step
        obs_raw, r, terminated, truncated, info = raw_env.step(action)
        vec_env.step(action[np.newaxis])

        r_tots.append(float(r))
        r_alts.append(float(info.get("r_alt", 0.0)))
        r_vels.append(float(info.get("r_vel", 0.0)))
        r_fuels.append(float(info.get("r_fuel", 0.0)))
        r_smooths.append(float(info.get("r_smooth", 0.0)))
        r_phys.append(float(info.get("r_physics", 0.0)))

        if terminated:
            outcome = info.get("outcome", "landed")
        done = terminated or truncated

    vec_env.close()

    return {
        "t":            np.array(ts),
        "y":            np.array(ys),
        "vy":           np.array(vys),
        "x":            np.array(xs),
        "z":            np.array(zs),
        "mass_frac":    np.array(mass_fracs),
        "target_y":     np.array(target_ys),
        "target_vy":    np.array(target_vys),
        "ekf_y":        np.array(ekf_ys),
        "pitch":        np.array(pitches),
        "yaw":          np.array(yaws),
        "r_total":      np.array(r_tots),
        "r_alt":        np.array(r_alts),
        "r_vel":        np.array(r_vels),
        "r_fuel":       np.array(r_fuels),
        "r_smooth":     np.array(r_smooths),
        "r_physics":    np.array(r_phys),
        "lstm_forecast":np.array(lstm_forecasts),
        "lstm_warned":  np.array(lstm_warned),
        "outcome":      outcome,
        "rocket":       raw_env._rocket,
    }
