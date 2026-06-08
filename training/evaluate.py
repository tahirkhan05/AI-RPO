"""
Evaluate a trained PPO model against the nominal baseline trajectory.

Uses VecNormalize.load() to apply the exact same observation normalisation
the agent saw during training — eliminates the BUG-011 normalisation mismatch.

Usage:
    python -m training.evaluate
    python -m training.evaluate --model models/ppo_v3_final
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env import RocketEnv
from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM
from simulation.trajectory import run as run_baseline


def _norm_path(model_path: str) -> str:
    """Derive the VecNormalize pkl path from the model path."""
    path = model_path.replace("models/ppo_", "models/vecnorm_")
    path = path.replace(".zip", "")
    if not path.endswith(".pkl"):
        path += ".pkl"
    return path


def run_episode(model_path: str, randomize: bool = False, seed: int = 99) -> dict:
    """
    Run one deterministic episode. Returns dict of state arrays.

    Normalisation is applied via VecNormalize.load() so it exactly matches
    training. The inner raw env is accessed directly for state recording.
    """
    norm_path = _norm_path(model_path)
    model = PPO.load(model_path, device="cpu")

    # Wrap in a VecEnv so VecNormalize can be applied
    vec_env = make_vec_env(lambda: RocketEnv(randomize=randomize, seed=seed), n_envs=1)
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False     # freeze running stats during eval
    vec_env.norm_reward = False  # we don't need reward normalisation for eval

    obs = vec_env.reset()
    inner_env = vec_env.venv.envs[0].unwrapped  # raw RocketEnv for state recording

    xs, ys, vxs, vys, masses, angles, ts, winds = [], [], [], [], [], [], [], []
    done = False

    while not done:
        x, y, vx, vy, mass = inner_env._state
        xs.append(x); ys.append(y); vxs.append(vx)
        vys.append(vy); masses.append(mass)
        angles.append(np.degrees(inner_env._angle_rad))
        ts.append(inner_env._t)
        winds.append(float(inner_env._wind._gust))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        done = bool(dones[0])

    vec_env.close()

    return {
        "time":      np.array(ts),
        "x":         np.array(xs),
        "y":         np.array(ys),
        "vx":        np.array(vxs),
        "vy":        np.array(vys),
        "mass":      np.array(masses),
        "angle_deg": np.array(angles),
        "wind":      np.array(winds),
    }


def plot_evaluation(result: dict, baseline, save_path: str) -> None:
    ts  = result["time"]
    ys  = result["y"]
    xs  = result["x"]
    ang = result["angle_deg"]

    bl_y = interp1d(baseline.time, baseline.y,
                    bounds_error=False, fill_value=(baseline.y[0], baseline.y[-1]))
    error = ys - bl_y(ts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Trained PPO Agent vs Nominal Baseline", fontsize=13)

    axes[0, 0].plot(baseline.time, baseline.y / 1000, "k--", lw=1.5, label="Baseline")
    axes[0, 0].plot(ts, ys / 1000, color="steelblue", lw=1.5, label="PPO Agent")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Altitude (km)", title="Altitude Tracking")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(baseline.x / 1000, baseline.y / 1000, "k--", lw=1.5, label="Baseline")
    axes[0, 1].plot(xs / 1000, ys / 1000, color="steelblue", lw=1.5, label="PPO Agent")
    axes[0, 1].set(xlabel="Downrange (km)", ylabel="Altitude (km)", title="Flight Path")
    axes[0, 1].set_ylim(bottom=0); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ts, ang, color="tomato", lw=1.2)
    axes[1, 0].axhline(85, color="k", ls="--", lw=0.8, label="Nominal 85 deg")
    axes[1, 0].set_ylim(0, 95)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Thrust angle (deg)",
                   title="Agent Control Actions")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts, error / 1000, color="goldenrod", lw=1.2)
    axes[1, 1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Error (km)",
                   title="Altitude Tracking Error (Agent - Baseline)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {save_path}")


def evaluate(model_path: str = "models/ppo_v3_final") -> None:
    print(f"Model:     {model_path}")
    print(f"VecNorm:   {_norm_path(model_path)}")
    print()

    baseline = run_baseline(DEFAULT_ROCKET, DEFAULT_SIM)
    result   = run_episode(model_path, randomize=False, seed=99)

    ts   = result["time"]
    ys   = result["y"]
    bl_y = interp1d(baseline.time, baseline.y,
                    bounds_error=False, fill_value=(baseline.y[0], baseline.y[-1]))
    error = ys - bl_y(ts)

    print(f"Episode steps:       {len(ts)}")
    print(f"Agent apogee:        {ys.max()/1000:.3f} km")
    print(f"Baseline apogee:     {baseline.apogee/1000:.3f} km")
    print(f"Max tracking error:  {abs(error).max()/1000:.3f} km")
    print(f"Mean tracking error: {abs(error).mean()/1000:.3f} km")
    print(f"Angle range:         {result['angle_deg'].min():.1f} to {result['angle_deg'].max():.1f} deg")
    print(f"Final mass:          {result['mass'][-1]:.2f} kg")

    plot_evaluation(result, baseline, "output_trained_agent.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_v3_final")
    args = parser.parse_args()
    evaluate(args.model)
