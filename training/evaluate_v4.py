"""
Phase 6 benchmark — compare PPO v3 (fixed nominal reference) vs
PPO v4 (PINN-guided per-episode reference).

Runs both models on the same set of test configurations and measures:
  1. Altitude tracking error vs PINN reference (the correct reference)
  2. Angle smoothness (variation in thrust angle)
  3. Physics consistency (how close to PINN prediction)

This is Table 1 / Figure 1 of the paper.

Usage:
    python -m training.evaluate_v4
    python -m training.evaluate_v4 --n_test 10
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
from simulation.target_trajectory import PhysicsGuidedTargetTrajectory
from simulation.trajectory import run as run_rk4


def _norm_path(model_path: str) -> str:
    path = model_path.replace("models/ppo_", "models/vecnorm_")
    path = path.replace(".zip", "")
    return path + ".pkl" if not path.endswith(".pkl") else path


def _run_episode(model_path: str, physics_guided: bool,
                 randomize: bool, seed: int) -> dict:
    """Run one deterministic episode, return state history + metrics."""
    norm_path = _norm_path(model_path)
    model = PPO.load(model_path, device="cpu")

    vec_env = make_vec_env(
        lambda: RocketEnv(randomize=randomize, seed=seed,
                          physics_guided=physics_guided),
        n_envs=1
    )
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    inner = vec_env.venv.envs[0].unwrapped

    ts, ys, vys, angles = [], [], [], []
    done = False

    while not done:
        x, y, vx, vy, mass = inner._state
        ts.append(inner._t)
        ys.append(y)
        vys.append(vy)
        angles.append(np.degrees(inner._angle_rad))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        done = bool(dones[0])

    vec_env.close()

    # Record the rocket config that was used (for PINN reference)
    return {
        "time":      np.array(ts),
        "y":         np.array(ys),
        "vy":        np.array(vys),
        "angle_deg": np.array(angles),
        "rocket":    inner._rocket,
        "wind_v_ref": inner._wind._cfg.v_ref,
    }


def _pinn_reference(result: dict, pinn_ref: PhysicsGuidedTargetTrajectory) -> np.ndarray:
    """Get PINN reference altitude for each timestep of an episode."""
    pinn_ref.set_episode(result["rocket"], result["wind_v_ref"])
    return np.array([pinn_ref.query(t)[0] for t in result["time"]])


def benchmark(v3_path: str, v4_path: str, n_test: int = 10) -> None:
    print(f"Benchmarking:")
    print(f"  v3 (fixed reference): {v3_path}")
    print(f"  v4 (PINN-guided):     {v4_path}")
    print(f"  Test configs:         {n_test} (nominal + {n_test-1} randomised)")
    print()

    pinn_ref = PhysicsGuidedTargetTrajectory()

    # Nominal config first, then randomised
    seeds = [99] + list(range(1, n_test))
    randomize_flags = [False] + [True] * (n_test - 1)

    v3_errors, v4_errors = [], []
    v3_angle_var, v4_angle_var = [], []
    v3_phys_errors, v4_phys_errors = [], []

    print(f"{'Config':<8} {'v3 max err':>12} {'v4 max err':>12} "
          f"{'v3 angle_var':>14} {'v4 angle_var':>14}")
    print("-" * 65)

    for i, (seed, rand) in enumerate(zip(seeds, randomize_flags)):
        cfg_label = "nominal" if not rand else f"rand:{seed}"

        r3 = _run_episode(v3_path, physics_guided=False, randomize=rand, seed=seed)
        r4 = _run_episode(v4_path, physics_guided=True,  randomize=rand, seed=seed)

        # PINN reference for v4's rocket config (the physically correct reference)
        pinn_y_v3 = _pinn_reference(r3, pinn_ref)
        pinn_y_v4 = _pinn_reference(r4, pinn_ref)

        err3 = np.abs(r3["y"] - pinn_y_v3)
        err4 = np.abs(r4["y"] - pinn_y_v4)

        angle_var3 = np.std(np.diff(r3["angle_deg"]))
        angle_var4 = np.std(np.diff(r4["angle_deg"]))

        v3_errors.append(err3)
        v4_errors.append(err4)
        v3_angle_var.append(angle_var3)
        v4_angle_var.append(angle_var4)

        print(f"{cfg_label:<8} {err3.max()/1000:>10.3f}km {err4.max()/1000:>10.3f}km "
              f"{angle_var3:>14.4f} {angle_var4:>14.4f}")

    print()
    print("=== Summary ===")

    v3_max = np.mean([e.max() for e in v3_errors])
    v4_max = np.mean([e.max() for e in v4_errors])
    v3_mean_err = np.mean([e.mean() for e in v3_errors])
    v4_mean_err = np.mean([e.mean() for e in v4_errors])
    v3_av = np.mean(v3_angle_var)
    v4_av = np.mean(v4_angle_var)

    print(f"{'Metric':<30} {'v3 (fixed ref)':>16} {'v4 (PINN-guided)':>18} {'Delta':>10}")
    print("-" * 78)
    print(f"{'Mean max altitude error':<30} {v3_max/1000:>13.3f}km {v4_max/1000:>15.3f}km "
          f"  {(v4_max-v3_max)/v3_max*100:>+7.1f}%")
    print(f"{'Mean avg altitude error':<30} {v3_mean_err/1000:>13.3f}km {v4_mean_err/1000:>15.3f}km "
          f"  {(v4_mean_err-v3_mean_err)/v3_mean_err*100:>+7.1f}%")
    print(f"{'Mean angle variation (std)':<30} {v3_av:>13.4f}° {v4_av:>15.4f}°  "
          f"  {(v4_av-v3_av)/v3_av*100:>+7.1f}%")

    _plot_comparison(seeds, randomize_flags, v3_errors, v4_errors,
                     v3_angle_var, v4_angle_var, v3_path, v4_path)


def _plot_comparison(seeds, rand_flags, v3_errs, v4_errs,
                     v3_av, v4_av, v3_path, v4_path) -> None:
    n = len(seeds)
    labels = ["nominal" if not r else f"s{s}" for s, r in zip(seeds, rand_flags)]
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PPO v3 (fixed ref) vs v4 (PINN-guided) — Benchmark", fontsize=13)

    # Max altitude error
    v3_max = [e.max()/1000 for e in v3_errs]
    v4_max = [e.max()/1000 for e in v4_errs]
    axes[0].bar(x - w/2, v3_max, w, label="v3 (fixed ref)", color="tomato",   alpha=0.8)
    axes[0].bar(x + w/2, v4_max, w, label="v4 (PINN-guided)", color="steelblue", alpha=0.8)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="Test config",
                ylabel="Max altitude error (km)", title="Tracking Error vs PINN Reference")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    # Angle variation
    axes[1].bar(x - w/2, v3_av, w, label="v3 (fixed ref)", color="tomato",   alpha=0.8)
    axes[1].bar(x + w/2, v4_av, w, label="v4 (PINN-guided)", color="steelblue", alpha=0.8)
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Test config",
                ylabel="Angle variation std (°)", title="Control Smoothness (lower = smoother)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("output_v3_vs_v4_benchmark.png", dpi=150, bbox_inches="tight")
    print("Benchmark plot -> output_v3_vs_v4_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", default="models/ppo_v3_final")
    parser.add_argument("--v4", default="models/ppo_v4_final")
    parser.add_argument("--n_test", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.v3, args.v4, args.n_test)
