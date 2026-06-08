"""
Phase 7 benchmark — compare v4 (ground truth) vs v5 (EKF observations).

Measures the "sensor noise cost": how much tracking performance degrades
when the agent sees EKF-estimated state instead of perfect ground truth.

This is the sim-to-real gap quantification. A small gap means the EKF is
good and the agent is robust to sensor noise — the system is deployable.

Usage:
    python -m training.evaluate_ekf
    python -m training.evaluate_ekf --n_test 10
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from simulation.env import RocketEnv
from simulation.target_trajectory import PhysicsGuidedTargetTrajectory


def _norm_path(model_path: str) -> str:
    path = model_path.replace("models/ppo_", "models/vecnorm_")
    path = path.replace(".zip", "")
    return path + ".pkl" if not path.endswith(".pkl") else path


def _run_episode(model_path: str, use_ekf: bool,
                 randomize: bool, seed: int) -> dict:
    """Run one deterministic episode, return state history."""
    norm_path = _norm_path(model_path)
    model = PPO.load(model_path, device="cpu")

    vec_env = make_vec_env(
        lambda: RocketEnv(randomize=randomize, seed=seed,
                          physics_guided=True, use_ekf=use_ekf),
        n_envs=1
    )
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    inner = vec_env.venv.envs[0].unwrapped

    ts, ys, vys, angles = [], [], [], []
    ekf_ys = []   # EKF estimate of y (None when use_ekf=False)
    done = False

    while not done:
        # True state (for tracking error measurement)
        x, y, vx, vy, mass = inner._state
        ts.append(inner._t)
        ys.append(y)
        vys.append(vy)
        angles.append(np.degrees(inner._angle_rad))

        # EKF state (for estimation error measurement)
        if use_ekf and inner._ekf is not None:
            ekf_ys.append(float(inner._ekf.state[1]))
        else:
            ekf_ys.append(y)  # same as true when no EKF

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        done = bool(dones[0])

    vec_env.close()

    return {
        "time":      np.array(ts),
        "y":         np.array(ys),
        "vy":        np.array(vys),
        "angle_deg": np.array(angles),
        "ekf_y":     np.array(ekf_ys),
        "rocket":    inner._rocket,
        "wind_v_ref": inner._wind._cfg.v_ref,
    }


def _pinn_reference(result: dict, pinn_ref: PhysicsGuidedTargetTrajectory) -> np.ndarray:
    pinn_ref.set_episode(result["rocket"], result["wind_v_ref"])
    return np.array([pinn_ref.query(t)[0] for t in result["time"]])


def benchmark(v4_path: str, v5_path: str, n_test: int = 10) -> None:
    print(f"Benchmarking sim-to-real gap:")
    print(f"  v4 (ground truth obs): {v4_path}")
    print(f"  v5 (EKF obs):          {v5_path}")
    print(f"  Test configs:          {n_test}")
    print()

    pinn_ref = PhysicsGuidedTargetTrajectory()

    seeds = [99] + list(range(1, n_test))
    randomize_flags = [False] + [True] * (n_test - 1)

    v4_errors, v5_errors = [], []
    v4_angle_var, v5_angle_var = [], []
    ekf_est_errors = []  # |ekf_y - true_y| — how good is EKF itself?

    print(f"{'Config':<8} {'v4 max err':>12} {'v5 max err':>12} "
          f"{'ekf_est_err':>13} {'v4 ang_var':>12} {'v5 ang_var':>12}")
    print("-" * 75)

    for seed, rand in zip(seeds, randomize_flags):
        cfg_label = "nominal" if not rand else f"rand:{seed}"

        r4 = _run_episode(v4_path, use_ekf=False, randomize=rand, seed=seed)
        r5 = _run_episode(v5_path, use_ekf=True,  randomize=rand, seed=seed)

        # Tracking error: agent trajectory vs PINN reference
        pinn_y4 = _pinn_reference(r4, pinn_ref)
        pinn_y5 = _pinn_reference(r5, pinn_ref)

        err4 = np.abs(r4["y"] - pinn_y4)
        err5 = np.abs(r5["y"] - pinn_y5)

        # EKF estimation error: how well does EKF track the true state?
        ekf_err = np.abs(r5["ekf_y"] - r5["y"])

        v4_errors.append(err4)
        v5_errors.append(err5)
        v4_angle_var.append(np.std(np.diff(r4["angle_deg"])))
        v5_angle_var.append(np.std(np.diff(r5["angle_deg"])))
        ekf_est_errors.append(ekf_err)

        print(f"{cfg_label:<8} {err4.max()/1000:>10.3f}km {err5.max()/1000:>10.3f}km "
              f"{ekf_err.mean():>11.2f}m "
              f"{v4_angle_var[-1]:>12.4f} {v5_angle_var[-1]:>12.4f}")

    print()
    print("=== Summary ===")

    v4_max = np.mean([e.max() for e in v4_errors])
    v5_max = np.mean([e.max() for e in v5_errors])
    v4_mean = np.mean([e.mean() for e in v4_errors])
    v5_mean = np.mean([e.mean() for e in v5_errors])
    v4_av = np.mean(v4_angle_var)
    v5_av = np.mean(v5_angle_var)
    ekf_rmse = np.sqrt(np.mean([np.mean(e**2) for e in ekf_est_errors]))

    print(f"{'Metric':<32} {'v4 (ground truth)':>18} {'v5 (EKF)':>12} {'Delta':>10}")
    print("-" * 76)
    print(f"{'Mean max altitude error':<32} {v4_max/1000:>15.3f}km {v5_max/1000:>9.3f}km "
          f"  {(v5_max-v4_max)/v4_max*100:>+7.1f}%")
    print(f"{'Mean avg altitude error':<32} {v4_mean/1000:>15.3f}km {v5_mean/1000:>9.3f}km "
          f"  {(v5_mean-v4_mean)/v4_mean*100:>+7.1f}%")
    print(f"{'Mean angle variation std':<32} {v4_av:>15.4f}° {v5_av:>9.4f}°  "
          f"  {(v5_av-v4_av)/v4_av*100:>+7.1f}%")
    print()
    print(f"EKF altitude estimation RMSE: {ekf_rmse:.3f} m  "
          f"(barometer s=5m -- EKF should be ~5x better)")
    sensor_sigma = 5.0
    print(f"EKF improvement over raw sensor: {sensor_sigma/ekf_rmse:.1f}x")

    _plot(seeds, randomize_flags, v4_errors, v5_errors,
          v4_angle_var, v5_angle_var, ekf_est_errors)


def _plot(seeds, rand_flags, v4_errs, v5_errs,
          v4_av, v5_av, ekf_errs) -> None:
    n = len(seeds)
    labels = ["nominal" if not r else f"s{s}" for s, r in zip(seeds, rand_flags)]
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sim-to-Real Gap: v4 (ground truth) vs v5 (EKF observations)", fontsize=13)

    # Max altitude error
    v4_max = [e.max()/1000 for e in v4_errs]
    v5_max = [e.max()/1000 for e in v5_errs]
    axes[0].bar(x - w/2, v4_max, w, label="v4 ground truth", color="steelblue", alpha=0.8)
    axes[0].bar(x + w/2, v5_max, w, label="v5 EKF",          color="darkorange", alpha=0.8)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="Test config",
                ylabel="Max altitude error (km)", title="Tracking Error vs PINN Reference")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    # Angle variation
    axes[1].bar(x - w/2, v4_av, w, label="v4 ground truth", color="steelblue", alpha=0.8)
    axes[1].bar(x + w/2, v5_av, w, label="v5 EKF",          color="darkorange", alpha=0.8)
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Test config",
                ylabel="Angle variation std (°)", title="Control Smoothness")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    # EKF estimation error per config
    ekf_means = [e.mean() for e in ekf_errs]
    axes[2].bar(x, ekf_means, color="mediumseagreen", alpha=0.8)
    axes[2].axhline(5.0, color="red", linestyle="--", label="Baro σ=5m")
    axes[2].set(xticks=x, xticklabels=labels, xlabel="Test config",
                ylabel="EKF altitude error (m)", title="EKF Estimation Quality")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="y")
    plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("output_ekf_benchmark.png", dpi=150, bbox_inches="tight")
    print("EKF benchmark plot -> output_ekf_benchmark.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4", default="models/ppo_v4_final")
    parser.add_argument("--v5", default="models/ppo_v5_final")
    parser.add_argument("--n_test", type=int, default=10)
    args = parser.parse_args()
    benchmark(args.v4, args.v5, args.n_test)
