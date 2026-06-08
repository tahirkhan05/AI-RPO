"""
PINN evaluation — compares trained PINN against RK4 ground truth.

Checks three things (as defined in notes/06_pinns.md Section 12):
  1. Trajectory accuracy  — PINN vs RK4 (max and mean error per variable)
  2. Physics residuals    — are the equations of motion satisfied?
  3. Interpolation        — do residuals stay small between training points?

Usage:
    python -m training.evaluate_pinn
    python -m training.evaluate_pinn --model models/pinn_v1.pt
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM
from simulation.pinn import RocketPINN
from simulation.trajectory import run as run_rk4


def load_model(path: str) -> tuple[RocketPINN, float]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    t_max = ckpt["t_max"]
    model = RocketPINN(hidden=128, layers=5)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, t_max


def evaluate_pinn(model_path: str = "models/pinn_v4.pt") -> None:
    print(f"Model: {model_path}")
    model, t_max = load_model(model_path)

    # ── 1. Trajectory accuracy ─────────────────────────────────────────────────
    traj = run_rk4(DEFAULT_ROCKET, DEFAULT_SIM)

    t_eval = torch.tensor(traj.time.astype(np.float32)).unsqueeze(-1)
    with torch.no_grad():
        pred = model.predict(t_eval).numpy()   # (N, 5)

    names = ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)", "mass (kg)"]
    truth = np.stack([traj.x, traj.y, traj.vx, traj.vy, traj.mass], axis=1)

    print()
    print("=== Trajectory Accuracy ===")
    print(f"{'Variable':<12} {'Max error':>12} {'Mean error':>12}  {'Note'}")
    print("-" * 60)
    for i, name in enumerate(names):
        err  = np.abs(pred[:, i] - truth[:, i])
        note = "(analytical — exact by construction)" if i == 4 else ""
        print(f"{name:<12} {err.max():>12.3f} {err.mean():>12.3f}  {note}")

    # ── 2. Physics residuals ───────────────────────────────────────────────────
    print()
    print("=== Physics Residuals ===")

    n_eval = 1_000
    t_col  = torch.linspace(0.01, t_max - 0.01, n_eval).unsqueeze(-1)
    t_col.requires_grad_(True)

    resid = model.residuals(t_col)   # (N, 5)
    resid_np = resid.detach().numpy()

    resid_names = [
        "R0: dx/dt - vx   (m/s)",
        "R1: dy/dt - vy   (m/s)",
        "R2: dvx/dt - ax  (m/s²)",
        "R3: dvy/dt - ay  (m/s²)",
    ]
    print(f"{'Residual':<30} {'Max |R|':>10} {'Mean |R|':>10}")
    print("-" * 52)
    for i, name in enumerate(resid_names):
        r = np.abs(resid_np[:, i])
        print(f"{name:<30} {r.max():>10.4f} {r.mean():>10.4f}")
    print("  R4 (mass) = 0 exactly — enforced analytically, not by the network")

    # ── 3. Plots ───────────────────────────────────────────────────────────────
    _plot_trajectory(traj, pred, "output_pinn_trajectory.png")
    _plot_residuals(resid_np, t_col.detach().numpy().squeeze(), "output_pinn_residuals.png")
    print()
    print("Trajectory plot -> output_pinn_trajectory.png")
    print("Residuals plot  -> output_pinn_residuals.png")


def _plot_trajectory(traj, pred: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PINN vs RK4 Ground Truth", fontsize=13)

    pairs = [
        (traj.time, traj.y / 1000,     pred[:, 1] / 1000,    "Time (s)", "Altitude (km)",  "Altitude"),
        (traj.time, traj.vy,            pred[:, 3],            "Time (s)", "vy (m/s)",       "Vertical velocity"),
        (traj.time, traj.vx,            pred[:, 2],            "Time (s)", "vx (m/s)",       "Horizontal velocity"),
        (traj.time, traj.mass,          pred[:, 4],            "Time (s)", "Mass (kg)",       "Mass"),
        (traj.x / 1000, traj.y / 1000, pred[:, 1] / 1000,    "Downrange (km)", "Alt (km)", "Flight path"),
        (traj.time, np.abs(traj.y - pred[:, 1]),              None,
         "Time (s)", "Error (m)", "Altitude error |PINN - RK4|"),
    ]

    for ax, pair in zip(axes.flat, pairs):
        if pair[2] is None:
            # error plot — only one series
            ax.plot(pair[0], pair[1], color="tomato", lw=1.2)
            ax.axhline(0, color="k", ls="--", lw=0.8)
        else:
            if len(pair) == 6:
                x_rk4, y_rk4, y_pinn, xl, yl, title = pair
            else:
                x_rk4 = pair[0]; y_rk4 = pair[1]; y_pinn = pair[2]
                xl = pair[3]; yl = pair[4]; title = pair[5]
            ax.plot(x_rk4, y_rk4, "k--", lw=1.5, label="RK4")
            ax.plot(x_rk4, y_pinn, color="steelblue", lw=1.2, label="PINN")
            ax.legend(fontsize=8)

        ax.set(xlabel=pair[-3] if len(pair) == 6 else pair[3],
               ylabel=pair[-2] if len(pair) == 6 else pair[4],
               title=pair[-1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_residuals(resid: np.ndarray, t: np.ndarray, path: str) -> None:
    labels = ["R0: dx/dt - vx", "R1: dy/dt - vy",
              "R2: dvx/dt - ax", "R3: dvy/dt - ay"]
    units  = ["m/s", "m/s", "m/s²", "m/s²"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Physics Residuals Across Flight (target: near zero)", fontsize=13)

    for i, ax in enumerate(axes.flat):
        ax.plot(t, resid[:, i], lw=1.0, color="steelblue")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set(xlabel="Time (s)", ylabel=units[i], title=labels[i])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pinn_v4.pt")
    args = parser.parse_args()
    evaluate_pinn(args.model)
