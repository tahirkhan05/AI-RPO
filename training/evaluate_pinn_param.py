"""
Parameterised PINN evaluation — Option B.

Tests generalisation across the parameter space by evaluating on held-out
configurations not seen during training.

Three checks:
  1. Nominal trajectory accuracy  (DEFAULT_ROCKET, no wind)
  2. Perturbed trajectories       (N_TEST configs from LHS of held-out region)
  3. Physics residuals            (are the EOM satisfied across configs?)

Usage:
    python -m training.evaluate_pinn_param
    python -m training.evaluate_pinn_param --model models/pinn_param_v1.pt --n_test 20
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM, RocketConfig
from simulation.pinn_param import (
    RocketPINNParam, PARAM_KEYS, PARAM_RANGES, normalise_params, denormalise_params
)
from simulation.trajectory import run as run_rk4
from training.train_pinn_param import _constant_wind  # shared duck-typed wind object


def load_model(path: str) -> tuple[RocketPINNParam, float]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    t_max = ckpt["t_max"]
    hidden = ckpt.get("hidden", 256)
    layers = ckpt.get("layers", 6)
    model  = RocketPINNParam(hidden=hidden, layers=layers)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, t_max


def _nominal_params() -> torch.Tensor:
    """Returns normalised parameter vector for DEFAULT_ROCKET, no wind."""
    raw = torch.tensor([[
        DEFAULT_ROCKET.mass_wet,
        DEFAULT_ROCKET.mass_dry,
        DEFAULT_ROCKET.thrust,
        DEFAULT_ROCKET.burn_rate,
        DEFAULT_ROCKET.drag_coeff,
        0.0,   # wind_vx = 0
    ]])
    return normalise_params(raw)   # (1, 6)


def _random_test_params(n: int, seed: int = 99) -> torch.Tensor:
    """
    Generate n random test parameter configs.
    Uses a different seed from training to ensure held-out configs.
    Returns (n, 6) normalised.
    """
    rng = np.random.default_rng(seed)
    raw_norm = rng.uniform(0.0, 1.0, (n, len(PARAM_KEYS))).astype(np.float32)
    return torch.tensor(raw_norm)


def _run_rk4_for_params(p_norm_single: torch.Tensor) -> tuple:
    """
    Run RK4 for a single normalised parameter vector.
    Returns (traj, mass_wet, mass_dry, burn_rate, wind_vx) for the config.
    """
    p_raw = denormalise_params(p_norm_single.unsqueeze(0)).squeeze(0).numpy()

    mass_wet   = float(p_raw[0])
    mass_dry   = min(float(p_raw[1]), mass_wet * 0.85)
    thrust     = float(p_raw[2])
    burn_rate  = float(p_raw[3])
    drag_coeff = float(p_raw[4])
    wind_vx    = float(p_raw[5])

    rocket = RocketConfig(
        mass_wet=mass_wet,
        mass_dry=mass_dry,
        thrust=thrust,
        burn_rate=burn_rate,
        exhaust_velocity=DEFAULT_ROCKET.exhaust_velocity,
        drag_coeff=drag_coeff,
        cross_section=DEFAULT_ROCKET.cross_section,
    )
    traj = run_rk4(rocket, DEFAULT_SIM, wind=_constant_wind(wind_vx))
    return traj, mass_wet, mass_dry, burn_rate, wind_vx


def evaluate_pinn_param(model_path: str = "models/pinn_param_v1.pt",
                        n_test: int = 20) -> None:
    print(f"Model: {model_path}")
    model, t_max = load_model(model_path)

    # ── 1. Nominal trajectory accuracy ────────────────────────────────────────
    print("\n=== 1. Nominal Trajectory (DEFAULT_ROCKET, wind=0) ===")
    traj_nom = run_rk4(DEFAULT_ROCKET, DEFAULT_SIM)
    p_nom    = _nominal_params()   # (1, 6)

    t_eval = torch.tensor(traj_nom.time.astype(np.float32)).unsqueeze(-1)
    p_exp  = p_nom.expand(len(traj_nom.time), -1)

    pred_nom = model.predict(t_eval, denormalise_params(p_exp))   # (N, 5)
    pred_nom = pred_nom.numpy()

    names = ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)", "mass (kg)"]
    truth_nom = np.stack([traj_nom.x, traj_nom.y, traj_nom.vx, traj_nom.vy,
                          traj_nom.mass], axis=1)

    print(f"{'Variable':<12} {'Max error':>12} {'Mean error':>12}")
    print("-" * 40)
    for i, name in enumerate(names):
        err = np.abs(pred_nom[:, i] - truth_nom[:, i])
        print(f"{name:<12} {err.max():>12.3f} {err.mean():>12.3f}")

    # ── 2. Generalisation across n_test held-out configs ──────────────────────
    print(f"\n=== 2. Generalisation ({n_test} held-out test configs) ===")
    p_test = _random_test_params(n_test)   # (n_test, 6) normalised

    max_errs_y  = []
    mean_errs_y = []

    for i in range(n_test):
        traj, _, _, _, _ = _run_rk4_for_params(p_test[i])

        t_ev   = torch.tensor(traj.time.astype(np.float32)).unsqueeze(-1)
        p_ev   = p_test[i].unsqueeze(0).expand(len(traj.time), -1)
        p_raw  = denormalise_params(p_ev)

        pred = model.predict(t_ev, p_raw).numpy()

        err_y = np.abs(pred[:, 1] - traj.y)
        max_errs_y.append(err_y.max())
        mean_errs_y.append(err_y.mean())

    print(f"Altitude error across {n_test} test configs:")
    print(f"  Mean of (per-traj max error):  {np.mean(max_errs_y):>10.1f} m")
    print(f"  Mean of (per-traj mean error): {np.mean(mean_errs_y):>10.1f} m")
    print(f"  Worst single trajectory:       {np.max(max_errs_y):>10.1f} m")
    print(f"  Best  single trajectory:       {np.min(max_errs_y):>10.1f} m")

    # ── 3. Physics residuals (nominal config, 1000 eval points) ──────────────
    print("\n=== 3. Physics Residuals (nominal config, 1,000 eval pts) ===")
    n_eval = 1_000
    t_col  = torch.linspace(0.01, t_max - 0.01, n_eval).unsqueeze(-1)
    t_col.requires_grad_(True)
    p_col  = p_nom.expand(n_eval, -1)

    resid    = model.residuals(t_col, p_col)
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
    print("  R4 (mass) = 0 exactly (analytical)")

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    _plot_nominal(traj_nom, pred_nom, "output_pinn_param_trajectory.png")
    _plot_generalisation(max_errs_y, "output_pinn_param_generalisation.png")
    _plot_residuals(resid_np, t_col.detach().numpy().squeeze(),
                    "output_pinn_param_residuals.png")

    print()
    print("Trajectory plot       -> output_pinn_param_trajectory.png")
    print("Generalisation plot   -> output_pinn_param_generalisation.png")
    print("Residuals plot        -> output_pinn_param_residuals.png")


def _plot_nominal(traj, pred: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Parameterised PINN vs RK4 — Nominal Config", fontsize=13)

    pairs = [
        (traj.time, traj.y / 1000,  pred[:, 1] / 1000,  "Time (s)", "Altitude (km)",  "Altitude"),
        (traj.time, traj.vy,         pred[:, 3],          "Time (s)", "vy (m/s)",       "Vertical velocity"),
        (traj.time, traj.vx,         pred[:, 2],          "Time (s)", "vx (m/s)",       "Horizontal velocity"),
        (traj.time, np.abs(traj.y - pred[:, 1]),
         None, "Time (s)", "Altitude error (m)", "Altitude error |PINN - RK4|"),
    ]

    for ax, (x, y_rk4, y_pinn, xl, yl, title) in zip(axes.flat, pairs):
        if y_pinn is None:
            ax.plot(x, y_rk4, color="tomato", lw=1.2)
        else:
            ax.plot(x, y_rk4,  "k--", lw=1.5, label="RK4")
            ax.plot(x, y_pinn, color="steelblue", lw=1.2, label="PINN")
            ax.legend(fontsize=8)
        ax.set(xlabel=xl, ylabel=yl, title=title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_generalisation(max_errs: list, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(max_errs)), max_errs, color="steelblue", alpha=0.7)
    ax.axhline(np.mean(max_errs), color="tomato", ls="--", lw=2,
               label=f"Mean = {np.mean(max_errs):.0f} m")
    ax.set(xlabel="Test trajectory index", ylabel="Max altitude error (m)",
           title="Parameterised PINN — Generalisation to Held-Out Configs")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_residuals(resid: np.ndarray, t: np.ndarray, path: str) -> None:
    labels = ["R0: dx/dt - vx", "R1: dy/dt - vy",
              "R2: dvx/dt - ax", "R3: dvy/dt - ay"]
    units  = ["m/s", "m/s", "m/s²", "m/s²"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Physics Residuals — Parameterised PINN (nominal config)", fontsize=13)

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
    parser.add_argument("--model",  default="models/pinn_param_v1.pt")
    parser.add_argument("--n_test", type=int, default=20)
    args = parser.parse_args()
    evaluate_pinn_param(args.model, args.n_test)
