"""
PINN training — Option A: nominal trajectory.

Trains RocketPINN to reproduce the DEFAULT_ROCKET / DEFAULT_SIM trajectory
while satisfying the rocket equations of motion at collocation points.

Loss:
  L_total = L_data + lambda_phys * L_physics

  L_data    : MSE between PINN [x, y, vx, vy] and RK4 ground truth
  L_physics : MSE of 4 kinematic residuals at randomly sampled collocation points

GPU is used here (unlike PPO) because:
  - PyTorch autograd over residuals is GPU-accelerated
  - Batch gradient descent on the wider network benefits from parallelism
  - The RTX 4050 is the right tool for this job

Two-phase training (BUG-015 / BUG-016 fix):
  Phase A (data only):   fit trajectory shape before adding physics pressure
  Phase B (warm lambda): gradually tighten physics from lambda=0.01 to 1.0

Usage:
    python -m training.train_pinn
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM
from simulation.pinn import RocketPINN
from simulation.trajectory import run as run_rk4


# ── Residual normalisation scales [R0, R1, R2, R3] ───────────────────────────
# Calibrated from actual derivative magnitudes along the nominal trajectory.
_R_SCALES = torch.tensor([
    100.0,   # R0: dx/dt - vx   (m/s)   — vx peak ~88 m/s
    900.0,   # R1: dy/dt - vy   (m/s)   — vy peak ~876 m/s
    10.0,    # R2: dvx/dt - ax  (m/s²)  — ax peak ~5 m/s²
    60.0,    # R3: dvy/dt - ay  (m/s²)  — ay peak ~59 m/s²
], dtype=torch.float32)

# ── Data normalisation scales [x, y, vx, vy] ─────────────────────────────────
_STATE_SCALES = torch.tensor([
    20_000.0,   # x  (m)
    35_000.0,   # y  (m)
    600.0,      # vx (m/s)
    900.0,      # vy (m/s)
], dtype=torch.float32)


def _build_training_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run RK4, subsample to N_data points.
    Returns (t_data (N,1), state_data (N,4)) — [x, y, vx, vy] only.
    Mass is excluded: the PINN computes it analytically.
    """
    traj  = run_rk4(DEFAULT_ROCKET, DEFAULT_SIM)
    N_data = 500
    idx   = np.linspace(0, len(traj.time) - 1, N_data, dtype=int)

    t_np     = traj.time[idx].astype(np.float32)
    state_np = np.stack([
        traj.x[idx], traj.y[idx], traj.vx[idx], traj.vy[idx]
    ], axis=1).astype(np.float32)

    t_data     = torch.tensor(t_np,     device=device).unsqueeze(-1)  # (N,1)
    state_data = torch.tensor(state_np, device=device)                 # (N,4)
    return t_data, state_data


def _data_loss(model: RocketPINN, t_data: torch.Tensor,
               state_data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MSE between PINN [x, y, vx, vy] and RK4 ground truth, normalised."""
    pred   = model.predict_grad(t_data)[:, :4]     # (N, 4) — grad-enabled, drop mass
    diff   = (pred - state_data) / scales.to(pred.device)
    return (diff ** 2).mean()


def _physics_loss(model: RocketPINN, t_max: float,
                  n_colloc: int, device: torch.device,
                  r_scales: torch.Tensor) -> torch.Tensor:
    """
    MSE of 4 physics residuals at randomly sampled collocation points.
    30% of points biased near burnout (~25s) where curvature is highest.
    """
    cfg      = DEFAULT_ROCKET
    n_uniform = int(0.7 * n_colloc)
    n_burnout = n_colloc - n_uniform

    t_unif = torch.rand(n_uniform, 1, device=device) * t_max
    t_burn = (cfg.mass_wet - cfg.mass_dry) / cfg.burn_rate   # ~25s
    t_near = t_burn + (torch.rand(n_burnout, 1, device=device) - 0.5) * 10.0
    t_near = t_near.clamp(0.01, t_max - 0.01)

    t_col = torch.cat([t_unif, t_near], dim=0)
    t_col.requires_grad_(True)

    resid      = model.residuals(t_col)                           # (N_c, 4)
    resid_norm = resid / r_scales.to(device).unsqueeze(0)
    return (resid_norm ** 2).mean()


def train_pinn(
    epochs_data: int = 5_000,
    epochs_phys: int = 20_000,
    lambda_phys: float = 0.01,
    lambda_phys_max: float = 1.0,
    n_colloc: int = 2_000,
    lr: float = 1e-3,
    save_path: str = "models/pinn_v4.pt",
    plot_every: int = 2_000,
) -> None:
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = epochs_data + epochs_phys

    print(f"Device:         {device}")
    print(f"Phase A (data): {epochs_data:,} epochs  (lambda=0, fit trajectory)")
    print(f"Phase B (phys): {epochs_phys:,} epochs  (lambda {lambda_phys} -> {lambda_phys_max})")
    print(f"Collocation:    {n_colloc} pts/batch  (30% near burnout)")
    print()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    model           = RocketPINN(hidden=128, layers=5).to(device)
    _R_SCALES_d     = _R_SCALES.to(device)
    _STATE_SCALES_d = _STATE_SCALES.to(device)

    traj_full = run_rk4(DEFAULT_ROCKET, DEFAULT_SIM)
    t_max     = float(traj_full.time[-1])
    t_data, state_data = _build_training_data(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5
    )

    loss_history = {"total": [], "data": [], "physics": [], "lambda": []}

    print(f"{'Epoch':>8} | {'Phase':>6} | {'L_total':>10} | "
          f"{'L_data':>10} | {'L_phys':>10} | {'lambda':>8}")
    print("-" * 72)

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if epoch <= epochs_data:
            lam     = 0.0
            phase   = "data"
            l_data  = _data_loss(model, t_data, state_data, _STATE_SCALES_d)
            l_phys  = torch.tensor(0.0, device=device)
            l_total = l_data
        else:
            frac    = (epoch - epochs_data) / epochs_phys
            lam     = lambda_phys * (lambda_phys_max / lambda_phys) ** frac
            phase   = "phys"
            l_data  = _data_loss(model, t_data, state_data, _STATE_SCALES_d)
            l_phys  = _physics_loss(model, t_max, n_colloc, device, _R_SCALES_d)
            l_total = l_data + lam * l_phys

        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(l_total.item())
        loss_history["data"].append(l_data.item())
        loss_history["physics"].append(l_phys.item())
        loss_history["lambda"].append(lam)

        if epoch % plot_every == 0 or epoch == 1 or epoch == epochs_data:
            print(
                f"{epoch:>8,} | {phase:>6} | {l_total.item():>10.6f} | "
                f"{l_data.item():>10.6f} | {l_phys.item():>10.6f} | {lam:>8.4f}"
            )

    print()
    torch.save({"model_state": model.state_dict(), "t_max": t_max}, save_path)
    print(f"Model saved -> {save_path}")
    _plot_loss(loss_history, "logs/pinn_loss_curve.png")
    print(f"Loss curve  -> logs/pinn_loss_curve.png")


def _plot_loss(history: dict, path: str) -> None:
    epochs = range(1, len(history["total"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(epochs, history["total"],   label="L_total",   lw=1.5)
    ax.semilogy(epochs, history["data"],    label="L_data",    lw=1.2, ls="--")
    ax.semilogy(epochs, history["physics"], label="L_physics", lw=1.2, ls=":")
    ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title="PINN Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_data",     type=int,   default=5_000)
    parser.add_argument("--epochs_phys",     type=int,   default=20_000)
    parser.add_argument("--lambda_phys",     type=float, default=0.01)
    parser.add_argument("--lambda_phys_max", type=float, default=1.0)
    parser.add_argument("--n_colloc",        type=int,   default=2_000)
    parser.add_argument("--lr",              type=float, default=1e-3)
    args = parser.parse_args()
    train_pinn(
        epochs_data=args.epochs_data,
        epochs_phys=args.epochs_phys,
        lambda_phys=args.lambda_phys,
        lambda_phys_max=args.lambda_phys_max,
        n_colloc=args.n_colloc,
        lr=args.lr,
    )
