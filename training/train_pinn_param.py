"""
Parameterised PINN training — Option B.

Trains RocketPINNParam to generalise across a family of rocket + wind configs,
so the PINN can predict any trajectory within the domain-randomisation envelope
without needing a new RK4 simulation.

Training data:
  - N_TRAJ trajectories via Latin Hypercube Sampling over 6D parameter space
  - Each trajectory: N_PTS_PER_TRAJ RK4 data points
  - Total data: N_TRAJ * N_PTS_PER_TRAJ points across the parameter family

Loss (same structure as Option A, but per-trajectory):
  L_data  = MSE over [x, y, vx, vy] for all data points across all trajectories
  L_phys  = MSE of 4 residuals at collocation points with randomly sampled params

Soft burnout switch used (see notes/06_pinns.md Section 17) — eliminates the
spike in autograd derivative at thrust cutoff.

Usage:
    python -m training.train_pinn_param
    python -m training.train_pinn_param --n_traj 50 --epochs_data 3000

[PAPER OPPORTUNITY: Latin Hypercube Sampling for physics-informed trajectory surrogates]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM, RocketConfig
from simulation.pinn_param import (
    RocketPINNParam, PARAM_KEYS, PARAM_RANGES, normalise_params
)
from simulation.trajectory import run as run_rk4


# ── Residual normalisation scales [R0, R1, R2, R3] ───────────────────────────
# Same calibration approach as Option A, but R2/R3 slightly wider due to
# thrust variation across the parameter family.
_R_SCALES = torch.tensor([
    100.0,   # R0: dx/dt - vx   (m/s)
    900.0,   # R1: dy/dt - vy   (m/s)
    15.0,    # R2: dvx/dt - ax  (m/s²) — wider than Option A (thrust varies ±20%)
    65.0,    # R3: dvy/dt - ay  (m/s²)
], dtype=torch.float32)

# ── State normalisation scales [x, y, vx, vy] ─────────────────────────────────
_STATE_SCALES = torch.tensor([
    25_000.0,   # x  (m) — wider due to wind drift
    35_000.0,   # y  (m)
    700.0,      # vx (m/s)
    900.0,      # vy (m/s)
], dtype=torch.float32)


def _lhs_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Latin Hypercube Sampling — n samples over 6D parameter space.
    Returns (n, 6) array in [0, 1] (normalised param space).

    LHS partitions each dimension into n equal intervals and takes exactly
    one sample per interval per dimension (then shuffles). This gives much
    better space coverage than pure random sampling, especially for n < 200.

    [PAPER OPPORTUNITY: LHS vs uniform random for PINN training data coverage]
    """
    n_dims  = len(PARAM_KEYS)
    samples = np.zeros((n, n_dims))
    for d in range(n_dims):
        # One point per interval: (k + U[0,1]) / n  for k in 0..n-1
        intervals = (np.arange(n) + rng.uniform(0, 1, n)) / n
        samples[:, d] = rng.permutation(intervals)
    return samples


def _build_training_data(n_traj: int, n_pts: int, device: torch.device,
                         rng: np.random.Generator) -> tuple:
    """
    Generate training dataset: n_traj trajectories from LHS parameter samples.

    Returns:
      t_data:     (n_traj * n_pts, 1)   time in seconds
      state_data: (n_traj * n_pts, 4)   [x, y, vx, vy]
      p_norm_data:(n_traj * n_pts, 6)   normalised parameters (repeated per traj)
    """
    print(f"Generating {n_traj} training trajectories via LHS...")
    p_norm_lhs = _lhs_sample(n_traj, rng)   # (n_traj, 6) in [0, 1]

    all_t, all_state, all_p = [], [], []

    for i in range(n_traj):
        pn = p_norm_lhs[i]   # (6,) normalised

        # Denormalise to physical units
        mass_wet   = PARAM_RANGES["mass_wet"][0]   + pn[0] * (PARAM_RANGES["mass_wet"][1]   - PARAM_RANGES["mass_wet"][0])
        mass_dry   = PARAM_RANGES["mass_dry"][0]   + pn[1] * (PARAM_RANGES["mass_dry"][1]   - PARAM_RANGES["mass_dry"][0])
        thrust     = PARAM_RANGES["thrust"][0]     + pn[2] * (PARAM_RANGES["thrust"][1]     - PARAM_RANGES["thrust"][0])
        burn_rate  = PARAM_RANGES["burn_rate"][0]  + pn[3] * (PARAM_RANGES["burn_rate"][1]  - PARAM_RANGES["burn_rate"][0])
        drag_coeff = PARAM_RANGES["drag_coeff"][0] + pn[4] * (PARAM_RANGES["drag_coeff"][1] - PARAM_RANGES["drag_coeff"][0])
        wind_vx    = PARAM_RANGES["wind_vx"][0]    + pn[5] * (PARAM_RANGES["wind_vx"][1]    - PARAM_RANGES["wind_vx"][0])

        # Ensure physical consistency: dry mass < wet mass
        mass_dry = min(mass_dry, mass_wet * 0.85)

        # exhaust_velocity not in our param set — use default
        rocket = RocketConfig(
            mass_wet=mass_wet,
            mass_dry=mass_dry,
            thrust=thrust,
            burn_rate=burn_rate,
            exhaust_velocity=DEFAULT_ROCKET.exhaust_velocity,
            drag_coeff=drag_coeff,
            cross_section=DEFAULT_ROCKET.cross_section,
        )

        # Constant wind: pass duck-typed wind object that returns wind_vx regardless
        # of altitude. This matches what the PINN force model assumes (constant wind_vx
        # from parameter vector), so training data and physics residuals are consistent.
        traj = run_rk4(rocket, DEFAULT_SIM, wind=_constant_wind(wind_vx))

        # Subsample to n_pts points
        idx = np.linspace(0, len(traj.time) - 1, n_pts, dtype=int)

        t_np = traj.time[idx].astype(np.float32)
        state_np = np.stack([traj.x[idx], traj.y[idx],
                             traj.vx[idx], traj.vy[idx]], axis=1).astype(np.float32)

        all_t.append(t_np)
        all_state.append(state_np)
        all_p.append(np.tile(pn, (n_pts, 1)).astype(np.float32))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_traj} done")

    t_cat     = np.concatenate(all_t,     axis=0)   # (n_traj*n_pts,)
    state_cat = np.concatenate(all_state, axis=0)   # (n_traj*n_pts, 4)
    p_cat     = np.concatenate(all_p,     axis=0)   # (n_traj*n_pts, 6)

    print(f"Training data: {len(t_cat):,} points ({n_traj} trajectories × {n_pts} pts)")

    return (
        torch.tensor(t_cat,     device=device).unsqueeze(-1),  # (N, 1)
        torch.tensor(state_cat, device=device),                  # (N, 4)
        torch.tensor(p_cat,     device=device),                  # (N, 6)
    )


def _constant_wind(vx: float):
    """
    Duck-typed wind model that always returns a constant vx m/s.
    Matches the trajectory.run() interface: wind.step(altitude, dt) -> float.
    Used to generate training trajectories that match the PINN force model
    (which uses constant wind_vx from the parameter vector).
    """
    class ConstantWind:
        def step(self, altitude: float, dt: float) -> float:
            return float(vx)

    return ConstantWind()


def _data_loss(model: RocketPINNParam, t_data: torch.Tensor,
               state_data: torch.Tensor, p_norm_data: torch.Tensor,
               scales: torch.Tensor) -> torch.Tensor:
    """MSE between PINN [x, y, vx, vy] and RK4 ground truth, normalised."""
    pred = model.predict_grad(t_data, p_norm_data)[:, :4]
    diff = (pred - state_data) / scales.to(pred.device)
    return (diff ** 2).mean()


def _physics_loss(model: RocketPINNParam, t_max: float,
                  n_colloc: int, device: torch.device,
                  r_scales: torch.Tensor,
                  rng_torch: torch.Generator) -> torch.Tensor:
    """
    MSE of 4 physics residuals at randomly sampled (t, p) collocation points.
    Parameters are sampled uniformly (not biased) since the parameter space
    is now jointly sampled and we want uniform coverage.
    30% of time points biased near burnout as in Option A.
    """
    n_uniform = int(0.7 * n_colloc)
    n_burnout = n_colloc - n_uniform

    t_unif = torch.rand(n_uniform, 1, device=device, generator=rng_torch) * t_max
    # Burnout time varies with parameters, centre bias at nominal ~25s
    t_near = 25.0 + (torch.rand(n_burnout, 1, device=device, generator=rng_torch) - 0.5) * 10.0
    t_near = t_near.clamp(0.01, t_max - 0.01)

    t_col = torch.cat([t_unif, t_near], dim=0)
    t_col.requires_grad_(True)

    # Sample random parameter configs for collocation
    p_col_norm = torch.rand(n_colloc, len(PARAM_KEYS), device=device,
                            generator=rng_torch)

    resid      = model.residuals(t_col, p_col_norm)          # (N_c, 4)
    resid_norm = resid / r_scales.to(device).unsqueeze(0)
    return (resid_norm ** 2).mean()


def train_pinn_param(  # noqa: C901
    n_traj: int = 200,
    n_pts: int = 300,
    epochs_data: int = 5_000,
    epochs_phys: int = 25_000,
    lambda_phys: float = 0.01,
    lambda_phys_max: float = 1.0,
    n_colloc: int = 2_000,
    lr: float = 8e-4,
    save_path: str = "models/pinn_param_v1.pt",
    plot_every: int = 2_000,
    seed: int = 42,
) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush each print immediately (needed when redirecting to log file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng_np = np.random.default_rng(seed)
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed)

    total_epochs = epochs_data + epochs_phys

    print(f"Device:              {device}")
    print(f"Training data:       {n_traj} trajectories × {n_pts} pts = {n_traj*n_pts:,} total")
    print(f"Phase A (data):      {epochs_data:,} epochs  (lambda=0)")
    print(f"Phase B (joint):     {epochs_phys:,} epochs  (lambda {lambda_phys} -> {lambda_phys_max})")
    print(f"Collocation:         {n_colloc} pts/batch  (30% near burnout=25s)")
    print()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    model = RocketPINNParam(hidden=256, layers=6).to(device)
    _R_SCALES_d     = _R_SCALES.to(device)
    _STATE_SCALES_d = _STATE_SCALES.to(device)

    traj_nominal = run_rk4(DEFAULT_ROCKET, DEFAULT_SIM)
    t_max        = float(traj_nominal.time[-1])

    t_data, state_data, p_norm_data = _build_training_data(
        n_traj, n_pts, device, rng_np
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=5e-6
    )

    loss_history = {"total": [], "data": [], "physics": [], "lambda": []}

    print(f"\n{'Epoch':>8} | {'Phase':>6} | {'L_total':>10} | "
          f"{'L_data':>10} | {'L_phys':>10} | {'lambda':>8}")
    print("-" * 72)

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if epoch <= epochs_data:
            lam     = 0.0
            phase   = "data"
            l_data  = _data_loss(model, t_data, state_data, p_norm_data, _STATE_SCALES_d)
            l_phys  = torch.tensor(0.0, device=device)
            l_total = l_data
        else:
            frac    = (epoch - epochs_data) / epochs_phys
            lam     = lambda_phys * (lambda_phys_max / lambda_phys) ** frac
            phase   = "phys"
            l_data  = _data_loss(model, t_data, state_data, p_norm_data, _STATE_SCALES_d)
            l_phys  = _physics_loss(model, t_max, n_colloc, device, _R_SCALES_d, rng_torch)
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
    torch.save({
        "model_state": model.state_dict(),
        "t_max": t_max,
        "hidden": 256,
        "layers": 6,
        "n_traj": n_traj,
        "n_pts": n_pts,
    }, save_path)
    print(f"Model saved -> {save_path}")
    _plot_loss(loss_history, "logs/pinn_param_loss_curve.png")
    print(f"Loss curve  -> logs/pinn_param_loss_curve.png")


def _plot_loss(history: dict, path: str) -> None:
    epochs = range(1, len(history["total"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(epochs, history["total"],   label="L_total",   lw=1.5)
    ax.semilogy(epochs, history["data"],    label="L_data",    lw=1.2, ls="--")
    ax.semilogy(epochs, history["physics"], label="L_physics", lw=1.2, ls=":")
    ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title="Parameterised PINN Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_traj",        type=int,   default=200)
    parser.add_argument("--n_pts",         type=int,   default=300)
    parser.add_argument("--epochs_data",   type=int,   default=5_000)
    parser.add_argument("--epochs_phys",   type=int,   default=25_000)
    parser.add_argument("--lambda_phys",   type=float, default=0.01)
    parser.add_argument("--lambda_phys_max", type=float, default=1.0)
    parser.add_argument("--n_colloc",      type=int,   default=2_000)
    parser.add_argument("--lr",            type=float, default=8e-4)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()
    train_pinn_param(
        n_traj=args.n_traj,
        n_pts=args.n_pts,
        epochs_data=args.epochs_data,
        epochs_phys=args.epochs_phys,
        lambda_phys=args.lambda_phys,
        lambda_phys_max=args.lambda_phys_max,
        n_colloc=args.n_colloc,
        lr=args.lr,
        seed=args.seed,
    )
