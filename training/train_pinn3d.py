"""
Train the 3D parameterised PINN on rocket trajectory families.

Extends train_pinn_param.py to 3D:
  - 7D parameter space (added wind_vz)
  - 300 LHS training trajectories (more for 7D coverage)
  - 6-output network: [x, y, z, vx, vy, vz]

Usage:
    python -m training.train_pinn3d
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn as nn

from simulation.config import DEFAULT_ROCKET, RocketConfig
from simulation.physics3d import rk4_step3d
from simulation.pinn3d import (
    RocketPINN3D, PARAM_RANGES_3D, PARAM_KEYS_3D, N_PARAMS_3D, normalise_params_3d
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

T_MAX    = 185.0
DT_DATA  = 0.05   # 20 Hz — denser than 2D for 3D accuracy
N_TRAJ   = 300
N_PTS    = 300    # collocation points per trajectory
EPOCHS_DATA = 5_000
EPOCHS_PHYS = 25_000
LR       = 8e-4

_STATE_SCALES = torch.tensor([25_000.0, 35_000.0, 20_000.0,
                               700.0, 900.0, 700.0])  # x,y,z,vx,vy,vz
_R_SCALES     = torch.tensor([100.0, 900.0, 100.0,
                               15.0, 65.0, 15.0])     # residual scales


def _lhs_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling over 7D parameter space."""
    samples = np.zeros((n, N_PARAMS_3D))
    for d in range(N_PARAMS_3D):
        intervals = (np.arange(n) + rng.uniform(0, 1, n)) / n
        samples[:, d] = rng.permutation(intervals)
    return samples


def _sample_to_params(s: np.ndarray) -> dict:
    """Map [0,1]^7 LHS sample to physical parameter dict."""
    p = {}
    for i, key in enumerate(PARAM_KEYS_3D):
        lo, hi, _ = PARAM_RANGES_3D[key]
        p[key] = lo + s[i] * (hi - lo)
    return p


def _run_trajectory(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run RK4 simulation for given params. Returns (times, states7)."""
    rocket = RocketConfig(
        mass_wet=params["mass_wet"], mass_dry=params["mass_dry"],
        thrust=params["thrust"], burn_rate=params["burn_rate"],
        exhaust_velocity=2500.0,
        drag_coeff=params["drag_coeff"], cross_section=0.05,
    )
    wind_vx = params["wind_vx"]
    wind_vz = params["wind_vz"]
    pitch = np.radians(85.0)

    state = np.array([0.0]*6 + [rocket.mass_wet])
    t = 0.0
    times, states = [t], [state.copy()]

    while t < T_MAX and state[1] >= -100.0:
        state = rk4_step3d(state, pitch, 0.0, DT_DATA, rocket, wind_vx, wind_vz)
        t += DT_DATA
        times.append(t)
        states.append(state.copy())

    return np.array(times, dtype=np.float32), np.array(states, dtype=np.float32)


def _build_dataset(n_traj: int, rng: np.random.Generator):
    """Generate training dataset: list of (t_tensor, state_tensor, p_norm_tensor)."""
    lhs = _lhs_sample(n_traj, rng)
    dataset = []
    for i in range(n_traj):
        params = _sample_to_params(lhs[i])
        times, states = _run_trajectory(params)
        p_raw = np.array([params[k] for k in PARAM_KEYS_3D], dtype=np.float32)
        p_norm = normalise_params_3d(p_raw)
        dataset.append({
            "t":     torch.tensor(times).unsqueeze(-1),
            "state": torch.tensor(states[:, :6]),  # drop mass col
            "p_raw": torch.tensor(p_raw),
            "p_norm": torch.tensor(p_norm),
        })
        if (i + 1) % 50 == 0:
            print(f"  Generated trajectory {i+1}/{n_traj}")
    return dataset


def _data_loss(model, dataset, device, n_sample: int = 50) -> torch.Tensor:
    indices = np.random.choice(len(dataset), min(n_sample, len(dataset)), replace=False)
    loss = torch.tensor(0.0, device=device)
    sc = _STATE_SCALES.to(device)
    for idx in indices:
        d = dataset[idx]
        t = d["t"].to(device)
        p_norm = d["p_norm"].unsqueeze(0).expand(len(t), -1).to(device)
        pred6 = model(t / model.t_max, p_norm)
        true6 = d["state"].to(device)
        err = (pred6 - true6) / sc
        loss = loss + (err ** 2).mean()
    return loss / len(indices)


def _physics_loss(model, dataset, device, n_traj: int = 20,
                  n_pts: int = N_PTS) -> torch.Tensor:
    indices = np.random.choice(len(dataset), min(n_traj, len(dataset)), replace=False)
    loss = torch.tensor(0.0, device=device)
    sc = _R_SCALES.to(device)
    for idx in indices:
        d = dataset[idx]
        t_all = d["t"]
        sel = torch.randint(0, len(t_all), (n_pts,))
        t_c = t_all[sel].clone().detach().to(device).requires_grad_(True)
        p_norm = d["p_norm"].unsqueeze(0).expand(n_pts, -1).to(device)
        R = model.residuals(t_c, p_norm)
        loss = loss + ((R / sc) ** 2).mean()
    return loss / len(indices)


def train_pinn3d():
    print("=== 3D Parameterised PINN Training ===")
    print(f"Trajectories: {N_TRAJ}  |  Collocation pts: {N_PTS}  |  "
          f"Epochs: {EPOCHS_DATA}+{EPOCHS_PHYS}")
    print()

    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RocketPINN3D(hidden=256, layers=6).to(device)
    # Set normalisation buffers
    for i, key in enumerate(PARAM_KEYS_3D):
        lo, hi, _ = PARAM_RANGES_3D[key]
        model.p_lo[i] = lo; model.p_hi[i] = hi
    model.t_max = torch.tensor(T_MAX)

    print("Generating training trajectories...")
    dataset = _build_dataset(N_TRAJ, rng)
    print(f"Dataset ready: {len(dataset)} trajectories\n")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_DATA+EPOCHS_PHYS)

    # Phase A: data only
    print("Phase A: data loss only")
    lam = 0.0
    for epoch in range(1, EPOCHS_DATA + 1):
        opt.zero_grad()
        L_data = _data_loss(model, dataset, device)
        L_data.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:5d}/{EPOCHS_DATA} | L_data={L_data.item():.6f}")

    # Phase B: data + physics
    print("\nPhase B: data + physics (warm lambda schedule)")
    for epoch in range(1, EPOCHS_PHYS + 1):
        lam = min(1.0, epoch / (EPOCHS_PHYS * 0.3)) * 0.01
        opt.zero_grad()
        L_data = _data_loss(model, dataset, device)
        L_phys = _physics_loss(model, dataset, device)
        loss = L_data + lam * L_phys
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if epoch % 2500 == 0:
            print(f"  Epoch {epoch:5d}/{EPOCHS_PHYS} | "
                  f"L_data={L_data.item():.6f} | L_phys={L_phys.item():.4f} | lam={lam:.4f}")

    # Save
    ckpt = {
        "model_state": model.state_dict(),
        "hidden": 256, "layers": 6,
        "t_max": T_MAX,
        "param_ranges": PARAM_RANGES_3D,
    }
    path = "models/pinn3d_param_v1.pt"
    torch.save(ckpt, path)
    print(f"\nModel saved -> {path}")


if __name__ == "__main__":
    train_pinn3d()
