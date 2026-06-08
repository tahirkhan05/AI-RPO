"""
Train the LSTM deviation forecaster.

Pipeline:
  1. Collect rollouts from v4 + v5 + v6ext agents
  2. Build windowed dataset
  3. Train LSTM with MAE loss + early stopping
  4. Save to models/lstm_forecaster_v1.pt

Usage:
    python -m training.train_lstm
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.env import RocketEnv
from simulation.env3d import RocketEnv3D
from simulation.rollout import collect_rollouts
from simulation.lstm_dataset import build_dataset
from simulation.lstm_forecaster import LSTMForecaster

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

EPOCHS        = 50
PATIENCE      = 7       # early stopping patience
LR            = 1e-3
N_EP_PER_MODEL = 150    # episodes per agent (150 × 3 agents = 450 total)


def _make_2d_env(seed=0):
    return RocketEnv(randomize=True, seed=seed, physics_guided=True)

def _make_3d_env(seed=0):
    return RocketEnv3D(randomize=True, seed=seed,
                       physics_guided=True, use_ekf=True)


def _collect_all():
    """Collect rollouts from available trained agents."""
    all_trajs = []

    agents = [
        # (model_path, vecnorm_path, env_factory, seed_offset, label)
        ("models/ppo_v4_final", "models/vecnorm_v4_final.pkl",
         _make_2d_env, 0,   "v4 2D"),
        ("models/ppo_v5_final", "models/vecnorm_v5_final.pkl",
         _make_2d_env, 200, "v5 2D+EKF"),
    ]

    # Add v6ext if trained
    if os.path.exists("models/ppo_v6ext_final.zip"):
        agents.append(("models/ppo_v6ext_final", "models/vecnorm_v6ext_final.pkl",
                        _make_3d_env, 400, "v6ext 3D"))
    elif os.path.exists("models/ppo_v6_final.zip"):
        agents.append(("models/ppo_v6_final", "models/vecnorm_v6_final.pkl",
                        _make_3d_env, 400, "v6 3D baseline"))

    for model_path, vecnorm_path, env_fn, offset, label in agents:
        if not os.path.exists(model_path + ".zip"):
            print(f"  Skipping {label} — model not found")
            continue
        print(f"  Collecting from {label}...")
        trajs = collect_rollouts(
            model_path, vecnorm_path, env_fn,
            n_episodes=N_EP_PER_MODEL, seed_offset=offset
        )
        all_trajs.extend(trajs)
        print(f"  {label}: {len(trajs)} episodes collected")

    return all_trajs


def train():
    print("=== LSTM Deviation Forecaster Training ===")
    print(f"Episodes per agent: {N_EP_PER_MODEL}")
    print(f"Epochs: {EPOCHS}  |  Patience: {PATIENCE}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Collecting rollout trajectories...")
    trajectories = _collect_all()
    print(f"Total episodes collected: {len(trajectories)}\n")

    if len(trajectories) == 0:
        print("ERROR: no trajectories collected — check model paths.")
        return

    print("Building dataset...")
    train_loader, val_loader, test_loader = build_dataset(trajectories)
    print()

    model = LSTMForecaster(hidden_size=128, num_layers=2, dropout=0.2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.L1Loss()   # MAE in normalised units (labels /1000 during training)
    LABEL_NORM = 1000.0     # scale labels to ~0-5 range for stable gradients

    best_val_mae = float("inf")
    patience_cnt = 0
    train_maes, val_maes = [], []

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        ep_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)                          # (batch, 1)
            loss = loss_fn(pred, y / LABEL_NORM)     # normalised MAE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        train_mae = ep_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y / LABEL_NORM).item()
        val_mae = val_loss / len(val_loader)

        train_maes.append(train_mae)
        val_maes.append(val_mae)
        sched.step()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train MAE={train_mae:.1f}m | val MAE={val_mae:.1f}m")

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            torch.save({
                "model_state": model.state_dict(),
                "hidden_size": 128, "num_layers": 2,
                "val_mae": val_mae,
            }, "models/lstm_forecaster_v1.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val MAE: {best_val_mae*LABEL_NORM:.1f}m")
    print("Model saved -> models/lstm_forecaster_v1.pt")

    # Test evaluation
    ckpt = torch.load("models/lstm_forecaster_v1.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y / LABEL_NORM).item()
    test_mae = test_loss / len(test_loader)
    print(f"Test MAE:      {test_mae*LABEL_NORM:.1f}m")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_maes, label="Train MAE")
    plt.plot(val_maes,   label="Val MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE (m)")
    plt.title("LSTM Deviation Forecaster — Training Curve")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/lstm_training_curve.png", dpi=120)
    plt.close()
    print("Loss curve saved -> logs/lstm_training_curve.png")

    print("\n[PAPER OPPORTUNITY: LSTM deviation forecasting]")
    print(f"  Forecast horizon: {model.horizon} steps (1 second)")
    print(f"  Test MAE: {test_mae:.1f}m on held-out episodes")
    print("  Claim: LSTM predicts trajectory deviation 1s ahead with <Xm error")


if __name__ == "__main__":
    train()
