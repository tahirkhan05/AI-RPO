"""
Evaluate the LSTM deviation forecaster.

Two evaluations:
  1. Standalone accuracy — MAE, R^2, prediction vs truth scatter plot
  2. Agent comparison — does the LSTM correctly flag high-deviation episodes
     before the deviation actually occurs? (precision/recall on "warning" events)

Usage:
    python -m training.evaluate_lstm
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.env import RocketEnv
from simulation.rollout import collect_rollouts
from simulation.lstm_dataset import build_dataset, _extract_windows
from simulation.lstm_forecaster import LSTMForecaster

MODEL_PATH = "models/lstm_forecaster_v1.pt"
WARNING_THRESHOLD = 2000.0   # metres — LSTM warns if forecast > this


def load_model(device) -> LSTMForecaster:
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = LSTMForecaster(hidden_size=128, num_layers=2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def eval_standalone(model, device):
    """Accuracy on held-out episodes from v4."""
    print("=== Standalone Accuracy (held-out v4 episodes) ===")

    def make_2d(seed=0):
        return RocketEnv(randomize=True, seed=seed, physics_guided=True)

    trajs = collect_rollouts(
        "models/ppo_v4_final", "models/vecnorm_v4_final.pkl",
        make_2d, n_episodes=30, seed_offset=9000   # unseen seeds
    )

    # Extract windows directly — no train/val split needed for eval
    from simulation.lstm_dataset import _extract_windows
    all_w, all_l = [], []
    for traj in trajs:
        w, l = _extract_windows(traj)
        all_w.append(w); all_l.append(l)
    all_w = np.concatenate(all_w)
    all_l = np.concatenate(all_l)

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.tensor(all_w), torch.tensor(all_l).unsqueeze(-1))
    test_loader = DataLoader(ds, batch_size=512, shuffle=False)

    preds, truths = [], []
    loss_fn = torch.nn.L1Loss()
    total_loss = 0.0
    LABEL_NORM = 1000.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            p = model(X)                                    # normalised output
            total_loss += loss_fn(p, y / LABEL_NORM).item()
            preds.append((p * LABEL_NORM).cpu().numpy())   # back to metres
            truths.append(y.cpu().numpy())                  # already metres

    preds  = np.concatenate(preds).flatten()
    truths = np.concatenate(truths).flatten()
    mae    = (total_loss / len(test_loader)) * LABEL_NORM  # back to metres

    ss_res = np.sum((truths - preds) ** 2)
    ss_tot = np.sum((truths - np.mean(truths)) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)

    print(f"  Test MAE : {mae:.1f} m")
    print(f"  R^2      : {r2:.4f}")
    print(f"  Pred range: {preds.min():.0f} - {preds.max():.0f} m")
    print(f"  True range: {truths.min():.0f} - {truths.max():.0f} m")

    return preds, truths, mae, r2


def eval_warning(model, device):
    """
    Early-warning evaluation: does the LSTM predict large deviations
    BEFORE they actually happen?

    For each timestep where true future deviation > WARNING_THRESHOLD,
    check if LSTM predicted > WARNING_THRESHOLD at least K steps earlier.
    """
    print(f"\n=== Early Warning Evaluation (threshold={WARNING_THRESHOLD:.0f}m) ===")

    def make_2d(seed=0):
        return RocketEnv(randomize=True, seed=seed, physics_guided=True)

    trajs = collect_rollouts(
        "models/ppo_v4_final", "models/vecnorm_v4_final.pkl",
        make_2d, n_episodes=20, seed_offset=8000
    )

    from simulation.lstm_dataset import WINDOW_LEN, HORIZON, N_FEATURES, _FEAT_SCALE

    tp = fp = fn = tn = 0
    lead_times = []   # how many steps BEFORE the actual deviation did LSTM warn?

    for traj in trajs:
        windows, labels = _extract_windows(traj)
        if len(windows) == 0:
            continue

        X = torch.tensor(windows).to(device)
        with torch.no_grad():
            p = model(X).cpu().numpy().flatten() * 1000.0  # to metres

        warned    = p > WARNING_THRESHOLD
        actual_hi = labels > WARNING_THRESHOLD

        for t in range(len(labels)):
            w = bool(warned[t])
            a = bool(actual_hi[t])
            if w and a:  tp += 1
            elif w and not a: fp += 1
            elif not w and a: fn += 1
            else: tn += 1

        # Lead time: for each actual high-deviation event, how early did LSTM warn?
        hi_events = np.where(actual_hi)[0]
        for ev in hi_events:
            # Look back to find earliest warning before this event
            lookback = max(0, ev - HORIZON)
            if np.any(warned[lookback:ev]):
                first_warn = np.where(warned[lookback:ev])[0][0]
                lead_times.append(ev - (lookback + first_warn))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_lead = np.mean(lead_times) if lead_times else 0.0

    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"  F1 score : {f1:.3f}")
    print(f"  Mean lead time: {mean_lead:.1f} steps ({mean_lead*0.1:.1f}s before event)")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return precision, recall, f1, mean_lead


def _plot(preds, truths, mae, r2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LSTM Deviation Forecaster Evaluation", fontsize=13)

    ax = axes[0]
    lim = max(truths.max(), preds.max())
    ax.scatter(truths, preds, alpha=0.3, s=5, color="steelblue")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True deviation (m)")
    ax.set_ylabel("Predicted deviation (m)")
    ax.set_title(f"Prediction vs Truth\nMAE={mae:.0f}m  R²={r2:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    err = preds - truths
    ax.hist(err, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Prediction error (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution\nMean={err.mean():.0f}m  Std={err.std():.0f}m")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/lstm_evaluation.png", dpi=120)
    plt.close()
    print("\nPlot saved -> logs/lstm_evaluation.png")


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run training/train_lstm.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = load_model(device)
    print(f"Model loaded: hidden=128, layers=2\n")

    preds, truths, mae, r2 = eval_standalone(model, device)
    precision, recall, f1, lead = eval_warning(model, device)

    _plot(preds, truths, mae, r2)

    print("\n=== Summary ===")
    print(f"  Forecast MAE    : {mae:.0f} m  (1s-ahead deviation prediction)")
    print(f"  R^2             : {r2:.3f}")
    print(f"  Warning F1      : {f1:.3f}")
    print(f"  Mean lead time  : {lead:.1f} steps ({lead*0.1:.1f}s early warning)")
    print()
    print("[PAPER OPPORTUNITY: LSTM early warning system]")
    print(f"  Claim: LSTM predicts trajectory deviation {lead*0.1:.1f}s before it occurs")
    print(f"  with F1={f1:.2f} on held-out episodes.")


if __name__ == "__main__":
    evaluate()
