"""
Build LSTM training dataset from rollout trajectories.

Each sample:
  window: (N, 5) — last N steps of [y, vy, target_y, error, mass_frac]
  label:  scalar — |y(t+K) - target_y(t+K)| in metres

Split is by episode to prevent data leakage between train/val/test.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

WINDOW_LEN = 20    # N: steps of history (2 seconds at dt=0.1)
HORIZON    = 10    # K: steps ahead to forecast (1 second)
N_FEATURES = 5     # [y, vy, target_y, error, mass_frac]

# Normalisation scales for each feature
_FEAT_SCALE = np.array([35_000.0, 900.0, 35_000.0, 35_000.0, 1.0],
                        dtype=np.float32)


def _extract_windows(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (window, label) pairs from a single trajectory."""
    y        = traj["y"]
    vy       = traj["vy"]
    ty       = traj["target_y"]
    mf       = traj["mass_frac"]
    err      = np.abs(y - ty)
    T        = len(y)

    windows, labels = [], []
    for t in range(WINDOW_LEN, T - HORIZON):
        win = np.stack([
            y [t-WINDOW_LEN:t],
            vy[t-WINDOW_LEN:t],
            ty[t-WINDOW_LEN:t],
            err[t-WINDOW_LEN:t],
            mf[t-WINDOW_LEN:t],
        ], axis=-1)                         # (N, 5)
        win = win / _FEAT_SCALE             # normalise

        future_err = float(np.abs(y[t + HORIZON] - ty[t + HORIZON]))
        windows.append(win)
        labels.append(future_err)

    if not windows:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), np.float32), np.zeros(0, np.float32)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)


def build_dataset(trajectories: list[dict],
                  train_frac: float = 0.70,
                  val_frac:   float = 0.15,
                  max_samples: int  = 200_000,
                  rng_seed: int     = 42
                  ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from episode trajectories.

    Split is by episode index (not timestep) to avoid leakage.
    """
    rng = np.random.default_rng(rng_seed)
    n   = len(trajectories)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    def _collect(indices):
        ws, ls = [], []
        for i in indices:
            w, l = _extract_windows(trajectories[i])
            ws.append(w); ls.append(l)
        if ws:
            return np.concatenate(ws), np.concatenate(ls)
        return np.zeros((0, WINDOW_LEN, N_FEATURES), np.float32), np.zeros(0, np.float32)

    X_tr, y_tr = _collect(train_idx)
    X_va, y_va = _collect(val_idx)
    X_te, y_te = _collect(test_idx)

    # Subsample training set if too large
    if len(X_tr) > max_samples:
        sel = rng.choice(len(X_tr), max_samples, replace=False)
        X_tr, y_tr = X_tr[sel], y_tr[sel]

    print(f"Dataset split — train: {len(X_tr):,}  val: {len(X_va):,}  test: {len(X_te):,}")

    def _loader(X, y, shuffle):
        ds = _TensorDataset(X, y)
        return DataLoader(ds, batch_size=512, shuffle=shuffle, num_workers=0)

    return _loader(X_tr, y_tr, True), _loader(X_va, y_va, False), _loader(X_te, y_te, False)


class _TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
