"""
LSTM Deviation Forecaster.

Input:  (batch, N=20, features=5) — sliding window of trajectory state
Output: (batch, 1) — predicted |y(t+K) - target_y(t+K)| in metres

Architecture:
  LSTM (2 layers, hidden=128) → Dropout(0.2) → FC(128→64) → FC(64→1, ReLU)

The ReLU output ensures predictions are non-negative (deviation >= 0).
"""

import torch
import torch.nn as nn

from simulation.lstm_dataset import WINDOW_LEN, HORIZON, N_FEATURES, _FEAT_SCALE

_LABEL_SCALE = 35_000.0   # metres — same as altitude scale


class LSTMForecaster(nn.Module):
    """
    LSTM-based trajectory deviation forecaster.
    Predicts how far off-track the rocket will be K steps ahead.
    """

    def __init__(self, input_size: int = N_FEATURES,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 dropout:     float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),   # smooth non-negative output, avoids dying ReLU
        )
        # Store metadata for inference
        self.register_buffer("label_scale", torch.tensor(_LABEL_SCALE))
        self.register_buffer("feat_scale",  torch.tensor(_FEAT_SCALE))
        self.window_len = WINDOW_LEN
        self.horizon    = HORIZON

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features) — normalised window
        returns: (batch, 1) — normalised predicted deviation
        """
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        last   = out[:, -1, :]         # (batch, hidden) — last timestep
        last   = self.dropout(last)
        return self.fc(last)           # (batch, 1)

    @torch.no_grad()
    def forecast(self, window_raw: torch.Tensor) -> float:
        """
        Single-window inference.
        window_raw: (N, 5) — raw (unnormalised) feature values
        Returns predicted deviation in metres.
        """
        x = (window_raw / self.feat_scale).unsqueeze(0)   # (1, N, 5)
        pred = self.forward(x)                              # (1, 1) normalised
        return float(pred.item()) * 1000.0                 # back to metres
