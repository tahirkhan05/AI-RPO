"""
3D Parameterised Physics-Informed Neural Network.

Predicts 3D rocket trajectories as a function of time and rocket/wind parameters.

Input:  (t, p) where p = [mass_wet, mass_dry, thrust, burn_rate, drag_coeff,
                           wind_vx, wind_vz]   (7 parameters)
Output: [x, y, z, vx, vy, vz]                  (6 states — mass analytical)

Same architecture as pinn_param.py but extended to 3D:
  - Input dim: 1 + 7 = 8
  - Output dim: 6 (added z, vz)
  - Hidden: 256 x 6 layers, tanh
  - Analytical mass (same formula)
  - Soft burnout switch (same sigmoid)

Physics residuals (6 ODEs):
  R0: dx/dt - vx   = 0
  R1: dy/dt - vy   = 0
  R2: dz/dt - vz   = 0
  R3: dvx/dt - ax  = 0
  R4: dvy/dt - ay  = 0
  R5: dvz/dt - az  = 0

See notes/09_phase8_3d.md Section 5 for design rationale.
"""

import math
import torch
import torch.nn as nn
import numpy as np

# ── Parameter ranges (added wind_vz vs 2D) ──────────────────────────────────
PARAM_RANGES_3D = {
    "mass_wet":  (80.0,   120.0,  100.0),
    "mass_dry":  (40.0,    60.0,   50.0),
    "thrust":    (4000.0, 6000.0, 5000.0),
    "burn_rate": (1.6,     2.4,    2.0),
    "drag_coeff":(0.32,    0.48,   0.4),
    "wind_vx":   (-15.0,  15.0,   0.0),
    "wind_vz":   (-10.0,  10.0,   0.0),
}
PARAM_KEYS_3D = ["mass_wet", "mass_dry", "thrust", "burn_rate",
                 "drag_coeff", "wind_vx", "wind_vz"]
N_PARAMS_3D = 7

_G       = 9.81
_RHO_SEA = 1.225
_H_SCALE = 8500.0
_BURNOUT_BETA = 50.0


def normalise_params_3d(p_raw: np.ndarray) -> np.ndarray:
    """Map physical param values → [0, 1] using training ranges."""
    out = np.zeros_like(p_raw)
    for i, key in enumerate(PARAM_KEYS_3D):
        lo, hi, _ = PARAM_RANGES_3D[key]
        out[i] = (p_raw[i] - lo) / (hi - lo)
    return out


def _soft_burning(t: torch.Tensor, t_burnout: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(_BURNOUT_BETA * (t_burnout - t.squeeze(-1)))


def _analytical_mass_3d(t: torch.Tensor, mass_wet: torch.Tensor,
                         mass_dry: torch.Tensor, burn_rate: torch.Tensor) -> torch.Tensor:
    t_flat = t.squeeze(-1)
    fuel_cap = mass_wet - mass_dry
    fuel_used = (burn_rate * t_flat).clamp(max=fuel_cap)
    return mass_wet - fuel_used


class RocketPINN3D(nn.Module):
    """
    3D Parameterised PINN: (t, p_7) → [x, y, z, vx, vy, vz].
    Mass computed analytically.
    """

    def __init__(self, hidden: int = 256, layers: int = 6) -> None:
        super().__init__()
        in_dim = 1 + N_PARAMS_3D   # 8
        net = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 6))   # [x, y, z, vx, vy, vz]
        self.net = nn.Sequential(*net)

        # Normalisation stats (set after training)
        self.register_buffer("t_max",    torch.tensor(185.0))
        self.register_buffer("p_lo",  torch.zeros(N_PARAMS_3D))
        self.register_buffer("p_hi",  torch.ones(N_PARAMS_3D))

    def _normalise_t(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.t_max

    def _normalise_p(self, p_raw: torch.Tensor) -> torch.Tensor:
        return (p_raw - self.p_lo) / (self.p_hi - self.p_lo + 1e-8)

    def forward(self, t_norm: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([t_norm, p_norm], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def predict(self, t: torch.Tensor, p_raw: torch.Tensor) -> torch.Tensor:
        """
        Physical-unit forward pass.
        t:     (N, 1) seconds
        p_raw: (N, 7) physical parameter values

        Returns (N, 7): [x, y, z, vx, vy, vz, mass]
        """
        t_norm = self._normalise_t(t)
        p_norm = self._normalise_p(p_raw)
        pred6 = self.forward(t_norm, p_norm)   # (N, 6)

        # Analytical mass
        mw = p_raw[:, 0]; md = p_raw[:, 1]; br = p_raw[:, 3]
        t_burn = (mw - md) / br
        mass = _analytical_mass_3d(t, mw, md, br)

        return torch.cat([pred6, mass.unsqueeze(-1)], dim=-1)  # (N, 7)

    def residuals(self, t: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute 6 physics residuals at collocation points.
        Returns (N, 6): [R0..R5]
        """
        t.requires_grad_(True)
        pred6 = self.forward(t / self.t_max, p_norm)

        # Unpack normalised params back to physical
        p_raw = p_norm * (self.p_hi - self.p_lo) + self.p_lo
        mw = p_raw[:, 0]; md = p_raw[:, 1]
        thrust = p_raw[:, 2]; br = p_raw[:, 3]
        Cd = p_raw[:, 4]
        wind_vx = p_raw[:, 5]; wind_vz = p_raw[:, 6]

        t_burn = (mw - md) / br
        burning = _soft_burning(t, t_burn)
        mass = _analytical_mass_3d(t, mw, md, br)

        x  = pred6[:, 0]; y  = pred6[:, 1]; z  = pred6[:, 2]
        vx = pred6[:, 3]; vy = pred6[:, 4]; vz = pred6[:, 5]

        # Time derivatives via autograd
        ones = torch.ones_like(x)
        dx_dt,  = torch.autograd.grad(x,  t, grad_outputs=ones, create_graph=True)
        dy_dt,  = torch.autograd.grad(y,  t, grad_outputs=ones, create_graph=True)
        dz_dt,  = torch.autograd.grad(z,  t, grad_outputs=ones, create_graph=True)
        dvx_dt, = torch.autograd.grad(vx, t, grad_outputs=ones, create_graph=True)
        dvy_dt, = torch.autograd.grad(vy, t, grad_outputs=ones, create_graph=True)
        dvz_dt, = torch.autograd.grad(vz, t, grad_outputs=ones, create_graph=True)

        # Physics: assume vertical launch (pitch≈90°, yaw=0) for PINN reference
        # The PINN learns the uncontrolled gravity-turn trajectory
        T_eff = thrust * burning
        rho = _RHO_SEA * torch.exp(-y.clamp(min=0) / _H_SCALE)

        rel_vx = vx - wind_vx
        rel_vz = vz - wind_vz
        airspeed = torch.sqrt(rel_vx**2 + vy**2 + rel_vz**2 + 1e-8)
        drag_mag = 0.5 * rho * airspeed**2 * Cd * 0.05  # cross_section=0.05 fixed

        # Near-vertical launch: cos(pitch)≈sin(launch), pitch from vertical
        # Use a fixed gravity-turn profile: thrust mostly vertical
        ax_phys = (- drag_mag * rel_vx / airspeed) / mass
        ay_phys = (T_eff     - drag_mag * vy      / airspeed) / mass - _G
        az_phys = (- drag_mag * rel_vz / airspeed) / mass

        R0 = dx_dt.squeeze(-1)  - vx
        R1 = dy_dt.squeeze(-1)  - vy
        R2 = dz_dt.squeeze(-1)  - vz
        R3 = dvx_dt.squeeze(-1) - ax_phys
        R4 = dvy_dt.squeeze(-1) - ay_phys
        R5 = dvz_dt.squeeze(-1) - az_phys

        return torch.stack([R0, R1, R2, R3, R4, R5], dim=-1)
