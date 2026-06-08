"""
Physics-Informed Neural Network for the nominal rocket trajectory.

Option A: maps t -> [x, y, vx, vy] for the single nominal trajectory
(DEFAULT_ROCKET, no wind, 85 deg launch angle).

KEY ARCHITECTURAL DECISION (BUG-016):
  Mass is NOT predicted by the network. It is computed analytically from time:
    mass(t) = mass_wet - burn_rate * t   for t <= burnout_time
    mass(t) = mass_dry                   for t >  burnout_time
  This enforces monotonicity as a hard structural constraint, eliminating the
  largest source of physics residual error in v1-v3. See notes/06_pinns.md.

The network is trained with two loss terms:
  L_data    — MSE against RK4 ground-truth [x, y, vx, vy] at data points
  L_physics — MSE of 4 kinematic residuals at collocation points via autograd

See notes/06_pinns.md for full derivation and real-number walkthrough.
"""

import math
import torch
import torch.nn as nn

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM

# ── Output normalisation scales: [x, y, vx, vy] ───────────────────────────────
_X_SCALE  = 20_000.0
_Y_SCALE  = 35_000.0
_VX_SCALE =    600.0
_VY_SCALE =    900.0

# ── Physical constants ─────────────────────────────────────────────────────────
_G       = 9.81
_RHO_SEA = 1.225
_H_SCALE = 8_500.0

# ── Precomputed burnout time (deterministic for nominal rocket) ────────────────
_BURNOUT_TIME = (DEFAULT_ROCKET.mass_wet - DEFAULT_ROCKET.mass_dry) / DEFAULT_ROCKET.burn_rate
_ANGLE_RAD    = math.radians(DEFAULT_SIM.launch_angle_deg)


def _analytical_mass(t: torch.Tensor) -> torch.Tensor:
    """
    Exact mass as a function of time — no network needed.
    Enforces monotonic decrease as a hard structural constraint.
    t shape: (N, 1) or (N,). Returns shape (N,).
    """
    t_flat = t.squeeze(-1)
    cfg    = DEFAULT_ROCKET
    fuel_remaining = (cfg.burn_rate * t_flat).clamp(max=cfg.mass_wet - cfg.mass_dry)
    return cfg.mass_wet - fuel_remaining


def _air_density(y: torch.Tensor) -> torch.Tensor:
    return _RHO_SEA * torch.exp(-y / _H_SCALE)


def _forces(vx: torch.Tensor, vy: torch.Tensor,
             y: torch.Tensor, mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Net force components / mass = acceleration [m/s²]."""
    cfg   = DEFAULT_ROCKET
    speed = torch.sqrt(vx**2 + vy**2).clamp(min=1e-6)
    rho   = _air_density(y)

    drag_mag = 0.5 * rho * speed**2 * cfg.drag_coeff * cfg.cross_section
    drag_x   = -drag_mag * (vx / speed)
    drag_y   = -drag_mag * (vy / speed)

    # Thrust is non-zero only while burning — use a soft switch via clamp
    # (avoids discontinuous gradient at burnout boundary)
    burning    = (mass > cfg.mass_dry + 0.01).float()
    thrust_x   = cfg.thrust * math.cos(_ANGLE_RAD) * burning
    thrust_y   = cfg.thrust * math.sin(_ANGLE_RAD) * burning

    ax = (thrust_x + drag_x) / mass
    ay = (thrust_y + drag_y) / mass - _G

    return ax, ay


class RocketPINN(nn.Module):
    """
    PINN for the nominal rocket trajectory.

    Input:  t_norm = t / t_max   in [0, 1]
    Output: [x, y, vx, vy]  (4 variables — mass is analytical, not predicted)

    Call .predict(t)   -> [x, y, vx, vy, mass] in physical units (N, 5)
    Call .residuals(t) -> [R0, R1, R2, R3]      physics residuals (N, 4)
    """

    def __init__(self, hidden: int = 128, layers: int = 5) -> None:
        super().__init__()
        self._t_max = DEFAULT_SIM.max_time

        # Tanh activations: smooth, infinitely differentiable.
        # Wider/deeper than v1-v3 because we no longer waste capacity on mass.
        net = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 4))   # 4 outputs: x, y, vx, vy
        self._net = nn.Sequential(*net)

        # Initialise output layer near zero so predictions start near origin —
        # reduces the distance the network must travel in Phase A.
        nn.init.zeros_(self._net[-1].bias)
        nn.init.xavier_uniform_(self._net[-1].weight, gain=0.1)

    def _decode(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass + decode to physical units. Returns (x, y, vx, vy, mass)."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        out  = self._net(t / self._t_max)   # (N, 4) in normalised units

        x  = out[:, 0] * _X_SCALE
        y  = out[:, 1] * _Y_SCALE
        vx = out[:, 2] * _VX_SCALE
        vy = out[:, 3] * _VY_SCALE

        mass = _analytical_mass(t)          # (N,) — exact, no gradient through mass

        return x, y, vx, vy, mass

    def predict(self, t: torch.Tensor) -> torch.Tensor:
        """
        Physical state [x, y, vx, vy, mass] at time(s) t (seconds).
        Returns shape (N, 5). No grad required.
        """
        with torch.no_grad():
            x, y, vx, vy, mass = self._decode(t)
        return torch.stack([x, y, vx, vy, mass], dim=1)

    def predict_grad(self, t: torch.Tensor) -> torch.Tensor:
        """Same as predict but keeps gradient graph — used inside residuals()."""
        x, y, vx, vy, mass = self._decode(t)
        return torch.stack([x, y, vx, vy, mass], dim=1)

    def residuals(self, t: torch.Tensor) -> torch.Tensor:
        """
        Physics residuals at collocation points t (must have requires_grad=True).

        Returns shape (N, 4):
          R0 = dx/dt  - vx      [m/s]
          R1 = dy/dt  - vy      [m/s]
          R2 = dvx/dt - ax      [m/s²]
          R3 = dvy/dt - ay      [m/s²]

        Mass residual R4 is identically zero by construction — no need to compute it.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x, y, vx, vy, mass = self._decode(t)

        def _grad(scalar_field: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                scalar_field, t,
                grad_outputs=torch.ones_like(scalar_field),
                create_graph=True,
                retain_graph=True,
            )[0].squeeze(-1)

        dx_dt  = _grad(x)
        dy_dt  = _grad(y)
        dvx_dt = _grad(vx)
        dvy_dt = _grad(vy)

        ax, ay = _forces(vx, vy, y, mass)

        R0 = dx_dt  - vx
        R1 = dy_dt  - vy
        R2 = dvx_dt - ax
        R3 = dvy_dt - ay

        return torch.stack([R0, R1, R2, R3], dim=1)   # (N, 4)
