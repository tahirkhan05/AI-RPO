"""
Parameterised Physics-Informed Neural Network for rocket trajectories.

Option B: maps (t, p) -> [x, y, vx, vy] where p is a 6-dim parameter vector
describing the rocket and wind configuration.

This PINN generalises across a *family* of trajectories, acting as a fast
physics-consistent surrogate model that replaces full RK4 simulation for
configurations it has seen during training.

Parameter vector p (all normalised to [0, 1]):
  p[0] = mass_wet          (80  – 120 kg)
  p[1] = mass_dry          (40  – 60  kg)
  p[2] = thrust            (4000 – 6000 N)
  p[3] = burn_rate         (1.6 – 2.4 kg/s)
  p[4] = drag_coeff        (0.32 – 0.48)
  p[5] = wind_vx           (-15 – +15 m/s)

Mass is still computed analytically — no network output for it.
Burnout is handled with a soft sigmoid switch to avoid discontinuity spikes
in the autograd derivative (see notes/06_pinns.md Section 17).

See notes/06_pinns.md for full derivation, architecture rationale, and paper angles.
"""

import math
import torch
import torch.nn as nn

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM

# ── Parameter ranges for normalisation ────────────────────────────────────────
# Each row: [min, max, nominal]  (nominal = DEFAULT_ROCKET value)
PARAM_RANGES = {
    "mass_wet":  (80.0,   120.0,  100.0),   # kg
    "mass_dry":  (40.0,    60.0,   50.0),   # kg
    "thrust":    (4000.0, 6000.0, 5000.0),  # N
    "burn_rate": (1.6,     2.4,    2.0),    # kg/s
    "drag_coeff":(0.32,    0.48,   0.4),    # dimensionless
    "wind_vx":   (-15.0,  15.0,   0.0),     # m/s
}

# Ordered as the network expects
PARAM_KEYS = ["mass_wet", "mass_dry", "thrust", "burn_rate", "drag_coeff", "wind_vx"]
N_PARAMS    = len(PARAM_KEYS)  # 6

# ── Output normalisation scales: [x, y, vx, vy] ───────────────────────────────
_X_SCALE  = 25_000.0   # slightly wider than Option A to accommodate wind drift
_Y_SCALE  = 35_000.0
_VX_SCALE =    700.0   # larger to handle wind-induced vx
_VY_SCALE =    900.0

# ── Physical constants ─────────────────────────────────────────────────────────
_G       = 9.81
_RHO_SEA = 1.225
_H_SCALE = 8_500.0

# ── Soft burnout switch ────────────────────────────────────────────────────────
# σ(β * (t_burnout - t)) — smooth approximation to the Heaviside step.
# β=50 means 90% → 10% transition happens over ~0.09 seconds — physically
# accurate enough while keeping the autograd derivative finite everywhere.
_BURNOUT_BETA = 50.0


def normalise_params(p_raw: torch.Tensor) -> torch.Tensor:
    """
    Convert raw physical parameter values to [0, 1] normalised range.
    p_raw: (..., 6) in physical units.
    Returns: (..., 6) in [0, 1].
    """
    lo = torch.tensor([PARAM_RANGES[k][0] for k in PARAM_KEYS],
                      dtype=torch.float32, device=p_raw.device)
    hi = torch.tensor([PARAM_RANGES[k][1] for k in PARAM_KEYS],
                      dtype=torch.float32, device=p_raw.device)
    return (p_raw - lo) / (hi - lo + 1e-8)


def denormalise_params(p_norm: torch.Tensor) -> torch.Tensor:
    """Inverse of normalise_params."""
    lo = torch.tensor([PARAM_RANGES[k][0] for k in PARAM_KEYS],
                      dtype=torch.float32, device=p_norm.device)
    hi = torch.tensor([PARAM_RANGES[k][1] for k in PARAM_KEYS],
                      dtype=torch.float32, device=p_norm.device)
    return p_norm * (hi - lo + 1e-8) + lo


def _analytical_mass_param(t: torch.Tensor, mass_wet: torch.Tensor,
                            mass_dry: torch.Tensor,
                            burn_rate: torch.Tensor) -> torch.Tensor:
    """
    Exact mass as a function of time and rocket parameters.
    Returns shape (N,) matching t.squeeze(-1).
    Unlike Option A, mass_wet/mass_dry/burn_rate can vary per trajectory.
    """
    t_flat   = t.squeeze(-1)           # (N,)
    fuel_cap = mass_wet - mass_dry
    fuel_used = (burn_rate * t_flat).clamp(max=fuel_cap)
    return mass_wet - fuel_used        # (N,)


def _soft_burning(t: torch.Tensor, t_burnout: torch.Tensor) -> torch.Tensor:
    """
    Smooth approximation to 'is the engine burning?'
    Returns values in (0, 1): ~1 before burnout, ~0 after.

    Using sigmoid: σ(β * (t_burnout - t))
    At t = t_burnout: output = σ(0) = 0.5
    At t = t_burnout - 0.1s: output ≈ σ(5) ≈ 0.993
    At t = t_burnout + 0.1s: output ≈ σ(-5) ≈ 0.007

    This avoids the infinite derivative spike that a hard step creates in autograd.
    """
    return torch.sigmoid(_BURNOUT_BETA * (t_burnout - t.squeeze(-1)))


def _air_density(y: torch.Tensor) -> torch.Tensor:
    return _RHO_SEA * torch.exp(-y / _H_SCALE)


def _forces_param(vx: torch.Tensor, vy: torch.Tensor, y: torch.Tensor,
                  mass: torch.Tensor, t: torch.Tensor,
                  thrust: torch.Tensor, burn_rate: torch.Tensor,
                  mass_wet: torch.Tensor, mass_dry: torch.Tensor,
                  drag_coeff: torch.Tensor, cross_section: torch.Tensor,
                  wind_vx: torch.Tensor,
                  angle_rad: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Net acceleration [m/s²] including thrust (soft-switch), drag, gravity, wind.
    All inputs are (N,) tensors.
    """
    t_burnout = (mass_wet - mass_dry) / burn_rate   # (N,) varying per config

    speed = torch.sqrt(vx**2 + vy**2).clamp(min=1e-6)
    rho   = _air_density(y)

    # Wind-relative velocity for drag computation
    vx_rel = vx - wind_vx
    speed_rel = torch.sqrt(vx_rel**2 + vy**2).clamp(min=1e-6)

    drag_mag = 0.5 * rho * speed_rel**2 * drag_coeff * cross_section
    drag_x   = -drag_mag * (vx_rel / speed_rel)
    drag_y   = -drag_mag * (vy    / speed_rel)

    # Soft burning switch — avoids discontinuity in autograd at burnout
    burning  = _soft_burning(t, t_burnout)           # (N,) in (0, 1)
    thrust_x = thrust * math.cos(angle_rad) * burning
    thrust_y = thrust * math.sin(angle_rad) * burning

    ax = (thrust_x + drag_x) / mass
    ay = (thrust_y + drag_y) / mass - _G

    return ax, ay


class RocketPINNParam(nn.Module):
    """
    Parameterised PINN for the rocket trajectory family.

    Input:  [t_norm, p_norm]  (7-dimensional: 1 time + 6 parameters)
    Output: [x, y, vx, vy]  (4 variables — mass still analytical)

    All inputs normalised before passing to network:
      t_norm = t / t_max ∈ [0, 1]
      p_norm = normalise_params(p_raw) ∈ [0, 1]^6

    Call .predict(t, p_raw)   -> [x, y, vx, vy, mass] (N, 5)
    Call .residuals(t, p_raw) -> [R0, R1, R2, R3]     (N, 4)
    """

    def __init__(self, hidden: int = 256, layers: int = 6) -> None:
        super().__init__()
        self._t_max = DEFAULT_SIM.max_time
        self._angle_rad = math.radians(DEFAULT_SIM.launch_angle_deg)
        self._cross_section = DEFAULT_ROCKET.cross_section  # fixed (not in param set)

        in_dim = 1 + N_PARAMS   # 7: t + 6 params

        net = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 4))   # [x, y, vx, vy]
        self._net = nn.Sequential(*net)

        nn.init.zeros_(self._net[-1].bias)
        nn.init.xavier_uniform_(self._net[-1].weight, gain=0.1)

    def _decode(self, t: torch.Tensor,
                p_norm: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass.
        t:      (N, 1) in seconds
        p_norm: (N, 6) in [0, 1]
        Returns: (x, y, vx, vy, mass) each (N,)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_norm = t / self._t_max
        inp    = torch.cat([t_norm, p_norm], dim=1)   # (N, 7)
        out    = self._net(inp)                        # (N, 4)

        x  = out[:, 0] * _X_SCALE
        y  = out[:, 1] * _Y_SCALE
        vx = out[:, 2] * _VX_SCALE
        vy = out[:, 3] * _VY_SCALE

        # Recover physical params for analytical mass
        p_raw    = denormalise_params(p_norm)
        mass_wet  = p_raw[:, 0]
        mass_dry  = p_raw[:, 1]
        burn_rate = p_raw[:, 3]

        mass = _analytical_mass_param(t, mass_wet, mass_dry, burn_rate)

        return x, y, vx, vy, mass

    def predict(self, t: torch.Tensor, p_raw: torch.Tensor) -> torch.Tensor:
        """
        Physical state [x, y, vx, vy, mass] at time(s) t for config p_raw.
        t:     (N,) or (N,1) in seconds
        p_raw: (N, 6) in physical units  OR  (1, 6) broadcast across t
        Returns (N, 5). No gradients.
        """
        with torch.no_grad():
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            if p_raw.shape[0] == 1 and t.shape[0] > 1:
                p_raw = p_raw.expand(t.shape[0], -1)
            p_norm = normalise_params(p_raw)
            x, y, vx, vy, mass = self._decode(t, p_norm)
        return torch.stack([x, y, vx, vy, mass], dim=1)

    def predict_grad(self, t: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        """Same as predict but keeps gradients — used during training."""
        x, y, vx, vy, mass = self._decode(t, p_norm)
        return torch.stack([x, y, vx, vy, mass], dim=1)

    def residuals(self, t: torch.Tensor,
                  p_norm: torch.Tensor) -> torch.Tensor:
        """
        Physics residuals at collocation points.
        t:      (N, 1) with requires_grad=True
        p_norm: (N, 6)  (no grad needed for params at collocation time)
        Returns (N, 4): [R0, R1, R2, R3]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x, y, vx, vy, mass = self._decode(t, p_norm)

        def _grad(field: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(
                field, t,
                grad_outputs=torch.ones_like(field),
                create_graph=True,
                retain_graph=True,
            )[0].squeeze(-1)

        dx_dt  = _grad(x)
        dy_dt  = _grad(y)
        dvx_dt = _grad(vx)
        dvy_dt = _grad(vy)

        # Recover physical params for force computation
        p_raw     = denormalise_params(p_norm)
        mass_wet  = p_raw[:, 0]
        mass_dry  = p_raw[:, 1]
        thrust    = p_raw[:, 2]
        burn_rate = p_raw[:, 3]
        drag_coef = p_raw[:, 4]
        wind_vx   = p_raw[:, 5]
        cross_sec = torch.full_like(wind_vx, self._cross_section)

        ax, ay = _forces_param(vx, vy, y, mass, t,
                               thrust, burn_rate, mass_wet, mass_dry,
                               drag_coef, cross_sec, wind_vx, self._angle_rad)

        R0 = dx_dt  - vx
        R1 = dy_dt  - vy
        R2 = dvx_dt - ax
        R3 = dvy_dt - ay

        return torch.stack([R0, R1, R2, R3], dim=1)   # (N, 4)
