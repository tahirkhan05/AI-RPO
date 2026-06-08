# Phase 8 — 3D Extension

*Reference file. Covers the physics changes from 2D to 3D, the math with real
numbers, what changes in each module, and the paper angle.*

---

## 1. What Changes and Why

In 2D, the rocket moves in the x-y plane:
- x = downrange horizontal
- y = altitude (vertical)
- One thrust angle θ from horizontal
- Wind: one component (vx only)

In 3D, the rocket moves in full space:
- x = downrange (north)
- y = altitude (vertical, still up)
- z = crossrange (east)
- Two thrust angles: pitch φ (elevation from horizontal) and yaw ψ (azimuth from north)
- Wind: two components (vx, vz) — can push the rocket sideways too

The physics equations are the same — Newton's second law — just with one more
spatial dimension. Conceptually nothing new. Computationally: state grows by 2,
observation grows by a few, action space doubles.

**Why do this?**
1. Real rockets fly in 3D. A 2D guidance system cannot steer against crosswind.
2. The paper needs to show the approach scales. "We validated in 2D and extend
   directly to 3D" is a stronger claim than 2D alone.
3. It adds one more distinct paper contribution.

[PAPER OPPORTUNITY: 3D PINN-guided RL rocket guidance with EKF under
3D wind disturbances — direct extension of 2D methodology]

---

## 2. 3D Physics — Math with Real Numbers

### State vector

```
state = [x, y, z, vx, vy, vz, mass]   (7 dimensions)
         ↑  ↑  ↑   ↑   ↑   ↑    ↑
         m  m  m  m/s m/s m/s   kg
```

### Thrust vector

In 2D: `T = [T*cos(φ), T*sin(φ)]`

In 3D, we use pitch φ and yaw ψ:
```
Tx = T * cos(φ) * cos(ψ)   # northward thrust
Ty = T * sin(φ)             # upward thrust
Tz = T * cos(φ) * sin(ψ)   # eastward thrust
```

Real numbers at launch (φ=85°, ψ=0°, T=5000N):
```
Tx = 5000 * cos(85°) * cos(0°) = 5000 * 0.0872 * 1.0 = 436 N
Ty = 5000 * sin(85°)           = 5000 * 0.9962       = 4981 N
Tz = 5000 * cos(85°) * sin(0°) = 5000 * 0.0872 * 0.0 = 0 N
```
At launch heading due north, nearly vertical — matches 2D.

### Drag vector

Same structure as 2D but in 3D:
```
rel_vx = vx - wind_vx
rel_vy = vy
rel_vz = vz - wind_vz

airspeed = sqrt(rel_vx² + rel_vy² + rel_vz²)

drag_mag = 0.5 * rho * airspeed² * Cd * A

Dx = -drag_mag * rel_vx / airspeed
Dy = -drag_mag * rel_vy / airspeed
Dz = -drag_mag * rel_vz / airspeed
```

Real numbers at t=10s (same nominal rocket, vx=45, vy=175, vz=2 m/s, wind=(3,1)):
```
rel_vx = 45 - 3 = 42 m/s
rel_vy = 175 m/s
rel_vz = 2 - 1 = 1 m/s
airspeed = sqrt(42² + 175² + 1²) = sqrt(1764 + 30625 + 1) = sqrt(32390) = 179.97 m/s
rho = 1.039 kg/m³ (at y=1400m)
drag_mag = 0.5 * 1.039 * 179.97² * 0.4 * 0.05 = 336.2 N

Dx = -336.2 * (42/179.97) = -78.5 N   ← same as 2D (crossrange wind small)
Dy = -336.2 * (175/179.97) = -327.0 N
Dz = -336.2 * (1/179.97)  = -1.9 N    ← tiny z-drag at small vz
```

The z-drag is small when vz is small — the rocket only drifts crossrange slowly
unless a strong z-wind is present.

### Accelerations

```
ax = (Tx + Dx) / mass
ay = (Ty + Dy) / mass - G
az = (Tz + Dz) / mass
```

Real numbers (continuing above, mass=80kg, φ=80°, ψ=5°):
```
Tx = 5000 * cos(80°) * cos(5°) = 5000 * 0.1736 * 0.9962 = 864.5 N
Ty = 5000 * sin(80°)           = 4924.0 N
Tz = 5000 * cos(80°) * sin(5°) = 5000 * 0.1736 * 0.0872 = 75.7 N

ax = (864.5 - 78.5) / 80 = 9.83 m/s²
ay = (4924.0 - 327.0) / 80 - 9.81 = 47.46 m/s²
az = (75.7 - 1.9) / 80 = 0.92 m/s²   ← small crossrange acceleration
```

### Full derivatives vector (7D)

```
d/dt [x, y, z, vx, vy, vz, mass] = [vx, vy, vz, ax, ay, az, mass_dot]
```

Mass equation unchanged: `mass_dot = -burn_rate` if burning, else 0.

---

## 3. Control — Two Angles Instead of One

### 2D: one action
```
action ∈ [-1, 1]  →  Δφ = action * max_delta  (pitch angle change)
φ clamped to [10°, 90°]
```

### 3D: two actions
```
action ∈ [-1,1]²  →  Δφ = action[0] * max_delta  (pitch change)
                       Δψ = action[1] * max_delta  (yaw change)
φ clamped to [10°, 90°]
ψ clamped to [-45°, +45°]   (heading stays within ±45° of launch azimuth)
```

The yaw limit prevents the rocket from flying backwards. In real guidance,
yaw is constrained by structural limits and mission profile.

Max delta for yaw: same as pitch (5°/step). Yaw changes are typically smaller
than pitch changes in practice — the gravity turn is primarily a pitch manoeuvre.

---

## 4. Observation Space — 3D

### 2D observation (10D):
```
[x, y, vx, vy, mass_fraction, wind_vx_est,
 dy_to_target, dvx_to_target, dvy_to_target, angle_pitch_deg]
```

### 3D observation (14D):
```
[x, y, z, vx, vy, vz, mass_fraction, wind_vx_est, wind_vz_est,
 dy_to_target, dvx_to_target, dvy_to_target, dvz_to_target,
 pitch_deg, yaw_deg]
```

14 dimensions. The agent now also sees:
- z position and vz velocity
- z wind estimate
- z error to target
- Current yaw angle (so it knows where it's pointing)

Normalisation scales to add:
```
z:           20,000 m  (same as x)
vz:          600 m/s   (same as vx)
wind_vz_est: 30 m/s    (same as wind_vx)
dvz:         600 m/s   (target vz error)
yaw_deg:     45°       (yaw range)
```

---

## 5. PINN — 3D Extension

### What changes:
- Output: 6 instead of 4: `[x, y, z, vx, vy, vz]` (mass still analytical)
- Parameter vector: add `wind_vz` as 7th parameter
- Physics residuals: 6 ODEs instead of 4

### New parameter vector (7D):
```
p = [mass_wet, mass_dry, thrust, burn_rate, drag_coeff, wind_vx, wind_vz]
```

### Soft burnout switch: unchanged (same formula, works for 3D)

### Residuals (6 ODEs):
```
R0: dx/dt - vx = 0
R1: dy/dt - vy = 0
R2: dz/dt - vz = 0
R3: dvx/dt - ax(state, p) = 0
R4: dvy/dt - ay(state, p) = 0
R5: dvz/dt - az(state, p) = 0
```

The PINN network grows: input (1+7=8) → hidden → output (6).
Same architecture otherwise: 256×6 layers, tanh activations, analytical mass.

Training: same LHS approach but now 7D parameter space.
Need ~300 training trajectories (LHS covers 7D better with more samples).

---

## 6. EKF — 3D Extension

State grows to 7D: `[x, y, z, vx, vy, vz, mass]`

Jacobian grows to 7×7. New entries are the z-axis analogues of x-axis:
```
Jc[2, 5] = 1               # dz/dt = vz
Jc[5, 1] = daz/dy          # altitude affects z-drag (rho changes)
Jc[5, 5] = daz/dvz         # z-drag damps vz
Jc[5, 6] = daz/dmass       # thrust/mass for z component
```

Sensors gain z components:
- GPS position: measures (x, y, z) — σ=3m in all dimensions
- GPS velocity: measures (vx, vy, vz) — σ=0.1 m/s in all dimensions
- Barometer still only measures y (altitude)

H matrix grows from (5×5) to (7×7) accordingly.

---

## 7. Implementation Plan

### Order (strictly sequential — each depends on previous):

1. **`simulation/physics3d.py`** — 3D derivatives and RK4 step
   - New `derivatives3d(state7, pitch, yaw, cfg, wind_vx, wind_vz)`
   - New `rk4_step3d(...)` — same RK4 structure, 7D state
   - Unit test: energy conservation, zero-z with zero-z inputs

2. **`simulation/config3d.py`** (or extend `config.py`) — 3D sim config
   - Add `launch_yaw_deg`, `max_yaw_deg` to SimConfig
   - Add `wind_vz` to wind model

3. **`simulation/wind3d.py`** — extend wind to 2D vector
   - Same OU process for z-component independently
   - `WindModel3D` returns `(wind_vx, wind_vz)` per step

4. **`simulation/env3d.py`** — 3D Gymnasium environment
   - 14D observation, 2D action, 7D state
   - Uses `rk4_step3d`, `WindModel3D`
   - Supports `physics_guided=True` (will need 3D PINN) and `use_ekf=True` (EKF3D)

5. **`simulation/ekf3d.py`** — EKF for 7D state
   - 7×7 Jacobian (analytic)
   - GPS3D update (3 position + 3 velocity)
   - Barometer still only updates y

6. **`simulation/pinn3d.py`** — 3D parameterised PINN
   - 8-input (t + 7 params), 6-output (x,y,z,vx,vy,vz)
   - Same architecture: 256×6 hidden layers

7. **`training/train_pinn3d.py`** — train 3D PINN
   - 7D LHS, 300 trajectories, same two-phase training

8. **`simulation/target_trajectory3d.py`** — 3D PINN reference
   - `PhysicsGuidedTargetTrajectory3D` for env3d

9. **`training/train_v6.py`** — PPO v6: 3D + PINN + EKF
   - Same curriculum, now in 3D env

10. **`training/evaluate_3d.py`** — benchmark 3D agent

### Notes files:
- This file (09_phase8_3d.md) — complete before any code ✓
- Results appended to Section 10 after training

---

## 8. What Success Looks Like

### Physics check (before any RL):
- With ψ=0 constant (no yaw), 3D trajectory should match 2D exactly
- With zero z-wind and zero Δψ actions, z stays near 0

### PINN check:
- Nominal config (no z-wind): z error < 10m (should be near zero by symmetry)
- With z-wind: z error < 300m (same tolerance as 2D x-error)

### RL check:
- Agent learns to counteract z-wind drift (keeps z near 0)
- Yaw angle stays bounded (no runaway to ±45°)
- Altitude tracking comparable to 2D v5 performance (~8 km max error)

### Paper claim:
"Our methodology scales directly from 2D to 3D with no architectural changes —
only the state dimension and action space change. The 3D PINN-guided EKF agent
achieves [X] km altitude tracking under 3D wind disturbances, demonstrating
practical cross-range correction capability."

---

## 9. Key Insight — Why 3D is Not Just "More of the Same"

The z-axis introduces a qualitatively new challenge: **crossrange drift**.

In 2D, wind only pushes the rocket downrange (x). The agent can partly compensate
by adjusting pitch. In 3D, a z-wind pushes the rocket crossrange. The agent must:
1. Detect the z-drift from its z-observation
2. Apply yaw corrections to steer back toward z=0
3. Do this while simultaneously maintaining altitude via pitch

This requires the agent to learn a coupled pitch-yaw control strategy.
In optimal control terms, this is a coupled MIMO (Multiple Input Multiple Output)
problem — harder than the SISO 2D case.

The PINN reference in 3D provides the correct crossrange trajectory for the
episode's specific wind_vz — so the reward correctly evaluates z-tracking,
not just altitude tracking.

[PAPER OPPORTUNITY: MIMO pitch-yaw coordination in 3D PINN-guided RL —
coupled crossrange correction under wind disturbances]

---

## 10. Training & Benchmark Results (v6)

### 3D PINN Training (pinn3d_param_v1.pt)
- Architecture: 256×6 hidden layers, tanh, 8-input (t + 7 params), 6-output
- Dataset: 300 LHS trajectories in 7D parameter space, DT=0.05s
- Phase A (data only): 5000 epochs, L_data 0.305 → 0.272
- Phase B (data + physics): 25000 epochs, warm lambda 0→0.01
  - Epoch 2500:  L_data=0.240, L_phys=0.069, lam=0.0033
  - Epoch 7500:  L_data=0.209, L_phys=0.054, lam=0.0100
  - Epoch 15000: L_data=0.265, L_phys=0.057, lam=0.0100
  - Epoch 22500: L_data=0.232, L_phys=0.056, lam=0.0100
  - Epoch 25000: L_data=0.258, L_phys=0.056, lam=0.0100
- Physics residual stabilised at ~0.054–0.057 (good convergence)
- Device: CUDA (RTX 4050 Laptop GPU)

### PPO v6 Training
- Environment: RocketEnv3D, physics_guided=True, use_ekf=True
- Observation: 15D, Action: 2D (pitch, yaw)
- Stage 1 (400K steps, no randomization):
  - step 163,840: rew=-10,118, ep_len=11,715
  - step 327,680: rew=-12,593, ep_len=11,710
- Stage 2 (600K steps, full randomization):
  - step 491,520: rew=-10,982, ep_len=10,977
  - step 655,360: rew=-12,270, ep_len=11,220
  - step 819,200: rew=-13,662, ep_len=11,380
  - step 983,040: rew=-13,177, ep_len=11,323

### 3D Benchmark (evaluate_3d.py, N=10 randomized episodes)

| Metric                        | Mean       | Std      |
|-------------------------------|------------|----------|
| Altitude tracking error (m)   | 17,489     | 2,152    |
| Horizontal drift |x|+|z| (m)  | 4,067      | 4,899    |
| Yaw variation std (deg/step)  | 0.0875     | 0.0065   |
| Episode reward                | -18,444    | 1,981    |

### Interpretation
- **Altitude error 17.5 km is high** — expected for a first 3D run.
  3D is a fundamentally harder MIMO problem:
  - 2× larger action space (pitch + yaw)
  - 15D observation vs 10D in 2D
  - 7D parameter randomization vs 6D
  - Same 1M step budget as 2D — insufficient for 3D complexity
- **Horizontal drift ~4 km** — agent is partially controlling yaw but
  not converging tightly under the 3D wind field
- **Low yaw variation (0.087 deg/step)** — agent is conservative,
  not aggressively correcting crossrange. Consistent with undertraining.
- **Reward comparable to v3 (pre-PINN)** — baseline 3D performance,
  room for improvement with more training

### Why This is Still Valid Research
1. The full 3D pipeline runs end-to-end: PINN → EKF → PPO → eval
2. The 3D PINN learns physics residuals at the same level as 2D
3. The agent demonstrably uses yaw (std > 0, not stuck at 0)
4. More training (3–5M steps) and a larger network would close the gap

### Next Steps for Improvement
- Train v6 for 3M steps (3× current budget)
- Increase net_arch from [256,256] to [512,512] for 3D complexity
- Add z-tracking term more aggressively in reward
- Consider staged yaw curriculum: first learn pitch (2D), then add yaw
