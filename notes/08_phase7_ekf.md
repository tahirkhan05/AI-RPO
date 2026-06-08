# Phase 7 — Extended Kalman Filter (EKF) State Estimator

*Reference file. Covers what EKF is, the math from first principles with real
numbers, what sensors we simulate, how EKF plugs into the existing pipeline,
and the paper angle.*

---

## 1. Why We Need This

Right now our RL agent is "cheating." In `_observe()`:

```python
x, y, vx, vy, mass = self._state   # ← perfect ground truth, no noise
```

In reality you never have this. Sensors lie:
- Barometer: altitude ±5 m (pressure fluctuations, temperature)
- Accelerometer: acceleration ±0.3 m/s² (vibration, bias drift)
- GPS: position ±3 m, velocity ±0.1 m/s, only 5-10 Hz
- No mass sensor exists — you estimate from burn time

The EKF is the answer: given noisy sensor readings and a physics model,
estimate the true state as accurately as possible.

**Why this matters for the paper:**
Once EKF is in place, we can evaluate the agent under realistic sensor noise
instead of god-mode ground truth. The comparison "ground truth vs EKF-estimated"
is a direct measurement of how much performance degrades when you move from
simulation to reality. That is the first step of sim-to-real validation.

[PAPER OPPORTUNITY: Performance degradation analysis from ground-truth to EKF-estimated
state in physics-guided RL for rocket guidance]

---

## 2. The Kalman Filter Idea — Intuition First

Imagine you're trying to estimate your position in a dark room.

**Dead reckoning (predict step):** You know where you were 1 second ago, and
you know your velocity. Predict: `x_now = x_before + v * dt`. But errors
accumulate — small velocity errors compound over time.

**Landmark correction (update step):** You briefly feel a wall. You know
where the wall is. Correct your position estimate toward what the landmark
implies. But the wall estimate is also noisy — maybe you misjudged where
your hand touched.

**The Kalman trick:** Weight how much to correct by comparing *how uncertain
you are about your prediction* (process noise) vs *how uncertain the sensor is*
(measurement noise). If your prediction is very certain, trust it more. If the
sensor is very accurate, trust it more. The optimal weighting is the Kalman gain.

---

## 3. The Full EKF Math — With Real Numbers

### State vector

We estimate: `x̂ = [x, y, vx, vy, mass]ᵀ` — same as our physics state.

### Step 1: State Prediction

At each timestep, use the rocket physics to predict the next state:
```
x̂_k|k-1 = f(x̂_k-1, u_k-1)
```
where f is our `rk4_step` function, u is the thrust angle.

Real numbers at t=10s, nominal rocket mid-burn:
```
state = [x=200m, y=1500m, vx=50m/s, vy=180m/s, mass=80kg]
angle = 80° (near-vertical)
After one dt=0.01s step:
  predicted_y ≈ 1500 + 180 * 0.01 + 0.5 * (ay) * 0.01²
  predicted_vy ≈ 180 + ay * 0.01
```

### Step 2: Covariance Prediction

The uncertainty in our state estimate also grows with each prediction step.
We track this with the covariance matrix P (5×5 for our 5-state system):
```
P_k|k-1 = F_k * P_k-1 * F_kᵀ + Q
```

Where:
- **F_k** = Jacobian of f with respect to state (linearisation of physics)
- **Q** = process noise covariance — how much does the model drift per step?
- **P** = current state uncertainty (diagonal entries = variance per state)

**Q matrix (process noise — how wrong is our physics model per step?):**
```
Q = diag([
    0.01,    # x variance added per step (m²) — wind not perfectly modelled
    0.01,    # y variance added per step (m²)
    0.1,     # vx variance (m²/s²) — drag uncertainty
    0.1,     # vy variance (m²/s²)
    1e-6,    # mass variance (kg²) — burn rate is well-known
])
```
These are small because our physics model is good.

**Physical meaning of Q:** Q encodes "how much does my prediction drift from
reality per step because my model isn't perfect?" Wind gusts, thrust variations,
atmospheric density uncertainty all go into Q.

### Step 3: Compute the Kalman Gain

```
K_k = P_k|k-1 * H_kᵀ * (H_k * P_k|k-1 * H_kᵀ + R)⁻¹
```

Where:
- **H** = observation matrix (which states do we actually measure?)
- **R** = measurement noise covariance (how noisy are the sensors?)
- **K** = Kalman gain (how much to trust the measurement vs the prediction)

**Intuition of K:**
- If R is large (noisy sensors): K is small → mostly trust the prediction
- If P is large (uncertain prediction): K is large → mostly trust the sensors
- K always lies between 0 and 1 per element — it's a weighted average

**Real number example:**

Suppose we have a barometer measuring altitude, R_alt = 25 m² (σ = 5m).
Current prediction uncertainty P_y = 100 m² (σ = 10m, accumulated over flight).
Then:
```
K_y = 100 / (100 + 25) = 0.8
```
Meaning: trust the sensor 80%, trust the prediction 20%.

If the sensor is very noisy R_alt = 2500 m² (σ = 50m):
```
K_y = 100 / (100 + 2500) = 0.038
```
Now we trust the sensor only 3.8% and the physics prediction 96.2%.

This is the power of EKF — it automatically decides when to trust sensors
based on their relative uncertainties.

### Step 4: State Update

```
x̂_k = x̂_k|k-1 + K_k * (z_k - H_k * x̂_k|k-1)
```

The term `(z_k - H_k * x̂_k|k-1)` is called the **innovation** — the
difference between what we measured and what we predicted we would measure.
If innovation = 0, the measurement confirms the prediction perfectly → no correction.
If innovation is large → measurement says we're wrong → big correction.

### Step 5: Covariance Update

```
P_k = (I - K_k * H_k) * P_k|k-1
```

After incorporating a measurement, our uncertainty decreases. This is the
fundamental reason the Kalman filter works: each measurement reduces P.
After enough measurements, P converges to a small, stable value.

---

## 4. Linearisation — Why "Extended"?

The standard Kalman filter requires the dynamics f to be linear.
Our rocket physics is NOT linear:
- Drag: `F_drag = 0.5 * ρ * v² * Cd * A` — v² is nonlinear
- Thrust direction: `ax = T*cos(θ)/m` — product of control × state
- Air density: `ρ = ρ₀ * exp(-y/H)` — exponential in state y

The EKF linearises f around the current estimate x̂ using the Jacobian F:
```
F = ∂f/∂x  evaluated at x̂_k-1
```

This is a 5×5 matrix for our system. Each entry is ∂(state_i_dot)/∂(state_j).

**Key Jacobian entries (non-zero, non-trivial ones):**

∂(ẋ)/∂(vx) = 1          (position rate depends on velocity — trivial)
∂(ẏ)/∂(vy) = 1          (same)
∂(v̇x)/∂(vx): drag term — involves ∂(drag_x)/∂(vx), which depends on speed
∂(v̇y)/∂(y):  air density changes with altitude → drag changes with y
∂(v̇x)/∂(mass): acceleration = Force/mass → ∂ax/∂m = -(Tx + Dx)/m²

We implement this Jacobian analytically — the partial derivatives of our
`derivatives()` function with respect to each state variable.

**Why not numerical Jacobian?**
We could estimate it by finite differences: F_ij ≈ (f(x + ε*eⱼ) - f(x)) / ε.
This works but costs 5 extra physics evaluations per step. The analytic Jacobian
is faster and more accurate. We implement it analytically.

---

## 5. Sensor Suite — What We Simulate

For our 2D rocket, we simulate these sensors:

| Sensor        | Measures       | Noise σ    | Rate    | Maps to states |
|---------------|---------------|------------|---------|----------------|
| Barometer     | Altitude y     | 5.0 m      | 100 Hz  | y              |
| Accelerometer | ax, ay         | 0.3 m/s²   | 100 Hz  | (used in model)|
| GPS position  | x, y           | 3.0 m      | 10 Hz   | x, y           |
| GPS velocity  | vx, vy         | 0.1 m/s    | 10 Hz   | vx, vy         |

Mass is NOT measured directly — estimated from:
```
m_est(t) = m_wet - burn_rate * min(t, t_burnout)
```
This is deterministic given the known burn rate, so mass uncertainty is small.

**H matrix (observation matrix):**

Which states does each sensor measure?
```
H = [[0, 1, 0, 0, 0],    # barometer → y
     [1, 1, 0, 0, 0],    # GPS pos   → x, y (two rows)
     [0, 0, 1, 1, 0]]    # GPS vel   → vx, vy (two rows)
```
(Simplified — actual H is built dynamically based on which sensors fire this step.)

**R matrix (measurement noise covariance):**
```
R_baro = [[25.0]]              # barometer: σ=5m → σ²=25m²
R_gps_pos = diag([9.0, 9.0])  # GPS pos: σ=3m
R_gps_vel = diag([0.01, 0.01]) # GPS vel: σ=0.1m/s
```

---

## 6. Where EKF Plugs Into the Pipeline

Current pipeline:
```
physics_step() → true_state → _observe() → agent
```

New pipeline:
```
physics_step() → true_state
                    ↓
              sensor_model()   ← adds realistic noise to true state
                    ↓
               EKF.update()   ← combines noisy measurements with physics prediction
                    ↓
              estimated_state → _observe() → agent
```

**Critical design decision:** The reward is still computed from the TRUE state
(for training stability). Only the OBSERVATION seen by the agent uses EKF estimates.
This decouples state estimation quality from reward quality during training —
we want the agent to learn correct behaviour, not to be penalised for EKF errors.

In deployment (real rocket), the true state is unavailable — only EKF estimates exist.

---

## 7. What the EKF Class Looks Like

```python
class EKF:
    def __init__(self, rocket: RocketConfig, dt: float):
        # Initial state uncertainty
        self.P = np.diag([100.0, 100.0, 10.0, 10.0, 1.0])
        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1, 1e-6])
        self.x = np.zeros(5)   # state estimate

    def predict(self, angle_rad: float, wind_vx: float) -> None:
        # 1. Propagate state through physics
        self.x = rk4_step(self.x, angle_rad, self.dt, self.rocket, wind_vx)
        # 2. Propagate covariance through Jacobian
        F = self._jacobian(self.x, angle_rad, wind_vx)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        # Innovation
        y = z - H @ self.x
        # Innovation covariance
        S = H @ self.P @ H.T + R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # State update
        self.x = self.x + K @ y
        # Covariance update
        self.P = (np.eye(5) - K @ H) @ self.P

    def _jacobian(self, state, angle_rad, wind_vx) -> np.ndarray:
        # 5×5 matrix: ∂derivatives/∂state
        # Derived analytically from our physics equations
        ...
```

---

## 8. The Jacobian — Derived from Our Physics

From `physics.py`, the derivatives function gives us:
```
ẋ  = vx
ẏ  = vy
v̇x = (Tx + Dx) / m
v̇y = (Ty + Dy) / m - g
ṁ  = -burn_rate  (if burning)
```

The Jacobian F = ∂f/∂x (5×5) at a given state:

Row 0 (∂ẋ/∂state):   [0, 0, 1, 0, 0]   (ẋ = vx, only depends on vx)
Row 1 (∂ẏ/∂state):   [0, 0, 0, 1, 0]   (ẏ = vy, only depends on vy)
Row 2 (∂v̇x/∂state):  non-trivial — drag depends on velocity and altitude
Row 3 (∂v̇y/∂state):  non-trivial — same, plus altitude dependence
Row 4 (∂ṁ/∂state):   [0, 0, 0, 0, 0]   (burn rate is constant, no dependence)

**Computing ∂v̇x/∂vx (the key drag derivative):**

```
v̇x = (Tx + Dx) / m
Dx = -0.5 * ρ * airspeed * Cd * A * (rel_vx / airspeed) * ρ
   = -0.5 * ρ * Cd * A * rel_vx

∂v̇x/∂vx = ∂Dx/∂vx / m
         = -0.5 * ρ * Cd * A / m  (at rel_vx = vx - wind_vx → ∂rel_vx/∂vx = 1)
```

Real number at t=5s, y=500m, nominal rocket:
```
ρ = 1.225 * exp(-500/8500) = 1.225 * 0.943 = 1.155 kg/m³
m = 90 kg (mid-burn)
Cd = 0.4, A = 0.05 m²
∂v̇x/∂vx = -0.5 * 1.155 * 0.4 * 0.05 / 90 = -0.000128 s⁻¹
```
Small but accumulates — this is why drag damps oscillations over long flights.

**Computing ∂v̇y/∂y (altitude-dependent drag):**

Since ρ = ρ₀ * exp(-y/H):
```
∂ρ/∂y = -ρ/H
∂Dy/∂y = ∂/∂y [-0.5 * ρ * v² * Cd * A * (vy/v)]
        = -0.5 * (∂ρ/∂y) * ... = 0.5 * ρ/H * v * Cd * A * (vy/v)
∂v̇y/∂y = ∂Dy/∂y / m
```

This term is small for low-altitude flight but grows at high altitudes where
∂ρ/∂y is proportionally larger relative to ρ.

---

## 9. Implementation Plan

### Files to create:
1. **`simulation/sensors.py`** — sensor noise models + measurement generation
2. **`simulation/ekf.py`** — EKF class with predict/update/jacobian

### Files to modify:
3. **`simulation/env.py`** — add `use_ekf: bool = False` flag; in `_observe()`,
   optionally replace true state with EKF estimate

### Files to create for training/evaluation:
4. **`training/train_v5.py`** — PPO v5 trained with EKF observations
5. **`training/evaluate_ekf.py`** — compare: v4 (ground truth) vs v5 (EKF) to
   measure the sim-to-real performance gap

### Notes:
6. **This file (notes/08_phase7_ekf.md)** — complete before any code ✓

---

## 10. Expected Results and Paper Claim

### What we expect:

Training with EKF will introduce more noise into the observation, making the
tracking reward harder to optimise. We expect:
- Slightly higher tracking error vs ground-truth agent (maybe 10-30% worse)
- Comparable control smoothness (EKF noise is small if tuned well)
- More robust policy — agent sees realistic observations, learns to handle uncertainty

### Performance under sensor noise (estimate):

| Agent | Observation | Expected max error |
|-------|-------------|-------------------|
| v4 | Ground truth | 4.892 km (measured) |
| v5 | EKF estimate | ~5.5–7.0 km (estimate) |

The gap (v4 → v5) is the "sensor noise cost" — the performance you lose
by not having perfect state information. A small gap = EKF is working well.
A large gap = either EKF is poorly tuned or the agent can't handle the noise.

### Paper claim:

"We evaluate our PINN-guided RL agent under two conditions: (1) ground-truth
state observation, and (2) EKF-estimated state from simulated IMU and barometer
sensors. The EKF agent achieves [X]% tracking accuracy relative to the ground-truth
agent, demonstrating practical deployability under realistic sensor conditions."

[PAPER OPPORTUNITY: Sim-to-real gap quantification via EKF state estimation —
ground-truth vs sensor-estimated observation for physics-guided RL guidance]

---

## 11. EKF Tuning — The Key Practical Challenge

EKF performance depends critically on Q and R choices:

**Q too small** (trust physics too much): EKF won't correct for model errors.
If thrust varies slightly from the nominal (common in real rockets), the estimate
drifts. The agent gets wrong observations. Tracking degrades.

**Q too large** (trust physics too little): EKF overweights sensors. If GPS is
noisy, the estimate jumps around. Observation is noisy. Agent control is erratic.

**R too small** (trust sensors too much): sensor noise passes directly into estimate.
**R too large** (trust sensors too little): estimate lags reality. Agent reacts late.

**Tuning approach:**
1. Start with Q based on known model uncertainties (burn rate variation ±5%,
   drag coefficient uncertainty ±10%)
2. Set R based on sensor datasheets (barometer σ=5m, GPS σ=3m)
3. Run EKF on known RK4 trajectories, measure estimation error
4. Adjust until RMSE < sensor noise level (EKF should be better than raw sensors)

This tuning process is itself a contribution — it demonstrates how to
calibrate a navigation filter for this class of vehicle.

[PAPER OPPORTUNITY: EKF tuning methodology for rocket state estimation —
Q/R covariance selection from physical uncertainty bounds]

---

## 12. Actual Training Results — Phase 7 v5

### Training curve

Stage 1 (400K steps, nominal, EKF active): -11629 → -12459
Stage 2 (600K steps, randomised, EKF active): -10727 → -6219

Stage 2 shows consistent improvement. Final reward -6219 vs v4's -5679 —
approximately 9% lower, which is the sensor noise cost during training.

### Benchmark results vs v4 (10 configurations)

```
Metric                     v4 (ground truth)   v5 (EKF)     Delta
Mean max altitude error         9.391 km        7.965 km    -15.2%
Mean avg altitude error         4.141 km        3.542 km    -14.5%
Mean angle variation std        0.0224°         0.0330°     +47.2%
EKF estimation RMSE             —               0.37 m      (vs baro 5m: 13.5× better)
```

### The unexpected result — EKF improves tracking

We predicted v5 would be slightly worse than v4 (sensor noise cost).
It is actually 15% better on tracking.

**Physical explanation:** The EKF is a recursive Bayesian filter — it
weights physics predictions and sensor measurements by their respective
uncertainties. The output is smoother than either input alone. Training
on smooth EKF estimates teaches the agent a smoother policy that happens
to track better, because:

1. High-frequency wind gusts appear as noise in ground truth but are
   smoothed by the EKF. The v5 agent doesn't react to individual gusts —
   it reacts to the EKF's estimate of the underlying trajectory, which is
   closer to the PINN reference.

2. The EKF's predict step implicitly "looks ahead" using physics — the
   estimate at time t already incorporates what physics says should happen
   at t+dt. This is a soft form of model predictive control.

**The cost:** Angle variation increases 47%. The EKF state at each step
differs slightly from ground truth, so the agent makes small corrections
that accumulate as control noise. This is the tradeoff: smoother trajectory,
noisier control.

### Paper Table 2

| Agent | Observation   | Max err   | Avg err  | Angle var | EKF RMSE |
|-------|--------------|-----------|----------|-----------|----------|
| v4    | Ground truth | 9.391 km  | 4.141 km | 0.0224°   | —        |
| v5    | EKF          | 7.965 km  | 3.542 km | 0.0330°   | 0.37 m   |
| Δ     |              | **-15.2%**| **-14.5%**| +47.2%  | 13.5× vs sensor |

**Paper claim:** "EKF-estimated state observations act as implicit policy
regularisation, improving altitude tracking by 14-15% at the cost of 47%
higher control variation. The EKF achieves 13.5× reduction in altitude
estimation error versus raw barometer measurements."

[PAPER OPPORTUNITY: EKF as implicit regularisation in PINN-guided RL —
sensor fusion improves trajectory tracking through observation smoothing]
