# Uncertainty Modeling & Domain Randomization

*Reference file. Covers wind models, atmospheric variation, thrust noise, and why domain randomization is essential for robust AI training.*

---

## 1. Why We Add Uncertainty

The Phase 1 simulation was deterministic — same inputs, same outputs every time.
Real rockets fly in a messy world:
- Wind pushes the rocket sideways
- Atmosphere is thicker or thinner depending on the day
- The engine doesn't always burn at exactly the rated thrust
- The rocket body isn't manufactured to perfect spec

If we train the AI only on the clean simulation, it learns a policy that works
in a perfect world but breaks the moment anything deviates. This is called the
**sim-to-real gap** — and it is the #1 reason RL policies fail outside the lab.

Domain randomization is the solution: deliberately vary the simulation parameters
during training so the AI is forced to learn a policy that works across a wide
range of conditions, not just one exact scenario.

---

## 2. Wind Model — Altitude-Varying + Gusts

We build the wind in two layers stacked on top of each other.

### 2.1 Layer 1 — Mean Wind Profile (altitude-varying)

Real horizontal wind speed varies with altitude. A standard engineering
approximation is the **power law wind profile**:

```
v_wind(h) = v_ref * (h / h_ref) ^ alpha

v_ref  — reference wind speed at reference height (m/s)
h_ref  — reference height, typically 10 m (standard met measurement height)
alpha  — shear exponent. Typically 0.14 for open terrain (1/7 law)
h      — current altitude (m)
```

**Worked example:**
```
v_ref = 10 m/s at h_ref = 10 m, alpha = 0.14

At h = 100 m:  v = 10 * (100/10)^0.14 = 10 * 10^0.14 = 10 * 1.38 = 13.8 m/s
At h = 1000 m: v = 10 * (1000/10)^0.14 = 10 * 100^0.14 = 10 * 1.91 = 19.1 m/s
At h = 10000m: v = 10 * (10000/10)^0.14 = 10 * 1000^0.14 = 10 * 2.63 = 26.3 m/s
```

Wind grows with altitude but not linearly — it flattens out at high altitudes
where air is too thin to sustain strong winds. Above ~30 km we cap it.

**Why this matters for the AI:** The agent observes altitude as part of its state.
If wind changes with altitude, the agent can learn to anticipate and pre-correct
rather than only react. This is a much richer control problem.

---

### 2.2 Layer 2 — Turbulent Gusts (Dryden model, simplified)

On top of the mean wind, we add random gusts. The **Dryden turbulence model**
is used in aerospace (MIL-SPEC standards) and models gusts as colored noise —
not pure random, but correlated over time (a gust at t=1s influences t=1.1s).

Simplified version we use: **Ornstein-Uhlenbeck (OU) process**

```
dw = -theta * w * dt + sigma * sqrt(dt) * N(0,1)

w      — current gust velocity (m/s)
theta  — mean-reversion rate. How quickly the gust decays back to zero.
sigma  — volatility. How strong the gusts are.
N(0,1) — standard normal random sample
```

**Intuition:** Imagine the gust velocity as a ball attached to a spring (theta)
in a noisy environment (sigma). It gets kicked randomly but always pulled back
toward zero. This creates realistic-looking wind bursts.

**Worked example (one step):**
```
w = 3.0 m/s, theta = 0.5, sigma = 1.5, dt = 0.01s
dw = -0.5 * 3.0 * 0.01 + 1.5 * sqrt(0.01) * N(0,1)
   = -0.015 + 1.5 * 0.1 * (say) 0.8
   = -0.015 + 0.12 = +0.105
w_next = 3.0 + 0.105 = 3.105 m/s
```

Total wind at any moment: `w_total(h,t) = v_wind(h) + w_gust(t)`

[PAPER OPPORTUNITY #3: Dryden vs OU vs white noise turbulence models — impact on RL policy robustness for rocket guidance]

---

## 3. Thrust Noise

The engine doesn't produce a perfectly constant thrust. Manufacturing tolerances,
propellant grain variation, and combustion instability all contribute.

We model this as multiplicative noise:

```
T_actual(t) = T_nominal * (1 + epsilon(t))

epsilon(t) ~ OU process with small sigma (e.g. sigma=0.03 → 3% fluctuation)
```

Multiplicative (not additive) because a 3% variation on a 5000 N engine
means ±150 N, not ±3 N. The noise scales with the signal.

---

## 4. Domain Randomization

At the start of each training episode, we draw rocket and environment parameters
from ranges around their nominal values. The AI never sees the same rocket twice.

### 4.1 What we randomize and why

| Parameter | Nominal | Range | Physical meaning |
|---|---|---|---|
| mass_wet | 100 kg | ±10% | Fuel load variation, payload variance |
| mass_dry | 50 kg | ±5% | Manufacturing tolerance |
| thrust | 5000 N | ±15% | Engine-to-engine variation |
| burn_rate | 2 kg/s | ±10% | Propellant flow rate variance |
| drag_coeff | 0.4 | ±20% | Surface finish, fin alignment |
| wind speed | 10 m/s | ±50% | Day-to-day weather variation |
| gust sigma | 1.5 | ±40% | Turbulence intensity variation |

### 4.2 Why randomizing BOTH rocket and environment matters

If we only randomize environment (wind):
- AI learns to handle wind but assumes rocket is perfect
- Real rocket with ±10% mass variation will confuse it

If we randomize both:
- AI learns to be robust to uncertainty in itself AND the world
- The policy generalizes — this is called a **universal policy**

The key insight: **the AI learns to handle its own imperfection, not just the weather.**

[PAPER OPPORTUNITY #4: Rocket parameter randomization as a proxy for manufacturing tolerances — how much variation can a PPO policy absorb before performance degrades?]

### 4.3 Uniform vs Gaussian sampling

Two choices for how to sample each parameter:

```
Uniform:  param = nominal * Uniform(1 - pct, 1 + pct)
          → any value in range is equally likely
          → can produce extreme edge cases

Gaussian: param = nominal * Normal(1.0, pct/3)
          → values near nominal are most likely
          → extreme cases rare but possible
          → more realistic (most rockets are near-nominal)
```

We use **Gaussian** because it better models real manufacturing distributions.
The "3-sigma" rule: 99.7% of samples within ±pct range.

---

## 5. How This Feeds Into RL Training

Each training episode now looks like this:

```
1. Sample randomized rocket config     ← domain randomization
2. Sample randomized wind params       ← domain randomization
3. Reset simulation to t=0
4. Agent observes state, takes action
5. Physics steps forward with wind forces applied
6. Repeat until landing or timeout
7. Compute episode reward
8. Agent updates policy based on reward
9. Go to step 1 (new randomized episode)
```

The agent never memorizes one trajectory — it must learn the underlying
physics well enough to control any rocket in any wind. This is the difference
between a lookup table and actual intelligence.

[PAPER OPPORTUNITY #5: Curriculum domain randomization — start narrow, widen ranges as agent improves. Does this converge faster than full randomization from the start?]

---

## 6. What Changes in the Plots

When we run a simulation with uncertainty active:

- **Altitude curve** — no longer a smooth hill. Small deviations from wind forces.
- **Speed curve** — shows micro-fluctuations from gusts and thrust noise.
- **Flight path** — drifts sideways (in the x direction) from wind.
- **New plot needed** — wind speed vs altitude and gust time series.

Running 50 randomized episodes and overlaying all their trajectories produces
a **trajectory envelope** — a spread of paths showing how much uncertainty
affects the flight. The AI's job is to keep the envelope tight around the target.

[PAPER OPPORTUNITY #6: Trajectory dispersion analysis — quantifying the envelope width as a function of domain randomization intensity]

---

## 7. First Dispersion Results (Baseline vs 20 Randomized Episodes)

Running 20 episodes with full rocket + wind randomization against our fixed baseline:

| Metric | Value |
|---|---|
| Baseline apogee (no wind, nominal rocket) | 30.80 km |
| Randomized apogee range | 24.39 — 44.41 km |
| Total spread | ~20 km |

**What the dispersion plot tells you:**

- **Left panel (flight paths):** The black line is the clean baseline. Colored lines are
  randomized episodes. Notice some go much higher (lighter rocket, stronger thrust) and
  some go lower (heavier rocket, weaker thrust). The horizontal spread shows wind pushing
  rockets to different downrange distances.

- **Right panel (altitude vs time):** The black curve is the baseline hill shape.
  Randomized episodes form a **spread envelope** around it — some peak earlier, some later,
  some land sooner. This envelope is what the AI must learn to navigate.

**The key insight:** The AI's job is to collapse this spread. A well-trained policy given
the same target trajectory should produce results that cluster tightly around the baseline
*even when the rocket parameters and wind are unknown*. Wide envelope = untrained.
Narrow envelope = robust trained policy.

**Why the spread is asymmetric (more episodes above baseline than below):**
The Gaussian perturbation allows mass to decrease AND thrust to increase simultaneously,
which compounds upward. The constraint that mass_dry < mass_wet * 0.85 limits the
downward cases. This asymmetry is physically realistic — performance gains from lighter
rockets compound more than losses from heavier ones (Tsiolkovsky logarithm effect).
