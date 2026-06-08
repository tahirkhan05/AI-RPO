# Reinforcement Learning — The Environment

*Reference file. Covers what RL is, how Gymnasium works, and every design decision
made in the AI-RPO environment: state space, action space, reward function, and
episode structure. Return here when training or writing papers.*

---

## 1. What Reinforcement Learning Actually Is

RL is a framework for teaching an agent to make decisions by trial and error.

```
At every timestep:
  1. Agent OBSERVES the current state of the world
  2. Agent CHOOSES an action
  3. World transitions to a new state
  4. Agent RECEIVES a reward (positive or negative number)
  5. Agent updates its policy to get more reward in the future
  6. Repeat
```

There is no dataset, no labels, no "correct answer" given upfront.
The agent discovers what works purely by doing things and seeing what happens.

**The three things you must design:**
- **State** — what the agent can see
- **Action** — what the agent can do
- **Reward** — what "good" means

Get these right and the agent learns the right thing.
Get them wrong and it learns something useless or weird.

---

## 2. The Gymnasium API — How RL Environments Work

Gymnasium (formerly OpenAI Gym) is the standard Python interface.
Every RL environment, regardless of what it simulates, follows the same pattern:

```python
env = RocketEnv()

obs, info = env.reset()          # start a new episode, get first observation
done = False

while not done:
    action = agent.choose(obs)   # agent picks an action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

**Four methods every environment must implement:**

| Method | What it does |
|---|---|
| `reset()` | Start a new episode. Returns (observation, info dict). |
| `step(action)` | Apply action, advance sim one timestep. Returns (obs, reward, terminated, truncated, info). |
| `observation_space` | Declares the shape and bounds of what the agent can see. |
| `action_space` | Declares the shape and bounds of what the agent can do. |

`terminated` = episode ended naturally (rocket landed).
`truncated` = episode hit the time limit before landing.

---

## 3. State Space — What the Agent Sees

The agent observes a vector of numbers at each timestep.
We call this the **observation vector**.

```
obs = [x, y, vx, vy, mass_fraction, wind_vx_est, dy_to_target, dvx_to_target, dvy_to_target, angle_deg]
```

| Variable | Meaning | Why included |
|---|---|---|
| x | Downrange position (m) | Agent needs to know where it is |
| y | Altitude (m) | Most important — drives all decisions |
| vx | Horizontal velocity (m/s) | Agent needs rate of change, not just position |
| vy | Vertical velocity (m/s) | Same — vy going negative means falling |
| mass_fraction | fuel_remaining / fuel_total (0→1) | Agent learns to be fuel-efficient |
| wind_vx_est | Estimated wind (m/s) | Agent can partially observe disturbance |
| dy_to_target | target_y - y | Error signal: how far off the target altitude |
| dvx_to_target | target_vx - vx | Error signal: velocity mismatch |
| dvy_to_target | target_vy - vy | Error signal: velocity mismatch |
| angle_deg | Current thrust angle (degrees) | Agent sees its own control state |

**Why angle_deg was added (BUG-011 lesson):**
Without the angle in the observation, the agent was flying blind — it could not
see that it had saturated to the 90° or 10° bound. The action space is a delta
(±5°/step). If the agent always commands +5° but can't see the current angle,
it keeps pressing +5° even when already at the bound, because nothing in the
observation changed. Adding angle_deg broke the saturation and allowed the policy
to learn nuanced corrections. Mean tracking error dropped from 7.7 km to 0.25 km
after this single change (combined with longer training).

**Why normalise?** Neural networks train better when inputs are in a consistent
range (roughly -1 to 1). We normalise each observation by its expected max value.

**Why mass_fraction and not raw mass?**
Raw mass varies with domain randomization (80–120 kg). Fraction is always 0→1
regardless of the randomized mass, so the agent can interpret it consistently.

**Why wind estimate and not true wind?**
In reality the agent cannot read the wind directly — it infers it from how the
rocket behaves. We give a noisy estimate to simulate this. Later (sensor fusion
phase) this will come from a Kalman filter. For now: true wind + small noise.

[PAPER OPPORTUNITY #7: Observation space ablation study — which state variables
contribute most to policy performance? Remove one at a time and measure degradation.]

---

## 4. Action Space — What the Agent Controls

The agent outputs one number every timestep:

```
action = thrust_angle_delta (radians)
```

This is a **delta** (change) not an absolute angle.
Current angle += action each step.

**Why delta and not absolute angle?**
- Absolute: agent says "point to 45°" — can jump discontinuously between steps
- Delta: agent says "rotate 2° left" — physically smooth, like a real TVC system
- Smooth actions are easier to learn and more realistic

**Action bounds:** clipped to [-max_delta, +max_delta] per step.
max_delta = 5° per step at dt=0.01s = 500°/s max rotation rate.
Real TVC systems: ~20°/s. We're generous for learning speed.

The full angle is also clipped to [-90°, +90°] from horizontal —
pointing downward (below horizontal) is never useful.

**Action space type:** `gym.spaces.Box(low=-1, high=1, shape=(1,))`
We normalise the action to [-1, 1] and scale internally.
This is standard practice — PPO works best with normalised action ranges.

---

## 5. Target Trajectory — What "Good" Looks Like

The agent is not just flying to maximum altitude — it is trying to follow
a **desired trajectory**. We define this as the baseline (no-wind, nominal rocket)
trajectory from Phase 1.

```
At each timestep t, the target state is:
  target_y(t)  = baseline_y[t]
  target_vy(t) = baseline_vy[t]
  target_vx(t) = baseline_vx[t]
```

The agent must steer the randomized, wind-affected rocket to match
what the clean nominal rocket would have done. This is the guidance problem.

---

## 6. Reward Function — What "Good" Means Numerically

The reward is computed every timestep. It has three components:

### 6.1 Trajectory tracking reward (dense)
```
r_track = -w1 * |dy_to_target| / y_scale
        + -w2 * |dvy_to_target| / vy_scale
```
Penalises deviation from the target altitude and vertical velocity.
Negative because deviation is bad.
Scaled so both terms contribute roughly equally.

### 6.2 Fuel efficiency reward (dense)
```
r_fuel = +w3 * mass_fraction
```
Rewards having more fuel remaining. Agent learns not to waste thrust
on unnecessary corrections. This directly competes with r_track —
the agent must find the efficient correction, not just any correction.

### 6.3 Stability penalty (dense)
```
r_stable = -w4 * |angle_delta|²
```
Penalises large control actions (large angle changes per step).
Smooth control is preferred. Prevents oscillation (the agent thrashing
the thrust angle back and forth to track the trajectory).

### 6.4 Terminal rewards (sparse — only at episode end)
```
if landed safely (vy not too fast at impact):
    r_terminal = +100
elif crashed (vy too fast at landing):
    r_terminal = -100
elif timeout (never landed):
    r_terminal = -50
```

**Total reward per step:**
```
r = r_track + r_fuel + r_stable
```
Terminal bonus added at the final step.

**Why dense rewards?** Sparse reward (only at landing) makes learning very hard
— the agent has to accidentally succeed before it knows what good looks like.
Dense rewards give a learning signal every single step.

[PAPER OPPORTUNITY #8: Reward function ablation — impact of each reward component
on convergence speed and final policy quality. Which term matters most?]

---

## 7. Episode Structure

```
reset():
  1. Sample randomized rocket config (domain randomization)
  2. Sample randomized wind config
  3. Create fresh WindModel
  4. Set initial state = [0, 0, 0, 0, rocket.mass_wet]
  5. Set thrust_angle = launch_angle (85° from horizontal)
  6. t = 0
  7. Return observation

step(action):
  1. Clip action to [-1, 1], scale to radians
  2. Update thrust_angle += scaled_action
  3. Get wind velocity at current altitude
  4. Call rk4_step() — advance physics
  5. Compute reward
  6. Check termination conditions
  7. Return (obs, reward, terminated, truncated, info)

Termination conditions:
  - y < 0 and t > 0.5s  → landed (terminated=True)
  - t > max_time        → truncated=True
```

---

## 8. Why PPO for This Problem

PPO (Proximal Policy Optimization) is our chosen algorithm. Here's why it fits:

| Property | Why it matters here |
|---|---|
| Continuous action space | Thrust angle is a real number, not a discrete choice |
| On-policy | Collects fresh experience each update — important when environment randomizes |
| Clipped objective | Prevents destructively large policy updates — stable training |
| Works out of the box | Less hyperparameter sensitivity than DDPG or SAC |

**The clip objective (the key PPO idea):**
```
L = min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t)

r_t = probability ratio: new_policy(a|s) / old_policy(a|s)
A_t = advantage estimate (how much better this action was than average)
ε   = clip range, typically 0.2
```

If the new policy tries to change too much (r_t far from 1.0), the clip
kicks in and limits the update. This is what makes PPO stable.

[PAPER OPPORTUNITY #9: PPO vs DDPG vs SAC for continuous rocket guidance —
convergence speed, sample efficiency, and final policy robustness comparison.]

---

## 9. The Full RL Loop in One Picture

```
┌─────────────────────────────────────────────────────┐
│                    TRAINING LOOP                    │
│                                                     │
│  Episode starts:                                    │
│    randomize rocket + wind                          │
│    ↓                                                │
│  Each step:                                         │
│    Agent sees obs → outputs angle_delta             │
│    Physics advances (RK4 + wind)                    │
│    Reward computed                                  │
│    ↓                                                │
│  Episode ends (landed or timeout):                  │
│    Store (obs, action, reward, next_obs) in buffer  │
│    ↓                                                │
│  After N episodes:                                  │
│    PPO updates neural network weights               │
│    ↓                                                │
│  Repeat with new randomized episode                 │
└─────────────────────────────────────────────────────┘
```

The neural network IS the policy. It takes obs as input and outputs
a probability distribution over actions. Training makes this distribution
sharper and better over time.

---

## 10. Gymnasium Compliance — What "Correct" Looks Like

Three things Gymnasium strictly requires that are easy to get wrong:

**1. `super().reset(seed=seed)` is mandatory**
Gymnasium tracks its own internal `_np_random` attribute. If you don't call
`super().reset(seed=seed)`, the official `check_env()` validator fails.
The symptom is: "Expects the random number generator to have been generated."
Fix: always call `super().reset(seed=seed)` as the first line of `reset()`.

**2. `terminated` and `truncated` must be Python `bool`, not `np.bool_`**
NumPy comparisons like `y <= 0.0` return `np.bool_`, not `bool`.
Gymnasium's type checks reject this. Always cast: `bool(y <= 0.0)`.

**3. Determinism tests must use fixed actions, not `action_space.sample()`**
`action_space.sample()` has its own internal RNG that is NOT reset when you
call `env.reset(seed=...)`. So two runs with the same env seed but using
`action_space.sample()` will produce different action sequences.
To test determinism: provide fixed pre-computed actions to both runs.

These three are standard Gymnasium pitfalls — worth knowing for any future
environment you build.

---

## 11. What the Agent Learns (Intuition)

After enough training episodes, the agent learns things like:

- "When vy is much less than target_vy, tilt more steeply to gain vertical speed"
- "When wind is pushing me right (positive wind_vx), compensate left"
- "When fuel is low, stop making big corrections — conserve"
- "Near apogee, reduce angle aggressively to avoid overshooting"

None of this is programmed. It emerges purely from maximising the reward signal
across thousands of randomized episodes. This is the core claim of the paper.
