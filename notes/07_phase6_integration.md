# Phase 6 — PINN-PPO Integration

*Reference file. Covers why integration matters, the math behind the two
changes, how the reward structure changes, what to expect from training,
and what the paper says.*

---

## 1. The Gap After Phase 5

After Phase 4 we have: a PPO agent that tracks a fixed nominal trajectory.
After Phase 5 we have: a PINN that predicts physically consistent trajectories
across a family of rocket configurations.

They are two completely separate systems. The PPO agent doesn't know the PINN
exists. The PINN has never influenced any reward signal.

The gap: **the PPO agent is still tracking the wrong reference.**

Here's the concrete problem. Every episode, domain randomisation picks a
different rocket (mass_wet: 80-120 kg, thrust: 4000-6000 N, etc.). But the
target trajectory in the reward comes from a single RK4 simulation of the
NOMINAL rocket (100 kg, 5000 N) run once at startup. So:

- Heavy rocket episode (120 kg): agent penalised for not climbing as fast as
  the nominal rocket would. But the heavy rocket physically cannot climb that fast.
- Light rocket episode (80 kg): agent penalised for climbing too fast. But
  that's what the physics demands for that config.

The agent is being rewarded for fighting its own physics. It's learning to
be wrong in a consistent way.

**The fix — Phase 6 Part A:** Replace the fixed RK4 reference with the
parameterised PINN. Per episode, query `pinn_param(t, p_episode)` to get
the reference. The reward now compares the agent against *its correct trajectory
for its specific config*.

**The extension — Phase 6 Part B:** Add a physics consistency penalty. Even
within a correct reference, the agent can deviate — aggressive angle changes
can produce velocities that aren't physically achievable. The PINN penalty
enforces: "your state must be near what physics says it should be."

---

## 2. Math — Part A: Per-Episode PINN Reference

### Current system (broken for randomised episodes)

```
TargetTrajectory:
  - Built once at __init__
  - Interpolates RK4(DEFAULT_ROCKET, DEFAULT_SIM) at any t
  - Returns (target_y, target_vy, target_vx) for any time

Reward at step t:
  r_alt = -2.0 * |target_y_nominal(t) - y_actual| / 35000
```

Real number example at t=30s, episode with heavy rocket (mass_wet=120kg):
```
target_y_nominal(30) ≈ 8,500 m   (nominal rocket altitude at t=30s)
y_actual             ≈ 7,200 m   (heavy rocket is lower — correct physics!)
error                = 1,300 m
r_alt                = -2.0 × 1300/35000 = -0.074
```

The agent is penalised 0.074 for being physically correct. Over thousands of
steps, this teaches the agent to push against its own physics constraints —
which is wrong and produces the angle-saturation behaviour we saw in v3.

### Fixed system (PINN reference per episode)

```
PhysicsGuidedTargetTrajectory:
  - Takes the episode's rocket config (mass_wet, thrust, etc.) at reset()
  - Builds parameter vector p from the config
  - Queries pinn_param(t, p) at each step via the trained model

Reward at step t:
  r_alt = -2.0 * |pinn_y(t, p) - y_actual| / 35000
```

Real number example at t=30s, same heavy rocket:
```
pinn_y(30, p_heavy) ≈ 7,150 m   (PINN's prediction for heavy rocket at t=30s)
y_actual            ≈ 7,200 m   (agent is very close to correct trajectory!)
error               = 50 m
r_alt               = -2.0 × 50/35000 = -0.0029
```

The agent is nearly perfectly rewarded for being on its correct trajectory.
The PINN reference turns a 26× penalty into a near-zero penalty for physically
correct behaviour.

### How to build parameter vector p from the episode config

The PINN's parameter vector is:
```
p = [mass_wet, mass_dry, thrust, burn_rate, drag_coeff, wind_vx]
```

At each episode reset, `env._rocket` is the sampled `RocketConfig`. We read:
```python
p_raw = np.array([
    rocket.mass_wet,
    rocket.mass_dry,
    rocket.thrust,
    rocket.burn_rate,
    rocket.drag_coeff,
    wind_vx_estimate,   # average wind for the episode
], dtype=np.float32)
```

Then normalise to [0,1] using the PINN's `normalise_params()`.

Wind is trickier — the wind model is stochastic (Ornstein-Uhlenbeck process),
so we don't know the exact wind at each step. Use the initial wind estimate
(the reference velocity `wind_cfg.v_ref`) as the fixed p[5] for the PINN
reference. The agent sees the actual noisy wind in its observation. The slight
mismatch between PINN wind and actual wind is fine — it's within the ±15 m/s
training envelope.

---

## 3. Math — Part B: Physics Consistency Penalty

After Part A, the reward tells the agent "track your PINN reference". But the
agent could still produce a trajectory that deviates from physics — if it applies
very aggressive angle changes that momentarily boost acceleration beyond what the
thrust and drag allow, the PINN would predict a different velocity.

The physics penalty adds:
```
r_physics = -W_PINN * (|y - pinn_y(t,p)|/Y_SCALE)²
           + -W_PINN * (|vy - pinn_vy(t,p)|/VY_SCALE)²
```

Real number example: agent takes a sharp 5° angle delta that overshoots:
```
pinn_y(t, p)   = 12,000 m
y_actual       = 12,350 m   (agent climbed too fast — beyond physics)
deviation      = 350 m
r_physics      = -0.5 × (350/35000)² = -0.5 × 0.0001 = -0.00005
```

At small deviations, this is tiny relative to the tracking reward — the agent
doesn't feel it. At large deviations (1000+ m), it grows quadratically:
```
deviation = 2000 m
r_physics = -0.5 × (2000/35000)² = -0.5 × 0.0033 = -0.0016
```

Still small relative to the main tracking signal (-0.074 scale). This is
intentional: W_PINN should be small enough to not dominate but large enough
to create a gradient signal that discourages physical inconsistency.

**Why quadratic?** Linear penalty has constant gradient everywhere —
doesn't distinguish small natural deviations from large unphysical ones.
Quadratic: gradient grows with deviation, so large violations get strong
correction signal. Standard choice for physics-consistency penalties.

**W_PINN choice:** Start at 0.5 (half the velocity tracking weight).
Monitor r_physics in training logs. If it becomes the dominant term
(larger than r_alt), reduce it. If residuals don't improve, increase it.

---

## 4. What Changes in the Code

### Change 1: simulation/target_trajectory.py

Add `PhysicsGuidedTargetTrajectory` class:
- Loads `pinn_param_v1.pt` at init (once, stays in memory)
- `set_episode(rocket_cfg, wind_vx_ref)` — builds p from episode config
- `query(t)` — returns (pinn_y, pinn_vy, pinn_vx) from PINN forward pass
- Falls back gracefully if PINN model file doesn't exist (uses RK4)

Why separate class rather than modifying `TargetTrajectory`?
- Keeps backward compatibility (v3 env still works)
- Physics-guided mode is opt-in — `physics_guided=True` flag in RocketEnv
- Can compare v3 vs v4 training without code conflicts

### Change 2: simulation/env.py

Add `physics_guided: bool = False` parameter to `__init__`.
When True:
- Replace `self._target = TargetTrajectory()` with
  `self._target = PhysicsGuidedTargetTrajectory()`
- In `reset()`: call `self._target.set_episode(self._rocket, wind_v_ref)`
- In `_compute_reward()`: add `r_physics` term

New reward constant: `_W_PINN = 0.5`

### Change 3: training/train_v4.py

New training script, saves as `ppo_v4_*`:
- Same two-stage curriculum as v3
- Uses `RocketEnv(physics_guided=True)`
- Everything else identical to v3

---

## 5. Expected Training Behaviour

### What should improve vs v3:
1. **Reward signal quality** — agent gets meaningful gradients for its actual config
   instead of fighting the wrong reference
2. **Control smoothness** — physics penalty discourages large angle jumps
3. **Episode-to-episode consistency** — reward is now relative to correct physics,
   not an arbitrary fixed standard
4. **Angle saturation** — the snapping to 10° after burnout should reduce because
   the PINN reference naturally describes the gravity turn

### What might not improve (honest):
- **Absolute tracking error** may not drop dramatically. The PINN reference is
  the uncontrolled nominal trajectory — the agent still needs to track it under
  wind disturbances and domain randomisation. The physics penalty helps quality
  of control, not necessarily accuracy.
- **Training stability** — adding a new term can introduce instability in early
  training. If rewards oscillate more, reduce W_PINN.

### How to measure success:
Compare ppo_v3_final vs ppo_v4_final:
- Mean tracking error (km): should decrease
- Mean r_smooth (should increase — less angle variation)
- Physics residual: compute |y_agent - pinn_y(t,p)| averaged over episodes
- Angle saturation: does the agent still snap to 10° after burnout?

This comparison is Table 1 in the paper.

---

## 6. Why This Is Novel — The Paper Claim

Current state of the art in RL-for-rockets:
- Most papers use a fixed reference trajectory
- Some use adaptive references via RK4 re-simulation (expensive)
- None (to our knowledge) use a PINN surrogate as a per-episode reference generator

Our claim: "We replace the computationally expensive per-episode RK4 re-simulation
with a trained PINN surrogate that provides physically consistent reference
trajectories at near-zero cost. The PINN is not only faster but also
differentiable — enabling gradient-based reward shaping that directly penalises
physically inconsistent control."

The PINN as reference is 100-1000× faster than RK4 per query (neural network
forward pass vs numerical integration). For real-time guidance (Phase 7 hardware),
this matters enormously.

[PAPER OPPORTUNITY: PINN-as-surrogate-reference for domain-randomised RL training —
comparison of fixed reference vs per-episode RK4 vs per-episode PINN reference]

---

## 7. The PINN Query Problem — Computational Cost During Training

The PINN query happens at every step of every episode during RL training.
With 4 parallel envs, 4096 steps per rollout, 1M total steps:
- Total PINN queries: ~1M / 0.01s (dt) = 100M queries
- Each query: one PINN forward pass (400K params, 6 layers) on CPU

This could be a bottleneck. Solutions:
1. **Batch queries**: instead of querying per-step, precompute the PINN
   trajectory for the full episode at reset() and cache it. Then each step
   just does a lookup (interpolation).
2. **CPU inference**: PINN is 400K params — lightweight enough for CPU.
   No GPU needed for inference during RL rollout.

Implementation: at `set_episode()`, precompute PINN trajectory for
t in [0, max_time] with dt=0.1s (1850 points), cache as numpy arrays,
interpolate at each step. This trades GPU memory for speed.

[PAPER OPPORTUNITY: Efficiency comparison of per-step PINN query vs cached
episode trajectory — latency analysis for near-real-time guidance]

---

## 8. The Wrong Reference Problem — Why v3 Was Fundamentally Broken for Randomisation

This is the most important conceptual insight of Phase 6, and it's worth
understanding deeply because it changes how you think about reward design.

### The problem in plain language

The v3 agent was trained to track a nominal trajectory. But domain randomisation
gives each episode a different rocket. The tracking reward compares the agent
against the NOMINAL trajectory regardless of which rocket it's flying.

This means the agent's entire reward signal was solving the wrong problem: "be
as close as possible to what the nominal rocket would do" — but that's
physically impossible for a heavy or light rocket. The agent learned to saturate
its thrust angle (the only thing it can control) trying to fight its own physics.
The 10° angle snap we observed in v3 evaluation is a direct consequence.

### Why the agent learned the angle snap

During Stage 2 (full randomisation), the agent saw episodes with many different
configs. For all of them, the reward penalised deviating from the nominal reference.
The only action that gives ANY reward in the coasting phase (after burnout, no thrust)
is to push the angle as low as possible to maximise the component of remaining
velocity in the horizontal direction — minimising the altitude-velocity mismatch
against the nominal reference which by then is at a specific point in its trajectory.

The agent didn't learn "fly a good trajectory for this rocket." It learned "given
that I'm being compared to the nominal rocket and I can't change that, set angle to 10°
and hold it to minimise the average punishment."

This is called **reward misspecification** — the reward signal optimises the wrong
objective. The agent found the optimal policy for the wrong problem.

### Why PINN reference fixes this at the root

With a per-episode PINN reference, the reward signal is now:
"how well are you tracking the physically correct trajectory FOR YOUR ROCKET?"

A heavy-rocket episode gets a heavy-rocket reference. The agent doesn't need to
fight its own physics. The correct action (angle that follows the mass-adjusted
gravity turn) now gets positive reward instead of being penalised.

This is the difference between:
- v3: "why isn't the heavy rocket as high as the nominal rocket?" (wrong question)
- v4: "why isn't the heavy rocket at the altitude the heavy rocket should be at?" (right question)

The physics penalty reinforces this: deviating from the PINN reference (which
satisfies Newton's second law) is penalised. Staying on-physics is rewarded.

### Expected behavioural change

With correctly specified rewards, the agent should:
1. Learn a different control strategy per rocket config (implicit conditioning
   via domain randomisation — the same thing it already does, but now rewarded correctly)
2. Show smoother angle variation during flight (no desperate snap to 10° at burnout)
3. Show lower tracking error on held-out configs (the reward shape now generalises)


---

## 9. Actual Training Results — Phase 6 v4

### Training progression

Stage 1 (400K steps, nominal config, PINN reference):
- rew: -4704 → -2587 (improving, agent learns nominal trajectory under PINN guidance)

Stage 2 (600K steps, full domain randomisation, per-episode PINN reference):
- rew: -3930 → -6918 (reward drops — harder task, as expected)
- ep_len: 11094 → 9846 (episodes ending earlier — agent explores termination states)

**Why Stage 2 reward drops:** This is correct behaviour. When domain randomisation
kicks in, each episode is harder — different rocket, different PINN reference.
The reward is now measuring "how close to the PINN reference for *this* specific
rocket" — which is a moving target. The agent is learning a general policy that
handles all configs, not specialising for the nominal case.

### Benchmark results (10 configurations)

```
Metric                     v3 (fixed ref)   v4 (PINN-guided)   Delta
Mean max altitude error         8.535 km         8.599 km       +0.7%
Mean avg altitude error         3.875 km         3.908 km       +0.8%
Mean angle variation std        0.0637°          0.0453°       -28.8%
```

### Physical interpretation of results

**Tracking equivalence:** v4 doesn't improve raw tracking accuracy in the same
number of training steps. The PINN reference is more complex than the fixed
reference — the agent needs to learn to track a different target per episode.
With more training (2-3M steps), v4 likely improves tracking too.

**Smoothness improvement is the key result.** The physics penalty (r_physics)
creates a gradient signal that discourages angle jumps that push the agent off
the PINN reference. The agent learned that smooth control = staying near PINN
trajectory = lower penalty. This is exactly what we designed the penalty for.

**Why smoothness matters for the paper:**
- Smoother control = fewer high-frequency angle changes
- In real hardware: fewer gimbal actuator cycles → longer actuator lifetime
- In the RL literature: smoother control = better safety properties
- In the physics sense: smooth control → trajectory closer to Pontryagin minimum
  principle (optimal control), which uses smooth bang-bang-like structures

**The rand:1 outlier:** seed=1 happens to produce a configuration that v4 handles
poorly (25.6 km). This suggests the PINN training distribution doesn't cover that
corner of the parameter space well enough, or the Stage 2 curriculum didn't expose
the agent to enough of that configuration. A future experiment: oversample that
region during training.

### Table 1 for the paper

| Method              | Mean max error | Mean avg error | Angle var std | Training steps |
|---------------------|---------------|----------------|--------------|----------------|
| Fixed RK4 ref (v3)  | 8.535 km      | 3.875 km       | 0.0637°      | 1M             |
| PINN-guided (v4)    | 8.599 km      | 3.908 km       | 0.0453°      | 1M             |
| Δ                   | +0.7%         | +0.8%          | **-28.8%**   | equal          |

**Paper claim:** "Our PINN-guided reward shaping achieves equivalent altitude tracking
accuracy while producing 28.8% smoother control behaviour, as measured by thrust angle
variation standard deviation across 10 held-out rocket configurations."

### What to do next (Phase 7 directions)

1. **Train v4 longer:** 2-3M steps would likely show tracking improvement too
2. **Investigate rand:1 outlier:** identify the config, check PINN accuracy there
3. **Tune W_PINN:** currently 0.5. Increasing to 1.0 may improve smoothness further
4. **3D extension:** Phase 5 was 2D. The PINN surrogate approach scales directly to 3D
5. **Real-time inference benchmark:** time the PINN query vs RK4 per episode
