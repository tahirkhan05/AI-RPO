# PPO Training — Concepts, Hyperparameters & Reading Results

*Reference file. Covers how PPO works internally, what every hyperparameter
does, how to read training curves, and what good vs bad training looks like.*

---

## 1. What PPO Is Actually Doing

PPO trains a neural network (the **policy**) that maps observations to actions.
It does this by repeatedly:

```
1. Collect N timesteps of experience using the CURRENT policy
2. Compute how good each action was (advantage estimation)
3. Update the network to make good actions more likely
4. Throw away the old experience and collect fresh data
5. Repeat
```

Step 3 is where "proximal" comes in. PPO clips how much the policy can change
in one update:

```
ratio = new_policy(a|s) / old_policy(a|s)

L_clip = min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)
```

- If ratio > 1+ε (new policy much more likely to take this action): clip
- If ratio < 1-ε (new policy much less likely): clip
- Otherwise: use the true gradient

**Why clip?** Unconstrained gradient steps can destroy a good policy in one
bad update. The clip says "don't change more than ε=20% per update step."
This makes PPO reliably stable — the #1 reason it's the default algorithm.

---

## 2. The Two Networks PPO Trains

PPO trains two separate networks simultaneously:

### Actor (policy network)
- Input: observation (9 numbers)
- Output: mean and std of a Gaussian distribution over actions
- The action is *sampled* from this distribution during training
  (exploration) and taken as the *mean* during evaluation (deterministic)

### Critic (value network)
- Input: observation (9 numbers)
- Output: a single number — estimated total future reward from this state
- Used to compute the **advantage**: how much better was this action
  compared to what we expected?

```
advantage = actual_return - critic_estimate
```

Positive advantage = this action was better than expected → increase probability
Negative advantage = worse than expected → decrease probability

Both networks are trained at every update step. The critic improves its
estimates, the actor improves its actions.

---

## 3. Hyperparameters — What Each One Does

### Core PPO parameters

| Parameter | Our value | What it controls |
|---|---|---|
| `learning_rate` | 3e-4 | Step size for gradient updates. Too high → unstable. Too low → slow. |
| `n_steps` | 2048 | Timesteps collected per update. More → better gradient estimates, slower updates. |
| `batch_size` | 64 | Mini-batch size for each gradient step. |
| `n_epochs` | 10 | How many passes over collected data per update. |
| `gamma` | 0.99 | Discount factor. 0.99 → rewards 100 steps away still matter. |
| `gae_lambda` | 0.95 | GAE smoothing. Balances bias vs variance in advantage estimates. |
| `clip_range` | 0.2 | The ε in the clip. 0.2 = max 20% policy change per update. |
| `ent_coef` | 0.01 | Entropy bonus. Encourages exploration. Higher → more random actions. |
| `vf_coef` | 0.5 | How much to weight critic (value) loss vs actor loss. |

### Network architecture
```
policy_kwargs = dict(net_arch=[64, 64])
```
Two hidden layers of 64 neurons each. This is small but appropriate — our
observation is only 9 numbers. Bigger networks train slower and can overfit.

**Why 64x64 and not deeper?**
The control problem is relatively low-dimensional. The rocket has 9 observable
quantities and 1 control output. A 64x64 network has ~5000 parameters — enough
to represent a smooth control surface without overfitting.

[PAPER OPPORTUNITY #10: Network architecture search for rocket guidance policy —
does depth or width matter more? 32x32 vs 64x64 vs 128x128 vs 64x64x64]

---

## 4. Training Schedule

We train in two stages:

### Stage 1 — Curriculum warmup (0 → 200K steps)
Domain randomization is OFF. The agent trains on the nominal rocket with
light wind only. This gives it a stable foundation before the hard cases.

### Stage 2 — Full randomization (200K → 500K+ steps)
Full domain randomization ON. The agent must now handle the full spread
of rocket and wind variation. Policy built in Stage 1 provides a warm start.

**Why curriculum?** Random initialisation + full randomization from step 0
can cause the agent to never discover a working trajectory. Starting easy
and increasing difficulty is called **curriculum learning**. It consistently
outperforms pure random-from-start.

[PAPER OPPORTUNITY #5 revisited: Curriculum domain randomization — does
staged difficulty increase convergence speed vs full randomization from start?]

---

## 5. What the Training Curves Tell You

Stable-Baselines3 logs these metrics. Here's how to read each one:

### `ep_rew_mean` — Mean episode reward
The most important metric. Should generally increase over time.

```
Early training (~0-100K steps):  very negative, agent crashing often
Mid training (~100-300K steps):  rising, agent learning basic control
Late training (~300K+ steps):    plateaus near max — agent converged
```

A flat line from the start = the agent isn't learning (reward too sparse,
learning rate too high, or bug in reward function).
A flat line after rising = normal convergence.
A falling line after rising = catastrophic forgetting (too high learning rate
or network too small for full randomization).

### `ep_len_mean` — Mean episode length
How many steps each episode lasts.

- Increasing = agent surviving longer (good early sign)
- Stabilising = agent reaching natural episode end (landing or timeout)
- Suddenly dropping = agent crashing more (regression)

### `entropy_loss` — Policy entropy
How random the agent's actions are.

- High early, decreasing over time = normal (agent explores, then exploits)
- Goes to zero too fast = agent got stuck in a local optimum (collapsed policy)
- Stays high forever = agent never converged (too high `ent_coef`)

### `value_loss` — Critic error
How wrong the critic's predictions are.

- Should decrease over training
- Spikes are normal at stage transitions (e.g. when full randomization kicks in)
- Persistently high = critic can't fit the value function (network too small)

### `approx_kl` — KL divergence between old and new policy
Measures how much the policy changed each update. Should stay below ~0.02.

- Consistently above 0.02 = updates too large, reduce learning rate
- At zero = policy not updating (lr too low or gradient vanished)

---

## 6. What "Converged" Looks Like for This Problem

A well-trained policy for our rocket problem should show:

- `ep_rew_mean` plateaued above -50 (dense reward per step is ~-0.5 to +0.3,
  over ~18000 steps per episode)
- Trajectory tracking within ±500m of target altitude at most timesteps
- Fuel efficiency above 60% (agent not wasting thrust on large corrections)
- Smooth thrust angle curve — no rapid oscillation
- Safe landing rate > 90% on randomized episodes

---

## 7. Saving and Loading Models

```python
# Save
model.save("models/ppo_rocket_stage1")

# Load
from stable_baselines3 import PPO
model = PPO.load("models/ppo_rocket_stage1", env=env)
```

Models are saved as `.zip` files containing network weights + hyperparameters.
Always save after each training stage — if training crashes you don't lose
the earlier checkpoint.

---

## 8. GPU vs CPU — What Changes

With CUDA available (RTX 4050):
- PyTorch moves tensors to GPU automatically via SB3's `device="cuda"` setting
- Matrix multiplications (forward + backward pass) run on GPU
- Data collection (simulation stepping) still runs on CPU

For our problem size (9-dim obs, 1-dim action, 64x64 network):
- GPU speedup is ~3-8x over CPU for training steps
- Total training time: ~5-15 min for 500K steps on GPU
- The simulation loop itself (Python/NumPy) is the bottleneck, not the GPU

**When GPU matters more:** bigger networks, pixel-based observations,
image-based environments. For our tabular rocket state, the 4050 is overkill
but still meaningfully faster.

---

## 9. RTX 4050 — Specifications Relevant to This Project

| Spec | Value | Relevance |
|---|---|---|
| VRAM | 6 GB | Our model uses <200 MB — no constraint |
| CUDA cores | 2560 | Plenty for small policy networks |
| Architecture | Ada Lovelace | CUDA 12.1 supported, PyTorch 2.x compatible |
| Expected training speed | ~100K steps/min | 500K steps ≈ 5 min |

This GPU is sufficient for all phases of AI-RPO including PINN training later.

### Why CPU is faster than GPU for this specific task — the detailed explanation

This question comes up every time someone first uses RL. The answer requires
understanding what a GPU actually does.

**What a GPU is:** Thousands of small cores that do the same operation in parallel.
A matrix multiply `W × batch` where W is 64×64 and batch is 128 vectors:
the GPU does all 128 rows simultaneously. That parallelism is the speedup.

**Our policy network:**
```
Layer 1: 10 → 64    (640 weights)
Layer 2: 64 → 64    (4,096 weights)
Output:  64 → 1     (64 weights)
Total: ~5,000 parameters
```

GPT-2 small has 117 million parameters. ImageNet CNNs have 25+ million.
Ours has 5,000. The GPU's parallel cores are 99% idle during our forward pass.

**The real bottleneck:** The training loop is:
```
Step 1: Collect 4096 timesteps by running the Python simulation   ← this is slow
Step 2: Run the PPO update (matrix math on 64x64 network)         ← this is fast
Repeat
```
Step 1 is pure Python/NumPy on CPU. GPU cannot touch it.
Step 2 is so small the GPU's data transfer overhead (CPU RAM → VRAM via PCIe bus)
costs more time than the actual computation would on CPU.

SB3 itself warns you: "You are trying to run PPO on the GPU, but it is primarily
intended to run on the CPU when not using a CNN policy."

**When GPU matters in this project:**
Phase 5 — PINNs. The PINN is a larger network trained on large batches of
collocation points, and the loss requires computing derivatives through the
physics residual via autograd. That is exactly the large-batch parallel gradient
computation that GPUs excel at. The RTX 4050 becomes genuinely important there.

---

## 10. v3 Training — What Changed and Why

*(This section records the fix pass that resolved BUG-011 — agent saturating angle bounds.)*

### What was wrong in v2 (600K steps, 9D observation)

The v2 agent had one critical blindspot: it could not see its own thrust angle.
The action space is a **delta** (±5°/step). If the agent always commands +5° but
never sees the current angle, it keeps pressing +5° even when already pinned at 90°
— because nothing in the 9-number observation changes to tell it the bound was hit.
The result: agent snaps to 90°, stays there, error grows monotonically to 14 km.

This is a classic **partial observability bug**. The agent's observation was
incomplete — it was missing a key piece of its own state.

**Lesson for any RL project:** If the agent controls something incrementally
(delta-based actions), that thing MUST be in the observation. Otherwise the
agent cannot close the loop on its own control state.

### What changed in v3

| Change | Why |
|---|---|
| Added thrust angle as 10th observation dimension | Agent can now see where it is in the control range |
| n_steps: 2048 → 4096 | Larger rollout buffer → better credit assignment over long episodes |
| stage1_steps: 200K → 400K | More time on the simple case before introducing full randomization |
| total_steps: 500K → 1M | More total training — v2 was still improving at 600K |
| Linear LR decay: 3e-4 → 1e-5 | Larger steps early (exploration), smaller steps late (fine-tuning) |
| ent_coef: 0.01 → 0.005 | Reduce exploration pressure so policy commits faster to a solution |
| VecNormalize in evaluate.py | Use the exact same normalisation pipeline as training — no manual pkl math |

### v3 results

| Metric | v2 (broken baseline) | v3 (fixed) | Target |
|---|---|---|---|
| Agent apogee | 20.6 km | 31.2 km | ~30.8 km (match baseline) |
| Baseline apogee | 30.8 km | 30.8 km | — |
| Max tracking error | 14.3 km | 0.56 km | < 2 km |
| Mean tracking error | 7.7 km | 0.25 km | < 0.5 km |
| Angle range | always 10°–90° (saturated) | 10°–90° (used, not stuck) | smooth curve |

Mean tracking error of 250m over a 30km flight = **0.8% relative error.**
For an unconstrained PPO baseline, this is the expected performance level for a
well-configured first run.

---

## 11. What the v3 Evaluation Plot Is Actually Showing

*(This is the honest analysis, not just the headline number. Important for paper methodology.)*

Look at the bottom-left panel: **Agent Control Actions (Thrust Angle)**.

The agent:
1. Starts at 90° (vertical — correct for launch)
2. Holds near 90° through burnout (~25s) — correct, thrust is vertical
3. Drops sharply from ~90° to ~10° after burnout
4. Stays pinned near 10° for most of the coast phase (25s–100s)
5. Briefly recovers late in the flight

This is **not physically elegant control.** A real gravity-turn manoeuvre would
produce a smooth continuous pitch-over starting shortly after launch. What we have
is: "hold vertical during burn, then snap to minimum angle and hold."

### Why the error is still small despite this

During the coasting phase (after burnout at ~25s), there is no thrust.
The rocket's trajectory is ballistic — governed entirely by gravity and drag,
not by the thrust angle. So the control angle literally does not matter for
trajectory accuracy during coasting. The agent discovered this: "after burnout,
the angle is irrelevant — just hold whatever."

The tracking error (0.25 km mean) is earned almost entirely from the 0–25s
powered phase, where the agent holds 90° correctly.

### What this means for the paper

This is actually the **perfect setup for Phase 5 (PINNs).**

The unconstrained PPO does not know that slamming to 10° is physically
unrealistic — it just knows the reward signal doesn't punish it.
When we add a PINN loss term that penalises physically inconsistent trajectories,
the agent will be forced to learn the smooth pitch-over. The contrast between:

> "Unconstrained PPO: snaps angle, gets lucky with ballistics"
> "PINN-constrained PPO: smooth continuous control, physically consistent"

...is exactly the contribution of Phase 5. The current v3 result is the
**baseline to beat**, not the final result.

[PAPER OPPORTUNITY: Section in results — Table 1 shows PPO-only vs PINN+PPO
side by side. The control curve comparison is a compelling figure.]

---

## 12. The Training Curve Problem (BUG-013) — Why It Matters for Papers

Our custom `ProgressCallback` reads `logger.name_to_value["train/ep_rew_mean"]`
inside `_on_rollout_end`. But SB3 writes episode stats AFTER `_on_rollout_end`
fires — so our callback always reads stale/missing data and prints "no ep yet"
for the entire run.

The policy is training correctly (KL > 0 throughout, final eval shows 0.25 km
error), but we have **no reward curve to show** for the paper.

### Why training curves matter for a research paper

A paper about RL needs to show:
1. **That learning happened** — reward increasing over time proves the policy improved
2. **Convergence behaviour** — where did it plateau? How stable was Stage 1 vs Stage 2?
3. **Algorithm comparison** — if we compare PPO vs DDPG later (Paper #9), we need
   both curves on the same axes

Without a visible training curve, reviewers will ask: "How do you know it converged
and not just got lucky on the evaluation seed?"

TensorBoard logs ARE being written (we set `tensorboard_log="logs/tensorboard"`)
so the data exists — we just cannot see it from our custom callback output.

### The fix (Phase 4 caveat, to be done before Phase 5 paper submission)

Two options:

**Option A — Fix the callback timing (correct, requires understanding SB3 internals):**
Move reward reading from `_on_rollout_end` to `_on_step()`, or use
`self.model.ep_info_buffer` (SB3's internal deque that accumulates episode stats
from all envs). This is accessible correctly at any time.

```python
# In ProgressCallback._on_step() or _on_rollout_end():
if len(self.model.ep_info_buffer) > 0:
    ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
```

**Option B — Use SB3's built-in logging (simpler, already working):**
Set `verbose=1` in the PPO constructor. SB3 prints its own formatted training
stats every `log_interval` updates. Not as custom, but guaranteed correct.

**Recommendation:** Fix Option A (ep_info_buffer) — it keeps our custom
callback format and produces the correct reward curve data for paper figures.
Do this before Phase 5 training so both Phase 4 and Phase 5 curves are clean.

---

## 13. Phase 4 Closure — What Is Done, What Is the Paper Claim

### What is done
- Physics simulation (Phase 1–2): validated, 19/19 tests passing
- RL environment (Phase 3): Gymnasium-compliant, 10D observation, reward balanced
- PPO training (Phase 4): 1M-step curriculum run, v3 model saved
- Evaluation: 0.25 km mean error on held-out episode, evaluation pipeline correct

### What is the honest paper claim from Phase 4

> "We trained a PPO-based trajectory tracking policy for a 2D rocket with
> domain-randomized parameters and wind disturbances. After 1M training steps
> with curriculum learning, the policy achieves a mean altitude tracking error
> of 250m (0.8% of apogee altitude = 30.8 km) on a nominal evaluation episode.
> The policy uses a 10-dimensional observation including thrust angle, with
> VecNormalize observation normalisation. This constitutes the unconstrained
> RL baseline for comparison against the PINN-constrained policy in Phase 5."

### What Phase 4 is NOT claiming
- That the control strategy is physically elegant (the pitch-over is abrupt)
- That the policy generalises to 3D (still 2D only)
- That it has been tested across many randomized evaluation seeds (only seed=99)
- That training curves show clean convergence (logging bug exists)

All four of these will be addressed: 3 by PINNs/Phase 5, 1 by the callback fix.

---

## 14. Project Uniqueness — Why This Is Novel Research

*(What was said in session — captured here so it's never lost.)*

Most published work falls into one of three buckets and does not cross them:

| Existing bucket | What they do | Gap |
|---|---|---|
| Pure RL rocket control | PPO/SAC for trajectory tracking | No physics constraint enforcement; policies can violate physics |
| PINNs for rocket trajectory prediction | Physics-informed NN for forward simulation | No RL, no control, no adaptation — static trajectory |
| Classical GNC | PID / optimal control | No learning, cannot adapt to manufacturing variation or wind uncertainty |

**AI-RPO combines all three simultaneously:**

1. PPO RL for adaptive closed-loop guidance under uncertainty
2. PINNs to enforce physical consistency as a differentiable constraint
3. Domain randomization to simulate manufacturing tolerances + atmospheric uncertainty
4. Curriculum learning to make training stable

The combination — **PINN-constrained PPO with domain randomization for physically
consistent adaptive rocket guidance** — is the contribution. No published paper
has exactly this architecture for rocket guidance.

This positions Paper #12 (Hybrid PINN-RL architecture, from skills.md) as the
**primary contribution paper**. Papers #1–11 are supporting evidence, methodology,
and ablation studies.

### The strongest paper narrative

```
Phase 1+2:  We built a physically accurate simulation with quantified uncertainty
Phase 3+4:  We trained an RL baseline — it works but control is physically crude
Phase 5:    We add PINNs — control becomes physically consistent, error drops further
            This is the contribution. Side-by-side comparison is the key figure.
Phase 6:    Dashboard making the full system demonstrable
```

Each phase is independently publishable (methodology papers, benchmarking papers),
and Phase 5 is the novel architecture paper.
