# AI-RPO Project — Learning Flow

A complete, ground-up map of every file in this project and how to study it.

---

## How the Project Was Built (Phases)

```
Phase 1  →  Physics Simulation (2D)
Phase 2  →  Uncertainty & Domain Randomization
Phase 3  →  RL Environment (Gymnasium)
Phase 4  →  PPO Training (v1 → v6 extended)
Phase 5  →  PINNs (Physics-Informed Neural Networks)
Phase 6  →  PINN + RL Integration
Phase 7  →  EKF (Extended Kalman Filter)
Phase 8  →  3D Upgrade (physics, env, pinn)
Phase 9  →  LSTM Forecaster
Phase 10 →  Live Dashboard
```

---

## Step-by-Step Learning Path

---

### Step 1 — Understand the Goal (Before Any Code)

| File | What it tells you |
|---|---|
| `CLAUDE.md` | Project rules, tech stack, pace of work, what NOT to do |
| `skills.md` | Full concept map — what you learn in each phase |
| `context/AI-Integrated Adaptive Path Optimization System...docx` | The original project spec — the "why" |
| `context/conversations.json` | Full chat history of how every decision was made |
| `context/conv_dump.txt` / `full_conv.txt` | Plain-text version of the same conversations |
| `context/AI-RPS.jsonl` | Machine-readable JSONL export of all conversations (used for bulk analysis) |
| `AI-RPS_System_Architecture.svg` | Bird's-eye diagram of the whole system |
| `notes/00_problems_and_decisions_log.md` | Every bug, wrong assumption, and design decision — read alongside each phase |

---

### Step 2 — Physics Core (Phase 1)

**Read first:**
- `notes/01_rocket_physics.md` — equations of motion, thrust, drag, RK4 integration
- `notes/02_reading_simulation_output.md` — **how to read every plot panel** (critical for beginners)

**Code files:**
| File | What it does |
|---|---|
| `simulation/config.py` | Defines `DEFAULT_ROCKET` and `DEFAULT_SIM` — all tunable parameters |
| `simulation/physics.py` | 2D equations of motion (Newton's 2nd law, thrust, drag, gravity) |
| `simulation/trajectory.py` | Runs the RK4 simulation loop, returns a result object |
| `simulation/visualize.py` | Plotting helpers used by all evaluate scripts |

**Tests:**
```bash
python -m pytest tests/test_physics.py -v
```

**Run & see output:**
```bash
python run_simulation.py
```
- Console prints: apogee (km), max speed (m/s), burnout time (s), downrange (km)
- Plot window opens: `output_summary.png`
  - Top-left: altitude vs time (the hill shape — rise, apogee, descent)
  - Top-right: speed vs time (double-hump — burn peak, dip at apogee, descent hump)
  - Bottom-left: mass vs time (straight drop to dry mass, then flat)
  - Bottom-right: flight path altitude vs downrange (the arch the rocket traces through space)

> See `notes/02_reading_simulation_output.md` for a full explanation of every panel and a checklist of what "physically correct" looks like.

---

### Step 3 — Uncertainty & Sensors (Phase 2)

**Read first:**
- `notes/03_uncertainty_and_domain_randomization.md` — atmospheric models, wind noise, domain randomization

**Code files:**
| File | What it does |
|---|---|
| `simulation/wind.py` | 2D wind disturbance (Dryden turbulence / OU process) |
| `simulation/sensors.py` | Adds realistic noise on top of the true simulation state |
| `simulation/domain_randomization.py` | Randomly varies rocket parameters each episode (mass, thrust, drag) |

**Tests:**
```bash
python -m pytest tests/test_wind.py -v
```

**Output:**
- `output_dispersion.png` — trajectory spread when domain randomization is active (many runs overlaid)
- No standalone run script; these modules are used automatically when the RL environment runs.

---

### Step 4 — RL Environment (Phase 3)

**Read first:**
- `notes/04_reinforcement_learning_environment.md` — state space, action space, reward function design

**Code files:**
| File | What it does |
|---|---|
| `simulation/env.py` | 2D Gymnasium environment — defines `reset()`, `step()`, observation, reward, done |
| `simulation/target_trajectory.py` | Generates the reference path the agent is trying to follow (2D) |

**Tests:**
```bash
python -m pytest tests/test_env.py -v
```

---

### Step 5 — PPO Training (Phase 4)

**Read first:**
- `notes/05_ppo_training.md` — PPO algorithm mechanics, reward shaping, two-stage training strategy

**Code files (read in this order — each version adds something new):**
| File | What changed |
|---|---|
| `training/config.py` | Shared hyperparameter config used by all train scripts |
| `training/callbacks.py` | Custom SB3 callbacks — saves checkpoints, logs reward to file |
| `training/train.py` | v1/v2 — baseline PPO, single stage |
| `training/train_v4.py` | v4 — two-stage curriculum (stage 1 easy, stage 2 full task) |
| `training/train_v5.py` | v5 — improved reward shaping |
| `training/train_v6.py` | v6 — further tuned hyperparameters |
| `training/train_v6_extended.py` | v6ext — longest run, best model (3M+ steps) |

**Run the best (latest) training:**
```bash
python -m training.train_v6_extended
```

**Logs produced:**
| Log file | What it contains |
|---|---|
| `logs/train_v4.log` | Per-episode reward, length, loss for v4 run |
| `logs/train_v5.log` | Same for v5 |
| `logs/train_v6.log` | Same for v6 |
| `logs/train_v6_extended.log` | Same for v6ext — most data here |
| `logs/tensorboard/` | Full tensorboard event files for all runs |

**View tensorboard:**
```bash
tensorboard --logdir logs/tensorboard
```
Then open `http://localhost:6006` in a browser.

**Saved models (in order of training progression):**
| Model file | What it is |
|---|---|
| `models/ppo_stage1.zip` + `vecnorm_stage1.pkl` | v1 stage 1 policy + normalizer |
| `models/ppo_final.zip` + `vecnorm_final.pkl` | v1 final policy |
| `models/ppo_final_v2.zip` + `vecnorm_final_v2.pkl` | v2 final policy |
| `models/ppo_v3_stage1.zip` + `vecnorm_v3_stage1.pkl` | v3 stage 1 |
| `models/ppo_v3_final.zip` + `vecnorm_v3_final.pkl` | v3 final |
| `models/ppo_v4_stage1.zip` + `vecnorm_v4_stage1.pkl` | v4 stage 1 |
| `models/ppo_v4_final.zip` + `vecnorm_v4_final.pkl` | v4 final |
| `models/ppo_v5_stage1/final.zip` + `vecnorm_v5_*.pkl` | v5 stage 1 / final |
| `models/ppo_v6_stage1/final.zip` + `vecnorm_v6_*.pkl` | v6 stage 1 / final |
| `models/ppo_v6ext_stage1/final.zip` + `vecnorm_v6ext_*.pkl` | **Best model** — v6 extended |
| `models/checkpoint_100000.zip` → `checkpoint_3015808.zip` | Mid-training snapshots (step counts in name) |

> **Rule:** Always load a `.zip` policy together with its matching `vecnorm_*.pkl`. The normalizer was fitted during training and must be used at evaluation time too.

**Evaluate & see output:**
```bash
python -m training.evaluate_v4
```
- Plot: `output_trained_agent.png` — agent trajectory overlaid on target trajectory
- Plot: `output_v3_vs_v4_benchmark.png` — side-by-side benchmark of v3 vs v4 agents

---

### Step 6 — PINNs (Phase 5)

**Read first:**
- `notes/06_pinns.md` — what PINNs are, how differential equations become loss terms, smoke tests

**Code files:**
| File | What it does |
|---|---|
| `simulation/pinn.py` | 2D PINN — neural net trained with physics residual loss |
| `simulation/pinn_param.py` | Parametric PINN — learns a family of trajectories across different configs |
| `training/train_pinn.py` | Trains the 2D PINN |
| `training/train_pinn_param.py` | Trains the parametric PINN |
| `training/evaluate_pinn.py` | Evaluates 2D PINN, generates trajectory + residual plots |
| `training/evaluate_pinn_param.py` | Evaluates parametric PINN, generates generalisation plots |

**Run:**
```bash
python -m training.train_pinn
python -m training.train_pinn_param
```

**Logs produced:**
| File | What it contains |
|---|---|
| `logs/pinn_loss_curve.png` | Training loss over epochs — physics residual + data loss |
| `logs/pinn_param_loss_curve.png` | Parametric PINN training loss |
| `logs/pinn_param_training.log` | Text log of parametric PINN training progress |

**Evaluate:**
```bash
python -m training.evaluate_pinn
python -m training.evaluate_pinn_param
```

**Output plots:**
| File | What it shows |
|---|---|
| `output_pinn_trajectory.png` | PINN-predicted trajectory vs ground truth simulation |
| `output_pinn_residuals.png` | Physics residual error — how well PINN obeys the equations |
| `output_pinn_param_trajectory.png` | Parametric PINN trajectory for a sampled config |
| `output_pinn_param_residuals.png` | Residuals for the parametric PINN |
| `output_pinn_param_generalisation.png` | Generalisation across unseen rocket configs |

**Saved models:**
| File | What it is |
|---|---|
| `models/pinn_v1.pt` → `pinn_v4.pt` | 2D PINN checkpoints across training runs |
| `models/pinn_smoke.pt` | Smoke-test PINN (minimal training, used to verify the pipeline works) |
| `models/pinn_param_v1.pt` | Parametric PINN — best version |
| `models/pinn_param_smoke.pt` | Smoke-test parametric PINN |

---

### Step 7 — PINN + RL Integration (Phase 6)

**Read first:**
- `notes/07_phase6_integration.md` — how the PINN and the RL agent are used together

**Code files:**
| File | What it does |
|---|---|
| `simulation/rollout.py` | Runs a full episode: agent acts, PINN provides physics residuals alongside simulation |
| `training/evaluate.py` | Full evaluation — loads policy + PINN, runs rollout, generates comparison plots |

**Evaluate:**
```bash
python -m training.evaluate
```
- Output: trajectory comparison plots (PINN-guided agent vs baseline)

---

### Step 8 — EKF State Estimation (Phase 7)

**Read first:**
- `notes/08_phase7_ekf.md` — Kalman filter theory, predict step, update step, noise matrices

**Code files:**
| File | What it does |
|---|---|
| `simulation/ekf.py` | 2D Extended Kalman Filter — estimates true state from noisy sensor data |
| `training/evaluate_ekf.py` | Runs EKF alongside simulation, plots estimated vs true state |

**Tests:**
```bash
python -m pytest tests/test_ekf.py -v
```

**Evaluate:**
```bash
python -m training.evaluate_ekf
```

---

### Step 9 — 3D Upgrade (Phase 8)

**Read first:**
- `notes/09_phase8_3d.md` — 3D equations of motion, 6-DOF, quaternion vs Euler angles

**Code files:**
| File | What it does |
|---|---|
| `simulation/physics3d.py` | 3D equations of motion (6-DOF) |
| `simulation/env3d.py` | 3D Gymnasium environment |
| `simulation/wind3d.py` | 3D wind disturbance model |
| `simulation/ekf3d.py` | 3D Extended Kalman Filter |
| `simulation/pinn3d.py` | 3D PINN |
| `simulation/target_trajectory3d.py` | 3D reference trajectory generator |
| `training/train_pinn3d.py` | Trains the 3D PINN |
| `training/evaluate_3d.py` | Evaluates the 3D agent/PINN |

**Logs & outputs:**
| File | What it shows |
|---|---|
| `logs/train_pinn3d.log` | 3D PINN training progress |
| `logs/benchmark_3d.png` | 3D agent performance benchmark |
| `output_v3_vs_v4_benchmark.png` | Side-by-side: v3 (2D) agent vs v4 (improved) agent |

**Saved models:**
| File | What it is |
|---|---|
| `models/pinn3d_param_v1.pt` | 3D parametric PINN |

---

### Step 10 — LSTM Forecaster (Phase 9)

**Read first:**
- `notes/10_phase9_lstm.md` — LSTM architecture, sequence prediction, forecasting horizon

**Code files:**
| File | What it does |
|---|---|
| `simulation/lstm_dataset.py` | Prepares rollout data into sequences for LSTM training |
| `simulation/lstm_forecaster.py` | LSTM model definition |
| `training/train_lstm.py` | Trains the LSTM forecaster |
| `training/evaluate_lstm.py` | Evaluates LSTM, plots forecast vs true trajectory |

**Logs & outputs:**
| File | What it shows |
|---|---|
| `logs/train_lstm.log` | LSTM training loss per epoch |
| `logs/lstm_training_curve.png` | Loss curve plot |
| `logs/lstm_evaluation.png` | Forecast accuracy plots |

**Saved model:**
- `models/lstm_forecaster_v1.pt`

---

### Step 11 — Live Dashboard (Phase 10)

**Read first:**
- `notes/11_phase10_dashboard.md` — Plotly Dash architecture, live update pattern

**Code files:**
| File | What it does |
|---|---|
| `dashboard/app.py` | Plotly Dash web app — displays live trajectory, reward, and state plots |
| `dashboard/run_episode.py` | Runs a live episode and streams data to the dashboard |

**Run:**
```bash
python dashboard/app.py
```
Open `http://localhost:8050` in a browser.

**Log:**
- `logs/dashboard.log` — startup and episode run events

---

## Complete File Index

Every file in the project, grouped by folder.

### Root
| File | Purpose |
|---|---|
| `run_simulation.py` | Entry point — run bare physics, print stats, show plot |
| `CLAUDE.md` | Project rules and guidelines |
| `skills.md` | Phase-by-phase concept and skill map |
| `flow.md` | This file |
| `AI-RPS_System_Architecture.svg` | Full system architecture diagram |
| `output_summary.png` | Phase 1 output — baseline trajectory 4-panel plot |
| `output_dispersion.png` | Phase 2 output — trajectory spread under domain randomization |
| `output_trained_agent.png` | Phase 4 output — trained agent vs target trajectory |
| `output_pinn_trajectory.png` | Phase 5 output — PINN predicted vs true trajectory |
| `output_pinn_residuals.png` | Phase 5 output — PINN physics residual error |
| `output_pinn_param_trajectory.png` | Phase 5 output — parametric PINN trajectory |
| `output_pinn_param_residuals.png` | Phase 5 output — parametric PINN residuals |
| `output_pinn_param_generalisation.png` | Phase 5 output — parametric PINN generalisation |
| `output_v3_vs_v4_benchmark.png` | Phase 8 output — v3 vs v4 agent comparison |

### simulation/
| File | Phase | Purpose |
|---|---|---|
| `config.py` | 1 | Rocket + sim config dataclasses |
| `physics.py` | 1 | 2D equations of motion |
| `trajectory.py` | 1 | 2D simulation loop (RK4) |
| `visualize.py` | 1 | Matplotlib helpers |
| `wind.py` | 2 | 2D wind disturbance model |
| `sensors.py` | 2 | Noisy sensor model |
| `domain_randomization.py` | 2 | Randomize rocket params per episode |
| `env.py` | 3 | 2D Gymnasium RL environment |
| `target_trajectory.py` | 3 | 2D reference trajectory generator |
| `pinn.py` | 5 | 2D PINN |
| `pinn_param.py` | 5 | Parametric PINN (2D) |
| `rollout.py` | 6 | Full episode runner (policy + PINN) |
| `ekf.py` | 7 | 2D Extended Kalman Filter |
| `physics3d.py` | 8 | 3D equations of motion |
| `env3d.py` | 8 | 3D Gymnasium environment |
| `wind3d.py` | 8 | 3D wind disturbance model |
| `ekf3d.py` | 8 | 3D Extended Kalman Filter |
| `pinn3d.py` | 8 | 3D PINN |
| `target_trajectory3d.py` | 8 | 3D reference trajectory generator |
| `lstm_dataset.py` | 9 | Sequence dataset for LSTM |
| `lstm_forecaster.py` | 9 | LSTM model definition |

### training/
| File | Phase | Purpose |
|---|---|---|
| `config.py` | 4 | Shared PPO hyperparameters |
| `callbacks.py` | 4 | SB3 callbacks — checkpointing, logging |
| `train.py` | 4 | PPO v1/v2 training script |
| `train_v4.py` | 4 | PPO v4 — two-stage curriculum |
| `train_v5.py` | 4 | PPO v5 — improved reward |
| `train_v6.py` | 4 | PPO v6 — tuned hyperparams |
| `train_v6_extended.py` | 4 | PPO v6ext — final best training run |
| `evaluate.py` | 6 | Evaluate PINN + RL integration |
| `evaluate_v4.py` | 4 | Evaluate v4 agent, produce output plot |
| `evaluate_pinn.py` | 5 | Evaluate 2D PINN, generate trajectory + residual plots |
| `evaluate_pinn_param.py` | 5 | Evaluate parametric PINN, generate generalisation plot |
| `evaluate_ekf.py` | 7 | Evaluate EKF — estimated vs true state |
| `evaluate_3d.py` | 8 | Evaluate 3D agent/PINN |
| `train_pinn.py` | 5 | Train 2D PINN |
| `train_pinn3d.py` | 8 | Train 3D PINN |
| `train_pinn_param.py` | 5 | Train parametric PINN |
| `train_lstm.py` | 9 | Train LSTM forecaster |
| `evaluate_lstm.py` | 9 | Evaluate LSTM forecaster |

### tests/
| File | Phase | What it tests |
|---|---|---|
| `test_physics.py` | 1 | Physics engine — forces, mass burn, RK4 correctness |
| `test_wind.py` | 2 | Wind model output ranges and statistics |
| `test_env.py` | 3 | RL environment — reset, step, observation shape, reward sign |
| `test_ekf.py` | 7 | EKF predict/update cycle, covariance growth |

**Run all tests at once:**
```bash
python -m pytest tests/ -v
```

### notes/
| File | Phase | What it explains |
|---|---|---|
| `00_problems_and_decisions_log.md` | All | Every bug, fix, and design decision across all phases |
| `01_rocket_physics.md` | 1 | Physics equations, numerical integration |
| `02_reading_simulation_output.md` | 1 | How to read every plot panel — start here if confused by any output |
| `03_uncertainty_and_domain_randomization.md` | 2 | Wind, sensors, randomization |
| `04_reinforcement_learning_environment.md` | 3 | State, action, reward, Gymnasium API |
| `05_ppo_training.md` | 4 | PPO algorithm, training stages, diagnostics |
| `06_pinns.md` | 5 | PINN theory, residual loss, parametric extension |
| `07_phase6_integration.md` | 6 | PINN + RL joint usage |
| `08_phase7_ekf.md` | 7 | EKF theory and implementation |
| `09_phase8_3d.md` | 8 | 3D physics, 6-DOF, quaternions |
| `10_phase9_lstm.md` | 9 | LSTM for trajectory forecasting |
| `11_phase10_dashboard.md` | 10 | Dashboard architecture |

### logs/
| File | Phase | What it contains |
|---|---|---|
| `train_v4.log` | 4 | PPO v4 training — reward, episode length, loss per step |
| `train_v5.log` | 4 | PPO v5 training log |
| `train_v6.log` | 4 | PPO v6 training log |
| `train_v6_extended.log` | 4 | PPO v6ext training log — most complete |
| `pinn_loss_curve.png` | 5 | 2D PINN loss curve over epochs |
| `pinn_param_loss_curve.png` | 5 | Parametric PINN loss curve |
| `pinn_param_training.log` | 5 | Parametric PINN text training log |
| `train_pinn3d.log` | 8 | 3D PINN training log |
| `train_lstm.log` | 9 | LSTM training loss per epoch |
| `lstm_training_curve.png` | 9 | LSTM loss curve plot |
| `lstm_evaluation.png` | 9 | LSTM forecast accuracy |
| `benchmark_3d.png` | 8 | 3D agent benchmark chart |
| `dashboard.log` | 10 | Dashboard startup and episode events |
| `tensorboard/` | 4–9 | Full tensorboard event data for all training runs |

**View tensorboard:**
```bash
tensorboard --logdir logs/tensorboard
```
Then open `http://localhost:6006`.

**Tensorboard subfolder naming:** Each subfolder is one training run. The name encodes `stage` + `version` + `run number`.

| Folder pattern | Meaning |
|---|---|
| `stage1_1/`, `stage1_2/` … | v1/v2 stage 1, run attempts 1, 2 … |
| `stage2_0/` | v1/v2 stage 2 |
| `stage1_v3_1/`, `stage2_v3_0/` | v3 stage 1 / stage 2 |
| `stage1_v4_1/` … `stage2_v4_0/` | v4 stage 1 (3 attempts) / stage 2 |
| `stage1_v5_1/`, `stage2_v5_0/` | v5 stage 1 / stage 2 |
| `stage1_v6_1/`, `stage2_v6_0/` | v6 stage 1 / stage 2 |
| `stage1_v6ext_1/`, `stage2_v6ext_0/` | v6 extended stage 1 / stage 2 — **best run** |

Each subfolder contains one `events.out.tfevents.*` file — this is the raw data tensorboard reads.

### models/
| File | Phase | What it is |
|---|---|---|
| `ppo_stage1.zip` + `vecnorm_stage1.pkl` | 4 | PPO v1 — end of stage 1 |
| `ppo_final.zip` + `vecnorm_final.pkl` | 4 | PPO v1 — final |
| `ppo_final_v2.zip` + `vecnorm_final_v2.pkl` | 4 | PPO v2 — final |
| `ppo_v3_stage1/final.zip` + `vecnorm_v3_*.pkl` | 4 | PPO v3 stage 1 / final |
| `ppo_v4_stage1/final.zip` + `vecnorm_v4_*.pkl` | 4 | PPO v4 stage 1 / final |
| `ppo_v5_stage1/final.zip` + `vecnorm_v5_*.pkl` | 4 | PPO v5 stage 1 / final |
| `ppo_v6_stage1/final.zip` + `vecnorm_v6_*.pkl` | 4 | PPO v6 stage 1 / final |
| `ppo_v6ext_stage1/final.zip` + `vecnorm_v6ext_*.pkl` | 4 | **Best model** — v6 extended |
| `checkpoint_100000.zip` → `checkpoint_3015808.zip` | 4 | Mid-training snapshots |
| `pinn_v1.pt` → `pinn_v4.pt` | 5 | 2D PINN iterations |
| `pinn_smoke.pt` | 5 | Smoke-test PINN (pipeline verification only) |
| `pinn_param_v1.pt` | 5 | Parametric PINN — best version |
| `pinn_param_smoke.pt` | 5 | Smoke-test parametric PINN |
| `pinn3d_param_v1.pt` | 8 | 3D parametric PINN |
| `lstm_forecaster_v1.pt` | 9 | Trained LSTM forecaster |

---

## Debugging an Output You Don't Understand

1. Look up the output file in the **Complete File Index** above to find its phase.
2. Read the matching `notes/` file for that phase.
3. Find the `training/evaluate_*.py` script that generated it and read the code.
4. Check `notes/00_problems_and_decisions_log.md` for any bugs logged in that phase.
5. If a plot looks physically wrong, use the checklist in `notes/02_reading_simulation_output.md`.
