# Phase 10 — Streamlit Dashboard

*Reference file. Covers what the dashboard shows, why each panel matters,
and the implementation plan.*

---

## 1. Purpose

The dashboard is the "front window" of the entire project. Everything we've
built — physics simulation, PINN reference, EKF estimation, PPO agent,
LSTM forecaster — runs invisibly. The dashboard makes it visible.

Three audiences:
1. **You (the researcher):** understand what the agent is doing in real time
2. **Paper reviewers:** a figure showing the live system is worth 1000 words
3. **Demo / presentation:** click a button, watch a rocket fly

---

## 2. Layout — Four Panels

```
┌─────────────────────────────────────────────────────┐
│  AI-RPO Dashboard          [Run Episode] [Settings] │
├──────────────────┬──────────────────────────────────┤
│  PANEL 1         │  PANEL 2                         │
│  Altitude vs     │  Phase space                     │
│  Time            │  (y vs vy)                       │
│  (agent, PINN    │                                  │
│   target, EKF)   │                                  │
├──────────────────┼──────────────────────────────────┤
│  PANEL 3         │  PANEL 4                         │
│  LSTM forecast   │  Reward breakdown                │
│  + warnings      │  (r_alt, r_vel, r_fuel,          │
│                  │   r_smooth, r_physics)            │
└──────────────────┴──────────────────────────────────┘
```

### Panel 1 — Altitude vs Time
- Agent trajectory (solid blue)
- PINN reference trajectory (dashed orange)
- EKF estimate (dotted green, slightly offset)
- Tracking error band (shaded region)
- Burnout marker (vertical dashed line)

**Why:** This is the primary performance metric. How close does the agent
follow the physics-informed reference?

### Panel 2 — Phase Space (y vs vy)
- Agent path through altitude-velocity space
- Reference path
- Start (green dot) and end (red dot)

**Why:** Phase space shows the full flight character at a glance.
A good agent traces a smooth curve from bottom-left to apogee and back.
A bad agent oscillates — visible as loops in phase space.

### Panel 3 — LSTM Deviation Forecast
- Running forecast: predicted deviation at t+1s (blue line)
- Actual deviation (orange line)
- Warning threshold line (red dashed)
- Warning events highlighted (red shading)

**Why:** This is the novel Phase 9 contribution made visual.
The forecast anticipates the deviation — you can see it rise before
the actual orange line reaches the threshold.

### Panel 4 — Reward Breakdown
- Stacked area chart: r_alt, r_vel, r_fuel, r_smooth, r_physics
- Cumulative reward curve overlay

**Why:** Lets you see which reward component dominates.
After burnout, r_fuel goes to zero. Near apogee, r_alt dominates.

---

## 3. Sidebar Controls

- **Model selector:** v3, v4, v5, v6, v6ext (dropdown)
- **Environment:** Fixed / Randomized (toggle)
- **Seed:** integer input
- **Physics guided:** checkbox
- **Use EKF:** checkbox
- **LSTM overlay:** checkbox
- **Run Episode:** button → triggers simulation

---

## 4. Implementation

Single file: `dashboard/app.py`

Uses Streamlit's `st.plotly_chart` for interactive plots (zoom, hover).
Episode runs synchronously on button click — no background threading needed
(episodes are fast: ~2s for 2D, ~5s for 3D).

LSTM overlay: maintain a rolling window of the last 20 steps, query model
at each step, plot forecast alongside actual deviation.

---

## 5. Implementation Plan

1. `dashboard/app.py` — main Streamlit app
2. `dashboard/run_episode.py` — wrapper that runs one episode and returns
   all data needed for plotting (t, y, vy, target_y, ekf_y, reward_components,
   lstm_forecasts, warnings)
3. `notes/11_phase10_dashboard.md` — this file

---

## 6. What Success Looks Like

- `streamlit run dashboard/app.py` opens in browser
- Clicking "Run Episode" shows all 4 panels updating
- LSTM panel shows forecast rising before actual deviation
- Can switch between v4/v5/v6ext and see different tracking quality
