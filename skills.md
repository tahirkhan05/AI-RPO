# AI-RPO Skills & Knowledge Map

A living document. Updated as we build each module.

---

## Phase 1 — Physics Simulation Core (Current)

### Concepts to learn
- Equations of motion for a rocket (Newton's 2nd law in 2D/3D)
- Thrust force, drag force, gravity
- Variable mass (Tsiolkovsky rocket equation)
- Euler angles vs quaternions for orientation
- Numerical integration: Euler method → Runge-Kutta (RK4)

### Code skills to build
- Structuring a physics simulation loop in Python
- Using SciPy `solve_ivp` for ODE integration
- Plotting trajectory with Matplotlib

### Paper opportunities from Phase 1
- Benchmark study: RK4 vs Euler vs adaptive solvers for rocket simulation accuracy
- Effect of variable mass modeling fidelity on trajectory prediction

---

## Phase 2 — Uncertainty & Environment Modeling (Upcoming)

### Concepts to learn
- Atmospheric models (ISA — International Standard Atmosphere)
- Wind disturbance modeling
- Thrust fluctuation noise
- Domain randomization for sim-to-real transfer

### Paper opportunities from Phase 2
- Domain randomization strategies for aerodynamic uncertainty
- Atmospheric model comparison for low-altitude rocket guidance

---

## Phase 3 — RL Environment Design (Upcoming)

### Concepts to learn
- Gymnasium (OpenAI Gym) environment API
- State space and action space design
- Reward function engineering
- PPO algorithm mechanics (clip objective, value function, entropy bonus)

### Paper opportunities from Phase 3
- Reward shaping strategies for fuel-efficient rocket guidance
- PPO vs DDPG for continuous thrust-vectoring control

---

## Phase 4 — AI Training (Upcoming)

### Concepts to learn
- Policy gradient methods
- Neural network architecture for control policies
- Hyperparameter tuning (learning rate, clip epsilon, GAE lambda)
- Training stability and convergence diagnostics

### Paper opportunities from Phase 4
- Sample efficiency comparison of RL algorithms on rocket guidance tasks
- Effect of network architecture depth on policy convergence

---

## Phase 5 — PINNs Integration (Upcoming)

### Concepts to learn
- What Physics-Informed Neural Networks are
- How to embed differential equations as loss terms
- Trade-off between data-driven and physics-driven learning

### Paper opportunities from Phase 5
- PINNs vs unconstrained NN for physically consistent trajectory prediction
- Hybrid PINN-RL architecture for constraint-aware control

---

## Phase 6 — Dashboard & Demo (Upcoming)

### Concepts to learn
- Plotly Dash or Streamlit for interactive visualization
- Real-time plot updates from simulation data
- API design for sim-to-UI data streaming

---

## Master Paper Opportunity List

| # | Topic | Phase | Type | Notes file |
|---|---|---|---|---|
| 1 | RK4 vs Euler vs adaptive solvers — accuracy/cost benchmark | 1 | Benchmarking | 01_rocket_physics |
| 2 | Simulation fidelity (constant g vs ISA) impact on RL policy | 1 | Analysis | 01_rocket_physics |
| 3 | Dryden vs OU vs white noise turbulence — policy robustness | 2 | Benchmarking | 03_uncertainty |
| 4 | Rocket parameter randomization as proxy for manufacturing tolerances | 2 | Methodology | 03_uncertainty |
| 5 | Curriculum domain randomization — narrow-to-wide vs full-random | 2 | Methodology | 03_uncertainty |
| 6 | Trajectory dispersion analysis — envelope width vs randomization intensity | 2 | Analysis | 03_uncertainty |
| 7 | Observation space ablation — which state variables matter most | 3 | Analysis | 04_rl_environment |
| 8 | Reward function ablation — impact of each component on convergence | 3 | Methodology | 04_rl_environment |
| 9 | PPO vs DDPG vs SAC — convergence, sample efficiency, robustness | 3-4 | Benchmarking | 04_rl_environment |
| 10 | Adaptive reward curriculum for RL rocket guidance | 4 | Methodology | 04_rl_environment |
| 11 | PINNs vs unconstrained NN for physically consistent trajectory prediction | 5 | Comparison | (future notes) |
| 12 | Hybrid PINN-RL architecture for constraint-aware control | 5 | Novel Architecture | (future notes) |
