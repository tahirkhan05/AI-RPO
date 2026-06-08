# Rocket Physics — Equations of Motion

*Reference file. Return here any time to revisit the physics behind the simulation.*

---

## 1. The Core Idea

A rocket is just Newton's Second Law with one twist: **the mass is changing**.

```
F = m * a       →       a = F / m(t)
```

As the engine burns fuel, the rocket gets lighter. A lighter rocket accelerates more from the same thrust. This is why the last seconds of a burn feel the most violent — same force, much less mass.

---

## 2. The Three Forces

### 2.1 Thrust (T)
The engine expels gas backward. By Newton's Third Law, the rocket is pushed forward.

```
T = v_exhaust * (dm/dt)
```

- `v_exhaust` — how fast the exhaust gases leave the nozzle (m/s). Typically 2,000–4,000 m/s.
- `dm/dt` — mass flow rate (kg/s). How much fuel is burned per second.
- Thrust is a **vector** — it has direction. We control that direction (thrust vectoring).

**Worked example:**
- v_exhaust = 2500 m/s
- dm/dt = 2 kg/s
- T = 2500 × 2 = **5,000 N**

---

### 2.2 Drag (D)
Air resistance. It always opposes the direction of motion.

```
D = 0.5 * ρ * v² * Cd * A
```

- `ρ` (rho) — air density (kg/m³). At sea level: ~1.225. Drops as altitude increases.
- `v` — speed (m/s)
- `Cd` — drag coefficient (dimensionless). For a rocket: ~0.3–0.5
- `A` — cross-sectional area (m²). Front-facing area of the rocket.

**Key insight:** Drag grows with v². Double the speed → 4× the drag. This is why rockets pitch early to get out of dense atmosphere fast.

**Worked example:**
- ρ = 1.225 kg/m³, v = 200 m/s, Cd = 0.4, A = 0.05 m²
- D = 0.5 × 1.225 × 200² × 0.4 × 0.05
- D = 0.5 × 1.225 × 40000 × 0.4 × 0.05 = **490 N**

---

### 2.3 Gravity (g)
Always pulls straight down.

```
F_gravity = m(t) * g(h)
```

- `m(t)` — current mass (decreasing over time)
- `g(h)` — gravitational acceleration. At sea level: 9.81 m/s².
  Technically weakens with altitude: g(h) = 9.81 × (R_earth / (R_earth + h))²
  For altitudes < 50 km, treating g as constant (9.81) is a reasonable simplification.

---

## 3. Net Force and Acceleration

In 2D (x = downrange horizontal, y = altitude):

```
θ = thrust angle from vertical (0° = straight up, 90° = horizontal)

F_thrust_x = T * sin(θ)
F_thrust_y = T * cos(θ)

F_drag_x = -D * (vx / v)     ← drag opposes velocity direction
F_drag_y = -D * (vy / v)

F_gravity_x = 0
F_gravity_y = -m(t) * g

Net force:
  Fx = F_thrust_x + F_drag_x
  Fy = F_thrust_y + F_drag_y + F_gravity_y

Acceleration:
  ax = Fx / m(t)
  ay = Fy / m(t)
```

---

## 4. The Tsiolkovsky Rocket Equation

This is the most important equation in rocketry. It tells you how much the rocket's velocity can change given its fuel.

```
Δv = v_exhaust * ln(m_initial / m_final)
```

- `Δv` — total velocity change achievable (m/s)
- `v_exhaust` — exhaust velocity (m/s)
- `m_initial` — starting mass (rocket + full fuel)
- `m_final` — dry mass (rocket with no fuel)
- `ln` — natural logarithm

**Worked example:**
- v_exhaust = 2500 m/s
- m_initial = 100 kg (50 kg rocket + 50 kg fuel)
- m_final = 50 kg
- Δv = 2500 × ln(100/50) = 2500 × ln(2) = 2500 × 0.693 = **1,732 m/s**

**Key insight:** To double the Δv, you don't double the fuel — you need to square the mass ratio. This is why multi-stage rockets exist.

---

## 5. Mass Over Time

As fuel burns, mass decreases linearly:

```
m(t) = m_initial - (dm/dt) * t
```

Burn stops when fuel is exhausted: `m(t) >= m_dry`

---

## 6. Numerical Integration — Why We Need It

The equations above are **differential equations** — they describe *rates of change*, not direct positions. We can't solve them in one step; we have to step forward through time.

### 6.1 Euler Method (simple, less accurate)
```
v(t + dt) = v(t) + a(t) * dt
x(t + dt) = x(t) + v(t) * dt
```
- Easy to implement.
- Error accumulates — inaccurate for long simulations.

### 6.2 Runge-Kutta 4th Order — RK4 (what we'll use)
RK4 computes 4 estimates of the slope within each time step and takes a weighted average. Much more accurate for the same step size.

```
k1 = f(t,        y)
k2 = f(t + dt/2, y + dt/2 * k1)
k3 = f(t + dt/2, y + dt/2 * k2)
k4 = f(t + dt,   y + dt   * k3)

y(t+dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

**Intuition:** Instead of trusting just the slope at the start of the interval (Euler), RK4 "looks ahead" to the middle and end of the interval and blends all four estimates. Think of it as taking a more careful average of where the trajectory is heading.

[PAPER OPPORTUNITY #1: Benchmark RK4 vs Euler vs adaptive solvers (SciPy `solve_ivp` with `RK45`, `DOP853`) for accuracy and compute cost in rocket trajectory simulation]

---

## 7. 2D State Vector

Everything the simulation needs to track at each timestep:

```
state = [x, y, vx, vy, m]

x   — downrange position (m)
y   — altitude (m)
vx  — horizontal velocity (m/s)
vy  — vertical velocity (m/s)
m   — current mass (kg)
```

The simulation advances this state vector forward in time using RK4.

---

## 8. Typical Starter Values (Small Research Rocket)

| Parameter | Value |
|---|---|
| Initial mass (wet) | 100 kg |
| Dry mass (no fuel) | 50 kg |
| Fuel mass | 50 kg |
| Thrust | 5,000 N |
| Exhaust velocity | 2,500 m/s |
| Burn rate (dm/dt) | 2 kg/s → burn time = 25 s |
| Drag coefficient (Cd) | 0.4 |
| Cross-section area | 0.05 m² |
| Launch angle | 85° from horizontal (near-vertical) |

---

## 9. The "One Timestep" Mental Model

**The entire simulation is just one question asked over and over:**
> "Given everything about the rocket right now, how does it change in the next 0.01 seconds?"

Do that 10,000 times and you have the full trajectory.

Here is one real timestep traced from scratch with our starter values:

```
t = 0s
state = [x=0m, y=0m, vx=0 m/s, vy=0 m/s, m=100kg]

--- THRUST ---
Angle θ = 85° from horizontal (nearly straight up, slight lean)
  Thrust_x = 5000 * cos(85°) = 5000 * 0.0872 =  436 N
  Thrust_y = 5000 * sin(85°) = 5000 * 0.9962 = 4981 N

--- DRAG ---
v = 0 m/s at launch → D = 0.5 * 1.225 * 0² * 0.4 * 0.05 = 0 N
(No drag at rest. Drag only matters once the rocket is moving.)

--- GRAVITY ---
F_gravity = 100 * 9.81 = 981 N downward

--- NET FORCE ---
Fx = 436 + 0        =  436 N
Fy = 4981 + 0 - 981 = 4000 N

--- ACCELERATION ---
ax = 436  / 100 = 4.36  m/s²
ay = 4000 / 100 = 40.0  m/s²   ← about 4g upward

--- NEXT STATE (dt = 0.01s) ---
vx = 0 + 4.36  * 0.01 = 0.0436 m/s
vy = 0 + 40.0  * 0.01 = 0.400  m/s
m  = 100 - 2   * 0.01 = 99.98  kg
x  = 0 + 0     * 0.01 = 0      m   (uses velocity from THIS step)
y  = 0 + 0     * 0.01 = 0      m
```

Next iteration, those new values feed in and the rocket is already 0.4 m/s upward. After 25 seconds of this, the fuel is gone and the rocket is flying free.

---

## 10. Why Drag Splits Into vx/v and vy/v

Drag is a scalar magnitude (a single number, e.g. 490 N). But it acts as a **vector** that opposes the direction of motion.

To turn the scalar into a vector pointing opposite to velocity, we use the **unit velocity vector**:

```
v     = sqrt(vx² + vy²)          ← total speed (magnitude)
vx/v  = fraction of motion that is horizontal
vy/v  = fraction of motion that is vertical
```

So:
```
Drag_x = -D * (vx / v)    ← pushes back on the horizontal component
Drag_y = -D * (vy / v)    ← pushes back on the vertical component
```

The minus sign is because drag always opposes motion.

**Worked example:**
```
vx = 150 m/s, vy = 200 m/s
v  = sqrt(150² + 200²) = sqrt(22500 + 40000) = sqrt(62500) = 250 m/s
D  = 490 N

Drag_x = -490 * (150/250) = -490 * 0.60 = -294 N
Drag_y = -490 * (200/250) = -490 * 0.80 = -392 N
```

The rocket is climbing faster than it's going sideways (vy > vx), so more of the drag fights the climb. Makes sense physically.

---

## 11. How This Connects to the AI Layer

This physics simulation is the **environment** the AI agent lives in. The agent:
- **Observes** the state vector [x, y, vx, vy, m] at each timestep
- **Decides** the thrust angle θ (and optionally magnitude)
- **Receives a reward** based on how close it stays to the target trajectory

The physics engine doesn't know about AI — it just advances state honestly. The AI sits on top and learns to steer.

[PAPER OPPORTUNITY #2: Effect of simulation fidelity (constant g vs altitude-varying g, constant ρ vs ISA model) on RL policy quality]
