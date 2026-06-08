# Reading Simulation Output — How to Interpret Results

*Reference file. Every time we run a simulation, come back here to understand what the numbers and plots are telling you physically.*

---

## 1. The Four Panels and What Each One Tells You

### 1.1 Altitude vs Time (top-left)
The most important panel. Shows the full life of the flight.

```
Shape you expect: a smooth hill — rises steeply during burn, 
                  curves over at apogee, descends more slowly.
```

Key moments to read off this plot:
- **Burnout** (dashed line) — where the engine cuts off. Altitude is still rising here because the rocket has momentum.
- **Apogee** — the peak. Rocket has zero vertical velocity at this exact point.
- **Landing** — where the curve hits y=0 again.

If the curve looks wrong:
- Flat top → apogee region, not an error
- Asymmetric descent (slower going down than up) → correct, drag + thinner air on the way up, thicker on the way down
- Hits ground before apogee → launch angle too shallow or thrust too low

---

### 1.2 Speed vs Time (top-right)
Speed = `sqrt(vx² + vy²)` — total magnitude, not just vertical.

```
Shape you expect: sharp rise during burn, dip near apogee, 
                  second smaller hump on descent.
```

Why the dip at apogee:
- At apogee the rocket is momentarily moving only horizontally (vy → 0)
- Speed hits a local minimum, not zero (unless perfectly vertical launch)
- Then gravity pulls it back down, speed builds again

Why the second hump is smaller than the first:
- During burn: thrust + gravity both contribute to speed gain
- During descent: only gravity accelerates it, drag fights back
- Air gets denser as it falls → more drag → limits terminal speed

**Mach reference:** 340 m/s = Mach 1 at sea level. Our peak (880 m/s) = ~Mach 2.6.
At high altitude air is thinner so Mach number is actually higher than this — relevant for drag modeling later.

---

### 1.3 Mass vs Time (bottom-left)
```
Shape you expect: straight diagonal drop from wet mass to dry mass,
                  then a flat horizontal line forever after.
```

- Slope of the diagonal = burn rate (kg/s). Steeper = faster burn.
- The knee (where diagonal meets flat) = burnout time exactly.
- Flat line should be perfectly horizontal — any drift means a bug in the mass integration.

Our values: 100 kg → 50 kg over 25s = 2 kg/s. ✓

---

### 1.4 Flight Path — Altitude vs Downrange (bottom-right)
The actual shape the rocket traces through space (not through time).

```
Shape you expect: asymmetric arch — steeper on the way up 
                  (near-vertical launch), shallower arc at top, 
                  more vertical on descent.
```

- **Steeper ascent side** = launch was nearly vertical (85°), so most thrust goes up not sideways
- **Asymmetry** = rocket gains downrange distance throughout the whole flight, even while descending
- Landing point = total downrange distance (8.5 km for our baseline)

This plot is what you'd compare against a target trajectory later. The AI's job is to make this curve match a desired shape.

---

## 2. Baseline Numbers to Remember

These are our reference values for the default rocket config. Any future change should be compared against these.

| Metric | Baseline Value | Physical meaning |
|---|---|---|
| Apogee | 30.80 km | Max altitude reached |
| Max speed | 879.9 m/s | ~Mach 2.6 at sea level equivalent |
| Burnout time | 25.0 s | 50 kg fuel ÷ 2 kg/s |
| Downrange at landing | 8.54 km | Total horizontal distance |
| Total flight time | ~185 s | From launch to ground impact |
| Steps simulated | 18,359 | At dt=0.01s |

---

## 3. The Speed Curve Double-Hump Explained

This is a subtle but important physical insight.

```
Phase 1 (0 → burnout): thrust accelerates the rocket faster than 
  drag + gravity can decelerate it → speed rises sharply.

Phase 2 (burnout → apogee): no more thrust. Gravity and drag both 
  decelerate. Speed falls. Hits local minimum near apogee.

Phase 3 (apogee → landing): gravity now works WITH motion (downward). 
  Speed builds again but drag limits it → terminal velocity on descent.
```

The ratio of peak speed to terminal speed is a direct measure of how much energy drag dissipates. In a vacuum both peaks would be equal.

**Paper angle:** Quantifying energy dissipation across flight phases — how much Δv is lost to drag vs. gravity, and how that changes with altitude-varying atmospheric density.

---

## 4. What "Physically Correct" Means as a Checklist

Before trusting any simulation run, verify:

- [ ] Altitude curve is a smooth hill with a single peak
- [ ] Speed peaks at or just before burnout (not after)
- [ ] Mass drops linearly then holds flat exactly at dry mass
- [ ] Flight path is asymmetric (steeper ascent than descent) for near-vertical launches
- [ ] Rocket lands (y returns to ~0) within `max_time`
- [ ] No negative altitude before ~0.1s (the ground-check grace period)
- [ ] Burnout time = fuel_mass / burn_rate (within one timestep)

If any of these fail, the physics engine has a bug — don't trust the AI training results built on top of it.

---

## 5. How These Plots Will Change When AI Takes Over

Right now the thrust angle is fixed at 85° for the whole flight. Once the RL agent is controlling the rocket:

- The **altitude curve** will change shape — agent may pitch aggressively to follow a target path
- The **speed curve** will be less smooth — control actions cause small fluctuations
- The **flight path** will try to match a desired trajectory instead of flying free
- A new panel will appear: **thrust angle over time** — showing the agent's decisions

The baseline plots here are the "no AI" reference. Every AI result gets compared against them.
