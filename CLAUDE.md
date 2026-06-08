# AI-RPO Project — Claude Guidelines

## Project
AI-Integrated Adaptive Path Optimization System for Rocket Propulsion.
Simulation-only (no hardware). Research-grade academic submission.

## Who I'm working with
Solo developer (previously a 3-person team, now solo + IoT person to join later).
Learning as we build — go step by step, explain the WHY behind every decision.
Budget guidance in INR.

## Pace and style
- Step by step, slow. Explain concepts before code.
- Point out research paper opportunities as they arise.
- No IoT/sensor/hardware modules for now — defer to later.
- Focus: physics simulation → RL environment → AI training → dashboard.
- Always create a `notes/` learning file alongside every new module (e.g. `notes/01_physics.md`).
  These are permanent concept references the user can return to at any time.
- Any important insight said in conversation — physical interpretation, "why" explanations,
  paper angles, checklist patterns — must be written into the relevant notes/ file immediately.
  Never leave useful explanation only in the chat.
- Every bug, error, wrong assumption, design decision, and constraint must be logged
  in notes/00_problems_and_decisions_log.md with: what broke, root cause, fix, lesson,
  paper angle if applicable. This applies to ALL phases.
- Walk through math with real numbers before writing code.
- Start every simulation module in 2D; upgrade to 3D only after 2D is solid and understood.

## Context folder
`context/` is the canonical project idea source of truth.
Files: AI-RPO spec (.docx) and conversations.json.
Re-read these when starting a new phase to stay aligned with the original vision.

## Research paper tracking
Target: 8-12 high-quality papers from this project.
Flag every potential paper contribution inline as: `[PAPER OPPORTUNITY: <topic>]`

## Tech stack
- Python (primary), C++ (deferred)
- PyTorch, Gymnasium, SciPy, Matplotlib/Plotly
- PPO (primary RL algo), DDPG (fallback)
- PINNs for physics-informed learning
- No ROS 2, no MATLAB, no hardware for now

## What NOT to do
- Don't jump ahead of the current phase
- Don't suggest hardware integrations as primary path
- Don't skip explaining physics/math intuition before implementing
- Don't add IoT/sensor simulation until the IoT person joins
