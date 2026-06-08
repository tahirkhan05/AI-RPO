"""
AI-RPO Dashboard — Streamlit app.

Run with:
    streamlit run dashboard/app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-RPO Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.title("AI-RPO — Adaptive Path Optimisation Dashboard")
st.caption("Physics-Informed RL Rocket Guidance | PINN + PPO + EKF + LSTM")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Episode Settings")

    MODEL_OPTIONS = {
        "v4 — 2D PINN-guided":         ("models/ppo_v4_final",    "models/vecnorm_v4_final.pkl",    "2D"),
        "v5 — 2D PINN + EKF":          ("models/ppo_v5_final",    "models/vecnorm_v5_final.pkl",    "2D"),
        "v6 — 3D baseline (1M steps)": ("models/ppo_v6_final",    "models/vecnorm_v6_final.pkl",    "3D"),
        "v6ext — 3D extended (3M)":    ("models/ppo_v6ext_final", "models/vecnorm_v6ext_final.pkl", "3D"),
    }

    # Filter to only available models
    available = {k: v for k, v in MODEL_OPTIONS.items()
                 if os.path.exists(v[0] + ".zip")}

    if not available:
        st.error("No trained models found. Run training scripts first.")
        st.stop()

    model_choice = st.selectbox("Model", list(available.keys()))
    model_path, vecnorm_path, dim = available[model_choice]

    randomize    = st.checkbox("Randomize rocket & wind", value=True)
    physics_guided = st.checkbox("PINN-guided reference", value=True)
    use_ekf      = st.checkbox("EKF state estimation",   value=(dim == "2D"))
    use_lstm     = st.checkbox("LSTM deviation forecast", value=True)
    seed         = st.number_input("Seed", value=42, step=1)

    run_btn = st.button("Run Episode", type="primary", use_container_width=True)

    st.divider()
    st.caption("Models available: " + ", ".join(k.split("—")[0].strip() for k in available))

# ── Run episode ───────────────────────────────────────────────────────────────
if run_btn:
    from dashboard.run_episode import run_episode

    if dim == "2D":
        from simulation.env import RocketEnv as EnvCls
        env_kwargs = dict(randomize=randomize, physics_guided=physics_guided,
                          use_ekf=use_ekf)
    else:
        from simulation.env3d import RocketEnv3D as EnvCls
        env_kwargs = dict(randomize=randomize, physics_guided=physics_guided,
                          use_ekf=use_ekf)

    lstm_path = "models/lstm_forecaster_v1.pt" if (
        use_lstm and os.path.exists("models/lstm_forecaster_v1.pt")) else None

    with st.spinner("Running episode..."):
        data = run_episode(
            model_path, vecnorm_path,
            EnvCls, env_kwargs,
            lstm_path=lstm_path,
            seed=int(seed),
        )
    st.session_state["data"] = data
    st.session_state["dim"]  = dim

# ── Plot ──────────────────────────────────────────────────────────────────────
if "data" in st.session_state:
    data = st.session_state["data"]
    dim  = st.session_state["dim"]
    t    = data["t"]

    outcome_color = {"landed_safe": "green", "crashed": "red", "timeout": "orange"}
    outcome = data["outcome"]
    st.markdown(
        f"**Outcome:** :{outcome_color.get(outcome,'gray')}[{outcome.upper()}]  |  "
        f"Total reward: **{data['r_total'].sum():.0f}**  |  "
        f"Duration: **{t[-1]:.1f}s**  |  "
        f"Apogee: **{data['y'].max()/1000:.2f} km**"
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Altitude vs Time",
            "Phase Space (altitude vs vertical velocity)",
            "LSTM Deviation Forecast",
            "Reward Components",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Altitude vs Time ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=data["y"]/1000,
        name="Agent", line=dict(color="royalblue", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=data["target_y"]/1000,
        name="PINN target", line=dict(color="orange", width=2, dash="dash")
    ), row=1, col=1)

    if np.any(data["ekf_y"] != data["y"]):
        fig.add_trace(go.Scatter(
            x=t, y=data["ekf_y"]/1000,
            name="EKF estimate", line=dict(color="green", width=1, dash="dot")
        ), row=1, col=1)

    # Tracking error band
    err = np.abs(data["y"] - data["target_y"])
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([(data["target_y"] + err)/1000,
                           (data["target_y"] - err)[::-1]/1000]),
        fill="toself", fillcolor="rgba(255,165,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Error band", showlegend=False,
    ), row=1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)

    # ── Panel 2: Phase space ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=data["vy"], y=data["y"]/1000,
        name="Agent path",
        mode="lines", line=dict(color="royalblue", width=1.5),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=data["target_vy"], y=data["target_y"]/1000,
        name="PINN path",
        mode="lines", line=dict(color="orange", width=1.5, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[data["vy"][0]], y=[data["y"][0]/1000],
        mode="markers", marker=dict(color="green", size=10, symbol="circle"),
        name="Start", showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[data["vy"][-1]], y=[data["y"][-1]/1000],
        mode="markers", marker=dict(color="red", size=10, symbol="x"),
        name="End", showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Vertical velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=2)

    # ── Panel 3: LSTM forecast ────────────────────────────────────────────
    actual_dev = np.abs(data["y"] - data["target_y"])

    fig.add_trace(go.Scatter(
        x=t, y=data["lstm_forecast"]/1000,
        name="LSTM forecast", line=dict(color="purple", width=2)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=actual_dev/1000,
        name="Actual deviation", line=dict(color="tomato", width=1.5, dash="dash")
    ), row=2, col=1)

    # Warning threshold line
    fig.add_hline(y=2.0, line_dash="dot", line_color="red",
                  annotation_text="Warning threshold (2km)",
                  annotation_position="top right", row=2, col=1)

    # Shade warning events
    warned_t = t[data["lstm_warned"]]
    if len(warned_t) > 0:
        for wt in warned_t:
            fig.add_vrect(x0=wt-0.05, x1=wt+0.05,
                          fillcolor="red", opacity=0.15, line_width=0,
                          row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Deviation (km)", row=2, col=1)

    # ── Panel 4: Reward breakdown ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=np.cumsum(data["r_alt"]),
        name="r_alt (cumul)", stackgroup="one",
        line=dict(color="steelblue"),
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=t, y=np.cumsum(data["r_vel"]),
        name="r_vel (cumul)", stackgroup="one",
        line=dict(color="seagreen"),
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=t, y=np.cumsum(data["r_fuel"]),
        name="r_fuel (cumul)", stackgroup="one",
        line=dict(color="gold"),
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=t, y=np.cumsum(data["r_smooth"]),
        name="r_smooth (cumul)", stackgroup="one",
        line=dict(color="salmon"),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative reward", row=2, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(t=40, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Stats table ───────────────────────────────────────────────────────
    rocket = data["rocket"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apogee", f"{data['y'].max()/1000:.2f} km")
    col2.metric("Mean tracking error", f"{np.mean(np.abs(data['y']-data['target_y']))/1000:.2f} km")
    col3.metric("LSTM warnings fired", f"{data['lstm_warned'].sum()}")
    col4.metric("Rocket mass", f"{rocket.mass_wet:.0f} kg wet / {rocket.mass_dry:.0f} kg dry")

    if dim == "3D":
        c1, c2 = st.columns(2)
        c1.metric("Max yaw", f"{np.abs(data['yaw']).max():.1f}°")
        c2.metric("Horizontal drift",
                  f"{(np.abs(data['x']).mean() + np.abs(data['z']).mean())/1000:.2f} km")

else:
    st.info("Configure settings in the sidebar and click **Run Episode** to start.")
    st.markdown("""
    ### What this dashboard shows
    - **Panel 1:** How closely the agent tracks the PINN physics reference trajectory
    - **Panel 2:** Phase space (altitude vs velocity) — smooth curve = good control
    - **Panel 3:** LSTM forecasts upcoming deviations 1 second ahead
    - **Panel 4:** Reward breakdown — see which objective dominates at each phase
    """)
