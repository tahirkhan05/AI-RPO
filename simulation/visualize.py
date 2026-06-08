"""
Plotting utilities. Each function takes a TrajectoryResult and returns a Figure.
Nothing is shown automatically — caller decides when to call plt.show() or savefig().
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simulation.trajectory import TrajectoryResult


def _burnout_line(ax: plt.Axes, result: TrajectoryResult) -> None:
    """Draw a vertical dashed line at burnout time if it occurred."""
    if result.burnout_time > 0:
        ax.axvline(result.burnout_time, color="gray", linestyle="--",
                   linewidth=0.8, label=f"Burnout t={result.burnout_time:.1f}s")


def plot_trajectory(result: TrajectoryResult) -> plt.Figure:
    """Altitude vs downrange distance — the flight path shape."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.x / 1000, result.y / 1000, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Downrange distance (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("2D Trajectory")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return fig


def plot_summary(result: TrajectoryResult) -> plt.Figure:
    """4-panel summary: altitude, speed, mass, and trajectory shape."""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Altitude over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(result.time, result.y / 1000, color="steelblue")
    _burnout_line(ax1, result)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title(f"Altitude  [apogee: {result.apogee/1000:.2f} km]")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Speed over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(result.time, result.speed, color="tomato")
    _burnout_line(ax2, result)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title(f"Speed  [peak: {result.max_speed:.1f} m/s]")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Mass over time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(result.time, result.mass, color="goldenrod")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Mass (kg)")
    ax3.set_title("Mass (fuel burn)")
    ax3.grid(True, alpha=0.3)

    # Trajectory shape
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(result.x / 1000, result.y / 1000, color="mediumseagreen")
    ax4.set_xlabel("Downrange (km)")
    ax4.set_ylabel("Altitude (km)")
    ax4.set_title("Flight Path")
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)

    return fig
