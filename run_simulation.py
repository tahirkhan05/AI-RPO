"""Entry point — run a baseline trajectory and display the summary plot."""

import matplotlib.pyplot as plt

from simulation.config import DEFAULT_ROCKET, DEFAULT_SIM
from simulation.trajectory import run
from simulation.visualize import plot_summary


def main() -> None:
    result = run(DEFAULT_ROCKET, DEFAULT_SIM)

    print(f"Apogee:        {result.apogee/1000:.3f} km")
    print(f"Max speed:     {result.max_speed:.1f} m/s")
    print(f"Burnout at:    {result.burnout_time:.1f} s")
    print(f"Downrange:     {result.x[-1]/1000:.3f} km")

    fig = plot_summary(result)
    plt.show()


if __name__ == "__main__":
    main()
