import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from wind_assess.utils import (
    load_power_curve_csv,
    plot_weibull_fit,
    plot_wind_rose,
)

import matplotlib
matplotlib.use("Agg")  # avoid GUI backend during tests


def test_power_curve_loader(tmp_path):
    # create a temporary synthetic CSV
    csv_path = tmp_path / "pc.csv"
    df = pd.DataFrame({
        "Wind Speed [m/s]": [0, 5, 10],
        "Power [kW]": [0, 500, 2000],
    })
    df.to_csv(csv_path, index=False)

    speeds, power = load_power_curve_csv(csv_path)

    assert len(speeds) == 3
    assert power[1] == 500


def test_weibull_plot():
    import matplotlib.pyplot as plt

    speeds = np.array([1, 2, 3, 4, 5])
    k, A = 2, 3

    fig, ax = plt.subplots()
    plot_weibull_fit(speeds, k, A, ax=ax)

    # ensure something was plotted
    assert len(ax.lines) > 0


def test_wind_rose_plot():
    import matplotlib.pyplot as plt

    directions = np.array([0, 90, 180, 270])
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    plot_wind_rose(directions, ax=ax)

    # a bar plot must exist in the axes
    assert len(ax.patches) > 0
