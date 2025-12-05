"""
run with:

pytest --cov=src tests/
"""
import sys
from pathlib import Path

#   Check if the src/ folder is importable when running pytest
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from wind_assess.utils import (
    load_power_curve_csv,
    plot_weibull_fit,
    plot_wind_rose,
)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")  # avoid GUI backend during tests


# Test 1: Power curve loader
def test_power_curve_loader(tmp_path):
    """
    Tests the CSV power curve loader.

    Steps:
        1. Create a temporary CSV file inside pytest's tmp_path.
        2. Write a simple power curve with known values.
        3. Call load_power_curve_csv() and verify:
            - length of arrays is correct
            - values match what was written
    """

    # create a temporary synthetic CSV
    csv_path = tmp_path / "pc.csv"
    df = pd.DataFrame({
        "Wind Speed [m/s]": [0, 5, 10],
        "Power [kW]": [0, 500, 2000],
    })
    df.to_csv(csv_path, index=False)

    speeds, power = load_power_curve_csv(csv_path)

    # Correctness check
    assert len(speeds) == 3
    assert power[1] == 500  #known value


# Test 2: Weibull plot
def test_weibull_plot():
    # Tests that plot_weibull_fit() successfully draws a Weibull curve.
    import matplotlib.pyplot as plt

    speeds = np.array([1, 2, 3, 4, 5])  # example dataset
    k, A = 2, 3 #arbitrary weibull parameter                      

    fig, ax = plt.subplots()
    plot_weibull_fit(speeds, k, A, ax=ax)

    # ensure something was plotted
    assert len(ax.lines) > 0


# Test 3: Wind Rose Plot
def test_wind_rose_plot():
    """
    Tests that plot_wind_rose() creates bar patches on a polar axis.

    Steps:
        1. Provide a small set of directions
        2. Create a polar Axes object
        3. Call the plotting function
        4. Verify that at least one patch (bar) exists

    """
    import matplotlib.pyplot as plt

    directions = np.array([0, 90, 180, 270])    # N, E, S, W
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    plot_wind_rose(directions, ax=ax)

    # a bar plot must exist in the axes
    assert len(ax.patches) > 0
