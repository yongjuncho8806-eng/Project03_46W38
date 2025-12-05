"""
Utility module for ERA5-based wind resource assessment

Functionalities:
-   Power curve loading 
-   Weibull/wind-rose plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

# Power curve loading
def load_power_curve_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load power curve from NREL CSV files.
    two columns: wind speed, power 

    Return: 
    speeds : np.ndarray, Wind speeds (m/s)
    power : np.ndarray, Turbine rated power (kW) for each speed bin
    """
    df = pd.read_csv(path)

    # Detect wind speed column
    speed_col = None
    power_col = None

    # Identify correct columns by name 
    for col in df.columns:
        lower = col.lower()
        if "wind speed" in lower or "windspeed" in lower or lower.startswith("v"):
            speed_col = col
        if "power" in lower:
            power_col = col

    # If identification fails
    if speed_col is None or power_col is None:
        # Fallback: assume first column is speed, second is power
        speed_col = df.columns[0]
        power_col = df.columns[1]

    speeds = df[speed_col].values.astype(float)
    power = df[power_col].values.astype(float)
    return speeds, power

# Weibull Distribution plotting
def plot_weibull_fit(speeds: np.ndarray, k: float, A: float, ax=None):
    """
    Plot histogram of speeds and fitted Weibull.

    Parameters
    speeds : array, 1D list/array of wind speeds.
    k, A : float, Weibull parameters.
    ax : matplotlib axis, If None, a new figure is created.
    
    Returns:
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    speeds = np.asarray(speeds)
    speeds = speeds[~np.isnan(speeds)]

    if ax is None:
        fig, ax = plt.subplots()

    # Histogram of data
    ax.hist(speeds, bins=40, density=True, alpha=0.6, label="Data")

    # Weibull probability density curve
    x = np.linspace(0, np.nanmax(speeds) * 1.2, 300)
    pdf = weibull_min(c=k, scale=A).pdf(x)
    ax.plot(x, pdf, linewidth=2, label=f"Weibull k={k:.2f}, A={A:.2f}")

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density [-]")
    ax.legend()
    return ax

#   Wind Rose Plot
def plot_wind_rose(directions_deg: np.ndarray, ax=None, n_sectors: int = 16):
    """
    Simple wind rose using a polar histogram of wind direction frequencies.

    Parameters:
    directions_deg : np.ndarray, Wind direction (deg) where 0 = North, 90 = East.
    n_sectors : int, Number of direction bins (16).

    Returns:
    ax : matplotlib.axes.Axes (polar)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

    dirs = np.deg2rad(directions_deg) # Convert deg to radians
    bins = np.linspace(0, 2 * np.pi, n_sectors + 1) # Define bin edges around circle

    counts, _ = np.histogram(dirs, bins=bins)   # count data per bin
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2

    # Plot bars
    ax.bar(centers, counts, width=widths, bottom=0.0, align="center")

    # Meteorological convention
    ax.set_theta_zero_location("N") # 0 deg = North
    ax.set_theta_direction(-1)      # Rotate Clockwise 
    ax.set_title("Wind rose")
    return ax
