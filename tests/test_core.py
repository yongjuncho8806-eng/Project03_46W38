"""
run with:

pytest --cov=src tests/
"""
# so far 88%
import sys
from pathlib import Path

#   Check if the src/ folder is importable when running pytest
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import xarray as xr
from wind_assess.core import WindResource

# Synthetic test dataset
def make_synthetic_dataset():
    # Create a tiny synthetic ERA5-like dataset for testing.
    time = np.array(["2000-01-01T00:00", "2000-01-01T01:00"], dtype="datetime64[ns]")
    lat = np.array([55.75, 55.50])
    lon = np.array([7.75, 8.00])

    # simple u/v values
    # u10 = 1m/s everywhere
    u10 = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "latitude", "longitude"))
    # v10 = 0 east - west
    v10 = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "latitude", "longitude"))
    # u100 = 2m/s everywhere
    u100 = xr.DataArray(2 * np.ones((2, 2, 2)), dims=("time", "latitude", "longitude"))
    v100 = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "latitude", "longitude"))

    # put all in xarray same as ERA5
    ds = xr.Dataset(
        {
            "u10": u10,
            "v10": v10,
            "u100": u100,
            "v100": v100,
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    return ds


# Test 1: Speed and direction computation
def test_speed_direction_computation():

    # For synthetic dataset: u10 = 1, v10 = 0 speed = 1 m/s everywhere
    ds = make_synthetic_dataset()
    wr = WindResource(ds)

    speed, direction = wr.get_speed_direction_at_point(
        lat=55.60, lon=7.90, height=10, start_year=2000, end_year=2000
    )

    # speed should be sqrt(1^2 + 0) = 1 everywhere
    assert np.allclose(speed.values, 1.0, atol=1e-6)

    # wind blowing toward east direction = 270° (coming from west)
    assert np.all(direction.values >= 260) and np.all(direction.values <= 280)


# Test 2: Power law exponent
def test_power_law_extrapolation():
    """
    Tests the shear exponent and speed extrapolation.

    In the synthetic dataset, speed10  = 1 m/s, speed100 = 2 m/s

    So:
        ratio = 2 / 1 = 2
        = log(ratio) / log(100/10)
        = log(2) / log(10)
        = log10(2) = 0.30103
    """
    ds = make_synthetic_dataset()
    wr = WindResource(ds)

    alpha = wr.estimate_alpha_at_point(55.65, 7.95)

    # Correct expected value
    expected_alpha = np.log10(2)  # 0.30103
    assert abs(alpha - expected_alpha) < 0.01

    # Extrapolation to 150m
    speed150 = wr.extrapolate_speed_power_law(
        lat=55.65, lon=7.95, z_target=150, z_ref=100
    )

    # expected speed = 2 * (150/100)^alpha
    expected_speed = 2 * (1.5 ** alpha)
    assert abs(speed150.values[0] - expected_speed) < 0.05


# Test 3: Weibull fitting 
def test_weibull_fit():
    
    ds = make_synthetic_dataset()
    wr = WindResource(ds)

    synthetic_speeds = np.array([1, 2, 3, 4, 5], dtype=float)
    k, A = wr.fit_weibull_1d(synthetic_speeds)
    
    # Return positive k and A
    assert k > 0
    assert A > 0


# Test 4: AEP Computation
def test_aep_computation():
    """
    Tests the AEP method using:
    - A simple linear power curve
    - Synthetic ERA5 data with known 100 m speed = 2 m/s

    Expected outcome:
    - AEP > 0
    - Computation runs without errors
    """
    ds = make_synthetic_dataset()
    wr = WindResource(ds)

    # simple linear synthetic power curve
    speed_bins = np.array([0, 5, 10])   # speed bins
    power_kw = np.array([0, 500, 2000])  # linear power curve

    aep = wr.compute_aep_from_power_curve(
        lat=55.6,
        lon=7.9,
        hub_height=100,   # matches u100 = 2 m/s in synthetic dataset
        speed_bins=speed_bins,
        power_kw=power_kw,
        year=2000,
    )

    assert aep > 0 # Should not be zero
