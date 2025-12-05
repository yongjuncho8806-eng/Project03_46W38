"""
run with: 

python -m examples.main
"""

# adding project root/src folder to python path so the wind_assess module can be
# imported when running main.py 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

# import WindResource class and helper functions from the wind_assess module
from wind_assess.core import WindResource
from wind_assess.utils import load_power_curve_csv, plot_weibull_fit, plot_wind_rose

# Define project paths (locate inputs & save outputs)
ROOT = Path(__file__).resolve().parents[1]  # project root directory
inputs = ROOT / "inputs"                    # input folder 
outputs = ROOT / "outputs"                  # output folder
outputs.mkdir(exist_ok=True)                # create if output folder is missing

# ERA5 input files
era5_files =[
    inputs / "1997-1999.nc",
    inputs / "2000-2002.nc",
    inputs / "2003-2005.nc",
    inputs / "2006-2008.nc",
]

#Turbine power curves 
pc_5mw = inputs / "NREL_Reference_5MW_126.csv"
pc_15mw = inputs / "NREL_Reference_15MW_240.csv"

# Horns Rev 1 coordinates
HORNS_REV_LAT = 55.5297     #55°31′47″N 
HORNS_REV_LON = 7.9061      #7°54′22″E

AEP_YEAR = 2002             #example year

def main():
    # Loading ERA5 data
    # used .from_files method to load multiple files and merge them into a single 
    # xarray .Dataset containing u10, v10, u100, and v100. 
    wr = WindResource.from_files([str(p) for p in era5_files])

    # Time series at 100m for Horns Rev 1
    # interpolate ERA5 u/v components horizontally to the Horns Rev 1 localtion
    # get the wind speed and direction from u/v
    # return hourly array for data range
    speed100, dir100 = wr.get_speed_direction_at_point(
        HORNS_REV_LAT,      # latitde
        HORNS_REV_LON,      # longitude
        height=100,         # wind at chosen height of 100m
        start_year=1997,
        end_year=2008,
    )

    # Compute and print the summary 
    print("Mean wind speed at 100m (m/s):", float(speed100.mean()))
    print("Mean wind direction at 100m (deg):", float(dir100.mean()))

    # Weibull fit for 100m wind speed
    speeds_np = speed100.values.reshape(-1) # Convert speeds to flat NumPy array
    k, A = wr.fit_weibull_1d(speeds_np)     # Fit Weibull distribution
    print(f"Weibull at 100m: k={k:.3f}, A={A:.3f} m/s") # print Weibull

    # Plot Histogram + fitted weibull distribution curve
    fig1, ax1 = plt.subplots()
    plot_weibull_fit(speeds_np, k, A, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(outputs / "weibull_100m_horns_rev.png", dpi=150)   # save to output

    # Wind rose at 100m Direction 
    dirs_np = dir100.values.reshape(-1)     # Convert Dir to flat Numpy array
    
    # Create polar coordinate figure for the wind rose
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, polar=True)

    # Plot wind rose 
    plot_wind_rose(dirs_np, ax=ax2, n_sectors=16)   # 16 dir sectors 
    fig2.tight_layout()
    fig2.savefig(outputs / "wind_rose_100m_horns_rev.png", dpi=150) # save to output

    # AEP Calc for NREL 5MW and 15MW at Horns Rev 1 for AEP_YEAR
    # 5MW at 90m HH
    speed_bins_5, power_5 = load_power_curve_csv(str(pc_5mw))
    aep_5_mwh = wr.compute_aep_from_power_curve(
        lat=HORNS_REV_LAT,
        lon=HORNS_REV_LON,
        hub_height=90.0,        #HH of 90m
        speed_bins=speed_bins_5,
        power_kw=power_5,
        year=AEP_YEAR,
        availability=1.0,
    )

    # 15MW at 150m HH
    speed_bins_15, power_15 = load_power_curve_csv(str(pc_15mw))
    aep_15_mwh = wr.compute_aep_from_power_curve(
        lat=HORNS_REV_LAT,
        lon=HORNS_REV_LON,
        hub_height=150.0,       # HH of 150m
        speed_bins=speed_bins_15,
        power_kw=power_15,
        year=AEP_YEAR,
        availability=1.0,
    )

    # print AEP using two WTGs at 2002 
    print(f"AEP of {AEP_YEAR} with NREL 5MW:  {aep_5_mwh:.2f} MWh")
    print(f"AEP of {AEP_YEAR} with NREL 15MW: {aep_15_mwh:.2f} MWh")

#   Python entry point
if __name__ == "__main__":
    main()
