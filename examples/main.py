
#adding project root/src folder to python path so the wind_assess module can be
#imported when running main.py 
#run with python -m examples.main
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

#import WindResource class and helper functions from the wind_assess module
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

# Horns Rev 1 approximate coordinates
HORNS_REV_LAT = 55.5297
HORNS_REV_LON = 7.9061

# Year for AEP example (must be within 1997–2008)
AEP_YEAR = 2005


def main():
    # --------------------------------------------------------------
    # 1. Load ERA5 data
    # --------------------------------------------------------------
    wr = WindResource.from_files([str(p) for p in era5_files])

    # --------------------------------------------------------------
    # 2. Time series at 100m for Horns Rev 1 (full period)
    # --------------------------------------------------------------
    speed100, dir100 = wr.get_speed_direction_at_point(
        HORNS_REV_LAT,
        HORNS_REV_LON,
        height=100,
        start_year=1997,
        end_year=2008,
    )

    print("100m speed mean [m/s]:", float(speed100.mean()))
    print("100m direction mean [deg]:", float(dir100.mean()))

    # --------------------------------------------------------------
    # 3. Weibull fit for 100m speeds
    # --------------------------------------------------------------
    speeds_np = speed100.values.reshape(-1)
    k, A = wr.fit_weibull_1d(speeds_np)
    print(f"Weibull at 100m: k={k:.3f}, A={A:.3f} m/s")

    fig1, ax1 = plt.subplots()
    plot_weibull_fit(speeds_np, k, A, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(outputs / "weibull_100m_horns_rev.png", dpi=150)

    # --------------------------------------------------------------
    # 4. Wind rose at 100m (full period)
    # --------------------------------------------------------------
    dirs_np = dir100.values.reshape(-1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, polar=True)
    plot_wind_rose(dirs_np, ax=ax2, n_sectors=16)
    fig2.tight_layout()
    fig2.savefig(outputs / "wind_rose_100m_horns_rev.png", dpi=150)

    # --------------------------------------------------------------
    # 5. AEP for NREL 5MW and 15MW at Horns Rev 1 for AEP_YEAR
    # --------------------------------------------------------------
    # 5MW: hub height ~90 m
    speed_bins_5, power_5 = load_power_curve_csv(str(pc_5mw))
    aep_5_mwh = wr.compute_aep_from_power_curve(
        lat=HORNS_REV_LAT,
        lon=HORNS_REV_LON,
        hub_height=90.0,
        speed_bins=speed_bins_5,
        power_kw=power_5,
        year=AEP_YEAR,
        availability=1.0,
    )

    # 15MW: hub height ~150 m
    speed_bins_15, power_15 = load_power_curve_csv(str(pc_15mw))
    aep_15_mwh = wr.compute_aep_from_power_curve(
        lat=HORNS_REV_LAT,
        lon=HORNS_REV_LON,
        hub_height=150.0,
        speed_bins=speed_bins_15,
        power_kw=power_15,
        year=AEP_YEAR,
        availability=1.0,
    )

    print(f"AEP {AEP_YEAR} - NREL 5MW @ Horns Rev:  {aep_5_mwh:.1f} MWh")
    print(f"AEP {AEP_YEAR} - NREL 15MW @ Horns Rev: {aep_15_mwh:.1f} MWh")

    # Optionally show plots interactively during development
    # plt.show()


if __name__ == "__main__":
    main()
