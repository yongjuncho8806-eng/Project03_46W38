"""
Core module for ERA5-based wind resource assessment

Functionalities:
- Load and parse multiple ERA5 files
- Compute wind speed and direction from u/v
- Interpolate time series at 10m and 100m for a given location
- Extrapolate wind speed to arbitrary height using power law
- Fit Weibull distribution for wind speed 
- Compute AEP for NREL 5MW / 15MW 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import xarray as xr
from scipy.stats import weibull_min

@dataclass
class WindResource:

    # Main class storing ERA5 data and providing wind resource methods.
   
    ds: xr.Dataset  # ERA 5 dataset with variables u10, v10, u100, v100, lat, log, time

    # Load ERA5 data
    @classmethod
    def from_files(cls, filepaths: Sequence[str]) -> "WindResource":
        """
        Load multiple ERA5 files and merge them along time

        Parameters
        filepaths : list of str Paths to NetCDF4 files 1997 - 2008
        Returns: WindResource
        """
        ds = xr.open_mfdataset(filepaths, combine="by_coords")
        return cls(ds)

    # Internal utilities
    def _subset_years(self, start_year: int, end_year: int) -> xr.Dataset:

        # Select a time subset between start_year and end_year (inclusive).
        start = f"{start_year}-01-01"
        end = f"{end_year}-12-31T23:00" # inclusive of the last hour
        return self.ds.sel(time=slice(start, end))

    def _interp_point(self, ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:

        # Interpolate the dataset to a single (lat, lon) point.
        # Uses bilinear interpolation on the 2x2 ERA5 grid.
        return ds.interp(latitude=lat, longitude=lon)

    # Basic speed / direction from u/v components
    @staticmethod
    def _speed_dir_from_uv(u: xr.DataArray, v: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute wind speed and direction from u, v components.
        Converts vector components to:
        Magnitude (wind speed)
        Meteorological direction (0 deg = North, 90 deg = East)

        Negative signs ensure direction is "where wind is blowing FROM".
        """
        speed = np.sqrt(u**2 + v**2)

        # meteorological direction: where wind is coming FROM
        # dir_rad = atan2(-u, -v)
        dir_rad = np.arctan2(-u, -v)    #meteorological convention
        direction = (np.degrees(dir_rad) + 360.0) % 360.0
        return speed, direction

    # time series at a specific location
    def get_speed_direction_at_point(
        self,
        lat: float,
        lon: float,
        height: int,
        start_year: int = 1997,
        end_year: int = 2008,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute wind speed and direction time series at given location and height.

        Parameters:
        lat, lon : float, Location inside the 2x2 ERA5 grid.
        height : int, 10m or 100m 
        start_year, end_year : int, Define the period to use.

        Returns:
        speed : xarray.DataArray, Wind speed (m/s) with time.
        direction : xarray.DataArray, Wind direction (deg]) with time.
        """
        ds_sub = self._subset_years(start_year, end_year)   # subset years
        ds_pt = self._interp_point(ds_sub, lat, lon)        # interpolate to coor

        # select u/v components
        if height == 10:
            u = ds_pt["u10"]
            v = ds_pt["v10"]
        elif height == 100:
            u = ds_pt["u100"]
            v = ds_pt["v100"]
        else:
            raise ValueError("height must be 10 or 100 m for this method.")

        # convert to speed, direction
        speed, direction = self._speed_dir_from_uv(u, v)
        return speed, direction

    # Shear (power law extrapolation)
    def estimate_alpha_at_point(
        self,
        lat: float,
        lon: float,
        start_year: int = 1997,
        end_year: int = 2008,
    ) -> float:
        """
        Estimate power law exponent alpha at a given point using 10m & 100m speeds.

        Based on:
        u(z)/u(zr) = (z/zr)^alpha
        with z=100m, zr=10m.

        Returns:
        alpha : float, Median alpha over the selected time period.
        """
        ds_sub = self._subset_years(start_year, end_year)
        ds_pt = self._interp_point(ds_sub, lat, lon)

        s10, _ = self._speed_dir_from_uv(ds_pt["u10"], ds_pt["v10"])
        s100, _ = self._speed_dir_from_uv(ds_pt["u100"], ds_pt["v100"])

        # avoid divide by zero
        valid = (s10 > 0.0) & (s100 > 0.0)
        ratio = (s100 / s10).where(valid)

        alpha = np.nanmedian(np.log(ratio) / np.log(100.0 / 10.0))
        return float(alpha)

    def extrapolate_speed_power_law(
        self,
        lat: float,
        lon: float,
        z_target: float,
        z_ref: float = 100.0,
        alpha: Optional[float] = None,
        start_year: int = 1997,
        end_year: int = 2008,
    ) -> xr.DataArray:
        """
        Compute wind speed time series at height z_target using power law profile.

        Parameters:
        lat, lon : float, Location inside the ERA5 grid.
        z_target : float, Target height(m).
        z_ref : float, Reference height(m) (default 100 m).
        alpha : float, If None, estimated from 10m & 100m speeds at the point.
        start_year, end_year : int, Period to use.

        Returns:
        speed_target : xarray.DataArray, Wind speed at z_target(m/s) with time
        """
        if alpha is None:
            alpha = self.estimate_alpha_at_point(lat, lon, start_year, end_year)

        # get speed at ref height
        if int(z_ref) == 10:
            s_ref, _ = self.get_speed_direction_at_point(
                lat, lon, height=10, start_year=start_year, end_year=end_year
            )
        elif int(z_ref) == 100:
            s_ref, _ = self.get_speed_direction_at_point(
                lat, lon, height=100, start_year=start_year, end_year=end_year
            )
        else:
            raise ValueError("z_ref must be 10 or 100 m for this implementation.")

        speed_target = s_ref * (z_target / z_ref) ** alpha
        return speed_target

    # Weibull fitting
    @staticmethod
    def fit_weibull_1d(speeds: np.ndarray) -> Tuple[float, float]:
        """
        Fit Weibull distribution to 1D speed array using MLE.

        Uses scipy.stats.weibull_min parameterisation:
        c = k(shape), scale = A (scale).

        Returns:
        k : float, weibull shape parameter.
        A : float, Weibull scale parameter.
        """
        speeds = np.asarray(speeds)
        speeds = speeds[~np.isnan(speeds)]
        if speeds.size == 0:
            raise ValueError("Cannot fit Weibull to empty array")

        k, loc, A = weibull_min.fit(speeds, floc=0.0)
        return float(k), float(A)

    def fit_weibull_at_point(
        self,
        lat: float,
        lon: float,
        height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        use_power_law: bool = False,
    ) -> Tuple[float, float]:
        """
        Fit Weibull at a given location, height and period.

        Parameters:
        lat, lon : float
        height : float, If height is 10 or 100 and use_power_law=False, use ERA5 directly.
        Otherwise, speeds are extrapolated using power law from 100m.
        start_year, end_year : int
        use_power_law : bool, If True and height != 10 or 100, speeds are extrapolated.

        Returns:
        k, A : float, Weibull shape and scale parameters.
        """
        if (height in (10, 100)) and (not use_power_law):
            speed, _ = self.get_speed_direction_at_point(
                lat, lon, height=int(height), start_year=start_year, end_year=end_year
            )
        else:
            speed = self.extrapolate_speed_power_law(
                lat=lat,
                lon=lon,
                z_target=height,
                z_ref=100.0,
                start_year=start_year,
                end_year=end_year,
            )

        speeds_np = speed.values.reshape(-1)
        return self.fit_weibull_1d(speeds_np)

    # AEP calculation
    def compute_aep_from_power_curve(
        self,
        lat: float,
        lon: float,
        hub_height: float,
        speed_bins: np.ndarray,
        power_kw: np.ndarray,
        year: int,
        availability: float = 1.0,
    ) -> float:
        """
        Compute AEP (MWh) for a turbine.

        Parameters
        lat, lon : float, Location
        hub_height : float, Turbine hub height (m).
        speed_bins : np.ndarray, Wind speed values (m/s) where power curve is defined.
        power_kw : np.ndarray, Corresponding turbine power (kW).
        year : int, Year within 1997–2008 to use.
        availability : float, Turbine availability factor (0–1). Default 1.0.

        Returns:
        aep_mwh : float, Annual Energy Production (MWh).
        """
        # Use that year only
        speed_hub = self.extrapolate_speed_power_law(
            lat=lat,
            lon=lon,
            z_target=hub_height,
            z_ref=100.0,
            start_year=year,
            end_year=year,
        )

        speeds = speed_hub.values.reshape(-1)
        # interpolate power curve
        power_interp_kw = np.interp(
            speeds,
            speed_bins,
            power_kw,
            left=0.0,
            right=0.0,
        )

        # Each ERA5 element = 1 hour, Energy (kWh) = power (kW) * hours
        total_kwh = np.nansum(power_interp_kw * availability)  # *1h
        aep_mwh = total_kwh / 1000.0 #kwh to mwh
        return float(aep_mwh)
