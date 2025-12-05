# Project03_46W38 – Wind Resource Assessment Based on Reanalysis Data
This repository contains my final project for **46W38 – Scientific Programming for Wind Energy**.

## Brief Overview of the Module
The wind_assess module provides tools to:
-    load multi-year ERA5 wind data (10 m and 100 m)
-    compute wind speed and wind direction time series
-    interpolate ERA5 data to any point inside the 2×2 grid
-    extrapolate wind to arbitrary heights using a power-law wind profile
-    fit Weibull distributions to wind speed data
-    plot wind roses and Weibull curves
-    compute Annual Energy Production (AEP) from turbine power curves (NREL 5 MW and NREL 15 MW)

The main entry point is the WindResource class, which wraps the ERA5 dataset and implements all core analysis steps. The examples/main.py script demonstrates a complete workflow from loading data to producing plots and calculating AEP.

## Module Architecture
The project follows the following src/ layout:

```
src/
 └── wind_assess/
      ├── __init__.py
      ├── core.py
      └── utils.py
```

### **core.py** – Main Functionality
Contains the WindResource class, which provides:

-    ERA5 file loading
-    wind speed & direction computation (from u/v)
-    vertical power-law extrapolation
-    Weibull parameter fitting
-    AEP calculation using turbine power curves

### **utils.py** – Helper Functions
Includes:

-    turbine power curve CSV loading
-    Weibull PDF plotting
-    wind rose plotting

### **__init__.py**
Exposes the WindResource class for simple importing.

## Architecture Diagram

```
examples/main.py
     |
wind_assess.core (WindResource)
     |
     |- Load ERA5 NetCDF files
     |- Interpolate to site coordinates
     |- Extrapolate wind to hub height
     |- Fit Weibull distribution
     |- Compute AEP from power curves
     |
wind_assess.utils
     |- Load power curve CSVs
     |- Plot Weibull fit
     |- Plot wind rose
```

## Class Description

### **WindResource (in src/wind_assess/core.py)**

Method	                            Description
from_files()	                    Loads and merges multiple ERA5 NetCDF files
get_speed_direction_at_point()	    Computes wind speed and direction at 10 m or 100 m
estimate_alpha_at_point()	        Estimates power-law shear exponent
extrapolate_speed_power_law()	    Extrapolates wind speed to any height
fit_weibull_1d()	                Fits Weibull distribution (k, A)
fit_weibull_at_point()	            Weibull fit at any height/location
compute_aep_from_power_curve()	    Computes annual energy production

## Repository Structure

```
Project03_46W38/
|- inputs/
    |- 1997-1999.nc
    |- 2000-2002.nc
    |- 2003-2005.nc
    |- 2006-2008.nc
    |- NREL_Reference_5MW_126.csv
    |- NREL_Reference_15MW_240.csv                     
|- outputs/
    |- weibull_100m_horns_rev.png
    |- wind_rose_100m_horns_rev.png             
|- src/
    |- wind_assess/
        |- __init__.py
        |- core.py
        |- utils.py      
|- examples/
    |- main.py            
|- tests/
    |- test_core.py
    |- test_utils.py                    
|- LICENSE
|- README.md
|- Project03_46W38_S253940 Final Report.docx
```

