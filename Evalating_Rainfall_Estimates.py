#%% Package imports
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#%% Define working directory
imerg_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/imergv07/data'
era5_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/era5'
cml_sat_15min_dir = '/home/kkumah/Projects/cml-stuff/out_15min_cml_rain_oper'
cml_sat_daily_dir = '/home/kkumah/Projects/cml-stuff/out_daily_cml_rain_oper' 

plot_dir = '/home/kkumah/Projects/cml-stuff/plots'

#%% Floating varibales and constants
all_imerg_files = sorted([os.path.join(imerg_dir, f) for f in os.listdir(imerg_dir) if f.endswith('.nc4')])

era5_file = os.path.join(era5_dir, 'ERA5_total_precipitation_2025_09_12_Ghana.nc')

cde_run_dte = datetime.today().strftime('%Y%m%d')

days = pd.date_range("2025-09-01", "2025-12-31", freq="D")

#%%Functions
def create_xarray(all_precip, all_time_index, lon, lat, attrs=None):
    """
    Create an xarray DataArray from the list of 2D precipitation arrays and add attributes.

    Parameters:
    - all_precip: List or array of 2D precipitation arrays
    - all_time_index: List of timestamps
    - lon: Array of longitudes
    - lat: Array of latitudes
    - attrs: Dictionary of attributes to add to the DataArray (optional)

    Returns:
    - precip_data: xarray DataArray with the specified attributes
    """
    # Create a pandas DatetimeIndex from the list of timestamps
    time_index = pd.to_datetime(all_time_index)
    
    # Create an xarray DataArray from the list of 2D precipitation arrays
    precip_data = xr.DataArray(
                data=all_precip,
                dims=["time", "lat", "lon"],
                coords={
                    "time": time_index,
                    "lat": lat,
                    "lon": lon
                }
            )

    # Add attributes if provided
    if attrs:
        precip_data.attrs.update(attrs)
    
    return precip_data
def pick_imerg_day_files(files, day_str):
    """
    day_str: 'YYYY-MM-DD'
    Select files whose path/name contains YYYYMMDD.
    """
    ymd = day_str.replace("-", "")
    return [fp for fp in files if ymd in os.path.basename(fp)]

def read_nc_imger_file(file_path):
    imerg_precip_data = xr.open_dataset(file_path)
    precip_aray = imerg_precip_data.precipitation.data     
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data['time'].values[0]
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')
    lon = imerg_precip_data.coords['lon'].values
    lat = np.flip(imerg_precip_data.coords['lat']).values    
    imerg_precip_data.close()    
    del(imerg_precip_data,imerg_time)    
    return precip_aray, imerg_time_index, lon,lat

def process_imerg(files, product):    

    all_imfn_prcp, all_imfn_tms = [], []

    for imf in files:
        prcp, tms, lon, lat = read_nc_imger_file(imf)
        all_imfn_prcp.append(prcp)
        all_imfn_tms.append(tms)

    img_lon, img_lat = lon, lat

    # sort by time
    sorted_indices = np.argsort(np.array(all_imfn_tms))
    all_imfn_prcp = np.array(all_imfn_prcp)[sorted_indices]
    all_imfn_tms = np.array(all_imfn_tms)[sorted_indices]

    # create half-hourly DataArray
    imerg_xarr_hh_data = create_xarray(
        all_imfn_prcp,
        all_imfn_tms,
        img_lon,
        img_lat,
        attrs={"product": product}
    )

    # daily date
    day = pd.to_datetime(all_imfn_tms[0]).normalize()

    # daily precipitation (mm/day)
    imerg_dd_precip = (
    (imerg_xarr_hh_data.sum(dim="time") * 0.5)
    .expand_dims(time=[day])
    )

    return imerg_xarr_hh_data, imerg_dd_precip

#%% Main processing
# 1) Read and preprocess IMERG data day by day
daily_xr = []
for day in days:
    day_str = day.strftime("%Y-%m-%d")
    imerg_day_files  = pick_imerg_day_files(all_imerg_files,  day_str)

    _,imerg_day_dat = process_imerg(imerg_day_files, product='IMERG-Late')
    daily_xr.append(imerg_day_dat)

imerg_daily_xarr = xr.concat(daily_xr, dim="time")

# 2) Read and preprocess ERA5 data
era5_data = xr.open_dataset(era5_file)
era5_data = era5_data.rename({'valid_time': 'time'})
era5_daily_data = era5_data['tp'] * 1000  # Convert from meters to mm
era5_daily_data = era5_daily_data.resample(time='1D').mean() * 24

#%% Begin evaluation and plotting
# 1) a) Spatial mean over Ghana: single value per day time series
imerg_ghana_mean = imerg_daily_xarr.mean(dim=['lat', 'lon'])
era5_ghana_mean = era5_daily_data.mean(dim=['latitude', 'longitude'])

# b) Spatial mean: 2d array
imerg_ghana_2dmean = imerg_daily_xarr.mean(dim='time')
era5_ghana_2dmean = era5_daily_data.mean(dim='time')

# c) zonal latitudinal mean
imerg_ghana_zonalmean = imerg_daily_xarr.mean(dim=['time', 'lon'])
era5_ghana_zonalmean = era5_daily_data.mean(dim=['time', 'longitude'])
