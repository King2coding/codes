#%%
"""
functions used globally
"""

#%%
# import packages
import warnings
warnings.filterwarnings("ignore")
import os
import datetime
from datetime import date
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import xarray as xr

from netCDF4 import Dataset
from scipy import stats
from scipy.stats import binned_statistic
from concurrent.futures import ProcessPoolExecutor

#%%
# flaoting variables
beam_positions = np.array(range(409))
nadir_beam_position = int(np.median(beam_positions))
reference_beam_positions = range(nadir_beam_position - 50, nadir_beam_position + 50)  # Middle 100 beam positions

limb_beam_positions = [pos for pos in beam_positions if pos not in reference_beam_positions]

# Parameters
latitude_bin_size = 5
bin_size = 1  # Temperature bin size in Kelvin
num_bins = 30

# Mapping of surface type IDs to names
surface_type_mapping = {
    0: 'water',
    1: 'snow-free land',
    2: 'snow-covered land',
    3: 'ice'
}

# Define latitude windows for Southern Hemisphere (SH) and Northern Hemisphere (NH)
latitude_windows = {
    'SH': {
        'window1': (-75, -61),
        'window2': (-61, -53)
    },
    'NH': {
        'window1': (61, 75),
        'window2': (53, 61)
    }
}


# Define the combinations of latitude windows and seasons
combinations = [
    ('SH', 'Summer', latitude_windows['SH']['window1']),
    ('SH', 'Summer', latitude_windows['SH']['window2']),
    ('SH', 'Autumn', latitude_windows['SH']['window1']),
    ('SH', 'Autumn', latitude_windows['SH']['window2']),
    ('SH', 'Winter', latitude_windows['SH']['window1']),
    ('SH', 'Winter', latitude_windows['SH']['window2']),
    ('SH', 'Spring', latitude_windows['SH']['window1']),
    ('SH', 'Spring', latitude_windows['SH']['window2'])
]

#%%
# the fucntions
# Customized surface type mapping based on hemisphere, season, and latitude range
def get_custom_surface_type_mapping(hemisphere, season, lat_range):
    seesns = ['Summer', 'Autumn', 'Winter', 'Spring']
    
    if hemisphere == 'SH':
        
        if (season in seesns) and (lat_range == (-75, -61)):
            return {
                0: 'water',
                2: 'snow-covered land',
                3: 'ice'
            }
        elif (season in ['Summer', 'Spring']) and (lat_range == (-61, -53)):
            return {
                0: 'water',                
            }
        elif (season in ['Autumn', 'Winter']) and (lat_range == (-61, -53)):
            return {
                0: 'water',
                3: 'ice'
            }
        
    elif hemisphere == 'NH':
        if (season in ['Winter', 'Autumn']) and (lat_range == (61, 75)):
            return {
                0: 'water',
                2: 'snow-covered land',
                3: 'ice'
            }
        elif (season in ['Spring', 'Summer']) and (lat_range == (61, 75)):
            return {
                0: 'water',
                2: 'snow-free land',                
            }
        
        elif (season in ['Spring', 'Summer']) and (lat_range == (53, 61)):
            return {
                0: 'water',
                2: 'snow-free land',                
            }
        
        elif (season in ['Winter', 'Autumn']) and (lat_range == (53, 61)):
            return {
                0: 'water',
                2: 'snow-covered land',    
                3: 'ice'            
            }       
#----------------------------------------------   
# Function to find season given month
def find_season(month, hemisphere):
    if hemisphere == 'Southern':
        season_month_south = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'}
        return season_month_south.get(month)
        
    elif hemisphere == 'Northern':
        season_month_north = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        return season_month_north.get(month)
    else:
        print('Invalid selection. Please select a hemisphere and try again')
#----------------------------------------------

# Function to calculate the month from DOY
def calculate_month(doy, year):
    """Calculate the month from DOY (day of year)."""
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    return date.month
#----------------------------------------------

# Organize folders by seasons
def organize_by_season_with_hemisphere(base_path, hemisphere, year):
    """Organize folders into seasons using find_season."""
    seasons = {"Summer": [], "Autumn": [], "Winter": [], "Spring": []}
    
    for folder_name in os.listdir(base_path):
        if folder_name.isdigit():  # Check if folder name is numeric (DOY)
            doy = int(folder_name)
            month = calculate_month(doy, year)
            season = find_season(month, hemisphere)
            if season:
                seasons[season].append(folder_name)
    
    return seasons
#----------------------------------------------
# Load files for a specific season
def load_files_for_season(base_path, season_folders):
    """Load all .nc files from specified folders."""
    nc_files = []
    for folder_name in season_folders:
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            nc_files.extend(
                [
                    os.path.join(folder_path, file)
                    for file in os.listdir(folder_path)
                    if file.endswith(".nc")
                ]
            )
    return nc_files
#----------------------------------------------

def get_elements_nadir(list_of_arrays, positions):
    elem = [n[:, positions].flatten()[~np.isnan(n[:, positions].flatten())] for n in list_of_arrays if not np.all(np.isnan(n[:, positions]))]
    return np.hstack(elem)
#----------------------------------------------

def create_a_lat_mask(lat_data, target_lat, tolerance):
    """
    Create a 2D mask where latitude values are close to the target latitude.
    
    Parameters:
    - lat_data: 2D array of latitude values.
    - target_lat: Target latitude value to match.
    - tolerance: Allowed deviation from the target latitude (default 0.5 degrees).
    
    Returns:
    - mask: 2D boolean array with True where the latitude values are close to target_lat.
    """
    # Ensure the input is a NumPy array
    lat_data = np.array(lat_data)
    
    # Create a mask with NaN-safe comparison
    mask = np.abs(lat_data - target_lat) <= tolerance
    mask = np.where(np.isnan(lat_data), False, mask)  # Exclude NaN areas
    
    return mask
#----------------------------------------------
def get_elements_limb(list_of_arrays, positions):
    limb_elem = [
        n[:, positions].flatten()[~np.isnan(n[:, positions].flatten())]
        for n in list_of_arrays if not np.all(np.isnan(n[:, positions]))
    ]
    if limb_elem:
        return np.hstack(limb_elem)
    else:
        return np.array([])
#----------------------------------------------

def create_histogram(temp_values, bin_size, temp_range):    
    # Create bins with a specific step size
    bins = np.arange(temp_range[0], 
                     temp_range[1] + bin_size, 
                     bin_size)

    # Compute the histogram
    hist, bin_edges, _ = binned_statistic(x=temp_values, 
                                          values=temp_values, 
                                          statistic='count', 
                                          bins=bins)
    
    return hist, bin_edges[:-1]
#----------------------------------------------

def get_group_data(files, ir_var, hem, seasn, lat_window):
    """
    Process a list of NetCDF files to extract and group infrared brightness temperature data 
    based on surface type and latitude window.
    
    Parameters:
    files (list of str): List of file paths to NetCDF files to be processed.
    ir_var (str): The variable name in the NetCDF files representing the infrared brightness temperature.
    lat_window (tuple of float): A tuple specifying the latitude range (min_lat, max_lat) to filter the data.
    
    Returns:
    tuple: A tuple containing:
        - group_arrays_dict (dict): A dictionary where keys are surface types ('water', 'snow-free land', 
          'snow-covered land', 'ice') and values are lists of numpy arrays containing the grouped brightness 
          temperature data.
        - data_ranges (dict): A dictionary where keys are surface types and values are tuples representing 
          the minimum and maximum brightness temperature values for each surface type.
    """
    count = 0
    surface_type_elements = get_custom_surface_type_mapping(hem, seasn, lat_window)
    print(f"Processing {list(surface_type_elements.values())} data for {hem} hemisphere in {seasn} season")

    group_arrays_dict = {surface_type_id: [] for surface_type_id in surface_type_elements.keys()}
    data_ranges = {}

    for file in sorted(files):
        data = xr.open_dataset(file)
        lats = data['latitude'][:, :].data
        brightness_temp_11um = data[ir_var].data
        cloud_probability = data['cloud_probability'].data
        surfact_type = data['land_class'].data

        for surface_type_id, surface_type_name in surface_type_elements.items():
            max_lat, min_lat = max(lat_window), min(lat_window)
            mask = ((lats > min_lat) & (lats <= max_lat)) & \
                   (cloud_probability >= 0.5) & (surfact_type == surface_type_id)
            mask = np.where((mask == True), 1, np.nan)
            group_data = np.where(mask == 1, brightness_temp_11um, np.nan)

            if not np.all(np.isnan(group_data)):
                group_arrays_dict[surface_type_id].append(group_data)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(files)} ({(count / len(files)) * 100:.2f}%) of the total {len(files)} files")

    for surface_type_id, group_arrays in group_arrays_dict.items():
        if group_arrays:
            data_min = min(np.nanmin(i) for i in group_arrays)
            data_max = max(np.nanmax(i) for i in group_arrays)
            data_ranges[surface_type_id] = (data_min, data_max)
        else:
            data_ranges[surface_type_id] = (None, None)

    return group_arrays_dict, data_ranges
#----------------------------------------------

def process_group_data_by_variable(variable_name, seasonal_files):
    group_arrays_dict = {}
    data_ranges = {}

    # Loop through each combination and process the data
    for hemisphere, season, lat_window in combinations:
        if season in seasonal_files.keys():
            season_files = seasonal_files[season]
        
            # Process SH data
            print(f"Processing {lat_window} window in {hemisphere} hemisphere in {season} season")
            key_sh = f"SH_{season}_{lat_window}"
            group_arrays_dict[key_sh], data_ranges[key_sh] = get_group_data(
                season_files, variable_name, 'SH', season, lat_window
            )
            
            # Define corresponding NH data based on SH info
            nh_season = {
                'Summer': 'Winter',
                'Autumn': 'Spring',
                'Winter': 'Summer',
                'Spring': 'Autumn'
            }[season]
            
            # Determine the corresponding NH latitude window based on SH latitude window
            sh_window_key = next(key for key, value in latitude_windows['SH'].items() if value == lat_window)
            nh_lat_window = latitude_windows['NH'][sh_window_key]

            print(f"Processing {nh_lat_window} window in NH hemisphere in {nh_season} season")
            
            key_nh = f"NH_{nh_season}_{nh_lat_window}"
            group_arrays_dict[key_nh], data_ranges[key_nh] = get_group_data(
                season_files, variable_name, 'NH', nh_season, nh_lat_window
            )
    
    return group_arrays_dict, data_ranges

#----------------------------------------------

def process_file(file, ir_var, surface_type_elements, lat_window):
    data = xr.open_dataset(file)
    lats = data['latitude'][:, :].data
    brightness_temp_11um = data[ir_var].data
    cloud_probability = data['cloud_probability'].data
    surfact_type = data['land_class'].data

    file_group_arrays_dict = {surface_type_id: [] for surface_type_id in surface_type_elements.keys()}

    for surface_type_id in surface_type_elements.keys():
        max_lat, min_lat = max(lat_window), min(lat_window)
        mask = ((lats > min_lat) & (lats <= max_lat)) & \
               (cloud_probability >= 0.5) & (surfact_type == surface_type_id)
        mask = np.where((mask == True), 1, np.nan)
        group_data = np.where(mask == 1, brightness_temp_11um, np.nan)

        if not np.all(np.isnan(group_data)):
            file_group_arrays_dict[surface_type_id].append(group_data)

    return file_group_arrays_dict

def get_group_data_parallel(files, ir_var, hem, seasn, lat_window):
    """
    Process a list of NetCDF files to extract and group infrared brightness temperature data 
    based on surface type and latitude window using parallel processing.
    
    Parameters:
    files (list of str): List of file paths to NetCDF files to be processed.
    ir_var (str): The variable name in the NetCDF files representing the infrared brightness temperature.
    lat_window (tuple of float): A tuple specifying the latitude range (min_lat, max_lat) to filter the data.
    
    Returns:
    tuple: A tuple containing:
        - group_arrays_dict (dict): A dictionary where keys are surface types ('water', 'snow-free land', 
          'snow-covered land', 'ice') and values are lists of numpy arrays containing the grouped brightness 
          temperature data.
        - data_ranges (dict): A dictionary where keys are surface types and values are tuples representing 
          the minimum and maximum brightness temperature values for each surface type.
    """

    count = 0
    surface_type_elements = get_custom_surface_type_mapping(hem, seasn, lat_window)
    print(f"Processing {list(surface_type_elements.values())} data for {hem} hemisphere in {seasn} season")

    group_arrays_dict = {surface_type_id: [] for surface_type_id in surface_type_elements.keys()}
    data_ranges = {}

    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_file, 
                                    sorted(files), 
                                    [ir_var]*len(files), 
                                    [surface_type_elements]*len(files), 
                                    [lat_window]*len(files), 
                                    [cor_dir]*len(files)))

    for file_group_arrays_dict in results:
        for surface_type_id, group_arrays in file_group_arrays_dict.items():
            group_arrays_dict[surface_type_id].extend(group_arrays)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(files)} ({(count / len(files)) * 100:.2f}%) of the total {len(files)} files")

    for surface_type_id, group_arrays in group_arrays_dict.items():
        if group_arrays:
            data_min = min(np.nanmin(i) for i in group_arrays)
            data_max = max(np.nanmax(i) for i in group_arrays)
            data_ranges[surface_type_id] = (data_min, data_max)
        else:
            data_ranges[surface_type_id] = (None, None)

    return group_arrays_dict, data_ranges

#------------------------------------------

def get_nadir_bins_and_histogram(array_data, bin_size, data_range):

    all_nadirs = get_elements_nadir(array_data, reference_beam_positions)
    all_nadirs = np.round(all_nadirs,3)
    # Create histograms with independent ranges
    nadir_hist, nadir_bins = create_histogram(all_nadirs, bin_size, data_range)
    valid_nadir_indices = nadir_hist > 0
    nadir_bins = nadir_bins[valid_nadir_indices]
    nadir_hist = nadir_hist[valid_nadir_indices]
    return nadir_bins, nadir_hist, all_nadirs

#----------------------------------------------

def process_to_get_nadir_stats(group_arrays_dict, data_ranges, bin_size):
    nadir_bins_by_srftype = {}
    nadir_hist_by_srftype = {}
    all_nadirs_by_srftype = {}

    for surf_type, group_arrays in group_arrays_dict.items():
        print(f"Processing {surf_type} data")
        surf_type_grp_array = group_arrays
        srf_type_data_ranges = data_ranges[surf_type]

        nadir_bins, nadir_hist, all_nadirs = get_nadir_bins_and_histogram(surf_type_grp_array, 
                                                                          bin_size, 
                                                                          srf_type_data_ranges)
        nadir_bins_by_srftype[surf_type] = nadir_bins
        nadir_hist_by_srftype[surf_type] = nadir_hist
        all_nadirs_by_srftype[surf_type] = all_nadirs

        del(surf_type, group_arrays, 
            surf_type_grp_array, srf_type_data_ranges)
    
    return nadir_bins_by_srftype, nadir_hist_by_srftype, all_nadirs_by_srftype

#----------------------------------------------
def process_nadir_stats_by_hem_season_lat_var(group_arrays_dict, data_ranges, combinations, bin_size):
    sh_nadir_bins_by_srftype, sh_nadir_hist_by_srftype, sh_all_nadirs_by_srftype = {}, {}, {}
    nh_nadir_bins_by_srftype, nh_nadir_hist_by_srftype, nh_all_nadirs_by_srftype = {}, {}, {}

    for hemisphere, season, lat_window in combinations:
        key_sh = f"SH_{season}_{lat_window}"        
        sh_nadir_bins_by_srftype[key_sh], sh_nadir_hist_by_srftype[key_sh], sh_all_nadirs_by_srftype[key_sh] = process_to_get_nadir_stats(
            group_arrays_dict[key_sh], 
            data_ranges[key_sh], 
            bin_size
        )

        # Define corresponding NH data based on SH info
        nh_season = {
            'Summer': 'Winter',
            'Autumn': 'Spring',
            'Winter': 'Summer',
            'Spring': 'Autumn'
        }[season]
        
        # Determine the corresponding NH latitude window based on SH latitude window
        sh_window_key = next(key for key, value in latitude_windows['SH'].items() if value == lat_window)
        nh_lat_window = latitude_windows['NH'][sh_window_key]
        key_nh = f"NH_{nh_season}_{nh_lat_window}"

        nh_nadir_bins_by_srftype[key_nh], nh_nadir_hist_by_srftype[key_nh], nh_all_nadirs_by_srftype[key_nh] = process_to_get_nadir_stats(
            group_arrays_dict[key_nh], 
            data_ranges[key_nh], 
            bin_size
        )

    return (sh_nadir_bins_by_srftype, sh_nadir_hist_by_srftype, sh_all_nadirs_by_srftype,
            nh_nadir_bins_by_srftype, nh_nadir_hist_by_srftype, nh_all_nadirs_by_srftype)

#----------------------------------------------

def get_limb_bins_and_histogram(group_data, bm_pos, data_range):
    all_limbs_at_i = get_elements_limb(group_data, bm_pos)
    all_limbs_at_i = np.round(all_limbs_at_i,3)              

    # Create histograms with independent ranges
    limb_hist, limb_bins = create_histogram(all_limbs_at_i, bin_size, data_range)

    # Find the indices where counts in `limb_hist` are greater than zero
    valid_limb_indices = limb_hist > 0
    limb_bins = limb_bins[valid_limb_indices]
    limb_hist = limb_hist[valid_limb_indices]

    return limb_bins, limb_hist, all_limbs_at_i

#----------------------------------------------
def get_LUT(group_data, nadir_bins, nadir_hist, data_range, lat_window, surface_type):

    # Initialize lookup table and list baskets
    lookup_table = []
    limb_bin_list =  {}
    limb_hist_list = {}
    nadir_bin_list = {} 
    nadir_hist_list = {}
    adjusted_limb_bins_list = {}
    all_limbs_at_i_list = {}

    for i in limb_beam_positions:
        # Adjust the range dynamically based on beam position edges
        start = max(0, i - 10)  # Ensure the range doesn't go below 0
        end = min(409, i + 10)  # Ensure the range doesn't exceed 409
        limb_pos_rng = range(start, end)

        # Collect all data within the current window
        limb_bins, limb_hist, all_limbs_at_i = get_limb_bins_and_histogram(group_data, limb_pos_rng, data_range) 
        all_limbs_at_i_list[i] = all_limbs_at_i
        limb_hist_list[i] = limb_hist
        limb_bin_list[i] = limb_bins 

        # Perform limb-to-nadir adjustment    
        adjusted_limb_bins, nadir_hist_norm = adjust_limb_bins_bob_method_upgraded(
                                              nadir_bins, nadir_hist,limb_bins, limb_hist) 
        adjusted_limb_bins = np.round(adjusted_limb_bins,3)
        adjusted_limb_bins_list[i] = adjusted_limb_bins
        nadir_hist_list[i] = nadir_hist_norm
        nadir_bin_list[i] = nadir_bins  

        # Populate lookup table
        for orig_tb, corr_tb in zip(limb_bins, adjusted_limb_bins):
            lookup_table.append({            
                "latitude_bin": f"{lat_window[0]}-{lat_window[1]}",
                # "cloud_probability": 0.5,
                "surface_type": surface_type,
                "beam_position": i,
                "original_tb": orig_tb,
                "corrected_tb": corr_tb,
                "corr_coeff": corr_tb/orig_tb
            }) 
    return lookup_table, limb_hist_list, limb_bin_list, nadir_hist_list, \
            nadir_bin_list, adjusted_limb_bins_list, all_limbs_at_i_list

#----------------------------------------------
def process_to_get_limb_stats(group_arrays_dict, data_ranges, 
                              nadir_bins_by_srftype, nadir_hist_by_srftype, 
                              lat_wind):
    
    lookup_table_by_srftype = {}
    limb_hist_list_by_srftype = {}
    limb_bin_list_by_srftype = {}
    nadir_hist_list_by_srftype = {}
    nadir_bin_list_by_srftype = {}
    adjusted_limb_bins_list_by_srftype = {}
    all_limbs_at_i_list_by_srftype = {}

    for surf_type, group_arrays in group_arrays_dict.items():

        surf_type_grp_array = group_arrays
        srf_type_data_ranges = data_ranges[surf_type]

        srf_type_nadir_bins = nadir_bins_by_srftype[surf_type]
        srf_type_nadir_hist = nadir_hist_by_srftype[surf_type]

        lookup_table, limb_hist_list, limb_bin_list, nadir_hist_list, \
        nadir_bin_list, adjusted_limb_bins_list, all_limbs_at_i_list = get_LUT(surf_type_grp_array, 
                                                                               srf_type_nadir_bins, 
                                                                               srf_type_nadir_hist, 
                                                                               srf_type_data_ranges, 
                                                                               lat_wind, surf_type)
        lookup_table_by_srftype[surf_type] = pd.DataFrame(lookup_table)
        limb_hist_list_by_srftype[surf_type] = limb_hist_list
        limb_bin_list_by_srftype[surf_type] = limb_bin_list
        nadir_hist_list_by_srftype[surf_type] = nadir_hist_list
        nadir_bin_list_by_srftype[surf_type] = nadir_bin_list
        adjusted_limb_bins_list_by_srftype[surf_type] = adjusted_limb_bins_list
        all_limbs_at_i_list_by_srftype[surf_type] = all_limbs_at_i_list

        print(f"Processed {surf_type} data")
    
    return (lookup_table_by_srftype, 
            limb_hist_list_by_srftype, limb_bin_list_by_srftype, 
            nadir_hist_list_by_srftype, nadir_bin_list_by_srftype, 
            adjusted_limb_bins_list_by_srftype, 
            all_limbs_at_i_list_by_srftype)

#----------------------------------------------

def process_surface_type(surf_type, group_arrays, srf_type_data_ranges, srf_type_nadir_bins, srf_type_nadir_hist, lat_wind):
    lookup_table, limb_hist_list, limb_bin_list, nadir_hist_list, \
    nadir_bin_list, adjusted_limb_bins_list, all_limbs_at_i_list = get_LUT(group_arrays, 
                                                                           srf_type_nadir_bins, 
                                                                           srf_type_nadir_hist, 
                                                                           srf_type_data_ranges, 
                                                                           lat_wind, surf_type)
    return (surf_type, pd.DataFrame(lookup_table), limb_hist_list, limb_bin_list, nadir_hist_list, 
            nadir_bin_list, adjusted_limb_bins_list, all_limbs_at_i_list)
#----------------------------------------------

def process_to_get_limb_stats_fast(group_arrays_dict, data_ranges, 
                                   nadir_bins_by_srftype, nadir_hist_by_srftype, 
                                   lat_wind):

    lookup_table_by_srftype = {}
    limb_hist_list_by_srftype = {}
    limb_bin_list_by_srftype = {}
    nadir_hist_list_by_srftype = {}
    nadir_bin_list_by_srftype = {}
    adjusted_limb_bins_list_by_srftype = {}
    all_limbs_at_i_list_by_srftype = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_surface_type, surf_type, group_arrays, data_ranges[surf_type], 
                            nadir_bins_by_srftype[surf_type], nadir_hist_by_srftype[surf_type], lat_wind)
            for surf_type, group_arrays in group_arrays_dict.items()
        ]

        for future in futures:
            surf_type, lookup_table, limb_hist_list, limb_bin_list, nadir_hist_list, \
            nadir_bin_list, adjusted_limb_bins_list, all_limbs_at_i_list = future.result()
            
            lookup_table_by_srftype[surf_type] = lookup_table
            limb_hist_list_by_srftype[surf_type] = limb_hist_list
            limb_bin_list_by_srftype[surf_type] = limb_bin_list
            nadir_hist_list_by_srftype[surf_type] = nadir_hist_list
            nadir_bin_list_by_srftype[surf_type] = nadir_bin_list
            adjusted_limb_bins_list_by_srftype[surf_type] = adjusted_limb_bins_list
            all_limbs_at_i_list_by_srftype[surf_type] = all_limbs_at_i_list

            print(f"Processed {surf_type} data")
    
    return (lookup_table_by_srftype, 
            limb_hist_list_by_srftype, limb_bin_list_by_srftype, 
            nadir_hist_list_by_srftype, nadir_bin_list_by_srftype, 
            adjusted_limb_bins_list_by_srftype, 
            all_limbs_at_i_list_by_srftype)

#----------------------------------------------

def process_limb_stats_by_hem_season_lat_var(group_arrays_dict, data_ranges, nadir_bins_by_srftype, nadir_hist_by_srftype, combinations):
    # Initialize dictionaries to store results
    lookup_table_by_srftype_sh, limb_hist_list_by_srftype_sh, limb_bin_list_by_srftype_sh = {}, {}, {}
    nadir_hist_list_by_srftype_sh, nadir_bin_list_by_srftype_sh, adjusted_limb_bins_list_by_srftype_sh = {}, {}, {}
    all_limbs_at_i_list_by_srftype_sh = {}

    lookup_table_by_srftype_nh, limb_hist_list_by_srftype_nh, limb_bin_list_by_srftype_nh = {}, {}, {}
    nadir_hist_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, adjusted_limb_bins_list_by_srftype_nh = {}, {}, {}
    all_limbs_at_i_list_by_srftype_nh = {}

    # Process limb stats for SH and NH
    for hemisphere, season, lat_window in combinations:
        key_sh = f"SH_{season}_{lat_window}"
        
        # Process SH data
        (lookup_table_by_srftype_sh[key_sh], limb_hist_list_by_srftype_sh[key_sh], limb_bin_list_by_srftype_sh[key_sh], 
         nadir_hist_list_by_srftype_sh[key_sh], nadir_bin_list_by_srftype_sh[key_sh], adjusted_limb_bins_list_by_srftype_sh[key_sh], 
         all_limbs_at_i_list_by_srftype_sh[key_sh]) = process_to_get_limb_stats_fast(
                                                             group_arrays_dict[key_sh], 
                                                             data_ranges[key_sh], 
                                                             nadir_bins_by_srftype[key_sh], 
                                                             nadir_hist_by_srftype[key_sh], 
                                                             lat_window)
        
        # Define corresponding NH data based on SH info
        nh_season = {
            'Summer': 'Winter',
            'Autumn': 'Spring',
            'Winter': 'Summer',
            'Spring': 'Autumn'
        }[season]
        
        # Determine the corresponding NH latitude window based on SH latitude window
        sh_window_key = next(key for key, value in latitude_windows['SH'].items() if value == lat_window)
        nh_lat_window = latitude_windows['NH'][sh_window_key]
        key_nh = f"NH_{nh_season}_{nh_lat_window}"

        # Process NH data
        (lookup_table_by_srftype_nh[key_nh], limb_hist_list_by_srftype_nh[key_nh], limb_bin_list_by_srftype_nh[key_nh],
         nadir_hist_list_by_srftype_nh[key_nh], nadir_bin_list_by_srftype_nh[key_nh], adjusted_limb_bins_list_by_srftype_nh[key_nh],
         all_limbs_at_i_list_by_srftype_nh[key_nh]) = process_to_get_limb_stats_fast(
                                                                    group_arrays_dict[key_nh],
                                                                    data_ranges[key_nh],
                                                                    nadir_bins_by_srftype[key_nh],
                                                                    nadir_hist_by_srftype[key_nh],
                                                                    nh_lat_window)

    return (lookup_table_by_srftype_sh, limb_hist_list_by_srftype_sh, limb_bin_list_by_srftype_sh, 
            nadir_hist_list_by_srftype_sh, nadir_bin_list_by_srftype_sh, adjusted_limb_bins_list_by_srftype_sh, 
            all_limbs_at_i_list_by_srftype_sh, lookup_table_by_srftype_nh, limb_hist_list_by_srftype_nh, 
            limb_bin_list_by_srftype_nh, nadir_hist_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, 
            adjusted_limb_bins_list_by_srftype_nh, all_limbs_at_i_list_by_srftype_nh)

#----------------------------------------------
def create_histogram_of_histogram(binss,hists):
    bin_edges = np.linspace(binss.min(), binss.max(), 100)  # Adjust number of bins
    binned_hist, _, _ = binned_statistic(binss, hists, 
                                         statistic='sum', bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, binned_hist
    
# Function to normalize histograms
def normalize_histogram(hist):
    total_count = np.sum(hist)
    return hist / total_count if total_count > 0 else hist
#----------------------------------------------

def generate_bins_with_fixed_step_and_count(start, end, step=1, total_bins=None):
    """
    Generate bins with a fixed step size and ensure a total number of bins.

    Parameters:
    - start (float): Minimum value in the range (from observational data).
    - end (float): Maximum value in the range (from observational data).
    - step (float): Step size for the bins (default is 1).
    - total_bins (int, optional): Desired total number of bins. If provided, adjusts range.

    Returns:
    - bins (np.ndarray): Array of bin edges with specified step size and total bins.
    """
    if step <= 0:
        raise ValueError("Step size must be positive.")
    
    # Adjust start and end if total_bins is specified
    if total_bins is not None:
        end = start + (total_bins - 1) * step
    
    # Create bins with the given step size
    bins = np.arange(start, end + step, step)
    
    return bins
#----------------------------------------------

def generate_bins_preserving_range(start, end, total_bins):
    # Ensure step size is fixed at 1
    step = bin_size
    # Compute adjusted total bins to fit within the original range
    new_total_bins = int((end - start) / step) + 1
    
    # Adjust total bins to the desired count
    if new_total_bins < total_bins:
        # Expand range to fit desired total bins
        end = start + (total_bins - 1) * step
    elif new_total_bins > total_bins:
        # Shrink range to fit desired total bins
        end = start + (total_bins - 1) * step

    # Generate bins
    bins = np.linspace(start, end, total_bins + 1)
    return bins
#----------------------------------------------

def generate_dist(data):
    valid_data = [i for i in data if not np.all(np.isnan(i))]
    temp_min = min([np.nanmin(i) for i in valid_data])
    temp_max = max([np.nanmax(i) for i in valid_data])
    temp_range = (temp_min, temp_max)
    num_bins = 100
    # bins = generate_bins_preserving_range(temp_min, temp_max, num_bins) 
    # bins = np.arange(temp_range[0], temp_range[1] + bin_size, bin_size)
    # Get the number of columns from the first 2D array
    num_columns = data[0].shape[1]
    hists = []
    
    for x in valid_data:
        for i in range(num_columns):
            # Flatten the column and exclude NaN values
            column_data = x[:, i].flatten()[~np.isnan(x[:, i].flatten())]
            if column_data.size > 0:  # Ensure there's valid data
                hist, _, _ = binned_statistic(x = column_data, 
                                              values = column_data, 
                                              statistic = 'count', 
                                              bins = num_bins,
                                              range = temp_range)
                # hist,_ = np.histogram(column_data, bins=bins)
                hists.append(hist / hist.sum(axis=0, keepdims=True))
                # hists.append(hist)
    del(x,i)
    
    # Initialize a list to store values for each column
    column_values = [[] for _ in range(num_columns)]

    # Loop through each 2D array in data
    for x in data:
        # Loop through each column
        for col in range(x.shape[1]):
            column_data = x[:, col].flatten()[~np.isnan(x[:, col].flatten())]
            if column_data.size > 0:
                # Append non-NaN values from the current column to the respective list
                column_values[col].extend(column_data)

    # Calculate the mean for each column
    column_means = [np.nanmean(values) if values else np.nan for values in column_values]

    return np.array(hists), column_means, temp_range

#----------------------------------------------
def generate_dist_(data):
    num_bins = 100
    valid_data = [i for i in data if not np.all(np.isnan(i))]   
    temp_min_ = min([np.nanmin(i) for i in valid_data])
    temp_max_ = max([np.nanmax(i) for i in valid_data])
    temp_range_ = (temp_min_, temp_max_)
    # Get the number of columns from the first 2D array
    num_columns = valid_data[0].shape[1]
    hists = []
    beam_position_means = []

    for i in range(num_columns):
        column_data_at_i = [] 
        # column_data_at_i = [[] for _ in range(num_columns)]   
        for x in valid_data:
            column_data = x[:, i].flatten()
            # Flatten the column and exclude NaN values
            column_data = column_data[~np.isnan(column_data)]
            if not np.all(np.isnan(column_data)):  # Ensure there's valid data
                column_data_at_i.append(column_data)
                # column_data_at_i[i].extend(column_data)

        column_data_at_i_arr = np.hstack(column_data_at_i)
        # column_data_at_i_arr = np.array(column_data_at_i[i])
        temp_min = np.nanmin(column_data_at_i_arr)
        temp_max = np.nanmax(column_data_at_i_arr)
        temp_range = (temp_min, temp_max)        
        # bins = generate_bins_with_fixed_step_and_count(temp_range_[0], temp_range_[1], 1, num_bins)        
        # hist,_ = np.histogram(column_data_at_i_arr, bins=bins)
        hist, bins, _ = binned_statistic(x = column_data, 
                                      values = column_data, 
                                      statistic = 'count', 
                                      bins = num_bins, 
                                      range = temp_range_)
        hists.append(hist / np.nansum(hist)) # hist.sum(axis=0, keepdims=True)

        # get means at each beam position
        beam_position_ir_mean = np.nanmean(column_data_at_i_arr)
        beam_position_means.append(beam_position_ir_mean)
                # hists.append(hist)
    del(x,i)
    return np.array(hists), beam_position_means, temp_range_, bins

#----------------------------------------------

def generate_distribution_with_means(data):
    """
    Generate histograms for IR Tbs stratified by beam positions and compute mean IR Tbs.

    Parameters:
    - data: List of 2D arrays, where each array represents beam positions vs. IR Tbs.
    - num_bins: Number of bins for the histograms.

    Returns:
    - hists: 2D array of histograms [beam position, bin counts].
    - bin_edges: Edges of the bins used for histograms.
    - beam_position_means: List of mean IR Tbs for each beam position.
    """
    valid_data = [i for i in data if not np.all(np.isnan(i))]
    temp_min = min([np.nanmin(i) for i in valid_data])
    temp_max = max([np.nanmax(i) for i in valid_data]) 

    bins = np.arange(temp_min, temp_max + bin_size, bin_size)
    num_beam_positions = valid_data[0].shape[1]
    hists = []
    beam_position_means = []

    for i in range(num_beam_positions):
        beam_data = np.hstack([x[:, i][~np.isnan(x[:, i])] for x in valid_data])
        # hist, _ = np.histogram(beam_data, bins=bins)
        # Compute the histogram
        hist, bin_edges, _ = binned_statistic(
        x=beam_data, 
        values=beam_data, 
        statistic='count', 
        bins=bins
        )
        hists.append(hist / hist.sum() if hist.sum() > 0 else hist)
        beam_position_mean = np.nanmean(beam_data) if beam_data.size > 0 else np.nan
        beam_position_means.append(beam_position_mean)

    return np.array(hists), bin_edges, beam_position_means

#----------------------------------------------

# Define custom colormap
def create_custom_cmap(mn,mx,nm):
    colors = [
        (1.0, 1.0, 1.0),  # White for the lowest values
        (0.8, 0.9, 1.0),  # Light blue
        (0.5, 0.7, 1.0),  # Sky blue
        (0.3, 0.5, 0.8),  # Medium blue
        (0.2, 0.4, 1.0),  # Dark blue
        (0.4, 0.6, 0.2),  # Light green
        (0.6, 0.8, 0.2),  # Yellow-green
        (0.8, 0.9, 0.4),  # Yellow
        (0.9, 0.8, 0.2),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.4, 0.0),  # Deep orange
        (1.0, 0.2, 0.2),  # Light red
        (1.0, 0.0, 0.0),  # Red
    ]
    # colors = [
    #     (1.0, 1.0, 1.0),  # White for the lowest values
    #     (0.8, 0.8, 1.0),  # Light blue
    #     (0.5, 0.7, 1.0),  # Sky blue
    #     (0.3, 0.5, 0.8),  # Medium blue
    #     (1.0, 1.0, 0.6),  # Yellow
    #     (1.0, 0.6, 0.0),  # Orange
    #     (1.0, 0.0, 0.0)   # Red for the highest values
    # ]

    # Create a linear segmented colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_poster_cmap", colors)

    # Define discrete boundaries (for a linear range, adjust as needed)
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(mn, mx, nm), ncolors=cmap.N, clip=True)

    return cmap, norm

#----------------------------------------------

def process_limb_positions_with_running_window(
    group_arrays, limb_beam_positions, nadir_bins, nadir_hist, temp_range,
    bin_size, lat_val, min_lat, max_lat, window_size=5
):
    """
    Process limb beam positions using a running window approach and populate a lookup table.

    Parameters:
    - group_arrays: List of arrays containing limb data.
    - limb_beam_positions: List of beam positions to process.
    - nadir_bins: Bin edges for nadir histograms.
    - nadir_hist: Nadir histogram data.
    - temp_range: Tuple of (min_temp, max_temp) for histogram binning.
    - bin_size: Size of each histogram bin.
    - lat_val: Current latitude value for lookup table.
    - min_lat, max_lat: Latitude bin bounds for lookup table.
    - window_size: Size of the running window (default is 5).

    Returns:
    - lookup_table: A list of dictionaries with computed corrections.
    - results: A dictionary containing processed histograms and bins.
    """
    # Initialize results and lookup table
    lookup_table = []
    results = {
        "all_limbs_at_i":{},
        "limb_hist_list": {},
        "limb_bin_list": {},
        "adjusted_limb_bins_list": {},
        "nadir_hist_list": {},
        "nadir_bin_list": {}
    }

    # Loop through limb beam positions in steps of `2*window_size + 1`
    step_size = 2 * window_size + 1
    for i in range(0, len(limb_beam_positions), step_size):
        # Define the range for the current window
        start = max(0, limb_beam_positions[i] - window_size)
        end = min(409, limb_beam_positions[i] + window_size + 1)  # +1 to include the last beam in range
        limb_pos_rng = range(start, end)

        # Collect all data within the current window
        all_limbs_at_i = get_elements_limb(group_arrays, limb_pos_rng)
        all_limbs_at_i = np.round(all_limbs_at_i, 3)

        # Create histograms for the entire window
        limb_hist, limb_bins = create_histogram(all_limbs_at_i, 
                                                bin_size, 100, 
                                                temp_range)

        # Perform limb-to-nadir adjustment
        adjusted_limb_bins, nadir_hist_norm = adjust_limb_bins_bob_method(
            nadir_bins, nadir_hist, limb_bins, limb_hist
        )
        adjusted_limb_bins = np.round(adjusted_limb_bins, 3)

        # Assign results to all positions in the current window
        for pos in limb_pos_rng:
            results["limb_hist_list"][pos] = limb_hist
            results["limb_bin_list"][pos] = limb_bins
            results["adjusted_limb_bins_list"][pos] = adjusted_limb_bins
            results["nadir_hist_list"][pos] = nadir_hist_norm
            results["nadir_bin_list"][pos] = nadir_bins
            results["all_limbs_at_i"][pos] = all_limbs_at_i

            # Populate the lookup table for each position
            for orig_tb, corr_tb in zip(limb_bins, adjusted_limb_bins):
                lookup_table.append({
                    "latitude": lat_val,
                    "latitude_bin": f"{min_lat}-{max_lat}",
                    "surface_type": 0,
                    "beam_position": pos,
                    "original_tb": orig_tb,
                    "corrected_tb": corr_tb,
                    "corr_coeff": corr_tb / orig_tb
                })

    return lookup_table, results

#-------------------------------------------------
def adjust_limb_bins_refined(nadir_bins, nadir_hist, limb_bins, limb_hist):
    # Normalize nadir_hist to match the sum of limb_hist
    hist_coef = np.sum(nadir_hist) / np.sum(limb_hist)
    nadir_hist_norm = nadir_hist / hist_coef  # Scaled nadir histogram

    adjusted_limb_bins = []  # To store the adjusted limb bin values
    remaining_samples = nadir_hist_norm.copy()  # Keep track of remaining samples in nadir_hist_norm

    # Loop through each limb bin
    for limb_count in limb_hist:
        collected_samples = 0  # Tracks the number of samples collected for the current limb bin
        weighted_tb_sum = 0  # Weighted sum for the current limb bin
        bin_index = 0  # Start from the first nadir bin

        while collected_samples < limb_count and bin_index < len(nadir_bins):
            available_samples = remaining_samples[bin_index]  # Samples available in the current nadir bin

            if available_samples > 0:
                if collected_samples + available_samples <= limb_count:
                    # Collect all samples from the current nadir bin
                    weight = available_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += available_samples
                    remaining_samples[bin_index] = 0  # Mark the nadir bin as exhausted
                else:
                    # Collect only the remaining required samples
                    needed_samples = limb_count - collected_samples
                    weight = needed_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += needed_samples
                    remaining_samples[bin_index] -= needed_samples  # Reduce the count in the nadir bin

            # Move to the next nadir bin
            bin_index += 1

        # Compute the adjusted value for the current limb bin
        adjusted_tb = weighted_tb_sum
        adjusted_limb_bins.append(adjusted_tb)

    return np.array(adjusted_limb_bins), nadir_hist_norm

#-------------------------------------------------

def adjust_limb_bins_bob_method(nadir_bins, nadir_hist, limb_bins, limb_hist):
    """
    Adjusts limb bins to match nadir bins using the method described by Bob,
    incorporating proportional scaling, pixel matching, and interpolation.

    Args:
        nadir_bins (array): Bin edges for the nadir histogram.
        nadir_hist (array): The histogram counts for nadir data.
        limb_bins (array): Bin edges for the limb histogram.
        limb_hist (array): The histogram counts for limb data.

    Returns:
        adjusted_limb_bins (array): Adjusted brightness temperatures for limb bins.
        nadir_hist_scaled (array): Proportionally scaled nadir histogram for matching.
    """
    # Step 1: Scale nadir histogram to match the total count of limb histogram
    hist_coef = np.sum(limb_hist) / np.sum(nadir_hist)
    nadir_hist_scaled = nadir_hist * hist_coef  # Proportional scaling of nadir histogram

    # Step 2: Initialize variables for adjustment
    adjusted_limb_bins = []  # To store the adjusted limb bin values
    remaining_samples = nadir_hist_scaled.copy()  # Keep track of remaining samples in nadir_hist_scaled

    # Step 3: Iterate through each limb bin
    for limb_bin, limb_count in zip(limb_bins, limb_hist):
        collected_samples = 0  # Tracks the number of samples collected for the current limb bin
        weighted_tb_sum = 0  # Weighted sum for the current limb bin
        bin_index = 0  # Start from the first nadir bin

        # Step 4: Collect samples to match the current limb bin count
        while collected_samples < limb_count and bin_index < len(nadir_bins):
            available_samples = remaining_samples[bin_index]  # Samples available in the current nadir bin

            if available_samples > 0:
                if collected_samples + available_samples <= limb_count:
                    # Collect all samples from the current nadir bin
                    weight = available_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += available_samples
                    remaining_samples[bin_index] = 0  # Mark the nadir bin as exhausted
                else:
                    # Collect only the remaining required samples
                    needed_samples = limb_count - collected_samples
                    weight = needed_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += needed_samples
                    remaining_samples[bin_index] -= needed_samples  # Reduce the count in the nadir bin

            # Move to the next nadir bin
            bin_index += 1

        # Handle gaps using interpolation if no match is found
        if collected_samples == 0:
            if bin_index == 0:
                # Use the next bin if at the start
                adjusted_tb = nadir_bins[bin_index]
            elif bin_index >= len(nadir_bins):
                # Use the previous bin if at the end
                adjusted_tb = nadir_bins[bin_index - 1]
            else:
                # Interpolate between neighboring bins
                adjusted_tb = (nadir_bins[bin_index - 1] + nadir_bins[bin_index]) / 2
        else:
            # Compute the adjusted value for the current limb bin
            adjusted_tb = weighted_tb_sum

        adjusted_limb_bins.append(adjusted_tb)

    return np.array(adjusted_limb_bins), nadir_hist_scaled
#------------------------------------------


def adjust_limb_bins_bob_method_upgraded(nadir_bins, nadir_hist, limb_bins, limb_hist):
    """
    Adjusts limb bins to match nadir bins using the method described by Bob,
    incorporating proportional scaling, pixel matching, and interpolation.
    This upgraded version ensures correction factors are >= 1.

    Args:
        nadir_bins (array): Bin edges for the nadir histogram.
        nadir_hist (array): The histogram counts for nadir data.
        limb_bins (array): Bin edges for the limb histogram.
        limb_hist (array): The histogram counts for limb data.

    Returns:
        adjusted_limb_bins (array): Adjusted brightness temperatures for limb bins.
        nadir_hist_scaled (array): Proportionally scaled nadir histogram for matching.
    """
    # Step 1: Scale nadir histogram to match the total count of limb histogram
    hist_coef = np.sum(limb_hist) / np.sum(nadir_hist)
    nadir_hist_scaled = nadir_hist * hist_coef  # Proportional scaling of nadir histogram

    # Step 2: Initialize variables for adjustment
    adjusted_limb_bins = []  # To store the adjusted limb bin values
    remaining_samples = nadir_hist_scaled.copy()  # Keep track of remaining samples in nadir_hist_scaled

    # Step 3: Iterate through each limb bin
    for limb_bin, limb_count in zip(limb_bins, limb_hist):
        collected_samples = 0  # Tracks the number of samples collected for the current limb bin
        weighted_tb_sum = 0  # Weighted sum for the current limb bin
        bin_index = 0  # Start from the first nadir bin

        # Step 4: Collect samples to match the current limb bin count
        while collected_samples < limb_count and bin_index < len(nadir_bins):
            available_samples = remaining_samples[bin_index]  # Samples available in the current nadir bin

            if available_samples > 0:
                if collected_samples + available_samples <= limb_count:
                    # Collect all samples from the current nadir bin
                    weight = available_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += available_samples
                    remaining_samples[bin_index] = 0  # Mark the nadir bin as exhausted
                else:
                    # Collect only the remaining required samples
                    needed_samples = limb_count - collected_samples
                    weight = needed_samples / limb_count
                    weighted_tb_sum += weight * nadir_bins[bin_index]
                    collected_samples += needed_samples
                    remaining_samples[bin_index] -= needed_samples  # Reduce the count in the nadir bin

            # Move to the next nadir bin
            bin_index += 1

        # Handle gaps using interpolation if no match is found
        if collected_samples == 0:
            if bin_index == 0:
                # Use the next bin if at the start
                adjusted_tb = nadir_bins[bin_index]
            elif bin_index >= len(nadir_bins):
                # Use the previous bin if at the end
                adjusted_tb = nadir_bins[bin_index - 1]
            else:
                # Interpolate between neighboring bins
                adjusted_tb = (nadir_bins[bin_index - 1] + nadir_bins[bin_index]) / 2
        else:
            # Compute the adjusted value for the current limb bin
            adjusted_tb = weighted_tb_sum

        # Ensure the correction factor is >= 1
        if adjusted_tb < limb_bin:
            adjusted_tb = limb_bin

        adjusted_limb_bins.append(adjusted_tb)

    return np.array(adjusted_limb_bins), nadir_hist_scaled
#------------------------------------------

# Example function to map adjusted bins back to group_data
def map_adjusted_bins_to_group(group_data, corrected_group_data, limb_pos_rng, adjusted_limb_bins, limb_bins):
    # Flatten the original IR BT values for the current limb_pos_rng
    original_irtbs = np.concatenate(
        [group_data[:, lpos].flatten() for lpos in limb_pos_rng 
         if lpos in beam_positions and lpos not in reference_beam_positions]
    )
    
    # Remove NaN values
    valid_irtbs_mask = ~np.isnan(original_irtbs)
    valid_irtbs = original_irtbs[valid_irtbs_mask]

    # Find the bin index for each value in valid_irtbs
    bin_indices = np.digitize(valid_irtbs, bins=limb_bins) - 1  # Adjust to 0-based index

    # Map to adjusted values
    corrected_irtbs = np.array([adjusted_limb_bins[idx] for idx in bin_indices])

    # Get global indices for these valid IR BTs in group_data
    global_indices = []
    for lpos in limb_pos_rng:
        if lpos in beam_positions and lpos not in reference_beam_positions:
            valid_indices = np.where(~np.isnan(group_data[:, lpos]))[0]  # Non-NaN indices
            for idx in valid_indices:
                if group_data[idx, lpos] in valid_irtbs:  # Check for matching IR BTs
                    global_indices.append((idx, lpos))  # Append global index as (row, col)

    # Assign corrected values back to group_data
    # corrected_group_data = group_data.copy()
    for (row, col), corrected_value in zip(global_indices, corrected_irtbs):
        corrected_group_data[row, col] = corrected_value

    return corrected_group_data
#------------------------------------------
# Function to retrieve correction factor from LUT
def get_correction(latwind,beam, surf_type, obs_temp, LUT):
    """
    Find the correction coefficient from the LUT based on input parameters.
    
    Parameters:
    lat (float): Latitude value to search for.    
    beam (int): Beam position to search for.
    obs_temp (float): Observed temperature to search for.
    LUT (pd.DataFrame): The lookup table with correction coefficients.
    
    Returns:
    float: The correction coefficient.
    """
    filt = LUT.copy()

    # Find the closest matching latitude window
    # lat_bin = filt['latitude'].iloc[(filt['latitude'] - lat).abs().idxmin()]
    # filt =filt[(filt['latitude'] == lat_bin)].reset_index()
    # Subset the data based on the provided latitude window (latwind)
    # lat_win = f"{latwind[0]}-{latwind[1]}"
    filt = filt[filt['latitude_bin'] == latwind].reset_index(drop=True)

    # Find the closest matching beam position
    beam_key = filt['beam_position'].iloc[(filt['beam_position'] - beam).abs().idxmin()]
    filt = filt[filt['beam_position'] == beam_key].reset_index(drop=True)

    # Find the closest matching surface type
    syrf_type_key = filt['surface_type'].iloc[(filt['surface_type'] - surf_type).abs().idxmin()]
    filt = filt[filt['surface_type'] == syrf_type_key].reset_index(drop=True)

    # Find the closest matching observed temperature
    temp_bin = filt['original_tb'].iloc[(filt['original_tb'] - obs_temp).abs().idxmin()]
    filt = filt[filt['original_tb'] == temp_bin].reset_index(drop=True)
    
    # Return the correction coefficient if a match is found
    if not filt.empty:
        return filt['corr_coeff'].values[0]
    else:
        return None  # Return None if no match is found
#------------------------------------------
# Apply corrections to the brightness temperature data
def apply_lut_corrections_slow(datafile, beams, target_lat, tol, LUT):
    """
    datafile: the input file to be adjusted 
    beams: the limb beam positions 
    target_lat: Target latitude value to match.
    tol: Allowed deviation from the target latitude (default 0.5 degrees).

    LUT: look up table    
    """
    dataset = xr.open_dataset(datafile)
    lats = dataset['latitude'].data
    #beams =  np.arange(dataset['temp_11_0um_nom'].shape[1])  # Beam indices
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs,np.nan)
    surfact_type = dataset['land_class'].data   
    # surfact_type = np.where((surfact_type == -128),np.nan,surfact_type)
    surfact_type_msk = np.where((surfact_type == 0),surfact_type, np.nan)

    brightness_temp = dataset['temp_11_0um_nom'].data    
    corrected_tb = brightness_temp.copy()  # Copy original data to apply corrections
    
    for i in range(brightness_temp.shape[0]):  # Loop over latitude rows
        for j in beams:  # Loop over beam positions
            lat = lats[i, j]
            cloud_prob = cloud_probs_msk[i, j]
            original_tb = brightness_temp[i, j]
            surface_type_val = surfact_type_msk[i, j]
            
            if (not np.isnan(lat) and (np.abs(lat - target_lat) <= tol)) and not\
                    np.isnan(cloud_prob) and not \
                    np.isnan(original_tb)and not \
                    np.isnan(surface_type_val)    :
              
                correction = get_correction(lat, j, original_tb, LUT)
                if correction is not None:
                    corrected_tb[i, j] = original_tb * correction  # Apply correction

    return corrected_tb

#------------------------------------------

def apply_lut_corrections_fast(datafile, lat_windows, LUT, outdir):
    """
    Apply corrections to brightness temperatures based on LUT and multiple conditions.

    Parameters:
    - datafile: Input file to be adjusted.
    - beams: Limb beam positions.
    - lat_windows: List of latitude windows for NH and SH.
    - LUT: Lookup table.
    - outdir: Output directory for corrected files.

    Returns:
    - corrected_tb: Corrected brightness temperatures.
    """
    dataset = xr.open_dataset(datafile)
    lats = dataset['latitude'].data
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    surfact_type = dataset['land_class'].data
    brightness_temp = dataset['temp_11_0um_nom'].data
    corrected_tb = brightness_temp.copy()  # Copy original data for corrections

    for lat_window in lat_windows:
        lat_wind_ = f"{lat_window[0]}-{lat_window[1]}"
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_msk = ((lats >= min_lat) & (lats <= max_lat))

        valid_mask = (
            lat_msk &
            (~np.isnan(cloud_probs_msk)) &
            (~np.isnan(brightness_temp)) &
            (~np.isnan(surfact_type))
        )
        # Get the indices where the mask is True
        valid_indices = np.argwhere(valid_mask)

        # Loop only over valid indices
        for i, j in valid_indices:
            if j in limb_beam_positions:
                lat = lats[i, j]
                original_tb = brightness_temp[i, j]
                surface_type_val = surfact_type[i, j]

                # print(j, original_tb, surface_type_val, lat_wind_)

                # Apply correction using the lookup table
                correction = get_correction(lat_wind_, int(j), surface_type_val, original_tb, LUT)
                corrected_tb[i, j] = original_tb * correction  # Apply correction

    # Save the corrected data to a new NetCDF file
    data_outfile = os.path.join(outdir, os.path.basename(datafile.replace('.nc', '_cor_.nc')))
    save_corrected_dataset(dataset, corrected_tb, data_outfile)

    return corrected_tb

#------------------------------------------
def process_index(index, brightness_temp, surfact_type, lat_wind_, LUT):
    i, j = index
    if j in limb_beam_positions:
        original_tb = brightness_temp[i, j]
        surface_type_val = surfact_type[i, j]

        # Apply correction using the lookup table
        correction = get_correction(lat_wind_, int(j), surface_type_val, original_tb, LUT)
        return (i, j, original_tb * correction)  # Return the corrected value
    return None

def process_index_helper(index, brightness_temp, surfact_type, lat_wind_, LUT):
    return process_index(index, brightness_temp, surfact_type, lat_wind_, LUT)

def apply_lut_corrections_fast_v2(datafile, lat_windows, LUT, outdir):
    """
    Apply corrections to brightness temperatures based on LUT and multiple conditions.

    Parameters:
    - datafile: Input file to be adjusted.
    - lat_windows: List of latitude windows for NH and SH.
    - LUT: Lookup table.
    - outdir: Output directory for corrected files.

    Returns:
    - corrected_tb: Corrected brightness temperatures.
    """
    # Open the dataset
    dataset = xr.open_dataset(datafile)
    lats = dataset['latitude'].data
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    surfact_type = dataset['land_class'].data
    brightness_temp = dataset['temp_11_0um_nom'].data
    corrected_tb = brightness_temp.copy()  # Copy original data for corrections

    for lat_window in lat_windows:
        lat_wind_ = f"{lat_window[0]}-{lat_window[1]}"
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_msk = ((lats >= min_lat) & (lats <= max_lat))
        valid_mask = (
            lat_msk &
            (~np.isnan(cloud_probs_msk)) &
            (~np.isnan(brightness_temp)) &
            (~np.isnan(surfact_type))
        )
        # Get the indices where the mask is True
        valid_indices = np.argwhere(valid_mask)

        # Use ProcessPoolExecutor for parallel processing of valid indices
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_index_helper, 
                                        valid_indices, 
                                        [brightness_temp]*len(valid_indices), 
                                        [surfact_type]*len(valid_indices), 
                                        [lat_wind_]*len(valid_indices), 
                                        [LUT]*len(valid_indices)))

        # Update the corrected_tb array with the results
        for result in results:
            if result is not None:
                i, j, corrected_value = result
                corrected_tb[i, j] = corrected_value

    # Save the corrected data to a new NetCDF file
    data_outfile = os.path.join(outdir, os.path.basename(datafile.replace('.nc', '_cor_fv2.nc')))
    save_corrected_dataset(dataset, corrected_tb, data_outfile)

    return corrected_tb
#------------------------------------------
# Function save corrected data to NetCDF
def save_corrected_dataset(dataset, corrected_tb, output_file):  
    cor_obs_diff = corrected_tb - dataset['temp_11_0um_nom'].data  
    dataset['temp_11_0um_nom_corrected'] = (dataset['temp_11_0um_nom'].dims, corrected_tb)
    dataset['temp_11_0um_nom_cor_obs_diff'] = (dataset['temp_11_0um_nom'].dims, cor_obs_diff)
    dataset.to_netcdf(output_file, mode='w', 
                      encoding={'temp_11_0um_nom_corrected': {'zlib': True, 'complevel': 5}})
    return dataset.close()
#------------------------------------------
# Define a function for processing a single file
def process_file(season_file, lat_windows, lookup_df, 
                 limb_beam_positions, cor_dir):
    # Apply LUT corrections
    cor_tb = apply_lut_corrections_fast(season_file, limb_beam_positions, 
                                        lat_windows, lookup_df, cor_dir)
    dat_ = xr.open_dataset(season_file)
    obs_tb = dat_['temp_11_0um_nom'].data
    lats_ = dat_['latitude'].data
    surfact_type = dat_['land_class'].data

    # Initialize dictionaries to store masked arrays for each surface type and hemisphere
    cor_tb_msk_dict = {'NH': {}, 'SH': {}}
    obs_tb_msk_dict = {'NH': {}, 'SH': {}}

    for lat_window in lat_windows:
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_mask = ((lats_ >= min_lat) & (lats_ <= max_lat))
        lat_mask = np.where(lat_mask == True, 1, np.nan)

        hemisphere = 'NH' if max_lat > 0 else 'SH'

        for surface_type_id, surface_type_name in surface_type_mapping.items():
            # Create surface type mask
            surfact_type_mask = np.where(surfact_type == surface_type_id, 1, np.nan)

            # Combine latitude and surface type masks
            combined_mask = lat_mask * surfact_type_mask

            # Apply the combined mask to the corrected and observed arrays
            cor_tb_msk = np.where(combined_mask == 1, cor_tb, np.nan)
            obs_tb_msk = np.where(combined_mask == 1, obs_tb, np.nan)

            # Store the masked arrays in the dictionaries
            if surface_type_id not in cor_tb_msk_dict[hemisphere]:
                cor_tb_msk_dict[hemisphere][surface_type_id] = []
                obs_tb_msk_dict[hemisphere][surface_type_id] = []

            cor_tb_msk_dict[hemisphere][surface_type_id].append(cor_tb_msk)
            obs_tb_msk_dict[hemisphere][surface_type_id].append(obs_tb_msk)

    return cor_tb_msk_dict, obs_tb_msk_dict, os.path.basename(season_file)
#------------------------------------------

# Parallel execution
def process_files_in_parallel(summer_files, lat_windows, lookup_df, limb_beam_positions, cor_dir):
    # Initialize dictionaries to store lists of arrays for each surface type and hemisphere
    corrected_arrays = {'NH': {surface_type_id: [] for surface_type_id in surface_type_mapping.keys()},
                        'SH': {surface_type_id: [] for surface_type_id in surface_type_mapping.keys()}}
    observed_arrays = {'NH': {surface_type_id: [] for surface_type_id in surface_type_mapping.keys()},
                       'SH': {surface_type_id: [] for surface_type_id in surface_type_mapping.keys()}}

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                process_file, season_file, lat_windows,
                lookup_df, limb_beam_positions, cor_dir
            ) for season_file in summer_files
        ]
        
        for idx, future in enumerate(futures):
            cor_tb_msk_dict, obs_tb_msk_dict, filename = future.result()
            
            for hemisphere in ['NH', 'SH']:
                for surface_type_id in surface_type_mapping.keys():
                    corrected_arrays[hemisphere][surface_type_id].extend(cor_tb_msk_dict[hemisphere][surface_type_id])
                    observed_arrays[hemisphere][surface_type_id].extend(obs_tb_msk_dict[hemisphere][surface_type_id])

            # Log progress for every 100 files
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(summer_files)}: {filename}")

    return corrected_arrays, observed_arrays
