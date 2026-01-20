#%% AVHRR IR TB Correction - Omid
import warnings
warnings.filterwarnings("ignore")
import sys
import gc
import os
import datetime
from datetime import date
import time
import pandas as pd
import numpy as np
from collections import defaultdict

import xarray as xr

from scipy.stats import binned_statistic


#%%
# Define Floating Varibales
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
#------------------------------------------------------------------------------

# Define latitude windows for Southern Hemisphere (SH) and Northern Hemisphere (NH)
latitude_windows = {
    'SH': {
        'window1': (-75, -61),
        'window2': (-61, -53),
        'window3': (-53, -41)
    },
    'NH': {
        'window1': (61, 75),
        'window2': (53, 61),
        'window3': (41, 53)
    }
}
#------------------------------------------------------------------------------

# Define the combinations of latitude windows and seasons
combinations = [
    ('SH', 'Summer', latitude_windows['SH']['window1']),
    ('SH', 'Summer', latitude_windows['SH']['window2']),
    ('SH', 'Summer', latitude_windows['SH']['window3']),
    ('SH', 'Autumn', latitude_windows['SH']['window1']),
    ('SH', 'Autumn', latitude_windows['SH']['window2']),
    ('SH', 'Autumn', latitude_windows['SH']['window3']),
    ('SH', 'Winter', latitude_windows['SH']['window1']),
    ('SH', 'Winter', latitude_windows['SH']['window2']),
    ('SH', 'Winter', latitude_windows['SH']['window3']),
    ('SH', 'Spring', latitude_windows['SH']['window1']),
    ('SH', 'Spring', latitude_windows['SH']['window2']),
    ('SH', 'Spring', latitude_windows['SH']['window3']),
]


#------------------------------------------------------------------------------
df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# read all LUTs
all_lut_files = sorted([os.path.join(df_dir, s) for s in os.listdir(df_dir) if (s.endswith('20.csv') or s.endswith('21.csv'))])

# Define a function to initialize the nested dictionary
def nested_dict():
    return defaultdict(dict)

# Initialize a nested dictionary
all_lut = defaultdict(nested_dict)

for file_path in all_lut_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    var = parts[0] + '_' + parts[1]
    hemisphere = parts[2]
    season = parts[3]
    all_lut[var][hemisphere][season] = pd.read_csv(file_path)
#%%
# Define Functions
def get_custom_surface_type_mapping(hemisphere, season, lat_range):
    seesns = ['Summer', 'Autumn', 'Winter', 'Spring']

     # -------------------------
    # Southern Hemisphere
    # -------------------------
    
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
                1: 'snow-free land'
            }
        
        elif (season in ['Autumn', 'Winter']) and (lat_range == (-61, -53)):
            return {
                0: 'water',
                3: 'ice'
            }
        
        elif (season in ['Summer', 'Spring']) and (lat_range == (-53, -45)):
            return {
                0: 'water',               
            }
        elif (season in ['Winter', 'Autumn']) and (lat_range == (-53, -45)):
            return {
                0: 'water',
                3: 'ice'
            }
        
    # -------------------------
    # Northern Hemisphere
    # -------------------------
        
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
                1: 'snow-free land'
            }
        
        elif (season in ['Spring', 'Summer']) and (lat_range == (53, 61)):
            return {
                0: 'water',
                1: 'snow-free land'
            }
        
        elif (season in ['Winter', 'Autumn']) and (lat_range == (53, 61)):
            return {
                0: 'water',
                2: 'snow-covered land',
                3: 'ice'
            }
        
        elif (season in ['Spring', 'Summer']) and (lat_range == (45, 53)):
            return {
                0: 'water',
                1: 'snow-free land'
            }
        
        elif (season in ['Winter', 'Autumn']) and (lat_range == (45, 53)):
            return {
                0: 'water',
                2: 'snow-covered land',
                3: 'ice'
            }

#-----------------------------------
def get_valid_indices_and_data(lat_window, surfact_type, brightness_temp, lut_df, lats, cloud_probs_msk, limb_beam_positions):
    """
    Extracts valid indices and corresponding brightness temperature, surface type, and j indices
    for a given temperature channel and latitude window.
    
    Parameters:
    - lat_window (tuple): Latitude range.
    - surfact_type (ndarray): Surface type array.
    - brightness_temp (ndarray): Brightness temperature array.
    - lut_df (pd.DataFrame): Lookup table DataFrame for the specific channel.
    - lats (ndarray): Latitude array.
    - cloud_probs_msk (ndarray): Cloud probability mask.
    - limb_beam_positions (set): Set of valid limb beam positions.

    Returns:
    - temp_tb (ndarray): Brightness temperature values for valid pixels.
    - surface_type_val (ndarray): Surface type values for valid pixels.
    - j_indices (ndarray): Beam position indices for valid pixels.
    """
    max_lat, min_lat = max(lat_window), min(lat_window)
    lat_msk = (lats > min_lat) & (lats <= max_lat)

    # Get valid surface types from the LUT for this latitude window
    valid_surface_types = set(lut_df[lut_df['latitude_bin'] == str(lat_window)]['surface_type'].unique())

    # Create valid mask
    valid_mask = (
        lat_msk &
        np.isin(surfact_type, list(valid_surface_types)) &  # Mask valid surface types
        (~np.isnan(cloud_probs_msk)) &
        (~np.isnan(brightness_temp))
    )

    valid_indices = np.argwhere(valid_mask)
    valid_valid_indices = valid_indices[np.isin(valid_indices[:, 1], limb_beam_positions)]

    if valid_valid_indices.size == 0:
        return None, None, None, None  # Skip processing if no valid indices found

    # Extract valid indices
    i_indices, j_indices = valid_valid_indices[:, 0], valid_valid_indices[:, 1]

    # Extract brightness temperature and surface type values
    temp_tb = brightness_temp[i_indices, j_indices]
    surface_type_val = surfact_type[i_indices, j_indices]

    return temp_tb, surface_type_val, i_indices,j_indices

#-----------------------------------
def get_correction_fast(latwind, beam, surf_type, obs_temp, lut_dict):
    """
    Fetch the correction coefficient from preprocessed LUT dictionary.
    Logs cases where no correction is found.
    """
    try:
        beam_dict = lut_dict.get(str(latwind), {})
        surf_dict = beam_dict.get(int(beam), {}).get(surf_type, {})

        if not surf_dict:
            print(f"DEBUG: Missing correction for latwind={latwind}, beam={beam}, surf_type={surf_type}, obs_temp={obs_temp}")
            return None  # Indicate no correction found
        
        temp_keys = np.array(list(surf_dict.keys()))  # Convert keys to array
        
        if temp_keys.size == 0:
            print(f"DEBUG: Empty temperature keys for latwind={latwind}, beam={beam}, surf_type={surf_type}, obs_temp={obs_temp}")
            return None
        
        temp_key = temp_keys[np.abs(temp_keys - obs_temp).argmin()]
        return surf_dict[temp_key]
    
    except Exception as e:
        print(f"ERROR: Exception in get_correction_fast for latwind={latwind}, beam={beam}, surf_type={surf_type}, obs_temp={obs_temp} | Error: {e}")
        return None  # Safe return
#-----------------------------------
def save_corrected_11_12_dataset(dataset, corrected_tb11, corrected_tb12, output_file):  
    # cor_obs_diff11 = corrected_tb11 - dataset['temp_11_0um_nom'].data  
    dataset['temp_11_0um_nom_corrected'] = (dataset['temp_11_0um_nom'].dims, corrected_tb11)
    # dataset['temp_11_0um_nom_cor_obs_diff'] = (dataset['temp_11_0um_nom'].dims, cor_obs_diff11)

    # cor_obs_diff12 = corrected_tb12 - dataset['temp_12_0um_nom'].data  
    dataset['temp_12_0um_nom_corrected'] = (dataset['temp_12_0um_nom'].dims, corrected_tb12)
    # dataset['temp_12_0um_nom_cor_obs_diff'] = (dataset['temp_12_0um_nom'].dims, cor_obs_diff12)
    # dataset.to_netcdf(output_file, mode='w', 
    #                   encoding={'temp_11_0um_nom_corrected': {'zlib': True, 'complevel': 9},
    #                             'temp_12_0um_nom_corrected': {'zlib': True, 'complevel': 9}})
    return dataset#.close()
#-----------------------------------
def process_file_vectorized(file_run, lat_windows, 
                            lut_11_nh_sh, lut_11_nh_sh_dict, 
                            lut_12_nh_sh, lut_12_nh_sh_dict, 
                            limb_beam_positions, cor_dir):
    """
    Processes a single AVHRR file and applies IR TB correction.
    Skips processing if the corrected file already exists.
    """
    # Define output file path
    outfile = os.path.join(cor_dir, os.path.basename(file_run).replace('.nc', '_corrected.nc'))

    # Skip processing if the output file already exists
    if os.path.exists(outfile):
        print(f"Skipping {os.path.basename(file_run)} (already processed)")
        return  # Exit function early

    # Open dataset and extract required data
    dataset = xr.open_dataset(file_run)
    lats = dataset['latitude'].data
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    surfact_type = dataset['land_class'].data
    brightness_temp_11 = dataset['temp_11_0um_nom'].data
    brightness_temp_12 = dataset['temp_12_0um_nom'].data
    corrected_tb_11 = brightness_temp_11.copy()
    corrected_tb_12 = brightness_temp_12.copy()

    # Iterate over lat_windows
    for lat_window in lat_windows:
        # Process 11 µm channel
        temp_11_tb, surface_type_val_11, i_indices_11, j_indices_11 = get_valid_indices_and_data(
            lat_window, surfact_type, brightness_temp_11, lut_11_nh_sh, lats, cloud_probs_msk, limb_beam_positions
        )

        if temp_11_tb is not None:
            correction_11 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_11, surface_type_val_11, temp_11_tb, lut_11_nh_sh_dict)
            corrected_tb_11[i_indices_11, j_indices_11] = temp_11_tb * correction_11

        # Process 12 µm channel
        temp_12_tb, surface_type_val_12, i_indices_12, j_indices_12 = get_valid_indices_and_data(
            lat_window, surfact_type, brightness_temp_12, lut_12_nh_sh, lats, cloud_probs_msk, limb_beam_positions
        )

        if temp_12_tb is not None:
            correction_12 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_12, surface_type_val_12, temp_12_tb, lut_12_nh_sh_dict)
            corrected_tb_12[i_indices_12,j_indices_12] = temp_12_tb * correction_12


    # outfile = os.path.join(cor_dir, os.path.basename(file_run).replace('.nc', '_corrected.nc'))

    save_corrected_11_12_dataset(dataset, corrected_tb_11, corrected_tb_12, outfile)
