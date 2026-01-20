#%% AVHRR IR TB Correction - Omid
import warnings
warnings.filterwarnings("ignore")
import re
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
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# read all LUTs
all_lut_files = sorted([os.path.join(df_dir, s) for s in os.listdir(df_dir) if s.endswith('.pkl')])

# Define a function to initialize the nested dictionary
def nested_dict():
    return defaultdict(dict)

# Initialize a nested dictionary
all_lut = {} #defaultdict(nested_dict)

for file_path in all_lut_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    var = parts[0] + '_' + parts[1]
    hemisphere = parts[2]
    season = parts[3]
    # Initialize hierarchy explicitly
    if var not in all_lut:
        all_lut[var] = {}

    if hemisphere not in all_lut[var]:
        all_lut[var][hemisphere] = {}

    # fle_read = pd.read_csv(file_path,engine='python')
    fle_read = pd.read_pickle(file_path)
    all_lut[var][hemisphere][season] = fle_read

#%%
# Define Functions
def np_describe(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return {
        "count": x.size,
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "min": x.min(),
        "25%": np.percentile(x, 25),
        "50%": np.percentile(x, 50),
        "75%": np.percentile(x, 75),
        "max": x.max(),
    }
#-----------------------------------
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

#-----------------------------------
def extract_year_and_doy(file_name):
        parts = file_name.split('.')
        d_parts = [part for part in parts if part.startswith('D')]
        if len(d_parts) == 0:
            raise ValueError(f"File name {file_name} is not in the expected format.")
        year_prefix = d_parts[0][1:3]
        day_of_year = int(d_parts[0][3:6])
        year = 1900 + int(year_prefix) if int(year_prefix) >= 98 else 2000 + int(year_prefix)
        return year, day_of_year

#-----------------------------------
# Function to calculate the month from day of year
def calculate_month(doy, year):
    """Calculate the month from DOY (day of year)."""
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    return date.month
#-----------------------------------


def parse_lat_window(lat):
    """
    Robust latitude window parser.

    Handles ALL of:
      "(61, 75)"
      "(61-75)"
      "61-75"
      "-75--61"
      "(-75, -61)"
      "(-75--61)"

    Returns: (min_lat, max_lat)
    """

    # Already numeric tuple
    if isinstance(lat, tuple):
        return tuple(sorted(map(int, lat)))

    s = str(lat).strip()

    # Extract all integers (with sign)
    nums = re.findall(r'-?\d+', s)

    if len(nums) != 2:
        raise ValueError(f"Unrecognized latitude_bin format: {lat}")

    a, b = map(int, nums)

    # Fix the classic "61-75" case where regex gives [61, -75]
    if (
        "-" in s
        and "--" not in s
        and not s.strip().startswith("-")
        and a > 0
        and b < 0
    ):
        b = abs(b)

    return tuple(sorted((a, b)))


#-----------------------------------
def lat_window_to_bin(lat_window):
    lo, hi = lat_window
    return f"{lo}-{hi}"
#-----------------------------------
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
def get_valid_indices_and_data(lat_window, lat_range, surfact_type, brightness_temp, lut_df, lats, cloud_probs_msk, limb_beam_positions):
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
    max_lat, min_lat = int(max(lat_range)), int(min(lat_range))
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
def save_corrected_11_12_dataset(dataset, corrected_tb11, corrected_tb12):  
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
def preprocess_lut(LUT):
    """
    Converts LUT DataFrame into a multi-index dictionary for fast lookup.
    """
    lut_dict = {}
    for _, row in LUT.iterrows():
        lat_bin = row['latitude_bin']
        beam = row['beam_position']
        surf_type = row['surface_type']
        obs_temp = row['original_tb']
        corr_coeff = row['corr_coeff']
        lut_dict.setdefault(lat_bin, {}).setdefault(beam, {}).setdefault(surf_type, {})[obs_temp] = corr_coeff
    return lut_dict

from collections import defaultdict

def preprocess_lut_fast(LUT):
    """
    Fast conversion of LUT DataFrame into nested dict:
    lut[lat_bin][beam][surf_type][obs_temp] = corr_coeff
    """

    lut_dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
        )
    )

    for row in LUT.itertuples(index=False):
        lut_dict[
            row.latitude_bin
        ][
            row.beam_position
        ][
            row.surface_type
        ][
            row.original_tb
        ] = row.corr_coeff

    return lut_dict
#-----------------------------------
def correct_file_vectorized(file_run): # , cor_dir
    """
    Processes a single AVHRR file and applies IR TB correction.
    Skips processing if the corrected file already exists.
    """
    # # Define output file path
    # outfile = os.path.join(cor_dir, os.path.basename(file_run).replace('.nc', '_corrected.nc'))

    # # Skip processing if the output file already exists
    # if os.path.exists(outfile):
    #     print(f"Skipping {os.path.basename(file_run)} (already processed)")
    #     return  # Exit function early

    #-----------------------------------
    # find season from file name
    # Extract the year and day of the year from the file name
    file_year, day_of_year = extract_year_and_doy(file_run)   
    
    # Check if the year is a leap year
    is_leap_year = (file_year % 4 == 0 and file_year % 100 != 0) or (file_year % 400 == 0)
    
    # Adjust for leap year if necessary
    if is_leap_year and day_of_year > 59:
        day_of_year -= 1
    
    # Calculate the month from the day of the year
    month = calculate_month(day_of_year, file_year)
    
    # Determine the season for the given month and hemisphere
    season = find_season(month, 'Southern')

    #-----------------------------------
    sh_seasn = season
    nh_seasn = {'Summer': 'Winter', 'Autumn': 'Spring', 
                'Winter': 'Summer', 'Spring': 'Autumn'}[sh_seasn]

    # Load LUTs for the season
    luts_11_nh, luts_11_sh = all_lut['temp_11']['NH'], all_lut['temp_11']['SH']
    luts_12_nh, luts_12_sh = all_lut['temp_12']['NH'], all_lut['temp_12']['SH']

    lut_11_nh_sh = pd.concat([luts_11_nh[nh_seasn], luts_11_sh[sh_seasn]], ignore_index=True)
    lut_12_nh_sh = pd.concat([luts_12_nh[nh_seasn], luts_12_sh[sh_seasn]], ignore_index=True)
    
    # lat_windows = [tuple(map(int, lat.split(','))) for lat in lut_12_nh_sh['latitude_bin'].unique()]
    lat_windows = [
    parse_lat_window(lat)
    for lat in lut_11_nh_sh['latitude_bin'].unique()
    ]

    # Precompute LUT dictionaries for fast lookup
    lut_11_nh_sh_dict, lut_12_nh_sh_dict = preprocess_lut_fast(lut_11_nh_sh), preprocess_lut_fast(lut_12_nh_sh)

    #-----------------------------------    

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
    for lat_bin in lat_windows:
        lat_window = lat_window_to_bin(lat_bin)

        # Process 11 µm channel
        temp_11_tb, surface_type_val_11, i_indices_11, j_indices_11 = get_valid_indices_and_data(
            lat_window, lat_bin, surfact_type, brightness_temp_11, lut_11_nh_sh, lats, cloud_probs_msk, limb_beam_positions
        )

        if temp_11_tb is not None:
            correction_11 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_11, surface_type_val_11, temp_11_tb, lut_11_nh_sh_dict)
            corrected_tb_11[i_indices_11, j_indices_11] = temp_11_tb * correction_11

        # Process 12 µm channel
        temp_12_tb, surface_type_val_12, i_indices_12, j_indices_12 = get_valid_indices_and_data(
            lat_window, lat_bin, surfact_type, brightness_temp_12, lut_12_nh_sh, lats, cloud_probs_msk, limb_beam_positions
        )

        if temp_12_tb is not None:
            correction_12 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_12, surface_type_val_12, temp_12_tb, lut_12_nh_sh_dict)
            corrected_tb_12[i_indices_12,j_indices_12] = temp_12_tb * correction_12


    # outfile = os.path.join(cor_dir, os.path.basename(file_run).replace('.nc', '_corrected.nc'))

    return save_corrected_11_12_dataset(dataset, corrected_tb_11, corrected_tb_12)


#%% Example funtion usage and plot
fle = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_AutoSnow_collocated_1998_2000_for_Kingsley'

file2cor = os.path.join(fle,'clavrx_NSS.GHRR.NK.D99364.S2243.E0025.B0848081.WI.hirs_avhrr_fusion.level2.nc')

cor_file = correct_file_vectorized(file2cor)

orig_m11 = cor_file['temp_11_0um_nom'].mean(dim='scan_lines_along_track_direction')
cor_m11 = cor_file['temp_11_0um_nom_corrected'].mean(dim='scan_lines_along_track_direction')


import matplotlib.pyplot as plt
f,x = plt.subplots()
orig_m11.plot(label='Original 11um',c='k',ls='-')
cor_m11.plot(label='Corrected 11um', c='k', ls=':', ax=x)
x.legend()
#  # Extract the year and day of the year from the file name
# file_year, day_of_year = extract_year_and_doy(file2cor)   

# # Check if the year is a leap year
# is_leap_year = (file_year % 4 == 0 and file_year % 100 != 0) or (file_year % 400 == 0)

# # Adjust for leap year if necessary
# if is_leap_year and day_of_year > 59:
#     day_of_year -= 1

# # Calculate the month from the day of the year
# month = calculate_month(day_of_year, file_year)

# # Determine the season for the given month and hemisphere
# season = find_season(month, 'Southern')

# #-----------------------------------
# sh_seasn = season
# nh_seasn = {'Summer': 'Winter', 'Autumn': 'Spring', 
#             'Winter': 'Summer', 'Spring': 'Autumn'}[season]