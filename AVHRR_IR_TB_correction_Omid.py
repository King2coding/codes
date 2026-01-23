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
import pickle
import numpy as np
from collections import defaultdict

import xarray as xr

from scipy.stats import binned_statistic


#%%
# Define Floating Varibales
NADIR_CENTER = 204

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
        'window3': (-53, -45)
    },
    'NH': {
        'window1': (61, 75),
        'window2': (53, 61),
        'window3': (45, 53)
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
path_to_lut = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/ir_correction_LUTs'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# read all LUTs
all_lut_files = sorted([os.path.join(path_to_lut, s) for s in os.listdir(path_to_lut) if \
                        (s.startswith('temp')) and (s.endswith('.pkl'))])

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
print("✓ Loaded All LUTs successfully")
# --------------------------------------------------
# Load GLOBAL geometry curves
# --------------------------------------------------
with open(os.path.join(path_to_lut, "global_geometry.pkl"), "rb") as f:
    global_curve = pickle.load(f)

# --------------------------------------------------
# Load SURFACE-specific geometry curves
# --------------------------------------------------
with open(os.path.join(path_to_lut, "surface_geometry.pkl"), "rb") as f:
    curve_lib = pickle.load(f)

print("✓ Loaded global_curve and curve_lib successfully")

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
def apply_surface_curve(
    curve_lib, var, hemisphere, season, lat_bin, surface, beam, obs_tb
):
    poly = curve_lib[var][hemisphere][season][lat_bin][surface]["poly"]
    corr = poly(beam - NADIR_CENTER)
    return obs_tb * corr

#-----------------------------------
def eval_geometry_curve(beam_position, coeffs):
    """
    Evaluate polynomial geometry correction.
    """
    x = beam_position - NADIR_CENTER
    return np.polyval(coeffs, x)
#-----------------------------------
# def apply_global_curve(
#     global_curve, var, hemisphere, season, lat_bin, beam, obs_tb
# ):
#     poly = global_curve[var][hemisphere][season][lat_bin]["poly"]
#     corr = poly(beam - NADIR_CENTER)
#     return obs_tb * corr

#-----------------------------------
def decide_correction_mode(
    hemisphere,
    season,
    lat_range,
    surface_name,
):
    """
    Decide the scientifically appropriate limb-correction strategy.

    Returns ONE of:
      - "LUT"
      - "SURFACE_CURVE"
      - "GLOBAL_CURVE"
      - "NONE"
    """

    # ----------------------------------
    # Expected surface climatology
    # ----------------------------------
    valid_map = get_custom_surface_type_mapping(
        hemisphere, season, lat_range
    )

    # ----------------------------------
    # Surface not climatologically expected
    # ----------------------------------
    if surface_name not in valid_map.values():
        return "GLOBAL_CURVE"

    # ==================================
    # Southern Hemisphere
    # ==================================
    if hemisphere == "SH":

        # Ocean-dominated viewing geometry
        if surface_name == "water":
            return "LUT"

        # Ice/snow are heterogeneous along limb paths
        if surface_name in ["ice", "snow-covered land"]:
            return "SURFACE_CURVE"

        return "GLOBAL_CURVE"

    # ==================================
    # Northern Hemisphere
    # ==================================
    if hemisphere == "NH":

        # Water is radiometrically stable
        if surface_name == "water":
            return "LUT"

        # ----------------------------------
        # Snow-covered land
        # ----------------------------------
        if surface_name == "snow-covered land":

            # Stable winter snowpack
            if season == "Winter" and lat_range in [(53, 61), (61, 75)]:
                return "LUT"

            # Transitional snow regimes
            return "SURFACE_CURVE"

        # ----------------------------------
        # Snow-free land
        # ----------------------------------
        if surface_name == "snow-free land":

            # Warm-season exposed land
            if season in ["Spring", "Summer"]:
                return "LUT"

            return "SURFACE_CURVE"

        # ----------------------------------
        # Ice
        # ----------------------------------
        if surface_name == "ice":
            return "SURFACE_CURVE"

    # ----------------------------------
    # Final fallback
    # ----------------------------------
    return "GLOBAL_CURVE"

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
def surface_curve_available(curve_lib, var, hemisphere, season, lat_bin, surface_code):
    """
    Check if a surface-specific curve exists.
    """
    try:
        return surface_code in curve_lib[var][hemisphere][season][lat_bin]
    except KeyError:
        return False
#-----------------------------------
def apply_channel_correction_grouped(
    *,
    var,
    temp_tb,
    surface_type_val,
    i_indices,
    j_indices,
    lat_bin,
    lat_window,
    hemisphere,
    season,
    lut_dict,
    curve_lib,
    global_curve,
    corrected_tb,
    surface_type_mapping,
):
    """
    Apply limb correction for ONE channel (temp_11 or temp_12)
    using grouped vectorized logic.
    """

    for surface_code in np.unique(surface_type_val):

        surface_name = surface_type_mapping[surface_code]

        mask = surface_type_val == surface_code
        if not np.any(mask):
            continue

        obs_tb = temp_tb[mask]
        beams  = j_indices[mask]
        i_idx  = i_indices[mask]
        j_idx  = j_indices[mask]

        surf_curve_ok = surface_curve_available(
            curve_lib, var, hemisphere, season, lat_window, surface_code
        )

        mode = decide_correction_mode(
            hemisphere=hemisphere,
            season=season,
            lat_range=lat_bin,
            surface_name=surface_name,            
        )

        # ----------------------------------
        # Apply correction in BULK
        # ----------------------------------
        if mode == "LUT":

            corr = np.vectorize(get_correction_fast)(
                lat_window,
                beams,
                surface_code,
                obs_tb,
                lut_dict,
            )
            corrected = obs_tb * corr

        elif mode == "SURFACE_CURVE":

            if surf_curve_ok:
                entry = curve_lib[var][hemisphere][season][lat_window][surface_code]
                coeffs = entry["coeffs"]
                # beam_center = entry["beam_center"]
                factors = eval_geometry_curve(beams, coeffs)
            else:
                entry = global_curve[var][hemisphere][season][lat_window]
                coeffs = entry["coeffs"]
                # beam_center = entry["beam_center"]
                factors = eval_geometry_curve(beams, coeffs)

            corrected = obs_tb * factors #poly(beams - NADIR_CENTER)

        elif mode == "GLOBAL_CURVE":

            entry = global_curve[var][hemisphere][season][lat_window]
            coeffs = entry["coeffs"]
            # beam_center = entry["beam_center"]
            factors = eval_geometry_curve(beams, coeffs)
            corrected = obs_tb * factors #poly(beams - NADIR_CENTER)

        else:
            corrected = obs_tb

        corrected_tb[i_idx, j_idx] = corrected
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

        # Hemisphere inferred from latitude window
        if int(lat_bin[1]) < 0:
            hemisphere = "SH"
        elif int(lat_bin[0]) > 0:
            hemisphere = "NH"
        else:
            # Equator-crossing window (rare but possible)
            hemisphere = None

        # Process 11 µm channel
        temp_11_tb, surface_type_val_11, i_indices_11, j_indices_11 = get_valid_indices_and_data(
            lat_window, lat_bin, surfact_type, brightness_temp_11, lut_11_nh_sh, lats, cloud_probs_msk, limb_beam_positions
        )

        if temp_11_tb is not None:
            # correction_11 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_11, surface_type_val_11, temp_11_tb, lut_11_nh_sh_dict)
            # corrected_tb_11[i_indices_11, j_indices_11] = temp_11_tb * correction_11

            apply_channel_correction_grouped(
            var="temp_11",
            temp_tb=temp_11_tb,
            surface_type_val=surface_type_val_11,
            i_indices=i_indices_11,
            j_indices=j_indices_11,
            lat_bin=lat_bin,
            lat_window=lat_window,
            hemisphere=hemisphere,
            season=season,
            lut_dict=lut_11_nh_sh_dict,
            curve_lib=curve_lib,
            global_curve=global_curve,
            corrected_tb=corrected_tb_11,
            surface_type_mapping=surface_type_mapping,
                )

        # Process 12 µm channel
        temp_12_tb, surface_type_val_12, i_indices_12, j_indices_12 = get_valid_indices_and_data(
            lat_window, lat_bin, surfact_type, brightness_temp_12, lut_12_nh_sh, lats, cloud_probs_msk, limb_beam_positions
                )

        if temp_12_tb is not None:
            # correction_12 = np.vectorize(get_correction_fast)(str(lat_window), j_indices_12, surface_type_val_12, temp_12_tb, lut_12_nh_sh_dict)
            # corrected_tb_12[i_indices_12,j_indices_12] = temp_12_tb * correction_12
            apply_channel_correction_grouped(
            var="temp_12",
            temp_tb=temp_12_tb,
            surface_type_val=surface_type_val_12,
            i_indices=i_indices_12,
            j_indices=j_indices_12,
            lat_bin=lat_bin,
            lat_window=lat_window,
            hemisphere=hemisphere,
            season=season,
            lut_dict=lut_12_nh_sh_dict,
            curve_lib=curve_lib,
            global_curve=global_curve,
            corrected_tb=corrected_tb_12,
            surface_type_mapping=surface_type_mapping,
                )


    # outfile = os.path.join(cor_dir, os.path.basename(file_run).replace('.nc', '_corrected.nc'))

    return save_corrected_11_12_dataset(dataset, corrected_tb_11, corrected_tb_12)


#%% Example funtion usage and plot
fle = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_AutoSnow_collocated_1998_2000_for_Kingsley'

file2cor = os.path.join(fle,'clavrx_NSS.GHRR.ND.D98001.S0001.E0155.B3444950.GC.hirs_avhrr_fusion.level2.nc')

cor_file = correct_file_vectorized(file2cor)

orig_m11 = cor_file['temp_11_0um_nom'].mean(dim='scan_lines_along_track_direction')
cor_m11 = cor_file['temp_11_0um_nom_corrected'].mean(dim='scan_lines_along_track_direction')


import matplotlib.pyplot as plt
f,x = plt.subplots()
orig_m11.plot(label='Original 11um',c='k',ls='-')
cor_m11.plot(label='Corrected 11um', c='k', ls=':', ax=x)
x.set_title('new-method')
x.legend()


#%% do some dist stts to be sure
fle = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_AutoSnow_collocated_1998_2000_for_Kingsley'

all_files_avhrr = [os.path.join(fle,f) for f in os.listdir(fle) if f.endswith('.nc')][:100]

cor_files = [correct_file_vectorized(file) for file in all_files_avhrr]

# orig_m11 = [cor_file['temp_11_0um_nom'] for cor_file in cor_files]#.data.flatten()
# # orig_m11 = orig_m11[~np.isnan(orig_m11)]
# cor_m11 = [cor_file['temp_11_0um_nom_corrected'] for cor_file in cor_files]#.data.flatten()
# # cor_m11 = cor_m11[~np.isnan(cor_m11)]


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
    num_beam_positions = data[0].shape[1]
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
def plot_discrete_ir_tb_distribution(orig_hists, orig_bin_edges, orig_beam_means,
                                     cor_hists, cor_bin_edges, cor_beam_means,
                                     beam_positions, surftp):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    """
    Plot the IR Tbs distribution stratified by beam positions with mean IR Tbs,
    mimicking a discrete plot style.
    """
    # Define discrete bins and use logarithmic normalization
    levels = np.logspace(-3, np.log10(0.25), 15)#np.logspace(-3, 0, 15)  # Logarithmic scale for contour levels
    cmap = plt.cm.get_cmap('RdYlBu_r', len(levels) - 1)  # Discrete colormap
    norm = mcolors.BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300, sharex=False, sharey=True) # 

    # Plot original histogram using contourf
    cf1 = axes[0].contourf(
        beam_positions, orig_bin_edges[:-1], orig_hists.T, levels=levels, cmap=cmap, norm=norm
    )
    axes[0].plot(beam_positions, orig_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB')
    axes[0].set_ylim(200,300)
    # axes[0].axhline(y=260, color='green', ls=':', lw=5, label='260 K')
    axes[0].set_title(f"Original IR Tbs Distribution ({surftp})", fontsize=16)
    axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[0].legend(frameon=False, fontsize=12)

    # Plot corrected histogram using contourf
    cf2 = axes[1].contourf(
        beam_positions, cor_bin_edges[:-1], cor_hists.T, levels=levels, cmap=cmap, norm=norm
    )
    axes[1].plot(beam_positions, cor_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)')
    axes[1].set_ylim(200,300)
    # axes[1].axhline(y=260, color='green', ls=':', lw=5, label='260 K')
    axes[1].set_title("Corrected IR Tbs Distribution", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Beam Positions", fontsize=14)
    axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[1].legend(frameon=False, fontsize=12)

    # Add a shared colorbar
    cbar = fig.colorbar(cf1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.05)
    # cbar.set_label("Normalized Percentage (Log Scale)", fontsize=14)
    cbar.set_ticks(levels)
    cbar.ax.tick_params(labelsize=10)
    # cbar.ax.set_xticklabels(
    # [f"{int(b):,}" if b >= 1 else f"{b:.2f}" for b in levels], fontsize=12
    # )
    cbar.ax.set_xticklabels([f"{v:.3f}" for v in levels])

    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()

orig_tb11_oc_lst, orig_tb11_snfl_lst = [], []
cor_tb11_oc_lst, cor_tb11_snfl_lst = [], []
import matplotlib.pyplot as plt
for i in cor_files:
    mnlat,mxlat = min(latitude_windows['NH']['window2']), max(latitude_windows['NH']['window2'])
    lat_vals = i['latitude'].data
    latmsk = (lat_vals > mnlat) & (lat_vals <= mxlat)
    cloud_probs = i['cloud_probability'].data
    cloudprobsmsk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    sftyp = i['land_class'].data
    oc_sftyp = sftyp == 0
    snfl_sftyp = sftyp == 1

    orig_tb11 = i['temp_11_0um_nom'].data
    cor_tb11 = i['temp_11_0um_nom_corrected'].data

    valid_surface_types = [0]

    # Create valid mask
    valid_mask_oc = (
        latmsk &
        np.isin(sftyp, list(valid_surface_types)) &  # Mask valid surface types &  # Mask valid surface types
        (~np.isnan(cloudprobsmsk)) &
        (~np.isnan(orig_tb11))
    )

    valid_surface_types = [1]


    valid_mask_snfl = (
        latmsk &
        np.isin(sftyp, list(valid_surface_types)) &  # Mask valid surface types
        (~np.isnan(cloudprobsmsk)) &
        (~np.isnan(orig_tb11))
    )

    orig_tb11_oc = np.where(valid_mask_oc, orig_tb11, np.nan)
    cor_tb11_oc = np.where(valid_mask_oc, cor_tb11, np.nan)

    orig_tb11_snfl = np.where(valid_mask_snfl, orig_tb11, np.nan)
    cor_tb11_snfl = np.where(valid_mask_snfl, cor_tb11, np.nan)

    orig_tb11_oc_lst.append(orig_tb11_oc)
    orig_tb11_snfl_lst.append(orig_tb11_snfl)

    cor_tb11_oc_lst.append(cor_tb11_oc)
    cor_tb11_snfl_lst.append(cor_tb11_snfl)
    


orig_hist_dist, org_bns, org_means = generate_distribution_with_means(orig_tb11_oc_lst)
cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(cor_tb11_oc_lst)


plot_discrete_ir_tb_distribution(
                            orig_hist_dist, org_bns, org_means,
                            cor_hist_dist, cor_bns, cor_means,
                            beam_positions, 'Ocean')


orig_hist_dist, org_bns, org_means = generate_distribution_with_means(orig_tb11_snfl_lst)
cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(cor_tb11_snfl_lst)


plot_discrete_ir_tb_distribution(
                            orig_hist_dist, org_bns, org_means,
                            cor_hist_dist, cor_bns, cor_means,
                            beam_positions, 'Snow free land')