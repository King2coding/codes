#%%
# import packages
from util_functions import *
from plot_functions import *
from collections import defaultdict
from multiprocessing import Pool
# from process_functions import process_season_v2  # Import the missing function

#%%
all_noaa_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_AutoSnow_collocated_1998_2000_for_Kingsley'

df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
plot_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/plots/Feb2025'
cor_dir =r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/all_corrected_avhrr'
miscellaneous_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/miscellaneous'

#%%
# floating variables
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""
all_noaa_files = sorted([os.path.join(all_noaa_data, s) for s in os.listdir(all_noaa_data) if s.endswith('.nc')])

file2run = next((f for f in all_noaa_files if "clavrx_NSS.GHRR.NJ.D98015.S1728.E1912.B156" in f), None)

# Organize files by season and hemisphere
# Organize folders into seasons based on SH
# Using the Southern Hemisphere (SH) to determine the seasons and corresponding files.
# Note: The same files can be used to define the Northern Hemisphere (NH) seasons.
# For example, summer in SH corresponds to winter in NH.
seasonal_files = organize_files_by_season_in_hemisphere(all_noaa_files,'SH',1998)


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
# main function calls eventually used in applying the correction
if __name__ == "__main__":
    with Pool(processes=10) as pool:
        for season, files in seasonal_files.items():
            process_season_v2(season, files, all_lut, pool)
def process_pixel_v2(args):
    i, j, brightness_temp_11, brightness_temp_12, surfact_type, lut_11_nh_sh, lut_12_nh_sh, lat_window = args
    temp_11_tb = brightness_temp_11[i, j]
    temp_12_tb = brightness_temp_12[i, j]
    surface_type_val = surfact_type[i, j]
    temp_11_cor_coeff = get_correction(str(lat_window), int(j), surface_type_val, temp_11_tb, lut_11_nh_sh)
    temp_12_cor_coeff = get_correction(str(lat_window), int(j), surface_type_val, temp_12_tb, lut_12_nh_sh)
    return (i, j, temp_11_tb * temp_11_cor_coeff, temp_12_tb * temp_12_cor_coeff)
#------------------------------------------

def process_file_v2(file_of_season, lut_11_nh_sh, lut_12_nh_sh, lat_windows, cor_dir, pool):
    dataset = xr.open_dataset(file_of_season)
    lats = dataset['latitude'].data
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    surfact_type = dataset['land_class'].data
    brightness_temp_11 = dataset['temp_11_0um_nom'].data
    brightness_temp_12 = dataset['temp_12_0um_nom'].data
    corrected_tb_11 = brightness_temp_11.copy()
    corrected_tb_12 = brightness_temp_12.copy()

    for lat_window in lat_windows:
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_msk = ((lats >= min_lat) & (lats <= max_lat))
        valid_mask = (
            lat_msk &
            (~np.isnan(cloud_probs_msk)) &
            (~np.isnan(brightness_temp_11)) &
            (~np.isnan(brightness_temp_12)) &
            (~np.isnan(surfact_type))
        )
        valid_indices = np.argwhere(valid_mask)
        valid_valid_indices = [index for index in valid_indices if index[1] in limb_beam_positions]

        results = pool.map(process_pixel_v2, [(i, j, brightness_temp_11, brightness_temp_12, surfact_type, lut_11_nh_sh, lut_12_nh_sh, lat_window) for i, j in valid_valid_indices])

        for i, j, corrected_11, corrected_12 in results:
            corrected_tb_11[i, j] = corrected_11
            corrected_tb_12[i, j] = corrected_12

    data_outfile = os.path.join(cor_dir, os.path.basename(file_of_season.replace('.nc', '_cor.nc')))
    save_corrected_dataset_v2(dataset, corrected_tb_11, corrected_tb_12, data_outfile)
#------------------------------------------

def process_season_v2(seas, seasn_files, all_lut, pool):
    start_time = time.time()
    sh_seasn = seas
    nh_seasn = {
        'Summer': 'Winter',
        'Autumn': 'Spring',
        'Winter': 'Summer',
        'Spring': 'Autumn'
    }[seas]

    luts_11_nh, luts_11_sh = all_lut['temp_11']['NH'], all_lut['temp_11']['SH']
    luts_12_nh, luts_12_sh = all_lut['temp_12']['NH'], all_lut['temp_12']['SH']

    lut_11_nh_sh = pd.concat([luts_11_nh[nh_seasn], luts_11_sh[sh_seasn]], ignore_index=True)
    lut_12_nh_sh = pd.concat([luts_12_nh[nh_seasn], luts_12_sh[sh_seasn]], ignore_index=True)
    lat_windows = [tuple(map(int, lat.strip('()').split(','))) for lat in lut_11_nh_sh['latitude_bin'].unique()]

    for file_of_season in seasn_files:
        process_file_v2(file_of_season, lut_11_nh_sh, lut_12_nh_sh, lat_windows, cor_dir, pool)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_seconds / 3600
    print(f"Completed processing for {seas} season")
    print('************************' * 100)
    print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds "
          f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")
    print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds "
        f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")

#%%
print('********* doing plot_ir_tb_distribution_with_means dist plots now *********')

for hemisphere in ['NH', 'SH']:
    # pull hemisphere data
    obs_data_hem = observed_arrays[hemisphere]
    cor_data_hem = corrected_arrays[hemisphere]

    # pull surface type data
    for i in surface_type_mapping.keys():
        sftype = surface_type_mapping[i]
        obs_by_srftyp_array = obs_data_hem[i]
        cor_by_srftyp_array = cor_data_hem[i]

        # Generate distribution with means
        orig_hist_dist, org_bns, org_means = generate_distribution_with_means(obs_by_srftyp_array)
        cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(cor_by_srftyp_array)

        #--------------------------------------------------------
        # Plot the distributions using different plot methods 
        savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_normal_version_{cde_run_dte}.png"    
        savefig  = os.path.join(plot_dir, savefig)
        plot_ir_tb_distribution_with_means_normal_version(
                                    orig_hist_dist, org_bns, org_means,
                                    cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)

        #--------------------------------------------------------

        savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_log_version_{cde_run_dte}.png"    
        savefig  = os.path.join(plot_dir, savefig)
        plot_ir_tb_distribution_with_means_log_version(
                                    orig_hist_dist, org_bns, org_means,
                                    cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)
        #--------------------------------------------------------

        savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_discrete-log_version_{cde_run_dte}.png"
        savefig  = os.path.join(plot_dir, savefig)
        plot_ir_tb_distribution_with_means_discrete_log(
                                    orig_hist_dist, org_bns, org_means,
                                    cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)
        #--------------------------------------------------------

        savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_contourf_version_{cde_run_dte}.png"        
        savefig  = os.path.join(plot_dir, savefig)
        plot_discrete_ir_tb_distribution(
                                    orig_hist_dist, org_bns, org_means,
                                    cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)