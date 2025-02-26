#%%
# import packages
from util_functions import *
from plot_functions import *
from collections import defaultdict
from multiprocessing import Pool
import multiprocessing as mp
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
if __name__ == "__main__":
    num_processes = 4  # Explicitly use 4 cores, one per season
    print(f"Using {num_processes} parallel processes")

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(process_season_vectorized, [(seas, seasn_files, all_lut, limb_beam_positions, cor_dir) 
                                      for seas, seasn_files in seasonal_files.items()])

#%%
# main function calls eventually used in applying the correction
# if __name__ == "__main__":
#     with Pool(processes=12) as pool:
#         for season, files in seasonal_files.items():
#             process_season_v2(season, files, all_lut, pool, cor_dir)


#%%
# cor_dir =r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/corrected'

# for seas, seasn_files in seasonal_files.items():
#     sh_seasn = seas
#     nh_seasn = {
#         'Summer': 'Winter',
#         'Autumn': 'Spring',
#         'Winter': 'Summer',
#         'Spring': 'Autumn'
#     }[seas]

#     luts_11_nh, luts_11_sh = all_lut['temp_11']['NH'], all_lut['temp_11']['SH']
#     luts_12_nh, luts_12_sh = all_lut['temp_12']['NH'], all_lut['temp_12']['SH']

#     lut_11_nh_sh = pd.concat([luts_11_nh[nh_seasn], luts_11_sh[sh_seasn]], ignore_index=True)
#     lut_12_nh_sh = pd.concat([luts_12_nh[nh_seasn], luts_12_sh[sh_seasn]], ignore_index=True)
#     lat_windows = [tuple(map(int, lat.strip('()').split(','))) for lat in lut_11_nh_sh['latitude_bin'].unique()]

#     lut_11_nh_sh_dict, lut_12_nh_sh_dict = preprocess_lut(lut_11_nh_sh), preprocess_lut(lut_12_nh_sh)

#     for file_of_season in seasn_files:
#         print(f"Starting processing for {seas} season")
#         start_time = time.time()
#         process_file_vectorized(file_of_season, lat_windows,
#                                 lut_11_nh_sh,lut_11_nh_sh_dict,
#                                 lut_12_nh_sh,lut_12_nh_sh_dict,
#                                 limb_beam_positions, cor_dir)
        
#         end_time = time.time()
#         elapsed_seconds = end_time - start_time
#         elapsed_minutes = elapsed_seconds / 60
#         elapsed_hours = elapsed_seconds / 3600
#         print(f"Completed processing for {seas} season")
#         print('************************' * 100)
#         print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds "
#             f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")
#         print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds "
#             f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")

#%%
# print('********* doing plot_ir_tb_distribution_with_means dist plots now *********')

# for hemisphere in ['NH', 'SH']:
#     # pull hemisphere data
#     obs_data_hem = observed_arrays[hemisphere]
#     cor_data_hem = corrected_arrays[hemisphere]

#     # pull surface type data
#     for i in surface_type_mapping.keys():
#         sftype = surface_type_mapping[i]
#         obs_by_srftyp_array = obs_data_hem[i]
#         cor_by_srftyp_array = cor_data_hem[i]

#         # Generate distribution with means
#         orig_hist_dist, org_bns, org_means = generate_distribution_with_means(obs_by_srftyp_array)
#         cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(cor_by_srftyp_array)

#         #--------------------------------------------------------
#         # Plot the distributions using different plot methods 
#         savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_normal_version_{cde_run_dte}.png"    
#         savefig  = os.path.join(plot_dir, savefig)
#         plot_ir_tb_distribution_with_means_normal_version(
#                                     orig_hist_dist, org_bns, org_means,
#                                     cor_hist_dist, cor_bns, cor_means,
#                                     beam_positions, savefig)

#         #--------------------------------------------------------

#         savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_log_version_{cde_run_dte}.png"    
#         savefig  = os.path.join(plot_dir, savefig)
#         plot_ir_tb_distribution_with_means_log_version(
#                                     orig_hist_dist, org_bns, org_means,
#                                     cor_hist_dist, cor_bns, cor_means,
#                                     beam_positions, savefig)
#         #--------------------------------------------------------

#         savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_discrete-log_version_{cde_run_dte}.png"
#         savefig  = os.path.join(plot_dir, savefig)
#         plot_ir_tb_distribution_with_means_discrete_log(
#                                     orig_hist_dist, org_bns, org_means,
#                                     cor_hist_dist, cor_bns, cor_means,
#                                     beam_positions, savefig)
#         #--------------------------------------------------------

#         savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_contourf_version_{cde_run_dte}.png"        
#         savefig  = os.path.join(plot_dir, savefig)
#         plot_discrete_ir_tb_distribution(
#                                     orig_hist_dist, org_bns, org_means,
#                                     cor_hist_dist, cor_bns, cor_means,
#                                     beam_positions, savefig)