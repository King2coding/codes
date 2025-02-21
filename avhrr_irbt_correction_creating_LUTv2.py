#%%
# import packages
from util_functions import *
from plot_functions import *

#%%
base_path = "/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14"  # Replace with your actual path
df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
plot_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/plots/Feb2025'
cor_dir =r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/corrected'
miscellaneous_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/miscellaneous'
path_to_1998_n14_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/frequency_analysis/data/avhrr/patmosx_l2_jan_1998/noaa-14/1998'
summer_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_1998_summer_for_Kingsley'
all_noaa_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_AutoSnow_collocated_1998_2000_for_Kingsley'
#%%
# floating variables
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""
# Select 1998 NOAA-14 files identified by "NJ" in the file name
all_noaa_files = sorted([
    os.path.join(all_noaa_data, s) 
    for s in os.listdir(all_noaa_data) 
    if s.endswith('.nc') and "NJ" in s and "D98" in s
])

# organize file by seasons and hemisphere


# Organize files by season and hemisphere
seasonal_files = organize_files_by_season_in_hemisphere(all_noaa_files,'SH',1998)

# # choose only 1 hemisphere
# hemisphere_sh = "Southern"  # Change to 'Northern' for the Northern Hemisphere
# # hemisphere_sh = "Northern"  # Change to 'Northern' for the Northern Hemisphere

# years = sorted([int(i) for i in os.listdir(base_path)])

# search_dir = os.path.join(base_path,str(years[0]))
# # Organize folders into seasons based on SH
# # Using the Southern Hemisphere (SH) to determine the seasons and corresponding files.
# # Note: The same files can be used to define the Northern Hemisphere (NH) seasons.
# # For example, summer in SH corresponds to winter in NH.
# seasons = organize_by_season_with_hemisphere(search_dir, hemisphere_sh, years[0])

# # Example: Load files for Summer
# # summer_files = load_files_for_season(search_dir, seasons["Summer"])

# seasonal_files = {}
# for season in seasons.keys():
#     seasonal_files[season] = load_files_for_season(search_dir, seasons[season])

# summer_files = [os.path.join(summer_data,s) for s in os.listdir(summer_data) if s.endswith('.nc')] 

file2run = next((f for f in all_noaa_files if "clavrx_NSS.GHRR.NJ.D98015.S1728.E1912.B156" in f), None)

cde_run_dte = str(date.today().strftime('%Y%m%d'))

nc_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(path_to_1998_n14_data)
    for filename in filenames if filename.endswith(".nc")
]

# sh_lat_wind = latitude_windows['SH']['window1']
# nh_lat_wind = latitude_windows['NH']['window1']

gc.collect()
#%%
print('********* The group data *********')

start_time = time.time()

print('processing group data for temp_11_0um_nom')

# Process each season individually
season = 'Summer'
season_files = seasonal_files[season]
print(f"Processing season: {season} with {len(season_files)} files")
group_array_df_dict, group_array_data_minmax_dict = process_group_data_by_hemisphere_season_latitude_df_method(
                                   'temp_11_0um_nom', season_files, season)

gc.collect()
#--------------------------------------------------------
print('********* The nadir stats *********')

hem_nadir_bins, hem_nadir_hist, all_hem_nadirs = grab_nadir_stats_elements(group_array_df_dict, 
                                                                           group_array_data_minmax_dict,
                                                                           list(group_array_df_dict.keys()))
gc.collect()

#--------------------------------------------------------
print('********* The limb stats *********')

hem_luts_by_srftyp, hem_limb_hist_by_srftyp, hem_limb_bns_by_srftyp,\
hem_nadir_hists_by_srftyp, hem_nadir_bins_by_srftyp,\
hem_adj_limb_bns_by_srftyp, all_hem_limb_by_srftyp = grab_limb_stats_elements(
                                                    group_array_df_dict, group_array_data_minmax_dict,
                                                    hem_nadir_bins, hem_nadir_hist,
                                                    list(group_array_df_dict.keys()))

gc.collect()

#--------------------------------------------------------
# Concatenate the DataFrames for each hemisphere and season
data_dict = {}

# Iterate over the keys in the hem_luts_by_srftyp dictionary
for key, value in hem_luts_by_srftyp.items():
    print(key)
    lat_w = key.split('_')[2]
    # Extract the hemisphere and season from the key
    hemisphere, season = key.split('_')[0], key.split('_')[1]

    # Create a new key for the data_dict dictionary
    new_key = f"{hemisphere}_{season}"

    # If the new key is not already in the data_dict dictionary, add it
    if new_key not in data_dict:
        data_dict[new_key] = []

    # Set the 'latitude_bin' column in each DataFrame
    for df in value:
        value[df]['latitude_bin'] = lat_w

    # Concatenate the DataFrames within the key and append to the data_dict list
    data_dict[new_key].append(pd.concat(value, ignore_index=True))
del(key,value)
gc.collect()

# Combine the DataFrames from each hemisphere and season into separate DataFrames
for key, value in data_dict.items():
    data_dict[key] = pd.concat(value, ignore_index=True)
del(key,value)
#--------------------------------------------------------

# save lut for each hemisphere and season
for ke, dta in data_dict.items():
    lut_name = os.path.join(df_dir,f'{ke}_{cde_run_dte}.csv')
    dta.to_csv(lut_name, index=False)
del(ke,dta)

gc.collect()

#--------------------------------------------------------
for ke, dta in data_dict.items():

    hem, sesn = ke.split('_')[0], ke.split('_')[1]    

    lat_wind = dta['latitude_bin'].unique()

    for l in lat_wind:
        print(l)
        sftyp_lut1 = dta[dta['latitude_bin'] == l]

        srftypes = sftyp_lut1['surface_type'].unique()

        for sf in srftypes:
            surf_str = surface_type_mapping[sf]
            sftyp_lut2 = sftyp_lut1[sftyp_lut1['surface_type'] == sf]
            plt_nme = f'{hem}_{sesn}_{l}_Distribution_of_correctiopn_coefficient_per_beam_pos_{surf_str}_{cde_run_dte}.png'
            plt_nme  = os.path.join(plot_dir,plt_nme)
            ttle = f'{hem} {sesn} {l}  {surf_str}: Distribution of correction coefficient per beam position'
            box_plot_of_corr_coeff(sftyp_lut2, 'beam_position', 
                                   'corr_coeff', np.arange(0, 410, 50).tolist(), 
                                   ttle, plt_nme) 
            gc.collect()

print(f"Completed processing for season: {season}")

#--------------------------------------------------------

# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600

# Print results
print(f"Elapsed time for getting group data: {elapsed_seconds:.2f} seconds "
      f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")
             
#%%
# print('********* The line plot *********')

# # some plots

# # Example usage:
# plot_lut_histograms_by_hemisphere('SH', limb_bin_list_by_srftype_sh, limb_hist_list_by_srftype_sh, 
#                                   adjusted_limb_bins_list_by_srftype_sh, nadir_bin_list_by_srftype_sh, 
#                                   nadir_hist_list_by_srftype_sh, plot_dir, cde_run_dte, sh_lat_wind)

# plot_lut_histograms_by_hemisphere('NH', limb_bin_list_by_srftype_nh, limb_hist_list_by_srftype_nh, 
#                                   adjusted_limb_bins_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, 
#                                   nadir_hist_list_by_srftype_nh, plot_dir, cde_run_dte, nh_lat_wind)"""
#%%
start_time = time.time()


print('processing group data for temp_11_0um_nom')
group_array_dict_elements_11 = {}

# Process each season individually
season = 'Summer'
season_files = seasonal_files[season]
print(f"Processing season: {season} with {len(season_files)} files")
group_array_dict_elements_11[season] = process_group_data_by_hemisphere_season_latitude_df_method(
                                   'temp_11_0um_nom', season_files, season)
print(f"Completed processing for season: {season}")

# arr_try = group_array_dict_elements_11[season][0]['SH_Summer_(-75, -61)'][0][0]
# arr2df = nonnan_to_df(arr_try)
#%%
season = 'Winter'
season_files = seasonal_files[season]
print(f"Processing season: {season} with {len(season_files)} files")
group_array_dict_elements_11[season] = process_group_data_by_hemisphere_season_latitude_v2(
                                   'temp_11_0um_nom', season_files, season)
print(f"Completed processing for season: {season}")
#%%
season = 'Spring'
season_files = seasonal_files[season]
print(f"Processing season: {season} with {len(season_files)} files")
group_array_dict_elements_11[season] = process_group_data_by_hemisphere_season_latitude_v2(
                                   'temp_11_0um_nom', season_files, season)
print(f"Completed processing for season: {season}")
#%%
season = 'Autumn'
season_files = seasonal_files[season]
print(f"Processing season: {season} with {len(season_files)} files")
group_array_dict_elements_11[season] = process_group_data_by_hemisphere_season_latitude_v2(
                                   'temp_11_0um_nom', season_files, season)
print(f"Completed processing for season: {season}")
#%%

print('processing group data for temp_11_0um_nom')    
group_array_dict_elements_11 = {}
for sees in seasonal_files.keys():
    print(f"Processing season: {sees} with {len(seasonal_files[sees])} files")
    season_files = seasonal_files[sees]
    group_array_dict_elements_11[sees] = process_group_data_by_hemisphere_season_latitude_df_method(                                       
                                       'temp_11_0um_nom', season_files, sees)
    
    gc.collect()



print('processing group data for temp_12_0um_nom')    
group_array_dict_elements_12 = {}
for sees in seasonal_files.keys():
    season_files = seasonal_files[sees]
    group_array_dict_elements_12[sees] = process_group_data_by_hemisphere_season_latitude_v2(
                                       
                                       'temp_12_0um_nom', seasonal_files, sees)

# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600

# Print results
print(f"Elapsed time for getting group data: {elapsed_seconds:.2f} seconds "
      f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")

#%%
print('********* The nadir stats *********')
# Get the indices of pixels meeting the conditions
# start time of code
start_time = time.time()

hem_group_dicts11, hem_group_data_rnges11 = group_array_dict_elements_11['Summer'][0], group_array_dict_elements_11['Summer'][1]

summer_hem_nadir_bins, hem_nadir_hist, all_hem_nadirs = grab_nadir_stats_elements(hem_group_dicts11, 
                                                                                  hem_group_data_rnges11,
                                                                                  list(hem_group_dicts11.keys()))
# Example usage for "temp_11_0um_nom"
(sh_nadir_bins_by_srftype_11, sh_nadir_hist_by_srftype_11, sh_all_nadirs_by_srftype_11,
nh_nadir_bins_by_srftype_11, nh_nadir_hist_by_srftype_11, nh_all_nadirs_by_srftype_11) = process_nadir_stats_by_hem_season_lat_var(
                                                                                         group_arrays_dict_11, 
                                                                                         data_ranges_11, 
                                                                                         combinations, 
                                                                                         bin_size
                                                                                        )

# Example usage for "temp_12_0um_nom"
(sh_nadir_bins_by_srftype_12, sh_nadir_hist_by_srftype_12, sh_all_nadirs_by_srftype_12,
nh_nadir_bins_by_srftype_12, nh_nadir_hist_by_srftype_12, nh_all_nadirs_by_srftype_12) = process_nadir_stats_by_hem_season_lat_var(
                                                                                         group_arrays_dict_12, 
                                                                                         data_ranges_12, 
                                                                                         combinations, 
                                                                                         bin_size
                                                                                        )

# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600

# Print results
print(f"Elapsed time for getting nadir stats: {elapsed_seconds:.2f} seconds "
      f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")

#%%
print('********* The limb stats *********')
# Initialize an array to store corrected limb data
# # start time of code
start_time = time.time()


# Example usage for "temp_11_0um_nom"
(lookup_table_by_srftype_sh_11, limb_hist_list_by_srftype_sh_11, limb_bin_list_by_srftype_sh_11, 
 nadir_hist_list_by_srftype_sh_11, nadir_bin_list_by_srftype_sh_11, adjusted_limb_bins_list_by_srftype_sh_11, 
 all_limbs_at_i_list_by_srftype_sh_11, lookup_table_by_srftype_nh_11, limb_hist_list_by_srftype_nh_11, 
 limb_bin_list_by_srftype_nh_11, nadir_hist_list_by_srftype_nh_11, nadir_bin_list_by_srftype_nh_11, 
 adjusted_limb_bins_list_by_srftype_nh_11, all_limbs_at_i_list_by_srftype_nh_11) = process_limb_stats_by_hem_season_lat_var(
    group_arrays_dict_11, data_ranges_11, sh_nadir_bins_by_srftype_11, sh_nadir_hist_by_srftype_11, combinations)

# Example usage for "temp_12_0um_nom"
(lookup_table_by_srftype_sh_12, limb_hist_list_by_srftype_sh_12, limb_bin_list_by_srftype_sh_12, 
 nadir_hist_list_by_srftype_sh_12, nadir_bin_list_by_srftype_sh_12, adjusted_limb_bins_list_by_srftype_sh_12, 
 all_limbs_at_i_list_by_srftype_sh_12, lookup_table_by_srftype_nh_12, limb_hist_list_by_srftype_nh_12, 
 limb_bin_list_by_srftype_nh_12, nadir_hist_list_by_srftype_nh_12, nadir_bin_list_by_srftype_nh_12, 
 adjusted_limb_bins_list_by_srftype_nh_12, all_limbs_at_i_list_by_srftype_nh_12) = process_limb_stats_by_hem_season_lat_var(
    group_arrays_dict_12, data_ranges_12, nh_nadir_bins_by_srftype_12, nh_nadir_hist_by_srftype_12, combinations)

# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting limb stats: {elapsed_seconds:.2f} seconds "
    f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")
                       

#%%
print('********* The lookup table *********')
# Convert lookup table to DataFrame and save it
# lookup_df = pd.DataFrame(lookup_table)
lut_full_sh = pd.concat(lookup_table_by_srftype_sh.values(), ignore_index=True)
lut_name = os.path.join(df_dir,f'sh_summer_LUT_all_surfaceTypes_{cde_run_dte}.csv')
lut_full_sh.to_csv(lut_name, index=False)
#--------------------------------------------------------

lut_full_nh = pd.concat(lookup_table_by_srftype_nh.values(), ignore_index=True)
lut_name = os.path.join(df_dir,f'nh_winter_LUT_all_surfaceTypes_{cde_run_dte}.csv')
lut_full_nh.to_csv(lut_name, index=False)
#--------------------------------------------------------

# combine sh and nh luts and save
lut_full = pd.concat([lut_full_sh, lut_full_nh], ignore_index=True)
lut_name = os.path.join(df_dir,f'sh_summer-nh_winter_LUT_all_surfaceTypes_{cde_run_dte}.csv')
lut_full.to_csv(lut_name, index=False)

#--------------------------------------------------------

for hem, hem_data in zip(['SH', 'NH'], [lookup_table_by_srftype_sh, lookup_table_by_srftype_nh]):
    print(hem)
    for i in surface_type_mapping.keys():
        if i in hem_data.keys():
            sftyp = surface_type_mapping[i]
            sftyp_lut = hem_data[i]
            plt_nme = f'{hem}_summer_mean_correction_coeff_at_beam_position_{sftyp}_{cde_run_dte}.png'
            plt_nme  = os.path.join(plot_dir, plt_nme)
            plt.figure()
            groupby_and_plot(sftyp_lut, 'beam_position', 'corr_coeff', sftyp,hem)       
            plt.savefig(plt_nme, dpi=300, bbox_inches='tight')

            plt_nme = f'{hem}_Distribution_of_correctiopn_coefficient_per_beam_pos_{sftyp}_{cde_run_dte}.png'
            plt_nme  = os.path.join(plot_dir,plt_nme)
            box_plot_of_corr_coeff(sftyp_lut, 
                                'beam_position', 
                                'corr_coeff', 
                                np.arange(0, 410, 50).tolist(), 
                                plt_nme)   
#--------------------------------------------------------
# 
# # End time of code"""