#%%
# import packages
from util_functions import *
from plot_functions import *
from collections import defaultdict

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
all_lut_files = sorted([os.path.join(df_dir, s) for s in os.listdir(df_dir) if s.endswith('.csv')])

# Initialize a nested dictionary
all_lut = defaultdict(lambda: defaultdict(dict))

for file_path in all_lut_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    var = parts[0] + '_' + parts[1]
    hemisphere = parts[2]
    season = parts[3]
    all_lut[var][hemisphere][season] = pd.read_csv(file_path)

#%%

luts_11_nh, luts_11_sh = all_lut['temp_11']['NH'], all_lut['temp_11']['SH']
luts_12_nh, luts_12_sh = all_lut['temp_12']['NH'], all_lut['temp_12']['SH']


for seas, seasn_files in seasonal_files.items():
    print(f"Season: {seas}")  

    start_time = time.time()
    sh_seasn = seas
    nh_seasn = {
            'Summer': 'Winter',
            'Autumn': 'Spring',
            'Winter': 'Summer',
            'Spring': 'Autumn'
        }[seas]

    # get the lut needed for the season

    lut_11_nh_sh = pd.concat([luts_11_nh[nh_seasn], luts_11_sh[sh_seasn]], ignore_index=True)
    lut_12_nh_sh = pd.concat([luts_12_nh[nh_seasn], luts_12_sh[sh_seasn]], ignore_index=True)

    lat_windows = [tuple(map(int, lat.strip('()').split(','))) for lat in lut_11_nh_sh['latitude_bin'].unique()]
    
    dataset = xr.open_dataset(file2run)
    lats = dataset['latitude'].data
    cloud_probs = dataset['cloud_probability'].data
    cloud_probs_msk = np.where(cloud_probs >= 0.5, cloud_probs, np.nan)
    surfact_type = dataset['land_class'].data
    brightness_temp_11 = dataset['temp_11_0um_nom'].data
    brightness_temp_12 = dataset['temp_12_0um_nom'].data
    corrected_tb_11 = brightness_temp_11.copy()  # Copy original data for corrections
    corrected_tb_12 = brightness_temp_12.copy()  # Copy original data for corrections

    for lat_window in lat_windows:
        lat_wind_ = f"{lat_window[0]}-{lat_window[1]}"
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_msk = ((lats >= min_lat) & (lats <= max_lat))

        valid_mask = (
            lat_msk &
            (~np.isnan(cloud_probs_msk)) &
            (~np.isnan(brightness_temp_11)) &
            (~np.isnan(brightness_temp_12)) &
            (~np.isnan(surfact_type))
        )
        # Get the indices where the mask is True
        valid_indices = np.argwhere(valid_mask)

        # Filter valid_indices to include only those where j is in limb_beam_positions
        valid_valid_indices = [index for index in valid_indices if index[1] in limb_beam_positions]

        # Vectorized approach for faster processing
        # valid_valid_indices = np.array(valid_valid_indices)
        # i_indices = valid_valid_indices[:, 0]
        # j_indices = valid_valid_indices[:, 1]

        # lat_values = lats[i_indices, j_indices]
        # original_tb_11_values = brightness_temp_11[i_indices, j_indices]
        # original_tb_12_values = brightness_temp_12[i_indices, j_indices]
        # surface_type_values = surfact_type[i_indices, j_indices]

        # corrections_11 = np.array([
        #     get_correction(lat_wind_, int(j), surf_type, tb, lut_11_nh_sh)
        #     for j, surf_type, tb in zip(j_indices, surface_type_values, original_tb_11_values)
        # ])

        # corrections_12 = np.array([
        #     get_correction(lat_wind_, int(j), surf_type, tb, lut_12_nh_sh)
        #     for j, surf_type, tb in zip(j_indices, surface_type_values, original_tb_12_values)
        # ])

        # corrected_tb_11[i_indices, j_indices] = original_tb_11_values * corrections_11
        # corrected_tb_12[i_indices, j_indices] = original_tb_12_values * corrections_12

        for i, j in valid_indices:
            if j in limb_beam_positions:

                lat = lats[i, j]
                original_tb = brightness_temp_11[i, j]
                surface_type_val = surfact_type[i, j]

                # print(j, original_tb, surface_type_val, lat_wind_)

                # Apply correction using the lookup table
                correction = get_correction(str(lat_window), int(j), surface_type_val, original_tb, lut_11_nh_sh)
                corrected_tb_11[i, j] = original_tb * correction  # Apply correction



            

        # Apply correction using the lookup table

    # Save the corrected data to a new NetCDF file
    data_outfile = os.path.join(cor_dir, os.path.basename(file2run.replace('.nc', '_cor_again_final.nc')))
    save_corrected_dataset_v2(dataset, corrected_tb_11, corrected_tb_12, data_outfile)

    # Apply the correction
    # corrected_arrays, observed_arrays = process_files_in_parallel(
    #     seasn_files, [(53,61),(-53,-61)], lookup_df=lut_full,
    #     limb_beam_positions=limb_beam_positions, cor_dir=cor_dir
    # )
    # End time of code
    end_time = time.time()
    # Compute elapsed time
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_seconds / 3600
    # Print results
    print(f"Completed processing for {seas} season")
    print('************************' * 100)
    print('************************' * 100)
    print('************************' * 100)
    print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds "
        f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")
    


#%%
print('********* The group data *********')

start_time = time.time()

group_arrays_dict_sh, data_ranges_sh = get_group_data(summer_files,
                                                      'temp_11_0um_nom', 
                                                      'SH', 
                                                      'Summer',
                                                      sh_lat_wind)

group_arrays_dict_nh, data_ranges_nh = get_group_data(summer_files,
                                                      'temp_11_0um_nom',
                                                      'NH',
                                                      'Winter',
                                                      nh_lat_wind)

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
print('********* The nadir stats *********')
# Get the indices of pixels meeting the conditions
# start time of code
start_time = time.time()

sh_nadir_bins_by_srftype, sh_nadir_hist_by_srftype, sh_all_nadirs_by_srftype = process_to_get_nadir_stats(
    group_arrays_dict_sh, 
    data_ranges_sh, 
    bin_size
)

nh_nadir_bins_by_srftype, nh_nadir_hist_by_srftype, nh_all_nadirs_by_srftype = process_to_get_nadir_stats(
    group_arrays_dict_nh, 
    data_ranges_nh, 
    bin_size
)

print(f"Elapsed time for getting limb stats: {elapsed_seconds:.2f} seconds "
    f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")

#%%
print('********* The limb stats *********')
# Initialize an array to store corrected limb data
# # start time of code
start_time = time.time()

(lookup_table_by_srftype_sh, limb_hist_list_by_srftype_sh, limb_bin_list_by_srftype_sh, 
 nadir_hist_list_by_srftype_sh, nadir_bin_list_by_srftype_sh, adjusted_limb_bins_list_by_srftype_sh, 
 all_limbs_at_i_list_by_srftype_sh) = process_to_get_limb_stats_fast(
                                                         group_arrays_dict_sh, 
                                                         data_ranges_sh, 
                                                         sh_nadir_bins_by_srftype, 
                                                         sh_nadir_hist_by_srftype, 
                                                         sh_lat_wind)

#--------------------------------------------------------
(lookup_table_by_srftype_nh, limb_hist_list_by_srftype_nh, limb_bin_list_by_srftype_nh,
 nadir_hist_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, adjusted_limb_bins_list_by_srftype_nh,
 all_limbs_at_i_list_by_srftype_nh) = process_to_get_limb_stats_fast(
                                                                group_arrays_dict_nh,
                                                                data_ranges_nh,
                                                                nh_nadir_bins_by_srftype,
                                                                nh_nadir_hist_by_srftype,
                                                                nh_lat_wind)
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
#%%
print('********* The line plot *********')

# some plots

# Example usage:
plot_lut_histograms_by_hemisphere('SH', limb_bin_list_by_srftype_sh, limb_hist_list_by_srftype_sh, 
                                  adjusted_limb_bins_list_by_srftype_sh, nadir_bin_list_by_srftype_sh, 
                                  nadir_hist_list_by_srftype_sh, plot_dir, cde_run_dte, sh_lat_wind)

plot_lut_histograms_by_hemisphere('NH', limb_bin_list_by_srftype_nh, limb_hist_list_by_srftype_nh, 
                                  adjusted_limb_bins_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, 
                                  nadir_hist_list_by_srftype_nh, plot_dir, cde_run_dte, nh_lat_wind)
         
#%%
print('********* The correction *********')
# limb_beam_positions, 
cor_tb_ex = apply_lut_corrections_fast(file2run,                                    
                                       [(61,75),(-75,-61)], 
                                       lut_full, 
                                       cor_dir)



# Example usage
start_time = time.time()
corrected_arrays, observed_arrays = process_files_in_parallel(
    summer_files, [(53,61),(-53,-61)], lookup_df=lut_full,
    limb_beam_positions=limb_beam_positions, cor_dir=cor_dir
)
# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600
# Print results
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