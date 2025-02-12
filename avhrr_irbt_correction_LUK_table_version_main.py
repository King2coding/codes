#%%
# import packages
from util_functions import *
from plot_functions import *

#%%
base_path = "/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14"  # Replace with your actual path
df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df'
plot_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/plots'
cor_dir =r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/corrected'
miscellaneous_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/miscellaneous'
path_to_1998_n14_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/frequency_analysis/data/avhrr/patmosx_l2_jan_1998/noaa-14/1998'
summer_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR_1998_summer_for_Kingsley'
#%%
# floating variables
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""
hemisphere_sh = "Southern"  # Change to 'Northern' for the Northern Hemisphere
hemisphere_sh = "Northern"  # Change to 'Northern' for the Northern Hemisphere

years = sorted([int(i) for i in os.listdir(base_path)])

search_dir = os.path.join(base_path,str(years[0]))
# Organize folders into seasons
seasons = organize_by_season_with_hemisphere(search_dir, hemisphere_sh, years[0])

# Example: Load files for Summer
# summer_files = load_files_for_season(search_dir, seasons["Summer"])
summer_files = [os.path.join(summer_data,s) for s in os.listdir(summer_data) if s.endswith('.nc')] 

file2run = next((f for f in summer_files if "clavrx_NSS.GHRR.NJ.D98015.S1728.E1912.B156" in f), None)

lat_intervals = np.arange(-90,90,5)

cde_run_dte = str(date.today().strftime('%Y%m%d'))

nc_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(path_to_1998_n14_data)
    for filename in filenames if filename.endswith(".nc")
]

sh_lat_wind, nh_lat_wind = (-75,-61), (61,75)

#%%
print('********* The group data *********')

start_time = time.time()

group_arrays_dict_sh, data_ranges_sh = get_group_data(summer_files,'temp_11_0um_nom',sh_lat_wind)

group_arrays_dict_nh, data_ranges_nh = get_group_data(summer_files,'temp_11_0um_nom',nh_lat_wind)

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
 all_limbs_at_i_list_by_srftype_sh) = process_to_get_limb_stats(group_arrays_dict_sh, 
                                                         data_ranges_sh, 
                                                         sh_nadir_bins_by_srftype, 
                                                         sh_nadir_hist_by_srftype, 
                                                         sh_lat_wind)

#--------------------------------------------------------
(lookup_table_by_srftype_nh, limb_hist_list_by_srftype_nh, limb_bin_list_by_srftype_nh,
 nadir_hist_list_by_srftype_nh, nadir_bin_list_by_srftype_nh, adjusted_limb_bins_list_by_srftype_nh,
 all_limbs_at_i_list_by_srftype_nh) = process_to_get_limb_stats(group_arrays_dict_nh,
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
cor_tb_ex = apply_lut_corrections_fast_v2(file2run, 
                                       limb_beam_positions, 
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