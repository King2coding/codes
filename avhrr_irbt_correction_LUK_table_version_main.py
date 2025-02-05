#%%
# import packages
from util_functions import *
from plot_functions import *

#%%
# Example Usage
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

sh_lat_wind, nh_lat_wind = (-53,-61), (53,61)
#%%
print('********* Building the LUT *********')

# # min_lat, max_lat = 5, 15
# # Initialize an empty list for storing lookup table entries
# lookup_table = []
# group_arrays = []
# group_arrays_cor = []
# limb_bin_list =  {}
# limb_hist_list = {}
# nadir_bin_list = {} 
# nadir_hist_list = {}
# adjusted_limb_bins_list = {}
# # '/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14/1998/039/clavrx_NSS.GHRR.NJ.D98039.S1804.E1906.B1603132.WI.hirs_avhrr_fusion.level2.nc'
# # read abd extarct relevant files
# # start time of code
# start_time = time.time()
# for file in sorted(summer_files):
#     file_nme = os.path.basename(file)
#     data = xr.open_dataset(file)
    
#     # Extract relevant parameters for the correction
#     lats = data['latitude'][:, :].data  
#     latitude =   data['latitude']

#     # Create stratification bins while preserving NaN values
#     latitude_bins = xr.apply_ufunc(
#         lambda x: (x // latitude_bin_size) * latitude_bin_size,
#         latitude,
#         dask="allowed"
#     )
#     latitude_bins = latitude_bins.where(~np.isnan(latitude_bins),drop=True)
#     lat_intervals = np.unique(latitude_bins.data.flatten())
#     brightness_temp_11um = data['temp_11_0um_nom'].data
#     brightness_temp_11um_cor = brightness_temp_11um.copy()
#     cloud_probability = data['cloud_probability'].data 
#     surfact_type = data['land_class'].data   
#     # surfact_type = np.where((surfact_type == -128),np.nan,surfact_type)
    
#     # for l in lat_intervals:
#     lat_val= -60 #l #latitude_bin_centers[20]
#     lat_wind = [lat_val - 15, lat_val + 15]
#     max_lat, min_lat = max(np.array(lat_wind)), min(np.array(lat_wind))#lat_val - -15, lat_val + -15      
    
#     mask = ((lats >= -61) & (lats <= -53)) & \
#            (cloud_probability >= 0.5) & \
#            (surfact_type == 0)

#     mask = np.where((mask == True),1,np.nan)       

#     # Assign values to the filtered array where the mask is True
#     group_data = np.where(mask==1,brightness_temp_11um, np.nan)

#     if not np.all(np.isnan(group_data)):
#         group_arrays.append(group_data)

# # End time of code
# end_time = time.time()
# # Compute elapsed time
# elapsed_seconds = end_time - start_time
# elapsed_hours = elapsed_seconds / 3600
# # Print results
# print(f"Elapsed time for getting group data: {elapsed_seconds:.2f} seconds ({elapsed_hours:.5f} hours)")

# data_min = [min(np.nanmin(i) for i in group_arrays)][0].round(3)#int(np.floor([min(np.nanmin(i) for i in group_arrays)][0]))
# data_max = [max(np.nanmax(i) for i in group_arrays)][0].round(3)#int(np.ceil([max(np.nanmax(i) for i in group_arrays)][0]))
# data_range = (data_min, data_max)  

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

# Example usage
sh_nadir_bins_by_srftype, sh_nadir_hist_by_srftype, sh_all_nadirs_by_srftype = process_to_get_nadir_stats(group_arrays_dict_sh, 
                                                                                                          data_ranges_sh, 
                                                                                                          bin_size)

nh_nadir_bins_by_srftype, nh_nadir_hist_by_srftype, nh_all_nadirs_by_srftype = process_to_get_nadir_stats(group_arrays_dict_nh, 
                                                                                                          data_ranges_nh, 
                                                                                                          bin_size)
    
# all_nadirs = get_elements_nadir(group_arrays, reference_beam_positions)#np.hstack(nadir_elem)
# all_nadirs = np.round(all_nadirs,3)
# # Create histograms with independent ranges
# nadir_hist, nadir_bins = create_histogram(all_nadirs, bin_size, 100, data_range)
# valid_nadir_indices = nadir_hist > 0
# nadir_bins = nadir_bins[valid_nadir_indices]
# nadir_hist = nadir_hist[valid_nadir_indices]
# End time of code
# end_time = time.time()
# # Compute elapsed time
# elapsed_seconds = end_time - start_time
# elapsed_minutes = elapsed_seconds / 60
# elapsed_hours = elapsed_seconds / 3600
# Print results
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
# for i in limb_beam_positions:
#     # limb_pos_rng = range(i-10, i+10)
#     # Adjust the range dynamically based on beam position edges
#     start = max(0, i - 10)  # Ensure the range doesn't go below 0
#     end = min(409, i + 10)  # Ensure the range doesn't exceed 409
#     limb_pos_rng = range(start, end)
#     all_limbs_at_i = get_elements_limb(group_arrays, limb_pos_rng)
#     all_limbs_at_i = np.round(all_limbs_at_i,3)
                
#     limb_temp_min = np.nanmin(all_limbs_at_i)
#     limb_temp_max = np.nanmax(all_limbs_at_i)
#     limb_temp_range = (limb_temp_min, limb_temp_max)
#     # Create histograms with independent ranges
#     limb_hist, limb_bins = create_histogram(all_limbs_at_i, 
#                                             bin_size, 100,
#                                             data_range)
#     # Find the indices where counts in `limb_hist` are greater than zero
#     valid_limb_indices = limb_hist > 0
#     limb_bins = limb_bins[valid_limb_indices]
#     limb_hist = limb_hist[valid_limb_indices]

#     limb_hist_list[i] = limb_hist
#     limb_bin_list[i] = limb_bins 
    
#     adjusted_limb_bins, nadir_hist_norm = adjust_limb_bins_bob_method(nadir_bins, nadir_hist,
#                                                 limb_bins, limb_hist) 
#     adjusted_limb_bins = np.round(adjusted_limb_bins,3)
#     adjusted_limb_bins_list[i] = adjusted_limb_bins
#     nadir_hist_list[i] = nadir_hist_norm
#     nadir_bin_list[i] = nadir_bins  

#     # Populate lookup table
#     for orig_tb, corr_tb in zip(limb_bins, adjusted_limb_bins):
#         lookup_table.append({
#             "latitude": lat_val,
#             "latitude_bin": f"{min_lat}-{max_lat}",
#             # "cloud_probability": 0.5,
#             "surface_type": 0,
#             "beam_position": i,
#             "original_tb": orig_tb,
#             "corrected_tb": corr_tb,
#             "corr_coeff": corr_tb/orig_tb
#         })                          

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
        box_plot_of_corr_coeff(sftyp_lut,plt_nme)
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
cor_tb_ex = apply_lut_corrections_fast(file2run, 
                                       limb_beam_positions, 
                                       [(53,61),(-53,-61)], 
                                       lut_full, 
                                       cor_dir)


# filt = lut_full.copy()

# # Find the closest matching latitude window
# # lat_bin = filt['latitude'].iloc[(filt['latitude'] - lat).abs().idxmin()]
# # filt =filt[(filt['latitude'] == lat_bin)].reset_index()
# # Subset the data based on the provided latitude window (latwind)
# lat_win = f"{53}-{61}"
# filt = filt[filt['latitude_bin'] == lat_win].reset_index(drop=True)

# # Find the closest matching beam position
# beam_key = filt['beam_position'].iloc[(filt['beam_position'] - 8).abs().idxmin()]
# filt = filt[filt['beam_position'] == beam_key].reset_index(drop=True)

# # Find the closest matching surface type
# syrf_type_key = filt['surface_type'].iloc[(filt['surface_type'] - 3).abs().idxmin()]
# filt = filt[filt['surface_type'] == syrf_type_key].reset_index(drop=True)

# # Find the closest matching observed temperature
# temp_bin = filt['original_tb'].iloc[(filt['original_tb'] - 235.938629).abs().idxmin()]
# filt = filt[filt['original_tb'] == temp_bin].reset_index(drop=True)

# filt['corrected_tb'].values[0]  
# lut_full = pd.read_csv(os.path.join(df_dir,'sh_summer-nh_winter_LUT_all_surfaceTypes_20250130.csv'))

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