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
var2process = 'temp_12_0um_nom'
var_nme_sve = var2process.replace('_0um_nom','')
start_time = time.time()

print(f'processing group data for {var2process}')

for season in seasonal_files.keys():
    # Process each season individually
    # season = 'Summer'
    season_files = seasonal_files[season]

    print(f"Processing season: {season} with {len(season_files)} files")

    group_array_df_dict, group_array_data_minmax_dict = process_group_data_by_hemisphere_season_latitude_df_method(
                                    var2process, season_files, season)

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
        # print(key)
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
        lut_name = os.path.join(df_dir,f'{var_nme_sve}_{ke}_{cde_run_dte}.csv')
        dta.to_csv(lut_name, index=False)
    del(ke,dta)

    gc.collect()

    #--------------------------------------------------------
    for ke, dta in data_dict.items():

        hem, sesn = ke.split('_')[0], ke.split('_')[1]    

        lat_wind = dta['latitude_bin'].unique()

        for l in lat_wind:
            # print(l)
            sftyp_lut1 = dta[dta['latitude_bin'] == l]

            srftypes = sftyp_lut1['surface_type'].unique()

            for sf in srftypes:
                surf_str = surface_type_mapping[sf]
                sftyp_lut2 = sftyp_lut1[sftyp_lut1['surface_type'] == sf]
                plt_nme = f'{var_nme_sve}_{hem}_{sesn}_{l}_Distribution_of_correctiopn_coefficient_per_beam_pos_{surf_str}_{cde_run_dte}.png'
                plt_nme  = os.path.join(plot_dir,plt_nme)
                ttle = f'{var_nme_sve} {hem} {sesn} {l}  {surf_str}: Distribution of correction coefficient per beam position'
                box_plot_of_corr_coeff(sftyp_lut2, 'beam_position', 
                                    'corr_coeff', np.arange(0, 410, 50).tolist(), 
                                    ttle, plt_nme) 
                gc.collect()

    print('********* The line plot *********')

    for ke in hem_limb_hist_by_srftyp.keys():
        # print(ke)
        hem, season, lat_w = ke.split('_')
        # gather all plot data
        limb_bn_lst_by_srftyp = hem_limb_bns_by_srftyp[ke]
        limb_hist_lst_by_srftyp = hem_limb_hist_by_srftyp[ke]
        adj_limb_bn_lst_by_srftyp = hem_adj_limb_bns_by_srftyp[ke]
        nadir_bn_lst_by_srftyp = hem_nadir_bins_by_srftyp[ke]
        nadir_hist_lst_by_srftyp = hem_nadir_hists_by_srftyp[ke]
        
        # Example usage:
        plot_lut_histograms_by_hemisphere(hem, season, var_nme_sve,
                                        limb_bn_lst_by_srftyp, limb_hist_lst_by_srftyp, 
                                        adj_limb_bn_lst_by_srftyp, nadir_bn_lst_by_srftyp, 
                                        nadir_hist_lst_by_srftyp, plot_dir, cde_run_dte, lat_w)
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

print('End time of code')