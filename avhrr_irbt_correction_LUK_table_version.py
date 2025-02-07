#%%
# import packages
from util_functions import *
from plot_functions import *

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import imageio
import cv2

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
hemisphere = "Southern"  # Change to 'Northern' for the Northern Hemisphere
years = sorted([int(i) for i in os.listdir(base_path)])

search_dir = os.path.join(base_path,str(years[0]))
# Organize folders into seasons
seasons = organize_by_season_with_hemisphere(search_dir, hemisphere, years[0])

# Example: Load files for Summer
# summer_files = load_files_for_season(search_dir, seasons["Summer"])
summer_files = [os.path.join(summer_data,s) for s in os.listdir(summer_data) if s.endswith('.nc')] 

lat_intervals = np.arange(-90,90,5)

cde_run_dte = str(date.today().strftime('%Y%m%d'))


# # save list of file names to txt
# output_filename = os.path.join(miscellaneous_dir,"summer_1998_file_list.txt")

# # Save the list to a text file
# with open(output_filename, 'w') as file:
#     for file_name in summer_files:
#         file.write(file_name + '\n')

nc_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(path_to_1998_n14_data)
    for filename in filenames if filename.endswith(".nc")
]
#%%
print('********* Building the LUT *********')

# min_lat, max_lat = 5, 15
# Initialize an empty list for storing lookup table entries
group_arrays = []
group_arrays_cor = []

lookup_table = []
limb_bin_list =  {}
limb_hist_list = {}
adjusted_limb_bins_list = {}
all_limb_at_i_list = {}

nadir_bin_list = {} 
nadir_hist_list = {}

# '/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14/1998/039/clavrx_NSS.GHRR.NJ.D98039.S1804.E1906.B1603132.WI.hirs_avhrr_fusion.level2.nc'
# read abd extarct relevant files
# start time of code
start_time = time.time()
for file in sorted(summer_files):
    file_nme = os.path.basename(file)
    data = xr.open_dataset(file)
    
    # Extract relevant parameters for the correction
    lats = data['latitude'][:, :].data  
    latitude =   data['latitude'][:,:]
    lon = data['longitude'][:,:]  # Define lon_el

    # Create stratification bins while preserving NaN values
    latitude_bins = xr.apply_ufunc(
        lambda x: (x // latitude_bin_size) * latitude_bin_size,
        latitude,
        dask="allowed"
    )
    latitude_bins = latitude_bins.where(~np.isnan(latitude_bins),drop=True)
    lat_intervals = np.unique(latitude_bins.data.flatten())
    brightness_temp_11um = data['temp_11_0um_nom'].data
    brightness_temp_11um_cor = brightness_temp_11um.copy()
    cloud_probability = data['cloud_probability'].data 
    surfact_type = data['land_class'].data   
    # surfact_type = np.where((surfact_type == -128),np.nan,surfact_type)
    
    # for l in lat_intervals:
    lat_val= -60 #l #latitude_bin_centers[20]
    # lat_wind = [lat_val - 5, lat_val + 5]
    # max_lat, min_lat = max(np.array(lat_wind)), min(np.array(lat_wind))#lat_val - -15, lat_val + -15  
    max_lat, min_lat = 61, 53 #-53,-61 

    # Create a mask based on the conditions
    # lat_msk  = create_a_lat_mask(lats, -35, 15)
    # lat_msk_ = (lats >= min_lat) & (lats <= max_lat)
    # cld_prob_msk = (cloud_probability >= 0.5)
    # surfact_type_msk = (surfact_type == 0)

    # msk_ = np.logical_and(lat_msk, cld_prob_msk)
    # msk_ = np.logical_and(msk_, surfact_type_msk)

    # plt.figure(dpi=250)
    # plt.imshow((msk_).T, cmap="binary")
    # plt.title("msk_")
    # plt.colorbar(orientation='horizontal')
    # plt.show()  

    # msk_ = np.logical_and((lat_msk, cld_prob_msk, surfact_type_msk))
    
    mask = ((lats >= min_lat) & (lats <= max_lat)) & \
           (cloud_probability >= 0.5) & \
           (surfact_type == 0)

    # mask = ((lats >= min_lat) & (lats <= max_lat)) & \
    #        (cloud_probability >= 0.5)


    mask = np.where((mask == True),1,np.nan)       

    # Assign values to the filtered array where the mask is True
    group_data = np.where(mask==1,brightness_temp_11um, np.nan)

    if not np.all(np.isnan(group_data)):
        group_arrays.append(group_data)
# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting group data: {elapsed_seconds:.2f} seconds ({elapsed_hours:.5f} hours)")

# Get the indices of pixels meeting the conditions
# mask_indices = np.where(mask == 1)  

min_r = min(np.nanmin(i) for i in group_arrays)#int(np.floor([min(np.nanmin(i) for i in group_arrays)][0]))
max_r = max(np.nanmax(i) for i in group_arrays)#int(np.ceil([max(np.nanmax(i) for i in group_arrays)][0]))
# start time of code
start_time = time.time()
all_nadirs = get_elements_nadir(group_arrays, reference_beam_positions)#np.hstack(nadir_elem)
all_nadirs = np.round(all_nadirs,3)
nadir_temp_min = np.nanmin(all_nadirs)
nadir_temp_max = np.nanmax(all_nadirs)
nadir_temp_range = (nadir_temp_min, nadir_temp_max)  
temp_range = (min_r, max_r)  

# Create histograms with independent ranges
nadir_hist, nadir_bins = create_histogram(all_nadirs, bin_size, temp_range)
valid_nadir_indices = nadir_hist > 0
nadir_bins = nadir_bins[valid_nadir_indices]
nadir_hist = nadir_hist[valid_nadir_indices]
# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting nadir stats: {elapsed_seconds:.2f} seconds ({elapsed_hours:.5f} hours)")

# Initialize an array to store corrected limb data
# corrected_limb_data = group_data.copy()
# start time of code

# lookup_table, res = process_limb_positions_with_running_window(
#                     group_arrays, limb_beam_positions, 
#                     nadir_bins, nadir_hist, temp_range,
#                     bin_size, lat_val, min_lat, max_lat, 
#                     window_size=5)

# for i in range(0, len(limb_beam_positions), step_size):
#     # Define the range for the current window
#     start = max(0, limb_beam_positions[i] - window_size)
#     end = min(409, limb_beam_positions[i] + window_size + 1)

#     print(start, end)

start_time = time.time()
for i in limb_beam_positions:
    # limb_pos_rng = range(i-10, i+10)
    # Adjust the range dynamically based on beam position edges
    start = max(0, i - 10)  # Ensure the range doesn't go below 0
    end = min(409, i + 10)  # Ensure the range doesn't exceed 409
    limb_pos_rng = range(start, end)
    all_limbs_at_i = get_elements_limb(group_arrays, 
                                        limb_pos_rng)
    all_limbs_at_i = np.round(all_limbs_at_i,3)
    all_limb_at_i_list[i] = all_limbs_at_i
  
    # Create histograms with independent ranges
    limb_hist, limb_bins = create_histogram(all_limbs_at_i, bin_size, temp_range)
    # Find the indices where counts in `limb_hist` are greater than zero
    valid_limb_indices = limb_hist > 0
    limb_bins = limb_bins[valid_limb_indices]
    limb_hist = limb_hist[valid_limb_indices]

    limb_hist_list[i] = limb_hist
    limb_bin_list[i] = limb_bins 
    
    adjusted_limb_bins, nadir_hist_norm = adjust_limb_bins_bob_method_upgraded(nadir_bins, nadir_hist,
                                                limb_bins, limb_hist) 
    adjusted_limb_bins = np.round(adjusted_limb_bins,3)
    adjusted_limb_bins_list[i] = adjusted_limb_bins
    nadir_hist_list[i] = nadir_hist_norm
    nadir_bin_list[i] = nadir_bins  

    # Populate lookup table
    for orig_tb, corr_tb in zip(limb_bins, adjusted_limb_bins):
        lookup_table.append({
            "latitude": lat_val,
            "latitude_bin": f"{min_lat}-{max_lat}",
            # "cloud_probability": 0.5,
            "surface_type": 2,
            "beam_position": i,
            "original_tb": orig_tb,
            "corrected_tb": corr_tb,
            "corr_coeff": corr_tb/orig_tb
        })                          
# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting limb stats: {elapsed_seconds:.2f} seconds ({elapsed_hours:.5f} hours)")

# Convert lookup table to DataFrame and save it
lookup_df = pd.DataFrame(lookup_table)
len(lookup_df[lookup_df['corr_coeff'] <1])

plt_nme = f'Distribution_of_correctiopn_coefficient_per_beam_pos_water_NH_{cde_run_dte}.png'
plt_nme  = os.path.join(plot_dir,plt_nme)

# lookup_df_w = lookup_df[lookup_df['surface_type'] == 0]
box_plot_of_corr_coeff(lookup_df,plt_nme)


plt_nme = f'Distribution_of_correctiopn_coefficient_per_beam_pos_snc_NH_{cde_run_dte}.png'
plt_nme  = os.path.join(plot_dir,plt_nme)
lookup_df_snc= lookup_df[lookup_df['surface_type'] == 2]
box_plot_of_corr_coeff(lookup_df_snc,plt_nme)


plt_nme = '_'.join(['mean_correction_coeff_at_beam_position-snc',cde_run_dte]) + '.png'
plt_nme  = os.path.join(plot_dir,plt_nme)

groupby_and_plot(lookup_df_snc, 'beam_position', 'corr_coeff', 'snc','NH')

plt.savefig(plt_nme, dpi=300, bbox_inches='tight')

# lookup_df = lookup_df[lookup_df['corrected_tb'] != 0]
lut_name = os.path.join(df_dir,'water-summer_LUT_'+cde_run_dte+'.csv' )
lookup_df.to_csv(lut_name, index=False)
print("***************** Lookup table saved. *******************")

lookup_df = pd.read_csv(os.path.join(df_dir,'water-summer_LUT_20250124.csv'))

mean_coeff = lookup_df.groupby('beam_position', as_index=True)['corr_coeff'].mean()
mean_coeff.plot(x='beam_position',y='corr_coeff')
# lookup_df.plot(x='beam_position',y='original_tb')


limb_elem = []
positions = [0,5,10,15,50,100,150,200, 300, 350, 384, 404, 408]
lposs = [p for p in positions if p in beam_positions]
for c in lposs:
    c_nt = []
    for n in group_arrays:
        limb_irtbs = n[:,c].flatten()
        limb_irtbs = limb_irtbs[~np.isnan(limb_irtbs)]
        cnt_num = len(limb_irtbs)
        # print((c,cnt_num))
        c_nt.append(int(cnt_num))
    limb_elem.append((c, np.nansum(c_nt)))



#%%
def return_int(dat):
    return np.array([i.astype(int) for i in dat])

import random
# Generate a sequence of numbers from 0 to total_count - 1
numbers = list(range(len(all_nadirs)))

# Shuffle the sequence
inx = random.shuffle(numbers)
# 
limb_ran_idx = numbers[:len(all_limb_at_i_list[i])]

nadir_subset = all_nadirs[limb_ran_idx]

nadir_hist_sub, nadir_bins_sub = create_histogram(nadir_subset, 
                                        bin_size, 100,
                                        temp_range)

limb_hst = limb_hist_list[i].copy()

# hist_coef_ = np.sum(nadir_hist_list[i]) / np.sum(limb_hst)
nadir_hist_norm_ = nadir_hist_list[i]#nadir_hist / hist_coef_

valid_limb_ind = limb_hist_list[i] > 0
limb_bns = limb_bin_list[i].copy()[valid_limb_ind]
limb_hst = limb_hst[valid_limb_ind]

adjusted_limb_bns = adjusted_limb_bins_list[i].copy()[valid_limb_ind]


nadir_hst_norm_ = nadir_hist_norm_.copy()[valid_limb_ind]
nadir_bns = nadir_bin_list[i].copy()[valid_limb_ind]

nadir_bns_sub = nadir_bins_sub.copy()[valid_limb_ind]
nadir_hst_sub = nadir_hist_sub.copy()[valid_limb_ind]

plt.plot(nadir_bns, return_int(nadir_hst_norm_), ls='--', c='k', label = 'all_nadir')
plt.plot(nadir_bns_sub, return_int(nadir_hst_sub), ls='-.', c='r', label = 'random samples from nadir')
plt.plot(limb_bns, return_int(limb_hst), 'gray', label =f'limb at {i}')
plt.plot(adjusted_limb_bns, return_int(limb_hst), 'g', ls='-.', label =f'limb_adj at {i}')
plt.xlim(210,300)
plt.yscale('log')
plt.xlabel('IR TB [K]')
plt.ylabel('Count')
plt.legend(frameon=False)
plt.title(f'lat at {0} +/- 5 deg S and beam position at {i}')

plt.plot(limb_bns, (limb_hst - nadir_hst_norm_), c = 'grey', ls = '-', label ='limb-nadir count')
# plt.plot(adjusted_limb_bns, (limb_hst - nadir_hst_norm_), c = 'g', ls = '-.', label ='limb_adjut - nadir count')
plt.legend(frameon=False)
plt.xlabel('IR TB [K]')
plt.ylabel('Count differences')
# plt.yscale('log')
plt.xlim(210,300)
plt.title(f'lat at {0} +/- 5 deg S and beam position at {i}')

#%%
print('********* The line plot *********')

# some plots

for b in [0,10,50,100, 400]:
    beam_pos = b
    plt_nme = '_'.join(['LUT_mehtod_line_plot_histogram-water',str(beam_pos),cde_run_dte]) + '.png'
    plt_nme  = os.path.join(plot_dir,plt_nme)
    ttle = f"Summer 1998 IR TB Distribution from {min_lat} - {max_lat} at {beam_pos} beam position"
    plt.figure()
    plt.plot(limb_bin_list[beam_pos],limb_hist_list[beam_pos], 
            ls = '-', c='k', label='limb')
    plt.plot(adjusted_limb_bins_list[beam_pos],
            limb_hist_list[beam_pos], ls = ':', 
            c='k', label='limb_adj')
    plt.plot(nadir_bin_list[beam_pos],nadir_hist_list[beam_pos], 
            ls = ':', c='r', label='nadir')
    plt.xlim(210,300)
    # Add text annotations for sums
    nadir_sum = np.nansum(nadir_hist_list[beam_pos])
    limb_sum = np.nansum(limb_hist_list[beam_pos])

    plt.text(0.05, 0.5, f"Nadir Sum: {nadir_sum:.2f}", 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', color='r')
    plt.text(0.05, 0.45, f"Limb Sum: {limb_sum:.2f}", 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', color='k')
    # plt.yscale('log')
    plt.legend(frameon=False, fontsize=15)
    plt.xlabel('IR TB [K]')
    plt.ylabel('Count')
    plt.grid(which='both', ls = '--', lw='0.5')
    plt.title(ttle, fontdict={'size':12, 'color':'k'})
    plt.savefig(plt_nme, bbox_inches='tight')
# limb_hist_list[beam_pos]
#%%
print('********* The correction *********')

# the correction
count = 0
corrected_arrays = []
observed_arrays = []
target_lat = -60  # Example target latitude
tolerance = 1.0  # Allow +/-1 degree deviation
for season_file in summer_files:

    output_filenme = os.path.basename(season_file).replace('.nc','_cor.nc')
    output_filenme = os.path.join(cor_dir,output_filenme)

    cor_tb = apply_lut_corrections_fast(season_file, limb_beam_positions,
                                   target_lat, tolerance, lookup_df)
    obs_tb = xr.open_dataset(season_file)['temp_11_0um_nom']
    lats_ = obs_tb['latitude'].data

    lat_mask = create_a_lat_mask(lats_, target_lat, tolerance)
    lat_mask = np.where(lat_mask == True,1,np.nan)       

    # Assign values to the filtered array where the mask is True
    cor_tb_msk = np.where(lat_mask == True,cor_tb, np.nan)
    obs_tb_msk = np.where(lat_mask == True,obs_tb, np.nan)

    corrected_arrays.append(cor_tb_msk)
    observed_arrays.append(obs_tb_msk)

    count += 1

    if count % 50 == 0:
        print(os.path.basename(season_file))


# Example usage
start_time = time.time()
corrected_arrays, observed_arrays = process_files_in_parallel(
    summer_files, target_lat=-60, tolerance=1.0, lookup_df=lookup_df,
    limb_beam_positions=limb_beam_positions, cor_dir=cor_dir
)
# End time of code
end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for applying correction: {elapsed_seconds:.2f} seconds ({elapsed_hours:.5f} hours)")

#%%
#%%
print('********* doing plot_ir_tb_distribution_with_means dist plots now *********')

savefig = '_'.join(['water-percent_dist_by_LUT_method3__',cde_run_dte]) + '.png'
savefig  = os.path.join(plot_dir,savefig)
orig_hist_dist, org_bns, org_means = generate_distribution_with_means(observed_arrays, num_bins=40)
cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(corrected_arrays, num_bins=40)

plot_ir_tb_distribution_with_means_normal_version(
    orig_hist_dist, org_bns, org_means,
    cor_hist_dist, cor_bns, cor_means,
    beam_positions, savefig)

plot_ir_tb_distribution_with_means_log_version(
    orig_hist_dist, org_bns, org_means,
    cor_hist_dist, cor_bns, cor_means,
    beam_positions, savefig)


plot_ir_tb_distribution_with_means_discrete_log(
    orig_hist_dist, org_bns, org_means,
    cor_hist_dist, cor_bns, cor_means,
    beam_positions, savefig)

plot_discrete_ir_tb_distribution(orig_hist_dist, org_bns, org_means,
                                   cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)
#%%
print('********* doing generate_dist dist plots now *********')

orig_hist_dist, orig_beam_ir_means, orig_temp_range = generate_dist(observed_arrays)
cor_hist_dist, cor_beam_ir_means, cor_temp_range = generate_dist(corrected_arrays)
# ,org_bns
savefig = '_'.join(['water-percent_dist_by_LUT_method1',cde_run_dte]) + '.png'
savefig  = os.path.join(plot_dir,savefig)
vmn = 0
vmx = 0.08
cmap, norm = create_custom_cmap(vmn, vmx, 15)
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 10), 
    constrained_layout=True, dpi=100, sharex=True, sharey=True
)

# Plot original histogram
im1 = axes[0].imshow(
    orig_hist_dist.T,
    extent=(beam_positions[0], beam_positions[-1], 
            orig_temp_range[0], orig_temp_range[1]),    
    cmap=cmap,
    norm=norm,
    aspect='auto',
    origin='lower'
)
axes[0].plot(
    beam_positions, orig_beam_ir_means, 
    ls='--', lw=2.5, c='k', label='Mean IR TB'
) # 
axes[0].set_title("Summer 1998: Original IR Tbs Distribution - 60S", fontsize=18)
axes[0].axhline(y=260, color='green', ls=':', lw=3, label='260 K') # 
axes[0].legend(loc='lower center', frameon=False, fontsize=12, ncol=2)
axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
axes[0].tick_params(which='both', labelsize=12)
axes[0].set_ylim(200,300)
# Plot corrected histogram
im2 = axes[1].imshow(
    cor_hist_dist.T,
    extent=(beam_positions[0], beam_positions[-1], 
            cor_temp_range[0], cor_temp_range[1]),   
    cmap=cmap,
    norm=norm,
    aspect='auto',
    origin='lower'
)
axes[1].plot(
    beam_positions, cor_beam_ir_means, 
    ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)'
)
axes[1].set_title("Summer 1998: Corrected IR Tbs Distribution - 60S", fontsize=18)
axes[1].axhline(y=260, color='green', ls=':', lw=3, label='260 K')
# axes[1].legend(loc='best', frameon=False, fontsize=12, ncol=2)
axes[1].set_xlabel("Beam Positions", fontsize=14)
axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
axes[1].tick_params(which='both', labelsize=15)
axes[1].set_ylim(200,300)

# Add a colorbar
bdrs = np.linspace(vmn, vmx, 15)
# Add a shared colorbar for both plots
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
                    fraction=0.08, pad=0.05, boundaries=bdrs)
cbar.ax.set_xticklabels(
    [f"{int(b):,}" if b >= 1 else f"{b:.3f}" for b in bdrs], fontsize=12
)
cbar.set_label("Normalized Histogram", fontsize=15)
plt.savefig(savefig, bbox_inches='tight')

#%%
print('********* doing generate_dist_ dist plots now *********')

orig_hist_dist, orig_beam_ir_means, orig_temp_range,org_bns = generate_dist_(observed_arrays)
cor_hist_dist, cor_beam_ir_means, cor_temp_range, cor_bns = generate_dist_(corrected_arrays)
# 
savefig = '_'.join(['water-percent_dist_by_LUT_method2',cde_run_dte]) + '.png'
savefig  = os.path.join(plot_dir,savefig)

vmn, vmx = 0, max(np.max(orig_hist_dist), np.max(cor_hist_dist)) * 0.30 
cmap, norm = create_custom_cmap(vmn, vmx, 15)
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 10), 
    constrained_layout=True, dpi=100, sharex=True, sharey=True
)

# Plot original histogram
im1 = axes[0].imshow(
    orig_hist_dist.T,
    extent=(beam_positions[0], beam_positions[-1], 
            orig_temp_range[0], orig_temp_range[1]),    
    cmap=cmap,
    norm=norm,
    aspect='auto',
    origin='lower'
)
axes[0].plot(
    beam_positions, orig_beam_ir_means, 
    ls='--', lw=2.5, c='k', label='Mean IR TB'
) # 
axes[0].set_title("Summer 1998: Original IR Tbs Distribution - 60S", fontsize=18)
axes[0].axhline(y=260, color='green', ls=':', lw=3, label='260 K') # 
axes[0].legend(loc='lower center', frameon=False, fontsize=12, ncol=2)
axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
axes[0].tick_params(which='both', labelsize=12)
axes[0].set_ylim(200,300)
# Plot corrected histogram
im2 = axes[1].imshow(
    cor_hist_dist.T,
    extent=(beam_positions[0], beam_positions[-1], 
            cor_temp_range[0], cor_temp_range[1]),   
    cmap=cmap,
    norm=norm,
    aspect='auto',
    origin='lower'
)
axes[1].plot(
    beam_positions, cor_beam_ir_means, 
    ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)'
)
axes[1].set_title("Summer 1998: Corrected IR Tbs Distribution - 60S", fontsize=18)
axes[1].axhline(y=260, color='green', ls=':', lw=3, label='260 K')
# axes[1].legend(loc='best', frameon=False, fontsize=12, ncol=2)
axes[1].set_xlabel("Beam Positions", fontsize=14)
axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
axes[1].tick_params(which='both', labelsize=15)
axes[1].set_ylim(200,300)

# Add a colorbar
bdrs = np.linspace(vmn, vmx, 15)
# Add a shared colorbar for both plots
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
                    fraction=0.08, pad=0.05, boundaries=bdrs)
cbar.ax.set_xticklabels(
    [f"{int(b):,}" if b >= 1 else f"{b:.3f}" for b in bdrs], fontsize=12
)
cbar.set_label("Normalized Histogram", fontsize=15)
plt.savefig(savefig, bbox_inches='tight')


#%%
start_time = time.time()

# plot distribution using already existing data
all_data_for_dist = [os.path.join(cor_dir, f) for f in os.listdir(cor_dir) if f.endswith('.nc')]

cor_tb_data_array_dict = {'NH': {}, 'SH': {}}
obs_tb_data_array_dict= {'NH': {}, 'SH': {}}

for c in all_data_for_dist:
    data = xr.open_dataset(c)
    obs_ir_tb = data['temp_11_0um_nom'].data
    cor_ir_tb = data['temp_11_0um_nom_corrected'].data
    lats = data['latitude'].data
    srfype = data['land_class'].data

    for lat_window in [(53,61),(-53,-61)]:
        max_lat, min_lat = max(lat_window), min(lat_window)
        lat_mask = ((lats >= min_lat) & (lats <= max_lat))
        lat_mask = np.where(lat_mask == True, 1, np.nan)

        hemisphere = 'NH' if max_lat > 0 else 'SH'

        for surface_type_id, surface_type_name in surface_type_mapping.items():
            # Create surface type mask
            surfact_type_mask = np.where(srfype == surface_type_id, 1, np.nan)

            # Combine latitude and surface type masks
            combined_mask = lat_mask * surfact_type_mask

            # Apply the combined mask to the corrected and observed arrays
            cor_tb_msk = np.where(combined_mask == 1, cor_ir_tb, np.nan)
            obs_tb_msk = np.where(combined_mask == 1, obs_ir_tb, np.nan)

            # Store the masked arrays in the dictionaries
            if surface_type_id not in cor_tb_data_array_dict[hemisphere]:
                cor_tb_data_array_dict[hemisphere][surface_type_id] = []
                obs_tb_data_array_dict[hemisphere][surface_type_id] = []

            cor_tb_data_array_dict[hemisphere][surface_type_id].append(cor_tb_msk)
            obs_tb_data_array_dict[hemisphere][surface_type_id].append(obs_tb_msk)

end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting limb stats: {elapsed_seconds:.2f} seconds "
    f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")    
#%%
start_time = time.time()

# the plot
for hemisphere in ['NH', 'SH']:
    # pull hemisphere data
    obs_data_hem = obs_tb_data_array_dict[hemisphere]
    cor_data_hem = cor_tb_data_array_dict[hemisphere]

    # pull surface type data
    for i in surface_type_mapping.keys():
        sftype = surface_type_mapping[i]
        obs_by_srftyp_array = obs_data_hem[i]
        cor_by_srftyp_array = cor_data_hem[i]

        # Generate distribution with means
        orig_hist_dist, org_bns, org_means = generate_distribution_with_means(obs_by_srftyp_array)
        cor_hist_dist, cor_bns, cor_means = generate_distribution_with_means(cor_by_srftyp_array)

        #--------------------------------------------------------
        savefig = f"{sftype}_{hemisphere}-percent_dist_by_LUT_contourf_version_{cde_run_dte}.png"        
        savefig  = os.path.join(plot_dir, savefig)
        plot_discrete_ir_tb_distribution(
                                    orig_hist_dist, org_bns, org_means,
                                    cor_hist_dist, cor_bns, cor_means,
                                    beam_positions, savefig)

end_time = time.time()
# Compute elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_seconds / 3600
# Print results
print(f"Elapsed time for getting limb stats: {elapsed_seconds:.2f} seconds "
    f"({elapsed_minutes:.2f} minutes) ({elapsed_hours:.5f} hours)")    
#%%
# some investigation and analysis of the latitudinal path of the beam position
data = xr.open_dataset(file)    
# Extract relevant parameters for the correction
lats = data['latitude'][:, :].data  
latitude =   data['latitude'][:,:]
lon = data['longitude'][:,:]
nadir_lat = latitude[:,204].data
nadir_lon = lon[:,204].data
nadir_lon = nadir_lon[~np.isnan(nadir_lon)]
# latitude[:,204].plot()
nadir_lat = nadir_lat[~np.isnan(nadir_lat)]

limb_lat_left = latitude[:,0].data
limb_lon_left = lon[:,0].data
limb_lon_left = limb_lon_left[~np.isnan(limb_lon_left)]

limb_lat_left = limb_lat_left[~np.isnan(limb_lat_left)]

limb_lat_right = latitude[:,408].data
limb_lon_right = lon[:,408].data
limb_lon_right = limb_lon_right[~np.isnan(limb_lon_right)]


limb_lat_right = limb_lat_right[~np.isnan(limb_lat_right)]


# Define the projection
projection = ccrs.PlateCarree()

# Create a figure and axis with the specified projection
fig, ax = plt.subplots(figsize=(10, 15), 
                       subplot_kw={'projection': projection},
                       dpi=500)

# Add land and ocean features
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
ax.add_feature(cfeature.OCEAN, zorder=0, edgecolor='black')

# Plot the nadir and limb latitude lines
ax.plot(nadir_lon, nadir_lat, transform=ccrs.PlateCarree(), 
        label='204 beam pos (Nadir) path', color='blue')
ax.plot(limb_lon_left, limb_lat_left, transform=ccrs.PlateCarree(), 
        label='0 beam pos (Left Limb) path', color='red')
ax.plot(limb_lon_right, limb_lat_right, transform=ccrs.PlateCarree(), 
        label='408 beam pos (Right Limb) path', color='green')

# Add gridlines

gl = ax.gridlines(draw_labels=True, linestyle='--', 
                  linewidth=0.8, alpha=0.5, color='gray')
gl.xlabel_style = {'size': 18}
gl.ylabel_style = {'size': 18}
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
# gl.yformatter = mticker.FixedFormatter(['-90', '-60', '-30', '0', '30', '60', '90'])

# Add a legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
          ncol=3, frameon = False, fontsize=13)

# Set the title
ax.set_title('Nadir and Limb Latitudinal path', fontsize=15)
plt.savefig(os.path.join(plot_dir, 
                         'nadir_and_limb_latitudinal_path.png'), 
                         bbox_inches='tight')
# Show the plot
plt.show()

#%%
file4movi = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14/1998/001'

# Get all .nc files in the directory
nc_files_ = sorted(glob.glob(os.path.join(file4movi, "*.nc")))

# Initialize lists to store the paths of the generated images
image_paths = []

# Loop through each file and generate the plot
for file in nc_files_:
    data = xr.open_dataset(file)
    lats = data['latitude'][:, :].data
    lon = data['longitude'][:, :].data

    nadir_lat = lats[:, 204]
    nadir_lon = lon[:, 204]
    limb_lat_left = lats[:, 0]
    limb_lon_left = lon[:, 0]
    limb_lat_right = lats[:, 408]
    limb_lon_right = lon[:, 408]

    # Extract date and time from the file name
    file_name = os.path.basename(file)
    date_str = int(file_name.split('.')[3][1:])
    # Convert year day number to date
    year = date_str // 1000
    day_of_year = date_str % 1000
    date_ = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    date_str = date_.strftime('%Y-%m-%d')
    # print(date_str)  # Output: 1998-01-01
    start_time_str = file_name.split('.')[4][1:]
    end_time_str = file_name.split('.')[5][1:]

    # Define the projection
    projection = ccrs.PlateCarree()

    # Create a figure and axis with the specified projection
    fig, ax = plt.subplots(figsize=(10, 15), subplot_kw={'projection': projection}, dpi=500)

    # Add land and ocean features
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, zorder=0, edgecolor='black')

    # Plot the nadir and limb latitude lines
    ax.plot(nadir_lon, nadir_lat, transform=ccrs.PlateCarree(), label='204 beam pos (Nadir) path', color='blue')
    ax.plot(limb_lon_left, limb_lat_left, transform=ccrs.PlateCarree(), label='0 beam pos (Left Limb) path', color='red')
    ax.plot(limb_lon_right, limb_lat_right, transform=ccrs.PlateCarree(), label='408 beam pos (Right Limb) path', color='green')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.8, alpha=0.5, color='gray')
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))

    # Add a legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=13)

    # Set the title
    ax.set_title(f'Nadir and Limb Latitudinal path - Date: {date_str} Time: {start_time_str}-{end_time_str}', fontsize=15)

    # Save the plot as an image
    image_path = os.path.join(plot_dir, f'nadir_and_limb_latitudinal_path_{date_str}_{start_time_str}_{end_time_str}.png')
    plt.savefig(image_path, bbox_inches='tight')
    image_paths.append(image_path)
    plt.close(fig)

# Create a movie from the images
# Define the codec and create a VideoWriter object
movie_path = os.path.join(plot_dir, 'nadir_and_limb_latitudinal_path_movie.mp4')
frame = cv2.imread(image_paths[0])
height, width, layers = frame.shape
video = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

for image_path in image_paths:
    video.write(cv2.imread(image_path))

video.release()
print(f"Movie saved at {movie_path}")


# plt.figure(dpi=250)
# plt.plot(nadir_lat, (limb_lat_left - limb_lat_right))
# plt.xlabel('Nadir latitude')
# plt.ylabel('Left and right limb latitude differences')
# plt.show()    

# plt.figure(dpi=250)
# plt.plot(nadir_lat, limb_lat_right)
# plt.xlabel('Nadir latitude')
# plt.ylabel('Left and right limb latitude differences')
# plt.show()   
# 
plt.figure(dpi=250)
plt.plot(nadir_lat, (nadir_lat - limb_lat_right))
plt.xlabel('Nadir latitude')
plt.ylabel('nadir and right limb latitude differences')
plt.show()     

plt.figure(dpi=250)
plt.plot(nadir_lat, (nadir_lat - limb_lat_left))
plt.xlabel('Nadir latitude')
plt.ylabel('nadir and left limb latitude differences')
plt.show() 
#%%
# filt=lookup_df.copy()

# # Find the closest matching latitude bin
# lat_bin = filt['latitude'].iloc[(filt['latitude'] - -60.00219).abs().idxmin()]
# filt =filt[(filt['latitude'] == int(lat_bin))].reset_index()

# # Find the closest matching beam position
# beam_key = filt['beam_position'].iloc[(filt['beam_position'] - 381).abs().idxmin()]
# filt =filt[(filt['beam_position'] == int(beam_key))].reset_index()

# # Find the closest matching observed temperature
# filt_reset = filt.reset_index()
# temp_bin = filt['original_tb'].iloc[(filt['original_tb'] - 267.59515).abs().idxmin()]
# filt =filt[(filt['original_tb'] == temp_bin)] 

# filt.shape


#%%
# # Example usage

from netCDF4 import Dataset
import seaborn as sns
import glob
import imageio
from datetime import datetime, timedelta


with Dataset(file) as nc:
    print(nc.variables.keys())
    # Extract the latitude values from the dataset
    avh_lat = nc['latitude'][:]    
    avh_lat = np.where((avh_lat.data == -999.0), np.nan, avh_lat.data)
    print(avh_lat[~np.isnan(avh_lat)])
    # Defining the mask to determine between northern and southern hemisphere latitude threshold.
    # We (mask) throw away this part of the datasets.
    avh_lat = np.where((avh_lat >= -5) & (avh_lat <= 5), avh_lat, np.nan)

    brightness_temp_11um = nc['temp_11_0um_nom'][:].data
    brightness_temp_11um = np.where(np.isnan(avh_lat), np.nan, 
                                    brightness_temp_11um)#.ravel()
    # brightness_temp_11um = np.tile(brightness_temp_11um.reshape(-1, 1), avh_lat.shape[1])


    cloud_probability = nc['cloud_probability'][:].data 
    cloud_probability = np.where(np.isnan(avh_lat), np.nan, 
                                    cloud_probability)#.ravel()
    # cloud_probability = np.tile(cloud_probability.reshape(-1, 1), avh_lat.shape[1])

    surfact_type = nc['land_class'][:].data 
    surfact_type = np.where(np.isnan(avh_lat), np.nan, 
                                    surfact_type)#.ravel()
    # surfact_type = np.tile(surfact_type.reshape(-1, 1), avh_lat.shape[1])
    
    # for l in lat_intervals:
    lat_val= 0 #l #latitude_bin_centers[20]
    lat_wind = [lat_val - 5, lat_val + 5]
    max_lat, min_lat = max(np.array(lat_wind)), min(np.array(lat_wind))#lat_val - -15, lat_val + -15  
    
    
    # mask = ((avh_lat >= min_lat) & (avh_lat <= max_lat)) & \
    #        (cloud_probability >= 0.5) & \
    #        (surfact_type == 0)

    mask = ((avh_lat >= min_lat) & (avh_lat <= max_lat)) & \
           (cloud_probability >= 0.5)

    mask = np.where((mask == True),1,np.nan)   


plt.figure(dpi=250)
plt.imshow((avh_lat).T, cmap="jet")
plt.title("avh_lat")
plt.colorbar(orientation='horizontal')
plt.show() 

plt.figure(dpi=250)
plt.imshow(((avh_lat >= min_lat) & (avh_lat <= max_lat)).T, cmap="binary")
plt.title("lat")
plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(dpi=250)
plt.imshow((cloud_probability).T, cmap="binary")
plt.title("cloud_probability")
plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(dpi=250)
plt.imshow((brightness_temp_11um).T, cmap="binary")
plt.title("brightness_temp_11um")
plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(dpi=250)
plt.imshow((surfact_type).T, cmap="binary")
plt.title("surfact_type")
plt.colorbar(orientation='horizontal')
plt.show()


#%%
def create_xarray(dat, lon, lat, attrs=None):
    """
    Create an xarray DataArray from the list of 2D precipitation arrays and add attributes.

    Parameters:
    - all_precip: List or array of 2D precipitation arrays
    - all_time_index: List of timestamps
    - lon: Array of longitudes
    - lat: Array of latitudes
    - attrs: Dictionary of attributes to add to the DataArray (optional)

    Returns:
    - precip_data: xarray DataArray with the specified attributes
    """
    # Create a pandas DatetimeIndex from the list of timestamps
    
    # Create an xarray DataArray from the list of 2D precipitation arrays
    precip_data = xr.DataArray(
        data=dat,
        dims=["lat", "lon"],
        coords={            
            "lat": lat,
            "lon": lon
        }
    )

    # Add attributes if provided
    if attrs:
        precip_data.attrs.update(attrs)
    
    return precip_data

# # Example usage
file_2plt= "/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/corrected/clavrx_NSS.GHRR.NJ.D98015.S1728.E1912.B1569394.WI.hirs_avhrr_fusion.level2_cor.nc"

datset = xr.open_dataset(file_2plt)
uncor_tb = datset['temp_11_0um_nom']
cor_tb = datset['temp_11_0um_nom_corrected']
diff_tb = datset['temp_11_0um_nom_cor_obs_diff']

lat_el = datset['latitude'].data
lat_el = lat_el[~np.isnan(lat_el)]
lon_el = datset['longitude'].data
lon_el = lon_el[~np.isnan(lon_el)]

# Plot for the latitude window -53, -61
lat_window_min, lat_window_max = -61, -53

# Create a mask for the latitude window
lat_mask = (datset['latitude'] >= lat_window_min) & (datset['latitude'] <= lat_window_max)

# Apply the mask to the data
uncor_tb_window = uncor_tb.where(lat_mask, drop=True)
cor_tb_window = cor_tb.where(lat_mask, drop=True)
diff_tb_window = diff_tb.where(lat_mask, drop=True)

# Plot the uncorrected temperature
# Define discrete bins and use logarithmic normalization
levels = np.linspace(200, 300, 15)  # Discrete levels for colorbar
cmap = plt.cm.get_cmap('jet', len(levels) - 1)  # Discrete colormap
norm = mcolors.BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)

plt.figure(figsize=(10, 10))
im = uncor_tb_window.plot(cmap=cmap, norm=norm, add_colorbar=False)
plt.title('Uncorrected Temperature (11um) for Latitude Window -53 to -61')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
plt.grid(which='major', linestyle='--', linewidth=0.5,axis='both')
# plt.xticks(ticks=np.arange(-180, 181, 30), labels=np.arange(-180, 181, 30), rotation=45)
# plt.yticks(ticks=np.arange(-61, -52, 1), labels=np.arange(-61, -52, 1))

# Add a colorbar with discrete levels
cbar = plt.colorbar(mappable=im,ticks=levels, orientation='horizontal', pad=0.1, fraction=0.05)
cbar.set_label('Temperature (K)')
cbar.ax.set_xticklabels([f"{int(v)}" for v in levels])

plt.show()

def plot_temperature_distribution(data_array, lat_window_min, lat_window_max, lon_el, lat_el, title, cmap, levels, norm, colorbar_label):
    """
    Plot temperature distribution for a given latitude window.

    Parameters:
    - data_array: xarray DataArray containing temperature data
    - lat_window_min: Minimum latitude for the window
    - lat_window_max: Maximum latitude for the window
    - lon_el: Array of longitudes
    - lat_el: Array of latitudes
    - title: Title for the plot
    - cmap: Colormap for the plot
    - levels: Discrete levels for the colorbar
    - norm: Normalization for the colormap
    - colorbar_label: Label for the colorbar
    """
    # Create a mask for the latitude window
    lat_mask = (data_array['latitude'] >= lat_window_min) & (data_array['latitude'] <= lat_window_max)

    # Apply the mask to the data
    data_window = data_array.where(lat_mask, drop=True)

    # Plot the temperature distribution
    plt.figure(figsize=(10, 10))
    im = data_window.plot(cmap=cmap, norm=norm, add_colorbar=False)
    plt.title(title)
    plt.grid(which='major', linestyle='--', linewidth=0.5,axis='both')

    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.xticks(ticks=np.arange(len(lon_el)), labels=np.round(lon_el, 2), rotation=45)
    # plt.yticks(ticks=np.arange(len(lat_el)), labels=np.round(lat_el, 2))

    # Add a colorbar with discrete levels
    cbar = plt.colorbar(mappable=im, ticks=levels, 
                        orientation='horizontal', pad=0.1, 
                        fraction=0.05)
    cbar.set_label(colorbar_label)
    cbar.ax.set_xticklabels([f"{int(v)}" for v in levels])

    plt.show()

# Define discrete bins and use logarithmic normalization
levels = np.linspace(200, 300, 15)  # Discrete levels for colorbar
cmap = plt.cm.get_cmap('jet', len(levels) - 1)  # Discrete colormap
norm = mcolors.BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)

# Plot the uncorrected temperature
# Convert the data arrays to Dask arrays
uncor_tb_dask = uncor_tb.chunk({'scan_lines_along_track_direction': 100, 'pixel_elements_along_scan_direction': 100})
cor_tb_dask = cor_tb.chunk({'scan_lines_along_track_direction': 100, 'pixel_elements_along_scan_direction': 100})
diff_tb_dask = diff_tb.chunk({'scan_lines_along_track_direction': 100, 'pixel_elements_along_scan_direction': 100})

# Use Dask to apply the mask
lat_mask_dask = ((uncor_tb_dask['latitude'] >= lat_window_min) & (uncor_tb_dask['latitude'] <= lat_window_max)).compute()
uncor_tb_window_dask = uncor_tb_dask.where(lat_mask_dask, drop=True)
cor_tb_window_dask = cor_tb_dask.where(lat_mask_dask, drop=True)
diff_tb_window_dask = diff_tb_dask.where(lat_mask_dask, drop=True)

# Compute the results
uncor_tb_window = uncor_tb_window_dask.compute()
cor_tb_window = cor_tb_window_dask.compute()
diff_tb_window = diff_tb_window_dask.compute()

plot_temperature_distribution(
    uncor_tb_window, lat_window_min=-61, lat_window_max=-53, lon_el=lon_el, lat_el=lat_el,
    title='Uncorrected Temperature (11um) for Latitude Window -53 to -61',
    cmap=cmap, levels=levels, norm=norm, colorbar_label='Temperature (K)'
)

plot_temperature_distribution(
    cor_tb_window, lat_window_min=-61, lat_window_max=-53, lon_el=lon_el, lat_el=lat_el,
    title='Corrected Temperature (11um) for Latitude Window -53 to -61',
    cmap=cmap, levels=levels, norm=norm, colorbar_label='Temperature (K)'
)

levels = np.linspace(-7, 7, 15)  # Discrete levels for colorbar
cmap = plt.cm.get_cmap('jet', len(levels) - 1)  # Discrete colormap
norm = mcolors.BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True)

plot_temperature_distribution(
    diff_tb_window, lat_window_min=-61, lat_window_max=-53, lon_el=lon_el, lat_el=lat_el,
    title='Corrected Temperature (11um) for Latitude Window -53 to -61',
    cmap=cmap, levels=levels, norm=norm, colorbar_label='Temperature (K)'
)
# Plot the corrected temperature
plt.figure(figsize=(10, 6))
cor_tb_window.plot(cmap='coolwarm', vmin=160, vmax=340)
plt.title('Corrected Temperature (11um) for Latitude Window -53 to -61')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks(ticks=np.arange(len(lon_el)), labels=np.round(lon_el, 2), rotation=45)
plt.yticks(ticks=np.arange(len(lat_el)), labels=np.round(lat_el, 2))
plt.show()

# Plot the difference between corrected and uncorrected temperature
plt.figure(figsize=(10, 6))
diff_tb_window.plot(cmap='coolwarm', vmin=-7, vmax=7)
plt.title('Difference (Corrected - Uncorrected) Temperature (11um) for Latitude Window -53 to -61')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks(ticks=np.arange(len(lon_el)), labels=np.round(lon_el, 2), rotation=45)
plt.yticks(ticks=np.arange(len(lat_el)), labels=np.round(lat_el, 2))
plt.show()