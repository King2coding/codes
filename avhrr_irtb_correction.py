#%%
# import packages
from util_functions import *

#%%
# paths
cor_dir =r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/corrected'
sample_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/miscellaneous'
day_dir_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/frequency_analysis/data/avhrr/patmosx_l2_jan_1998/noaa-14/1998/001'
path_to_n14 = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/frequency_analysis/data/avhrr/patmosx_l2_jan_1998/noaa-14/1998'

plot_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/plots'
# Example Usage
base_path = "/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14"  # Replace with your actual path

# %%

#%%
# floating variables
sample_file = os.path.join(sample_dir,'clavrx_NSS.GHRR.ND.D98001.S2305.E0059.B3446264.GC.hirs_avhrr_fusion.level2.nc')

day_files = sorted([os.path.join(day_dir_files,x) for x in os.listdir(day_dir_files) if x.endswith('.nc')])

nc_files = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(path_to_n14)
    for filename in filenames if filename.endswith(".nc")
]

hemisphere = "Southern"  # Change to 'Northern' for the Northern Hemisphere
years = sorted([int(i) for i in os.listdir(base_path)])

cmap, norm = create_custom_cmap(0, 0.04, 15)

search_dir = os.path.join(base_path,str(years[0]))
# Organize folders into seasons
seasons = organize_by_season_with_hemisphere(search_dir, hemisphere, years[0])

# Example: Load files for Summer
summer_files = load_files_for_season(search_dir, seasons["Summer"])

lat_intervals = np.arange(-90,90,5)

cde_run_dte = str(date.today().strftime('%Y%m%d'))

#%%
print('***************** begin process *****************')
# min_lat, max_lat = 5, 15
group_arrays = []
group_arrays_cor = []
limb_bin_list, limb_hist_list = [],[]
nadir_bin_list, nadir_hist_list = [],[]
adj_limb_bins_list = []

# '/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/1998-2000/l2_subsets/1e132ab/noaa-14/1998/039/clavrx_NSS.GHRR.NJ.D98039.S1804.E1906.B1603132.WI.hirs_avhrr_fusion.level2.nc'
# read abd extarct relevant files
for file in sorted(summer_files):
    file_nme = os.path.basename(file)
    data = xr.open_dataset(file)
    
    # Extract latitude brightness temperature, and cloud probability
    lats = data['latitude'][:, :].data  
    latitude =   data['latitude']
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
    
    # for l in lat_intervals:
    lat_val= -60 #l #latitude_bin_centers[20]
    lat_wind = [lat_val - 15, lat_val + 15]
    max_lat, min_lat = max(np.array(lat_wind)), min(np.array(lat_wind))#lat_val - -15, lat_val + -15  

    # Check if the latitude range overlaps with the valid values in lats
    if not ((min_lat >= np.nanmin(lats)) and (max_lat <= np.nanmax(lats))):
        print(f"Skipping latitude range: min_lat={min_lat}, max_lat={max_lat} (out of bounds)")
        continue  # Skip this iteration 

    # Create a mask based on the conditions
    mask = ((lats >= min_lat) & (lats <= max_lat)) & (cloud_probability >= 0.5)

    mask = np.where(mask,1,np.nan)    

    if np.any(mask):  # If the mask has valid data
        if lat_val % 100 == 0:  # Print selectively
            print(f"Processing valid latitude range: min_lat={min_lat}, max_lat={max_lat}")
        # Additional data processing here
        # For example:
        # result = process_data(mask)
    else:  # If no valid data in the mask
        print(f"No valid data in latitude range: min_lat={min_lat}, max_lat={max_lat}")
        continue   

    # Assign values to the filtered array where the mask is True
    group_data = np.where(mask==1,brightness_temp_11um, np.nan)

    # Get the indices of pixels meeting the conditions
    mask_indices = np.where(mask == 1)  

    # plt.figure(dpi=1000)
    # plt.imshow(mask.T, cmap="gray")
    # plt.title("Mask")
    # plt.colorbar(orientation='horizontal')
    # plt.show()       

    # Subset nadir and limb data
    nadir_data = group_data[:, reference_beam_positions]  # Nadir beams (middle 100)
    nadir_data_flat = nadir_data.flatten()
    nadir_data_flat = nadir_data_flat[~np.isnan(nadir_data_flat)]
    nadir_temp_min = np.nanmin(nadir_data)
    nadir_temp_max = np.nanmax(nadir_data)
    nadir_temp_range = (nadir_temp_min, nadir_temp_max)        
    # Create histograms with independent ranges
    nadir_hist, nadir_bins = create_histogram(nadir_data_flat, bin_size, nadir_temp_range)

    # Get the indices for nadir beam positions (non-NaN values in group_data)
    # Ensure reference_beam_positions is a list or array of global column indices
    # reference_beam_positions = np.array(reference_beam_positions)
    # Create a mask for non-NaN values specifically for reference beam positions
    # reference_mask = ~np.isnan(group_data[:, reference_beam_positions])
    # Get the row indices and the local column indices (relative to reference_beam_positions)
    # row_indices, local_col_indices = np.where(reference_mask)
    # Map the local column indices back to global column indices
    # global_col_indices = reference_beam_positions[local_col_indices]
    # Combine the row indices and global column indices
    # reference_indices = (row_indices, global_col_indices)       

    # Initialize an array to store corrected limb data
    corrected_limb_data = group_data.copy()

    for i in limb_beam_positions:
        limb_pos_rng = range(i-5,i+5)

        limb_pos_irtbs = np.concatenate(
        [group_data[:, lpos].flatten() for lpos in limb_pos_rng 
        if lpos in beam_positions and lpos not in reference_beam_positions]
        )
        limb_pos_irtbs = limb_pos_irtbs[~np.isnan(limb_pos_irtbs)]

        if limb_pos_irtbs.size == 0:
            continue           
        limb_temp_min = np.nanmin(limb_pos_irtbs)
        limb_temp_max = np.nanmax(limb_pos_irtbs)
        limb_temp_range = (limb_temp_min, limb_temp_max)
        # Create histograms with independent ranges
        limb_hist, limb_bins = create_histogram(limb_pos_irtbs.flatten(), 
                                                bin_size, limb_temp_range)
        limb_hist_list.append(limb_hist)
        limb_bin_list.append(limb_bins)
        
        adjusted_limb_bins, nadir_hist_norm = adjust_limb_bins_refined(nadir_bins, nadir_hist,
                                                    limb_bins, limb_hist) 
        adj_limb_bins_list.append(adjusted_limb_bins)

        nadir_hist_list.append(nadir_hist_norm)
        nadir_bin_list.append(nadir_bins)
        
        # Map adjusted values back to group_data
        group_data_ = map_adjusted_bins_to_group(group_data, corrected_limb_data,limb_pos_rng, 
                                                adjusted_limb_bins, limb_bins)
        
        adjusted_mask = np.where(~np.isnan(group_data_),1,np.nan)

        # Update brightness_temp_11um_cor
        cumulative_mask = ~np.isnan(brightness_temp_11um_cor)  # Track already corrected regions
        brightness_temp_11um_cor = np.where(adjusted_mask == 1 & ~cumulative_mask, 
                                            group_data_, brightness_temp_11um_cor)                  

    # Convert group data to numpy array
    if not np.all(np.isnan(group_data)):
        group_arrays.append(group_data)
        group_arrays_cor.append(group_data_)
    else:
        print(file)
    
    # # Add the new variable to the dataset
    # data['temp_11_0um_nom_cor'] = (data['temp_11_0um_nom'].dims, brightness_temp_11um_cor)

    # # Save the updated dataset to a NetCDF file with zlib compression
    # output_file = os.path.join(cor_dir, file_nme)
    # data.to_netcdf(output_file, mode='w', 
    #                encoding={'brightness_temp_11um_cor': {'zlib': True, "complevel": 5}})

print('******************* process end *****************')
#%%
print('********* doing dist plots now *********')
orig_hist_dist, orig_beam_ir_means, orig_temp_range = generate_dist(group_arrays)
cor_hist_dist, cor_beam_ir_means, cor_temp_range = generate_dist(group_arrays_cor)

savefig = '_'.join(['percent_dist',cde_run_dte]) + '.png'
savefig  = os.path.join(plot_dir,savefig)
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 10), 
    constrained_layout=True, dpi=500, sharex=True, sharey=True
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
    ls='--', lw=2.5, c='k', label='Mean IR TB (Orig)'
)
axes[0].set_title("Summer 1998: Original IR Tbs Distribution - 60S", fontsize=18)
axes[0].axhline(y=260, color='green', ls=':', lw=2, label='260 K')
axes[0].legend(loc='best', frameon=False, fontsize=12)
axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
axes[0].tick_params(which='both', labelsize=12)

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
axes[1].axhline(y=260, color='green', ls=':', lw=2, label='260 K')
axes[1].legend(loc='best', frameon=False, fontsize=12)
axes[1].set_xlabel("Beam Positions", fontsize=14)
axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
axes[1].tick_params(which='both', labelsize=12)
# Add a colorbar
bdrs = np.linspace(0, 0.04, 15)
# Add a shared colorbar for both plots
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
                    fraction=0.03, pad=0.1, boundaries=bdrs)
cbar.ax.set_xticklabels(
    [f"{int(b):,}" if b >= 1 else f"{b:.3f}" for b in bdrs], fontsize=12
)
cbar.set_label("Normalized Histogram", fontsize=14)
plt.savefig(savefig, bbox_inches='tight')

#%%
print('********* doing line dist plots now *********')

# Combine all bins and histograms into a single array
combined_limb_bins = np.hstack(limb_bin_list)
combined_limb_hist = np.hstack(limb_hist_list)
combined_limb_binned, combined_limb_histsos = create_histogram_of_histogram(combined_limb_bins,
                                                                            combined_limb_hist)

combined_nadir_bins = np.hstack(nadir_bin_list)
combined_nadir_hist = np.hstack(nadir_hist_list)
combined_nadir_binned, combined_nadir_histsos = create_histogram_of_histogram(combined_nadir_bins,
                                                                              combined_nadir_hist)

combined_adj_bins = np.hstack(adj_limb_bins_list)
combined_adj_binned, combined_adj_histsos = create_histogram_of_histogram(combined_adj_bins,
                                                                            combined_limb_hist)

plt_nme = '_'.join(['line_plot_histogram',cde_run_dte]) + '.png'
plt_nme  = os.path.join(plot_dir,plt_nme)
plt.figure()
plt.plot(combined_limb_binned,combined_limb_histsos, ls = '-', c='k', label='limb')
plt.plot(combined_adj_binned,combined_adj_histsos, ls = ':', c='k', label='limb_adj')
plt.plot(combined_nadir_binned,combined_nadir_histsos, ls = ':', c='r', label='nadir')
plt.legend(frameon=False, fontsize=15)
plt.xlabel('IR TB [K]')
plt.ylabel('Count')
plt.grid(which='both', ls = '--', lw='0.5')
plt.title(f"Distribution over range({i-5}, {i+5}) beam positions", 
            fontdict={'size':12, 'color':'k'})
plt.savefig(plt_nme, bbox_inches='tight')

#%%
# nc_dat = Dataset(sample_file)
# nc_lat = nc_dat['latitude']
# nc_lat = np.where(nc_dat['latitude'][:].mask, np.nan, 
#                   nc_dat['latitude'][:].data)

# nc_lat_msk = np.where(np.logical_and(nc_lat <=90, nc_lat >=60),nc_lat, np.nan)

# # nc_lat_msk = nc_lat[np.logical_and(nc_lat <=15, nc_lat >=5)]
# nc_11m = nc_dat['temp_11_0um_nom'][:].data

# # nc_11m_sub = nc_11m[:].data[np.logical_and(nc_lat <=90, nc_lat >=60)]

# nc_11m_sub = np.where(~np.isnan(nc_lat_msk), nc_11m, np.nan)
# # data = data.assign_coords(beam_position=("pixel_elements_along_scan_direction", beam_positions))

# data = data.set_coords(['latitude', 'longitude'])
# # Assign latitude as a coordinate
# data = data.assign_coords(latitude=data["latitude"])


#----------------------------------------------------------
# Function to create histograms
# Define custom colormap
# def create_custom_cmap():
#     # colors = [
#     #     (1.0, 1.0, 1.0),  # White for the lowest values
#     #     (0.8, 0.9, 1.0),  # Light blue
#     #     (0.5, 0.7, 1.0),  # Sky blue
#     #     (0.3, 0.5, 0.8),  # Medium blue
#     #     (0.2, 0.4, 1.0),  # Dark blue
#     #     (0.4, 0.6, 0.2),  # Light green
#     #     (0.6, 0.8, 0.2),  # Yellow-green
#     #     (0.8, 0.9, 0.4),  # Yellow
#     #     (0.9, 0.8, 0.2),  # Yellow-orange
#     #     (1.0, 0.6, 0.0),  # Orange
#     #     (1.0, 0.4, 0.0),  # Deep orange
#     #     (1.0, 0.2, 0.2),  # Light red
#     #     (1.0, 0.0, 0.0),  # Red
#     #     (0.8, 0.0, 0.0),  # Dark red
#     #     (0.6, 0.0, 0.0)   # Deep red for the highest values
#     # ]
#     colors = [
#         (1.0, 1.0, 1.0),  # White for the lowest values
#         (0.8, 0.8, 1.0),  # Light blue
#         (0.5, 0.7, 1.0),  # Sky blue
#         (0.3, 0.5, 0.8),  # Medium blue
#         (1.0, 1.0, 0.6),  # Yellow
#         (1.0, 0.6, 0.0),  # Orange
#         (1.0, 0.0, 0.0)   # Red for the highest values
#     ]

#     # colors = [
#     #         (0.0, 0.0, 0.8),  # Dark blue
#     #         (0.3, 0.5, 0.8),  # Medium blue
#     #         (0.5, 0.7, 1.0),  # Sky blue
#     #         (0.8, 0.9, 1.0),  # Light blue
#     #         (1.0, 1.0, 0.6),  # Yellow
#     #         (1.0, 0.6, 0.0),  # Orange
#     #         (1.0, 0.0, 0.0)   # Red for the highest values
#         # ]
#     cmap = mcolors.LinearSegmentedColormap.from_list("custom_poster_cmap", colors)
#     norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 0.2, 19), ncolors=cmap.N, clip=True)
#     # Define a log scale normalization
#     # norm = mcolors.LogNorm(vmin=0.1, vmax=400)  # Adjust range to match your data
#     # Define logarithmic boundaries for discrete intervals
#     # boundaries = np.logspace(-1.2, np.log10(0.2), 18 + 1)
#     # np.logspace(-1.2, 1, 18 + 1)  # Adjust the range as needed (e.g., 0.1 to 400)

#     # Create a discrete colormap and norm
#     # cmap = mcolors.LinearSegmentedColormap.from_list("custom_discrete_log_cmap", colors, N=len(boundaries) - 1)
#     # norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries) - 1, clip=True)
#     return cmap, norm
# # Function to create discrete colormap and normalization
# def create_discrete_log_cmap_and_norm(boundaries):
#     # Define colors for each bin
#     # colors = [
#     #     (1.0, 1.0, 1.0),  # White
#     #     (0.8, 0.8, 1.0),  # Light blue
#     #     (0.5, 0.7, 1.0),  # Sky blue
#     #     (0.3, 0.5, 0.8),  # Medium blue
#     #     (1.0, 1.0, 0.6),  # Yellow
#     #     (1.0, 0.6, 0.0),  # Orange
#     #     (1.0, 0.0, 0.0)   # Red
#     # ]

#     colors = [
#         (1.0, 1.0, 1.0),  # White for the lowest values
#         (0.8, 0.9, 1.0),  # Light blue
#         (0.5, 0.7, 1.0),  # Sky blue
#         (0.3, 0.5, 0.8),  # Medium blue
#         (0.2, 0.4, 1.0),  # Dark blue
#         (0.4, 0.6, 0.2),  # Light green
#         (0.6, 0.8, 0.2),  # Yellow-green
#         (0.8, 0.9, 0.4),  # Yellow
#         (0.9, 0.8, 0.2),  # Yellow-orange
#         (1.0, 0.6, 0.0),  # Orange
#         (1.0, 0.4, 0.0),  # Deep orange
#         (1.0, 0.2, 0.2),  # Light red
#         (1.0, 0.0, 0.0),  # Red
#         # (0.8, 0.0, 0.0),  # Dark red
#         # (0.6, 0.0, 0.0)   # Deep red for the highest values
#     ]

#     # Create discrete colormap and BoundaryNorm
#     cmap = mcolors.ListedColormap(colors)
#     norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(colors), clip=True)
#     return cmap, norm

# # Function to create custom colormap and normalization
# def create_log_cmap_and_norm(vmin, vmax):
#     colors = [
#         (1.0, 1.0, 1.0),  # White for the lowest values
#         (0.8, 0.8, 1.0),  # Light blue
#         (0.5, 0.7, 1.0),  # Sky blue
#         (0.3, 0.5, 0.8),  # Medium blue
#         (1.0, 1.0, 0.6),  # Yellow
#         (1.0, 0.6, 0.0),  # Orange
#         (1.0, 0.0, 0.0)   # Red for the highest values
#     ]
#     cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
#     norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
#     return cmap, norm

# Plotting function
# def plot_distribution(ax, data, beam_positions, title, bin_size, temp_range):
    
#     bins = np.arange(temp_range[0], temp_range[1] + bin_size, bin_size)
#     # hist = np.array([np.histogram(data[:, i], bins=bins)[0] for i in range(data.shape[1])])
#     # hist = np.array([np.histogram(data[:, i].flatten()[~np.isnan(data[:, i].flatten())], bins=bins)[0] for i in range(data.shape[1])])

#     hists = []
#     for x in data:
#         for i in range(x.shape[1]):
#             hist,_ = np.histogram(x[:, i].flatten()[~np.isnan(x[:, i].flatten())], bins=bins)
#             # hists.append(hist / hist.sum(axis=0, keepdims=True))
#             hists.append(hist)

#     hist_percent = np.array(hists)
#     # hist,_ = np.histogram(data.flatten(), bins=bins)\
#     # Normalize histogram to percentages
#     # hist_percent = hist / hist.sum(axis=0, keepdims=True)

#     # cmap, norm = create_custom_cmap()

#     # cmap, norm = create_log_cmap_and_norm(1, 400)
#     # Define log-scale boundaries for the bins
#     log_boundaries = np.logspace(0, 2.6, num=13)  # Adjust boundaries to match your data
#     cmap, norm = create_discrete_log_cmap_and_norm(log_boundaries)


#     # Plot
#     im = ax.imshow(
#         hist_percent.T, 
#         extent=(beam_positions[0], beam_positions[-1], 
#         temp_range[0], temp_range[1]),
#         aspect='auto', 
#         origin='lower', 
#         cmap=cmap,
#         norm=norm
#     )

#     ax.set_title(title, fontsize=20)
#     ax.set_xlabel("Beam Positions", fontsize=20)
#     ax.set_ylabel("IR Tbs (K)", fontsize=20)
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     return im, hist_percent, norm, log_boundaries

