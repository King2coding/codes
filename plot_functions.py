#%%
# import functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import imageio

import cv2

from datetime import datetime as datme, timedelta as tmdelta
#%%
# flaoting variables
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
#%%
def create_custom_cmap(mn,mx,nm):
    colors = [
        (1.0, 1.0, 1.0),  # White for the lowest values
        (0.8, 0.9, 1.0),  # Light blue
        (0.5, 0.7, 1.0),  # Sky blue
        (0.3, 0.5, 0.8),  # Medium blue
        (0.2, 0.4, 1.0),  # Dark blue
        (0.4, 0.6, 0.2),  # Light green
        (0.6, 0.8, 0.2),  # Yellow-green
        (0.8, 0.9, 0.4),  # Yellow
        (0.9, 0.8, 0.2),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.4, 0.0),  # Deep orange
        (1.0, 0.2, 0.2),  # Light red
        (1.0, 0.0, 0.0),  # Red
    ]
    # colors = [
    #     (1.0, 1.0, 1.0),  # White for the lowest values
    #     (0.8, 0.8, 1.0),  # Light blue
    #     (0.5, 0.7, 1.0),  # Sky blue
    #     (0.3, 0.5, 0.8),  # Medium blue
    #     (1.0, 1.0, 0.6),  # Yellow
    #     (1.0, 0.6, 0.0),  # Orange
    #     (1.0, 0.0, 0.0)   # Red for the highest values
    # ]

    # Create a linear segmented colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_poster_cmap", colors)

    # Define discrete boundaries (for a linear range, adjust as needed)
    norm = mcolors.BoundaryNorm(boundaries=np.linspace(mn, mx, nm), ncolors=cmap.N, clip=True)

    return cmap, norm

#-------------------------------------------------
def plot_ir_tb_distribution_with_means_log_version(
        orig_hists, orig_bin_edges, orig_beam_means,
        cor_hists, cor_bin_edges, cor_beam_means,
        beam_positions, save_path):
    """
    Plot the IR Tbs distribution stratified by beam positions with mean IR Tbs using a discrete log colormap.

    Parameters:
    - orig_hists: 2D histogram array for the original data.
    - orig_bin_edges: Edges of the bins used for original histograms.
    - orig_beam_means: List of mean IR Tbs for original data.
    - cor_hists: 2D histogram array for the corrected data.
    - cor_bin_edges: Edges of the bins used for corrected histograms.
    - cor_beam_means: List of mean IR Tbs for corrected data.
    - beam_positions: List of beam positions corresponding to histogram rows.
    - save_path: Path to save the plot.
    """
    # Define the value range and bins for the colormap
    vmn = max(np.min(orig_hists[orig_hists > 0]), np.min(cor_hists[cor_hists > 0]))
    vmx = max(np.max(orig_hists), np.max(cor_hists))
    cmap, norm = create_custom_cmap(vmn, vmx, 15)  # Use the provided custom colormap function

    # Use LogNorm for logarithmic scaling
    log_norm = mcolors.LogNorm(vmin=vmn, vmax=vmx, clip=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300, sharex=True, sharey=True)

    # Plot original histogram
    im1 = axes[0].imshow(
        orig_hists.T,
        extent=(beam_positions[0], beam_positions[-1], orig_bin_edges[0], orig_bin_edges[-1]),
        cmap=cmap,
        norm=log_norm,  # Apply logarithmic normalization
        aspect='auto',
        origin='lower'
    )
    axes[0].plot(beam_positions, orig_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB')
    axes[0].set_ylim(210, 300)
    axes[0].axhline(y=260, color='green', ls=':', lw=5, label='260 K')

    axes[0].set_title("Original IR Tbs Distribution", fontsize=16)
    axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[0].legend(frameon=False, fontsize=12)

    # Plot corrected histogram
    im2 = axes[1].imshow(
        cor_hists.T,
        extent=(beam_positions[0], beam_positions[-1], cor_bin_edges[0], cor_bin_edges[-1]),
        cmap=cmap,
        norm=log_norm,  # Apply logarithmic normalization
        aspect='auto',
        origin='lower'
    )
    axes[1].plot(beam_positions, cor_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)')
    axes[1].set_ylim(210, 300)
    axes[1].axhline(y=260, color='green', ls=':', lw=5, label='260 K')

    axes[1].set_title("Corrected IR Tbs Distribution", fontsize=16)
    axes[1].set_xlabel("Beam Positions", fontsize=14)
    axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[1].legend(frameon=False, fontsize=12)

    # Add a discrete, logarithmic colorbar
    bdrs = np.logspace(np.log10(vmn), np.log10(vmx), 15)  # Define discrete log-spaced bins
    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.08, ticks=bdrs)
    cbar.set_label("Normalized Percentage (Log Scale)", fontsize=14)
    cbar.ax.set_xticklabels([f"{v:.2e}" for v in bdrs])  # Format ticks for readability

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
#------------------------------------------

def plot_ir_tb_distribution_with_means_discrete_log(
        orig_hists, orig_bin_edges, orig_beam_means,
        cor_hists, cor_bin_edges, cor_beam_means,
        beam_positions, save_path):
    """
    Plot the IR Tbs distribution stratified by beam positions with mean IR Tbs using a discrete log colormap.

    Parameters:
    - orig_hists: 2D histogram array for the original data.
    - orig_bin_edges: Edges of the bins used for original histograms.
    - orig_beam_means: List of mean IR Tbs for original data.
    - cor_hists: 2D histogram array for the corrected data.
    - cor_bin_edges: Edges of the bins used for corrected histograms.
    - cor_beam_means: List of mean IR Tbs for corrected data.
    - beam_positions: List of beam positions corresponding to histogram rows.
    - save_path: Path to save the plot.
    """
    # Define the value range and bins for the colormap
    vmn = max(np.min(orig_hists[orig_hists > 0]), np.min(cor_hists[cor_hists > 0]))
    vmx = max(np.max(orig_hists), np.max(cor_hists))
    cmap, _ = create_custom_cmap(vmn, vmx, 15)  # Use the provided custom colormap function

    # Define discrete boundaries for the colorbar
    bdrs = np.logspace(np.log10(vmn), np.log10(vmx), 15)  # Define discrete log-spaced bins
    norm = mcolors.BoundaryNorm(boundaries=bdrs, ncolors=cmap.N, clip=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300, sharex=True, sharey=True)

    # Plot original histogram
    im1 = axes[0].imshow(
        orig_hists.T,
        extent=(beam_positions[0], beam_positions[-1], orig_bin_edges[0], orig_bin_edges[-1]),
        cmap=cmap,
        norm=norm,  # Apply discrete normalization
        aspect='auto',
        origin='lower'
    )
    axes[0].plot(beam_positions, orig_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB')
    axes[0].set_ylim(210, 300)
    axes[0].axhline(y=260, color='green', ls=':', lw=5, label='260 K')

    axes[0].set_title("Original IR Tbs Distribution", fontsize=16)
    axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[0].legend(frameon=False, fontsize=12)

    # Plot corrected histogram
    im2 = axes[1].imshow(
        cor_hists.T,
        extent=(beam_positions[0], beam_positions[-1], cor_bin_edges[0], cor_bin_edges[-1]),
        cmap=cmap,
        norm=norm,  # Apply discrete normalization
        aspect='auto',
        origin='lower'
    )
    axes[1].plot(beam_positions, cor_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)')
    axes[1].set_ylim(210, 300)
    axes[1].axhline(y=260, color='green', ls=':', lw=5, label='260 K')

    axes[1].set_title("Corrected IR Tbs Distribution", fontsize=16)
    axes[1].set_xlabel("Beam Positions", fontsize=14)
    axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[1].legend(frameon=False, fontsize=12)

    # Add a discrete colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.08, boundaries=bdrs, ticks=bdrs)
    cbar.set_label("Normalized Percentage (Log Scale)", fontsize=14)
    cbar.ax.set_xticklabels([f"{v:.5f}" if v < 0.01 else f"{v:.3f}" for v in bdrs])  # Format tick labels for readability
    # cbar.ax.set_xticklabels([f"{v:.4f}" for v in bdrs])  # Display all ticks as fixed decimal  # Use general formatting

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

#------------------------------------------

def plot_ir_tb_distribution_with_means_normal_version(
        orig_hists, orig_bin_edges, orig_beam_means,
        cor_hists, cor_bin_edges, cor_beam_means,
        beam_positions, save_path):
    """
    Plot the IR Tbs distribution stratified by beam positions with mean IR Tbs.

    Parameters:
    - orig_hists: 2D histogram array for the original data.
    - orig_bin_edges: Edges of the bins used for original histograms.
    - orig_beam_means: List of mean IR Tbs for original data.
    - cor_hists: 2D histogram array for the corrected data.
    - cor_bin_edges: Edges of the bins used for corrected histograms.
    - cor_beam_means: List of mean IR Tbs for corrected data.
    - beam_positions: List of beam positions corresponding to histogram rows.
    - save_path: Path to save the plot.
    """
    vmn, vmx = 0, max(np.max(orig_hists), np.max(cor_hists)) #* 0.95 
    cmap, norm = create_custom_cmap(vmn, vmx, 15)  

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300, sharex=True, sharey=True)

    # Plot original histogram
    im1 = axes[0].imshow(
        orig_hists.T,
        extent=(beam_positions[0], beam_positions[-1], orig_bin_edges[0], orig_bin_edges[-1]),
        cmap=cmap,
        norm=norm,
        aspect='auto',
        origin='lower'
    )
    axes[0].plot(beam_positions, orig_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB')
    axes[0].set_ylim(210,300)
    axes[0].axhline(y=260, color='green', ls=':', lw=5, label='260 K') # 

    axes[0].set_title("Original IR Tbs Distribution", fontsize=16)
    axes[0].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[0].legend(frameon=False, fontsize=12)

    # Plot corrected histogram
    im2 = axes[1].imshow(
        cor_hists.T,
        extent=(beam_positions[0], beam_positions[-1], cor_bin_edges[0], cor_bin_edges[-1]),
        cmap=cmap,
        norm=norm,
        aspect='auto',
        origin='lower'
    )
    axes[1].plot(beam_positions, cor_beam_means, ls='--', lw=2.5, c='k', label='Mean IR TB (Corrected)')
    axes[1].set_ylim(210,300)
    axes[1].axhline(y=260, color='green', ls=':', lw=5, label='260 K') # 

    axes[1].set_title("Corrected IR Tbs Distribution", fontsize=16)
    axes[1].set_xlabel("Beam Positions", fontsize=14)
    axes[1].set_ylabel("IR Tbs (K)", fontsize=14)
    axes[1].legend(frameon=False, fontsize=12)

    # Add a shared colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.08)
    cbar.set_label("Normalized Percentage", fontsize=14)   

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    # plt.close()
#------------------------------------------

def plot_discrete_ir_tb_distribution(orig_hists, orig_bin_edges, orig_beam_means,
                                     cor_hists, cor_bin_edges, cor_beam_means,
                                     beam_positions, save_path):
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
    axes[0].set_title("Original IR Tbs Distribution", fontsize=16)
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
    # cbar.ax.set_xticklabels(
    # [f"{int(b):,}" if b >= 1 else f"{b:.2f}" for b in levels], fontsize=12
    # )
    cbar.ax.set_xticklabels([f"{v:.3f}" for v in levels])

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
#------------------------------------------


def plot_distribution_at_beam_position(limb_bn, limb_cnt, limb_adj,
                                       nadir_bn, nadir_cnt, 
                                       beam_pos, ttle, fignme):
    f,ax = plt.subplots(dpi=250)
    ax.plot(limb_bn[beam_pos],limb_cnt[beam_pos], 
            ls = '-', c='k', label='limb')
    ax.plot(limb_adj[beam_pos], limb_cnt[beam_pos], ls = ':', 
            c='k', label='limb_adj')
    ax.plot(nadir_bn[beam_pos],nadir_cnt[beam_pos], 
            ls = ':', c='r', label='nadir')
    plt.xlim(210, 300)
    # Add text annotations for sums
    nadir_sum = np.nansum(nadir_cnt[beam_pos])
    limb_sum = np.nansum(limb_cnt[beam_pos])

    ax.text(0.05, 0.5, f"Nadir Sum: {nadir_sum:.1f}", 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', color='r')
    ax.text(0.05, 0.45, f"Limb Sum: {limb_sum:.1f}", 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', color='k')
    # plt.yscale('log')
    ax.legend(frameon=False, fontsize=15)
    ax.set_xlabel('IR TB [K]')
    ax.set_ylabel('Count')
    ax.grid(which='both', ls = '--', lw='0.5')
    ax.set_title(ttle, fontdict={'size':12, 'color':'k'})
    f.savefig(fignme, bbox_inches='tight')

#------------------------------------------

def plot_lut_histograms_by_hemisphere(hemisphere, limb_bin_list_by_srftype, limb_hist_list_by_srftype, 
                                      adjusted_limb_bins_list_by_srftype, nadir_bin_list_by_srftype, 
                                      nadir_hist_list_by_srftype, plot_dir, cde_run_dte, lat_wind):
    season = "summer" if hemisphere == "SH" else "winter"
    for id in surface_type_mapping.keys():
        sftyp = surface_type_mapping[id]
        for b in [0, 10, 50, 100, 400]:
            beam_pos = b
            plt_nme = f'LUT_method_line_plot_histogram_{hemisphere}_{sftyp}_{beam_pos}_{cde_run_dte}.png'
            plt_nme = os.path.join(plot_dir, plt_nme)
            ttle = f"{hemisphere.upper()} {season.capitalize()} 1998 IR TB Distribution \n beam position={beam_pos} (+/- 10) \n latitude window = {lat_wind} \n surface type= {sftyp}"

            plot_distribution_at_beam_position(limb_bin_list_by_srftype[id], 
                                               limb_hist_list_by_srftype[id], 
                                               adjusted_limb_bins_list_by_srftype[id],
                                               nadir_bin_list_by_srftype[id], 
                                               nadir_hist_list_by_srftype[id],                                       
                                               beam_pos, ttle, plt_nme)
            
#------------------------------------------
def groupby_and_plot(dat, grb_col, plt_col, srftyp, hem):
    dat_grpby = dat.groupby(grb_col, as_index=True)[plt_col].mean()
    dat_grpby.plot(kind='line', xlabel='Beam position',
                        ylabel='Mean correction coefficient',
                        title=f'Mean correction coefficient per beam position for {srftyp} ({hem})',
                        figsize=(10, 6), grid=True) 
    
#------------------------------------------
def box_plot_of_corr_coeff(lut, xcol, ycol, xticks, savefig):
    """
    A box plot (or box-and-whisker plot) is a standardized way of displaying 
    the distribution of data based on a five-number summary: minimum, first 
    quartile (Q1), median, third quartile (Q3), and maximum. Here's what each 
    component represents:

    Box: The box itself represents the interquartile range (IQR), which is the 
    range between the first quartile (Q1) and the third quartile (Q3). This is 
    where the middle 50% of the data points lie.

    Median Line: A line inside the box shows the median (Q2) of the data.

    Whiskers: The lines extending from the box (whiskers) show the range of the 
    data, typically up to 1.5 times the IQR from the quartiles. Data points 
    outside this range are considered outliers.

    Outliers: Individual points outside the whiskers are plotted as dots and 
    represent outliers.
    """
    new_rows = []
    for p in reference_beam_positions:
        new_row = lut.iloc[0].copy()
        new_row['beam_position'] = p
        for col in new_row.index:
            if col != 'beam_position':
                new_row[col] = np.nan
        new_rows.append(new_row)
    lut_cpy = pd.concat([lut, pd.DataFrame(new_rows)], ignore_index=True)
    lut_cpy.sort_values(by='beam_position', inplace=True)

    # Create a boxplot to show the distribution of corr_coeff per beam_position
    fig, ax = plt.subplots(figsize=(20, 10))

    mean_values = lut_cpy.groupby('beam_position')['corr_coeff'].mean().values
    sns.boxplot(x=xcol, y=ycol, 
                data=lut_cpy, showfliers=False, ax=ax,
                medianprops={'color': 'red', 'linewidth': 5})

    # Add custom legends for the median line and mean points
    median_line = Line2D([0], [0], color='r', linewidth=2.5, label='Median')
    mean_point = Line2D([0], [0], color='k', marker='o', 
                        linestyle='None', markersize=4, label='Mean')
    ax.legend(handles=[median_line, mean_point], frameon=False, fontsize=20, loc='best')
    ax.scatter(range(len(mean_values)), mean_values, 
               color='k', s=2, zorder=5)

    ax.set_title('Distribution of correction coefficient per beam_position', 
                 fontsize=20)
    ax.set_xlabel('Beam Position', fontsize=20)
    ax.set_ylabel('Correction coefficient', fontsize=20)
    # Set x-ticks at intervals of 10, ensuring the gap is maintained
    xticks = xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(savefig, dpi=500, bbox_inches='tight')
    plt.show()

#------------------------------------------
def plot_latitudinal_path(file, projection_type='PlateCarree', hemisphere='global'):
    data = xr.open_dataset(file)
    lats = data['latitude'][:, :].data
    lon = data['longitude'][:, :].data
    nadir_lat = lats[:, 204]
    nadir_lat = nadir_lat[~np.isnan(nadir_lat)]
    nadir_lon = lon[:, 204]
    nadir_lon = nadir_lon[~np.isnan(nadir_lon)]
    limb_lat_left = lats[:, 0]
    limb_lat_left = limb_lat_left[~np.isnan(limb_lat_left)]
    limb_lon_left = lon[:, 0]
    limb_lon_left = limb_lon_left[~np.isnan(limb_lon_left)]
    limb_lat_right = lats[:, 408]
    limb_lat_right = limb_lat_right[~np.isnan(limb_lat_right)]
    limb_lon_right = lon[:, 408]
    limb_lon_right = limb_lon_right[~np.isnan(limb_lon_right)]
    # Filter data based on hemisphere
    if hemisphere == 'NH':
        mask = nadir_lat >= 45
    elif hemisphere == 'SH':
        mask = nadir_lat <= -45
    else:
        mask = np.ones_like(nadir_lat, dtype=bool)
    nadir_lat = nadir_lat[mask]
    nadir_lon = nadir_lon[mask]
    limb_lat_left = limb_lat_left[mask]
    limb_lon_left = limb_lon_left[mask]
    limb_lat_right = limb_lat_right[mask]
    limb_lon_right = limb_lon_right[mask]
    # Define the projection
    if projection_type == 'PolarStereographic':
        projection = ccrs.Stereographic(central_latitude=45 if hemisphere == 'NH' else -45)
    else:
        projection = ccrs.PlateCarree()
    # Create a figure and axis with the specified projection
    fig, ax = plt.subplots(figsize=(10, 15), subplot_kw={'projection': projection}, dpi=500)
    # Plot the nadir and limb latitude lines
    ax.plot(nadir_lon, nadir_lat, transform=ccrs.PlateCarree(), label='204 beam pos (Nadir) path', color='blue')
    ax.plot(limb_lon_left, limb_lat_left, transform=ccrs.PlateCarree(), label='0 beam pos (Left Limb) path', color='red')
    ax.plot(limb_lon_right, limb_lat_right, transform=ccrs.PlateCarree(), label='408 beam pos (Right Limb) path', color='green')
    ax.coastlines()
    # Add land and ocean features
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, zorder=0, edgecolor='black')
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.8, alpha=0.5, color='gray')
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 15))
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    # Add a legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=13)
    # Set the title
    ax.set_title(f'Nadir and Limb Latitudinal path - {hemisphere}', fontsize=15)
    # return fig

#------------------------------------------
def create_movie_from_nc_files(file_dir, plot_dir):
    
    # Get the paths of all the .nc files in the directory
    nc_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.nc')]
    # Initialize lists to store the paths of the generated images
    image_paths_nh = []
    image_paths_sh = []

    for file in nc_files:
        file_name = os.path.basename(file)
        date_str = int(file_name.split('.')[3][1:])
        year = date_str // 1000
        day_of_year = date_str % 1000
        date_ = datme(year, 1, 1) + tmdelta(days=day_of_year - 1)
        date_str = date_.strftime('%Y-%m-%d')
        start_time_str = file_name.split('.')[4][1:]
        end_time_str = file_name.split('.')[5][1:]

        for hemisphere in ['NH', 'SH']:
            fig = plot_latitudinal_path(file, 
                                        projection_type='PolarStereographic', 
                                        hemisphere=hemisphere)
            image_path = os.path.join(plot_dir, 
                                      f'nadir_and_limb_latitudinal_path_{hemisphere}_{date_str}_{start_time_str}_{end_time_str}.png')
            plt.savefig(image_path, bbox_inches='tight')
            if hemisphere == 'NH':
                image_paths_nh.append(image_path)
            else:
                image_paths_sh.append(image_path)
            plt.close(fig)

    # Create a movie from the images for NH
    movie_path_nh = os.path.join(plot_dir, 
                                 'nadir_and_limb_latitudinal_path_movie_NH.mp4')
    frame = cv2.imread(image_paths_nh[0])
    height, width, layers = frame.shape

    video_nh = cv2.VideoWriter(movie_path_nh, 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               1, 
                               (width, height)
                               )

    for image_path in image_paths_nh:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading {image_path}")
            continue
        resized_frame = cv2.resize(frame, (width, height))
        video_nh.write(resized_frame)

    video_nh.release()
    print(f"NH Movie saved at {movie_path_nh}")

    # Create a movie from the images for SH
    movie_path_sh = os.path.join(plot_dir, 
                                 'nadir_and_limb_latitudinal_path_movie_SH.mp4')
    frame = cv2.imread(image_paths_sh[0])
    height, width, layers = frame.shape
    video_sh = cv2.VideoWriter(movie_path_sh, 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               1, 
                               (width, height)
                               )

    for image_path in image_paths_sh:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading {image_path}")
            continue
        resized_frame = cv2.resize(frame, (width, height))
        video_sh.write(resized_frame)

    video_sh.release()
    print(f"SH Movie saved at {movie_path_sh}")
#------------------------------------------
