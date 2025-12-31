#%% Package imports
import gc
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from datetime import datetime, timedelta

from rasterio.enums import Resampling
import rioxarray
#%% Define working directory
imerg_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/imergv07/data'
era5_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/era5'
cml_sat_15min_dir = '/home/kkumah/Projects/cml-stuff/out_15min_cml_rain_oper'
cml_sat_daily_dir = '/home/kkumah/Projects/cml-stuff/out_daily_cml_rain_oper' 

plot_dir = '/home/kkumah/Projects/cml-stuff/plots'

#%% Floating varibales and constants
all_imerg_files = sorted([os.path.join(imerg_dir, f) for f in os.listdir(imerg_dir) if f.endswith('.nc4')])

era5_file = os.path.join(era5_dir, 'ERA5_total_precipitation_2025_09_12_Ghana.nc')

cde_run_dte = datetime.today().strftime('%Y%m%d')

days = pd.date_range("2025-09-01", "2025-12-29", freq="D")

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

#%%Functions
def create_xarray(all_precip, all_time_index, lon, lat, attrs=None):
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
    time_index = pd.to_datetime(all_time_index)
    
    # Create an xarray DataArray from the list of 2D precipitation arrays
    precip_data = xr.DataArray(
                data=all_precip,
                dims=["time", "lat", "lon"],
                coords={
                    "time": time_index,
                    "lat": lat,
                    "lon": lon
                }
            )

    # Add attributes if provided
    if attrs:
        precip_data.attrs.update(attrs)
    
    return precip_data
def pick_imerg_day_files(files, day_str):
    """
    day_str: 'YYYY-MM-DD'
    Select files whose path/name contains YYYYMMDD.
    """
    ymd = day_str.replace("-", "")
    return [fp for fp in files if ymd in os.path.basename(fp)]

def read_nc_imger_file(file_path):
    imerg_precip_data = xr.open_dataset(file_path)
    precip_aray = imerg_precip_data.precipitation.data     
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data['time'].values[0]
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')
    lon = imerg_precip_data.coords['lon'].values
    lat = np.flip(imerg_precip_data.coords['lat']).values    
    imerg_precip_data.close()    
    del(imerg_precip_data,imerg_time)    
    return precip_aray, imerg_time_index, lon,lat

def process_imerg(files, product):    

    all_imfn_prcp, all_imfn_tms = [], []

    for imf in files:
        prcp, tms, lon, lat = read_nc_imger_file(imf)
        all_imfn_prcp.append(prcp)
        all_imfn_tms.append(tms)

    img_lon, img_lat = lon, lat

    # sort by time
    sorted_indices = np.argsort(np.array(all_imfn_tms))
    all_imfn_prcp = np.array(all_imfn_prcp)[sorted_indices]
    all_imfn_tms = np.array(all_imfn_tms)[sorted_indices]

    # create half-hourly DataArray
    imerg_xarr_hh_data = create_xarray(
        all_imfn_prcp,
        all_imfn_tms,
        img_lon,
        img_lat,
        attrs={"product": product}
    )

    # daily date
    day = pd.to_datetime(all_imfn_tms[0]).normalize()

    # daily precipitation (mm/day)
    imerg_dd_precip = (
    (imerg_xarr_hh_data.sum(dim="time") * 0.5)
    .expand_dims(time=[day])
    )

    return imerg_xarr_hh_data, imerg_dd_precip


def _standardize_latlon_names(da):
    """Rename common lat/lon variants to x/y for rioxarray."""
    rename_map = {}

    if "lat" in da.dims:
        rename_map["lat"] = "y"
    if "latitude" in da.dims:
        rename_map["latitude"] = "y"

    if "lon" in da.dims:
        rename_map["lon"] = "x"
    if "longitude" in da.dims:
        rename_map["longitude"] = "x"

    if rename_map:
        da = da.rename(rename_map)

    return da


def _ensure_crs(da, crs="EPSG:4326"):
    """Ensure CRS exists."""
    if not da.rio.crs:
        da = da.rio.write_crs(crs, inplace=False)
    return da


def _ensure_monotonic_coords(da):
    """Ensure lat decreases north→south and lon increases west→east."""
    if da.y[0] < da.y[-1]:
        da = da.sortby("y", ascending=False)
    if da.x[0] > da.x[-1]:
        da = da.sortby("x", ascending=True)
    return da


def harmonize_to_era5(
    source_da: xr.DataArray,
    era5_da: xr.DataArray,
    method="average",   # average is correct for precipitation
):
    """
    Harmonize a source DataArray (e.g. IMERG) onto ERA5 grid.
    """

    # --- Standardize dimension names
    src = _standardize_latlon_names(source_da)
    tgt = _standardize_latlon_names(era5_da)

    # --- CRS
    src = _ensure_crs(src)
    tgt = _ensure_crs(tgt)

    # --- Coordinate sanity
    src = _ensure_monotonic_coords(src)
    tgt = _ensure_monotonic_coords(tgt)

    # --- Match grid (NO reprojection needed)
    src_on_era5 = src.rio.reproject_match(
        tgt,
        resampling=Resampling.average if method == "average" else Resampling.bilinear,
    )

    # --- Restore conventional names
    src_on_era5 = src_on_era5.rename({"y": "latitude", "x": "longitude"})

    return src_on_era5

def compute_pdf_elements_from_array(data_array, bins):
    """
    Compute PDF elements (PDFc and PDFv) from a 2D numpy array.

    Parameters:
    - data_array: 2D numpy array containing the data.
    - bins: List of bin edges.

    Returns:
    - pdf_df: A dictionary containing bins, PDFc, and PDFv.
    """
    pdfc = []  # PDF by occurrence
    pdfv = []  # PDF by volume
    bin_labels = []  # Bin labels for the output

    # Flatten the 2D array to 1D
    flattened_data = data_array.flatten()

    # Remove NaN values from the flattened data
    flattened_data = flattened_data[~np.isnan(flattened_data)]

    total_count = len(flattened_data)
    total_volume = 0

    # Loop through bins to compute PDFc and PDFv
    for i, bn in enumerate(bins):
        if i == 0:
            bin_data = flattened_data[flattened_data <= bn]
        else:
            bin_data = flattened_data[(flattened_data > bins[i - 1]) & (flattened_data <= bn)]

        # PDFc: Percentage of occurrences in the bin
        bin_count = len(bin_data)
        pdfc.append((bin_count / total_count)) # * 100

        # PDFv: Percentage of volume in the bin
        if bin_count > 0:
            bin_mean = np.mean(bin_data)
            bin_volume = bin_count * bin_mean
        else:
            bin_volume = 0

        total_volume += bin_volume
        pdfv.append(bin_volume)

        # Add bin label
        if i == 0:
            bin_labels.append(f"<= {bn}")
        else:
            bin_labels.append(f"{bins[i - 1]} - {bn}")

    # Normalize PDFv to percentages
    pdfv = [(volume / total_volume) for volume in pdfv] # * 100

    # Create a dictionary with bin, pdfc, and pdfv
    pdf_dict = {
        "bin": bins,
        "pdfc": pdfc,
        "pdfv": pdfv
    }

    del(flattened_data, bin_data
        ,total_count, total_volume
        ,bin_count, bin_mean, bin_volume
        ,bin_labels)

    gc.collect()

    return pdf_dict

def plot_precip_1x2_compare_ghana(
        da_list,
        lons, lats,
        titles,
        vmin, vmax,
        bbox=(-4.0, 1.25, 4.5, 11.25),
        cmap_name="nipy_spectral",
        coast_lw=1.5,
        border_lw=1.5,
    ):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # ------------------------------------------------------------
    # SAME DISCRETE COLORMAP (UNCHANGED)
    # ------------------------------------------------------------
    def discrete_nipy(vmin, vmax):
        mpl_cm = cm.get_cmap(cmap_name, 21)
        ncolors = mpl_cm(np.linspace(0, 1, 21))[:20]

        ncolors[0] = [128/256, 100/256, 128/256, 1]
        ncolors[2] = ncolors[1].copy()
        ncolors[1] = [0.7, 0.1, 0.7, 1]

        newcmap = ListedColormap(ncolors)
        levels = np.linspace(vmin, vmax, len(ncolors))
        norm = BoundaryNorm(levels, newcmap.N)
        return newcmap, norm, levels

    cmap, norm, levels = discrete_nipy(vmin, vmax)

    # ------------------------------------------------------------
    # FIGURE + AXES
    # ------------------------------------------------------------
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(
        1, 2, figsize=(13, 6),
        subplot_kw=dict(projection=proj)
    )

    ims = []

    for i, (ax, da, title) in enumerate(zip(axs, da_list, titles)):

        ax.set_extent(bbox, crs=proj)

        ax.coastlines(resolution="10m", linewidth=coast_lw, color="black")
        ax.add_feature(
            cfeature.BORDERS.with_scale("10m"),
            linewidth=border_lw,
            edgecolor="black"
        )

        # --------------------------------------------------------
        # GRIDLINES + LABELS
        # --------------------------------------------------------
        gl = ax.gridlines(
            crs=proj,
            draw_labels=True,
            linewidth=0.6,
            linestyle="--",
            color="gray",
            alpha=0.5
        )

        gl.top_labels = False
        gl.right_labels = False

        # Only left panel shows latitude labels
        gl.left_labels = (i == 0)
        gl.bottom_labels = True

        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}

        # --------------------------------------------------------
        # DATA PLOT
        # --------------------------------------------------------
        im = ax.contourf(
            lons, lats, da,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=proj,
            extend="max"
        )

        ax.set_title(title, fontsize=18, fontweight="bold")
        ims.append(im)

    # ------------------------------------------------------------
    # SHARED COLORBAR
    # ------------------------------------------------------------
    cbar = fig.colorbar(
        ims[0],
        ax=axs,
        orientation="horizontal",
        fraction=0.06,
        pad=0.08
    )

    ticks = np.linspace(vmin, vmax, 9).round().astype(int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Rainfall (mm/day)", fontsize=15)

    # ------------------------------------------------------------
    # REDUCE WHITESPACE
    # ------------------------------------------------------------
    # Leave room at the bottom for colorbar
    fig.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.92,
    bottom=0.2,   # 👈 THIS is the key control
    wspace=0.005
    )

    return fig, axs

import numpy as np
import matplotlib.pyplot as plt

def plot_latitude_profiles(
    imerg_da,
    era5_da,
    lat_name="latitude",
    xlim=None,
    title="Latitudinal Mean Precipitation",
):
    """
    Plot IMERG and ERA5 precipitation profiles vs latitude.
    """

    # --- Extract data safely ---
    lat = imerg_da[lat_name].values
    imerg_vals = imerg_da.values
    era5_vals = era5_da.values

    # --- Ensure numpy arrays ---
    lat = np.asarray(lat)
    imerg_vals = np.asarray(imerg_vals)
    era5_vals = np.asarray(era5_vals)

    # --- Sort by latitude (south → north for plotting clarity) ---
    sort_idx = np.argsort(lat)
    lat = lat[sort_idx]
    imerg_vals = imerg_vals[sort_idx]
    era5_vals = era5_vals[sort_idx]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 7), dpi=140)

    ax.plot(
        imerg_vals, lat,
        color="tab:blue",
        linewidth=2.5,
        label="IMERG"
    )

    ax.plot(
        era5_vals, lat,
        color="tab:orange",
        linewidth=2.5,
        label="ERA5"
    )

    # Agro-climatic zone boundaries (°N)
    zone_bounds = [5.5, 7.5, 8.5]

    for z in zone_bounds:
        ax.axhline(
            y=z,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

    # Zone label positions: (label, y_position)
    zone_labels = [
    ("Coastal Savanna", 5.0),
    ("Forest Zone", 6.6),
    ("Transitional Zone", 8.0),
    ("Guinea Savanna", 10.0),
    ]

    for label, y in zone_labels:
        ax.text(
            0.98, y,
            label,
            fontsize=12,
            fontweight="bold",
            color="dimgray",
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform()
        )

    # --- Labels & title ---
    ax.set_xlabel("Rainfall (mm/day)", fontsize=14)
    ax.set_ylabel("Latitude (°N)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # --- Limits ---
    if xlim is not None:
        ax.set_xlim(xlim)

    # --- Grid & ticks ---
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=12)

    # --- Legend ---
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    plt.show()


#%% Main processing
# 1) Read and preprocess IMERG data day by day
daily_xr = []
for day in days:
    day_str = day.strftime("%Y-%m-%d")
    imerg_day_files  = pick_imerg_day_files(all_imerg_files,  day_str)

    _,imerg_day_dat = process_imerg(imerg_day_files, product='IMERG-Late')
    daily_xr.append(imerg_day_dat)

imerg_daily_xarr = xr.concat(daily_xr, dim="time")

# 2) Read and preprocess ERA5 data
# era5_data = xr.open_dataset(era5_file)
# era5_data = era5_data.rename({'valid_time': 'time'})
# era5_daily_data = era5_data['tp'] * 1000  # Convert from meters to mm
# era5_daily_data = era5_daily_data.resample(time='1D').mean() * 24

era5_data = xr.open_dataset(era5_file)
era5_data = era5_data.rename({'valid_time': 'time'})

era5_daily_data = (
    era5_data['tp']
    .groupby('time.date')
    .sum(dim='time')
    .rename({'date': 'time'})
    * 1000.0
)

# 3) Harmonize IMERG data to ERA5 grid
imerg_daily_agg = harmonize_to_era5(
    imerg_daily_xarr,
    era5_daily_data,
)

# harmonize the data in time ensuring we are comparing same days
# Convert ERA5 time to datetime64[ns]
era5_daily_data = era5_daily_data.assign_coords(
    time=pd.to_datetime(era5_daily_data.time.values)
)

# Normalize IMERG time (safe even if already normalized)
imerg_daily_agg = imerg_daily_agg.assign_coords(
    time=pd.to_datetime(imerg_daily_agg.time.values).normalize()
)

# Now intersect safely
common_times = np.intersect1d(
    imerg_daily_agg.time.values,
    era5_daily_data.time.values
)

# Subset
imerg_daily_agg = imerg_daily_agg.sel(time=common_times)
era5_daily_data = era5_daily_data.sel(time=common_times)

#%% Begin evaluation and plotting
# 1) a) Spatial mean over Ghana: single value per day time series
imerg_ghana_mean = imerg_daily_agg.mean(dim=['latitude', 'longitude'])
era5_ghana_mean = era5_daily_data.mean(dim=['latitude', 'longitude'])

# b) Spatial mean: 2d array
imerg_ghana_2dmean = imerg_daily_agg.mean(dim='time')
era5_ghana_2dmean = era5_daily_data.mean(dim='time')

# c) zonal latitudinal mean
imerg_ghana_zonalmean = imerg_daily_agg.mean(dim=['time', 'longitude'])
era5_ghana_zonalmean = era5_daily_data.mean(dim=['time', 'longitude'])

# ---------------------
# Their plotting
# ---------------------
# 1) a) time series
da1 = imerg_ghana_mean          # rename appropriately
da2 = era5_ghana_mean           # rename appropriately

fig, ax = plt.subplots(figsize=(10, 4), dpi=140)

# Plot both on the same axis
ax.plot(
    da1.time.values,
    da1.values,
    label="IMERG",
    color="tab:blue",
    linewidth=1.8
)

ax.plot(
    da2.time.values,
    da2.values,
    label="ERA5",
    color="tab:orange",
    linewidth=1.8
)

# --- Axis formatting ---
# ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("Rainfall (mm/day)", fontsize=13)

# Force identical y-ticks
yticks = np.arange(0, 21, 5)
ax.set_yticks(yticks)
ax.set_ylim(0, 20)

# Grid & ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=12, length=6, direction='in')
ax.tick_params(axis='both', which='minor', length=3, direction='in')
ax.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.6)

# Legend
ax.legend(fontsize=12, loc="upper right")

# Major ticks → months
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Minor ticks → every 7 days
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))

ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))   # 15

# Tick styling
ax.tick_params(axis='x', which='major', labelsize=16, pad=10)
ax.tick_params(axis='x', which='minor', labelsize=12, pad=3)

# Optional: rotate slightly for safety
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

plt.tight_layout()
plt.show()

# 2) b) Spatial mean maps
fig, axs = plot_precip_1x2_compare_ghana(
    da_list=[imerg_ghana_2dmean, era5_ghana_2dmean],
    lons=imerg_ghana_2dmean.longitude.values,
    lats=imerg_ghana_2dmean.latitude.values,
    titles=["IMERG", "ERA5"],
    vmin=0,
    vmax=6
)

# 2) c) Zonal mean plots
plot_latitude_profiles(
    imerg_ghana_zonalmean,   # your IMERG 1D DataArray
    era5_ghana_zonalmean,    # your ERA5 1D DataArray
    xlim=(0, 6),
    title="Ghana Latitudinal Mean Precipitation"
)
# %% 2) DO PDF ANALYSIS
print("Creating Precipitation PDFs for GPCP v3.2, GPCP v3.3 and ERA5...")
bins = [0.2 * (2 ** i) for i in range(10)]

imerg_pdf = compute_pdf_elements_from_array(imerg_daily_agg.values, bins)
era5_pdf = compute_pdf_elements_from_array(era5_daily_data.values, bins)


#%% 3) Do scatter plot analysis

from sklearn.linear_model import LinearRegression

def add_regression_line(ax, x, y, color, xlim):

    # Convert xarray → numpy
    if hasattr(x, "values"):
        x = x.values
    if hasattr(y, "values"):
        y = y.values

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 2:
        return np.nan, np.nan, mask

    x_clean = x[mask]
    y_clean = y[mask]

    model = LinearRegression()
    model.fit(x_clean.reshape(-1, 1), y_clean.reshape(-1, 1))

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    x_line = np.linspace(xlim[0], xlim[1], 200)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line, y_line,
        color=color, linestyle='-', linewidth=1.5,
        label='Regression Line'
    )

    return slope, intercept, mask

# FIGURE SETUP
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=140)

# Define hemisphere axis limits
xlims = (0., 20)
ylims = (0., 20)


plots = [
    (axs[0], imerg_ghana_mean, era5_ghana_mean, 'k', 'ERA5 vs IMERG', xlims, ylims),
    (axs[1], era5_ghana_mean, imerg_ghana_mean, 'k', 'IMERG vs ERA5', xlims, ylims),
    (axs[2], era5_ghana_mean, era5_ghana_mean, 'k', 'ERA5 vs ERA5', xlims, ylims) # (axs[1, 0], chgps_NH_means['zonal_means'], merra2_NH_means['zonal_means'], 'MERRA2', 'red', 'SH', SH_xlim, SH_ylim),
    
]

for ax, x, y, color, lab, xlim, ylim in plots:

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ticks = np.arange(0, 21, 5)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Scatter
    ax.scatter(x, y, s=45, color=color, edgecolor='black', alpha=0.85,
               label=lab)

    # 1:1 LINE across fixed limits
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]],
            color='black', linestyle='--', linewidth=1.5, label='1:1 Line')

    # Regression line
    slope, intercept, mask = add_regression_line(ax, x, y, color, xlim)

    # Stats using same cleaned subset
    # Ensure numeric
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    mask = np.isfinite(x) & np.isfinite(y)#~(x.isna() | y.isna())
    x_clean = x[mask].astype(float).values
    y_clean = y[mask].astype(float).values

    # Must be at least 2 values to compute correlation
    if len(x_clean) < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
    bias = (y_clean.mean() / x_clean.mean()) - 1

    # Regression equation text
    eq_text = f"y = {slope:.2f}x + {intercept:.2f}"

    # Metrics block
    ax.text(0.03, 0.97,
            f"Corr = {corr:.2f}\nBias = {bias:.1%}\n{eq_text}",
            transform=ax.transAxes,
            fontsize=15, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.65, edgecolor='none'))

    # Axes + ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=15, length=7, direction='in')
    ax.tick_params(axis='both', which='minor', length=4, direction='in')
    ax.grid(which='major', linestyle='--', linewidth=0.6, color='grey')

    # Legend
    ax.legend(fontsize=12, loc='lower right')

# Axis labels
axs[0].set_ylabel("ERA5 (mm/day)", fontsize=16)
axs[0].set_xlabel("IMERG (mm/day)", fontsize=16)
axs[1].set_ylabel("IMERG (mm/day)", fontsize=16)
axs[1].set_xlabel("ERA5 (mm/day)", fontsize=16)
axs[2].set_ylabel("ERA5 (mm/day)", fontsize=16)
axs[2].set_xlabel("ERA5 (mm/day)", fontsize=16)

plt.tight_layout()
plt.show()
gc.collect()