#%%
import os
from unittest import result
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl


#%%
# floating variables
df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
all_df_files = sorted([os.path.join(df_dir, s) for s in os.listdir(df_dir) if s.endswith('.pkl')])
all_df_files_11um = [f for f in all_df_files if '_11_' in f]
all_df_files_12um = [f for f in all_df_files if '_12_' in f]

surface_type_mapping = {
    0: 'water',
    1: 'snow-free land',
    2: 'snow-covered land',
    3: 'ice'
}

NADIR_MIN = 150
NADIR_MAX = 250

NADIR_CENTER = 204
NADIR_HALF_WIDTH = 50

NADIR_MIN = NADIR_CENTER - NADIR_HALF_WIDTH  # 154
NADIR_MAX = NADIR_CENTER + NADIR_HALF_WIDTH  # 254
MEAN_LW = 4
MEDIAN_LW = 2.5

# Minimum samples per beam to include in fit
MIN_SAMPLES = 30

# Polynomial orders to test
POLY_ORDERS = [2, 3, 4, 6]

# Surface types to process
SURFACE_TYPES = [
    "water",
    "ice",
    "snow_free_land",
    "snow_covered_land",
]

# Plot appearance
SURFACE_COLORS = {
    "water": "tab:blue",
    "ice": "tab:cyan",
    "snow-free land": "tab:brown",
    "snow-covered land": "tab:purple",
}

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

#%%
# Define fucntions
#---------------------------------
#---------------------------------
def plot_polynomial_diagnostics(
    df,
    season,
    surfaces,
    poly_orders=(2, 3, 4, 6),
    title_prefix=""
):
    """
    Plot raw medians + fitted polynomial curves
    for multiple surfaces and polynomial orders.
    """

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for ax, order in zip(axes, poly_orders):

        for surface in surfaces:
            stats = prepare_fit_stats(df, surface, season)
            if stats is None or len(stats) < order + 2:
                continue

            result = fit_polynomial(stats, order)
            if result is None:
                continue

            coeffs, x_fit, y_fit, rms = result

            color = SURFACE_COLORS.get(surface, "black")
            
            # -----------------------------
            # Raw medians (LIMB ONLY)
            # -----------------------------
            mask = (
                (stats["beam_position"] < NADIR_MIN) |
                (stats["beam_position"] > NADIR_MAX)
            )

            ax.plot(
                stats.loc[mask, "beam_position"] - NADIR_CENTER,
                stats.loc[mask, "median_corr"],
                "o",
                ms=3,
                alpha=0.35,
                color=color,
            )

            ax.axvspan(
            NADIR_MIN - NADIR_CENTER,
            NADIR_MAX - NADIR_CENTER,
            color="0.85",
            alpha=0.6,
            zorder=0
                )

            # fitted curve
            ax.plot(
                x_fit,
                y_fit,
                lw=2.6,
                color=color,
                label=f"{surface} (RMS={rms:.3f})"
            )

        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"Polynomial order {order}")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{title_prefix}{season}: Polynomial curve fitting diagnostics",
        fontsize=15, y=0.93
    )
    fig.supxlabel("Beam position (relative to nadir)")
    fig.supylabel("Correction coefficient")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
#---------------------------------
def fit_polynomial(stats_df, order):
    """
    Weighted polynomial fit to median correction coefficients,
    excluding nadir region from fit AND from plotted curve.

    Uses NumPy's classic polynomial convention:
    - np.polyfit (descending powers)
    - np.polyval (descending powers)

    Parameters
    ----------
    stats_df : DataFrame
        Must contain:
        - beam_position
        - median_corr
        - n_samples
    order : int
        Polynomial order

    Returns
    -------
    coeffs : ndarray
        Polynomial coefficients (descending powers)
    x_fit : ndarray
        Beam-centered x grid
    y_fit : ndarray
        Fitted curve
    rms : float
        RMS residual
    """

    # -----------------------------
    # Remove nadir beams FOR FITTING
    # -----------------------------
    fit_mask = (
        (stats_df["beam_position"] < NADIR_MIN) |
        (stats_df["beam_position"] > NADIR_MAX)
    )
    fit_df = stats_df.loc[fit_mask].copy()

    # Not enough limb beams → skip
    if len(fit_df) < order + 2:
        return None

    x = fit_df["beam_position"].values - NADIR_CENTER
    y = fit_df["median_corr"].values
    w = np.sqrt(fit_df["n_samples"].values)

    # Polynomial fit
    coeffs = np.polyfit(x, y, deg=order, w=w)
    coeffs = np.asarray(coeffs).ravel()

    # --------------------------------------------------
    # Evaluate polynomial on full beam range
    # but BREAK the curve at nadir using NaNs
    # --------------------------------------------------
    beam_eval = np.arange(
        stats_df["beam_position"].min(),
        stats_df["beam_position"].max() + 1
    )

    x_fit = beam_eval - NADIR_CENTER
    y_fit = np.polyval(coeffs, x_fit)

    # Insert NaNs in nadir region to prevent line connection
    nadir_mask = (beam_eval >= NADIR_MIN) & (beam_eval <= NADIR_MAX)
    y_fit[nadir_mask] = np.nan

    # RMS on fitted (limb-only) points
    y_hat = np.polyval(coeffs, x)
    rms = np.sqrt(np.mean((y - y_hat) ** 2))

    return coeffs, x_fit, y_fit, rms
#---------------------------------
def prepare_fit_stats(df, surface, season):
    """
    Prepare per-beam statistics for curve fitting.

    Uses MEDIAN correction coefficient to reduce noise.
    Explicitly removes nadir beams (SAFE for fitting).
    """

    sub = df[
        (df["surface_type_desc"] == surface) &
        (df["season"] == season)
    ]

    if sub.empty:
        return None

    stats = (
        sub.groupby("beam_position")["corr_coeff"]
           .agg(
               median_corr="median",
               n_samples="count"
           )
           .reset_index()
           .sort_values("beam_position")
    )

    # ✅ SAFE nadir removal (NO column duplication)
    mask = (
        (stats["beam_position"] >= NADIR_MIN) &
        (stats["beam_position"] <= NADIR_MAX)
    )

    stats = stats.loc[~mask].copy()

    return stats

#---------------------------------
def compute_stats(df, group_cols):
    """
    Compute mean and median correction coefficient for given grouping.

    Parameters
    ----------
    df : pandas.DataFrame
        Input LUT dataframe
    group_cols : list of str
        Columns to group by (must include 'beam_position')

    Returns
    -------
    pandas.DataFrame
        Aggregated dataframe with mean and median correction coefficients
    """
    return (
        df
        .groupby(group_cols, as_index=False)
        .agg(
            mean_corr=("corr_coeff", "mean"),
            median_corr=("corr_coeff", "median"),
        )
    )

#---------------------------------
# 🔹 GENERIC PLOTTING HELPER
def plot_mean_median(
    ax,
    beam,
    mean,
    median,
    color,
    label,
):
    """
    Plot mean (thick) and median (thin) correction coefficient.

    Mean  : thick solid line
    Median: thin dashed line

    Parameters
    ----------
    ax : matplotlib axis
    beam : array-like
        Beam positions
    mean : array-like
        Mean correction coefficient
    median : array-like
        Median correction coefficient
    color : str
        Line color
    label : str
        Base label (surface / land type)
    """
    ax.plot(
        beam, mean,
        color=color,
        lw=MEAN_LW,
        label=f"{label} (mean)"
    )

    ax.plot(
        beam, median,
        color=color,
        lw=MEDIAN_LW,
        ls="--",
        label=f"{label} (median)"
    )

#---------------------------------
# 🔹 COMMON AXIS DECORATION (important)
def decorate_axis(ax, season):
    """
    Apply common axis decorations:
    - nadir shading
    - labels
    - grid
    """
    ax.axvspan(
        NADIR_MIN,
        NADIR_MAX,
        color="gray",
        alpha=0.15,
        label="Nadir reference"
    )

    ax.set_title(f"{season}: Correction coefficient vs beam position")
    ax.set_xlabel("Beam position")
    ax.set_ylabel("Correction coefficient")
    ax.grid(alpha=0.3)

#---------------------------------
def summarize_by_beam(df):
    """
    Compute mean and median correction coefficient per beam position.
    """
    return (
        df.groupby("beam_position")["corr_coeff"]
          .agg(mean="mean", median="median")
          .reset_index()
    )
#-------------------------------
def mask_nadir(
    summary_df,
    nadir_min=154,
    nadir_max=254,
    mean_col="mean_corr",
    median_col="median_corr",
):
    """
    Mask nadir reference beams so they are not plotted.
    """
    mask = (
        (summary_df["beam_position"] >= nadir_min) &
        (summary_df["beam_position"] <= nadir_max)
    )

    summary_df = summary_df.copy()
    summary_df.loc[mask, [mean_col, median_col]] = np.nan

    return summary_df

#-------------------------------
# 🔹 Main plotting function (STAGE 1)
import matplotlib.pyplot as plt

def plot_geometry_by_season(df):
    """
    STAGE 1 — Geometry × Season
    Mean (thick) and median (thin dashed) correction coefficient vs beam position.
    """

    seasons = ["Autumn", "Spring", "Summer", "Winter"]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 7),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for ax, season in zip(axes, seasons):
        df_season = df[df["season"] == season]

        summary = summarize_by_beam(df_season)
        summary = mask_nadir(summary)

        # Mean: thick line
        ax.plot(
            summary["beam_position"],
            summary["mean"],
            color="black",
            linewidth=2.5,
            label="Mean"
        )

        # Median: thin dashed
        ax.plot(
            summary["beam_position"],
            summary["median"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Median"
        )

        # Shade nadir reference region
        ax.axvspan(
            NADIR_MIN, NADIR_MAX,
            color="0.9",
            alpha=0.8,
            zorder=0
        )

        ax.set_title(season)
        ax.grid(alpha=0.3)

    # Shared labels
    fig.supxlabel("Beam position", fontsize=15)
    fig.supylabel("Correction coefficient", 
                  x=0.002, fontsize=15,)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#-------------------------------
def plot_surface_season_panels(surface_type, stats_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, season in zip(axes, SEASONS):
        df_plot = stats_dict[(season, surface_type)]

        ax.plot(
            df_plot["beam_position"],
            df_plot["mean_corr"],
            color="black",
            lw=2.8,
            label="Mean"
        )

        ax.plot(
            df_plot["beam_position"],
            df_plot["median_corr"],
            color="black",
            lw=1.4,
            ls="--",
            label="Median"
        )

        ax.axvspan(NADIR_MIN, NADIR_MAX, color="gray", alpha=0.15)
        ax.set_title(season)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Correction coefficient")
    axes[2].set_ylabel("Correction coefficient")
    axes[2].set_xlabel("Beam position")
    axes[3].set_xlabel("Beam position")

    fig.suptitle(
        f"{surface_type.capitalize()} surface: Correction coefficient vs beam position",
        fontsize=14,
        y=0.98
    )

    fig.legend(
        ["Mean", "Median"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

#-------------------------------
def compute_surface_season_stats(df, surface):
    """
    Compute masked mean/median correction coefficients
    for a given surface type, split by season.
    """
    out = {}
    for season in ["Autumn", "Spring", "Summer", "Winter"]:
        sub = df[
            (df["surface_type_desc"] == surface) &   # ✅ FIX HERE
            (df["season"] == season)
        ]

        if sub.empty:
            out[season] = None
            continue

        stats = (
            compute_stats(sub, ["beam_position"])
            .sort_values("beam_position")
        )

        stats = mask_nadir(stats)
        out[season] = stats

    return out
#%%
dfs_list = []

for file in all_df_files_11um:
    # --- read file ---
    # df = pd.read_csv(file)
    df = pd.read_pickle(file)

    # --- land / water simplification ---
    df["land_type"] = df["surface_type"].map(
        lambda x: "Water" if (x == 0 or x == 3) else "Land"
    )

    df["surface_type_desc"] = df["surface_type"].map(surface_type_mapping)

    # --- extract season from filename ---
    # Example: temp_11_NH_Autumn_20250220.csv
    fname = os.path.basename(file)
    season = fname.split("_")[3]
    df["season"] = season

    dfs_list.append(df)

dfs = pd.concat(dfs_list, ignore_index=True)

df_53_61_nh = dfs[dfs['latitude_bin'] == '53-61'].copy()

df_53_61_sh = dfs[dfs['latitude_bin'] == '-61--53'].copy()


#%%===============================

# 🔹 STAGE 1 — GEOMETRY × SEASON
# ✔ Answers: Does limb behavior vary seasonally?
df = df_53_61_nh
plot_geometry_by_season(df)

#-------------------------------



#-------------------------------


# %%
df = df_53_61_nh
# ---------------------------------------
# GLOBAL reference: all seasons, all surfaces
# ---------------------------------------
stats_global = (
    compute_stats(df, ["beam_position"])
    .sort_values("beam_position")
)

stats_global = mask_nadir(stats_global)

stats_all_surfaces = {}

for season in ["Autumn", "Spring", "Summer", "Winter"]:
    stats_all_surfaces[season] = compute_stats(
        df[df["season"] == season],
        ["beam_position"]
    ).sort_values("beam_position")

stats_masked = {
    season: mask_nadir(df)
    for season, df in stats_all_surfaces.items()
}


SEASONS = ["Autumn", "Spring", "Summer", "Winter"]
SEASON_COLORS = {
    "Autumn": "#E69F00",
    "Spring": "#009E73",
    "Summer": "#D55E00",
    "Winter": "#0072B2",
}

NADIR_MIN = 154
NADIR_MAX = 254

GLOBAL_MEAN_LW   = 3.2
GLOBAL_MEDIAN_LW = 2.0

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 4.8),
    sharey=True
)

ax_mean, ax_median = axes

# ---------------------------
# Mean panel
# ---------------------------
for season in SEASONS:
    df = stats_masked[season]
    ax_mean.plot(
        df["beam_position"],
        df["mean_corr"],
        color=SEASON_COLORS[season],
        lw=2.8,
        label=season
    )

# Global mean (geometry-only reference)
ax_mean.plot(
    stats_global["beam_position"],
    stats_global["mean_corr"],
    color="black",
    lw=GLOBAL_MEAN_LW,
    label="Global mean"
)

ax_mean.set_title("Mean correction coefficient")
ax_mean.set_xlabel("Beam position")
ax_mean.set_ylabel("Correction coefficient")

# ---------------------------
# Median panel
# ---------------------------
for season in SEASONS:
    df = stats_masked[season]
    ax_median.plot(
        df["beam_position"],
        df["median_corr"],
        color=SEASON_COLORS[season],
        lw=1.6
    )

# Global median (geometry-only reference)
ax_median.plot(
    stats_global["beam_position"],
    stats_global["median_corr"],
    color="black",
    lw=GLOBAL_MEDIAN_LW,
    ls="--",
    label="Global median"
)

ax_median.set_title("Median correction coefficient")
ax_median.set_xlabel("Beam position")

# ---------------------------
# Shared decorations
# ---------------------------
for ax in axes:
    ax.axvspan(
        NADIR_MIN, NADIR_MAX,
        color="gray",
        alpha=0.15
    )
    ax.grid(alpha=0.3)

# ---------------------------
# Legend (global, clean)
# ---------------------------
handles, labels = ax_mean.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,   # was 4 → now 5
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

fig.subplots_adjust(top=0.85)
plt.show()

# stats_stage1 = compute_stats(
#     df,
#     group_cols=["season", "beam_position"]
# )

# for season in sorted(stats_stage1["season"].unique()):
#     fig, ax = plt.subplots(figsize=(13, 4))

#     s = (
#         stats_stage1
#         [stats_stage1["season"] == season]
#         .sort_values("beam_position")
#     )

#     plot_mean_median(
#         ax,
#         s["beam_position"],
#         s["mean_corr"],
#         s["median_corr"],
#         color="black",
#         label="All surfaces"
#     )

#     decorate_axis(ax, season)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

# ===============================

#%% ===============================

# 🔹 STAGE 2 — GEOMETRY × SURFACE TYPE × SEASON
# ✔ Answers: How does limb behavior vary by surface type and season?
# Does surface physics modulate limb correction differently across seasons?

# df = df_53_61_nh
# stats_surface = {}

# for season in SEASONS:
#     for sfc in df["surface_type_desc"].unique():
#         key = (season, sfc)
#         stats_surface[key] = (
#             compute_stats(
#                 df[
#                     (df["season"] == season) &
#                     (df["surface_type_desc"] == sfc)
#                 ],
#                 ["beam_position"]
#             )
#             .sort_values("beam_position")
#         )

# stats_surface_masked = {
#     k: mask_nadir(v.copy())
#     for k, v in stats_surface.items()
# }

# for sfc in ["water", "snow-free land", "snow-covered land", "ice"]:
#     plot_surface_season_panels(sfc, stats_surface_masked)


# ===============================
df = df_53_61_nh

# ---------------------------------------
# GLOBAL reference: all data, all conditions
# ---------------------------------------
# STAGE 2A — Water + Ice combined
stats_global = (
    compute_stats(df, ["beam_position"])
    .sort_values("beam_position")
)

# Mask nadir region (same logic as before)
stats_global = mask_nadir(stats_global)

SURFACE_COLORS = {
    "water": "#1f77b4",   # blue
    "ice":   "#d62728",   # red
}

MEAN_LW   = 2.8
MEDIAN_LW = 1.4
GLOBAL_MEAN_LW   = 3.2
GLOBAL_MEDIAN_LW = 2.0

stats_water = compute_surface_season_stats(df, "water")
stats_ice   = compute_surface_season_stats(df, "ice")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
axes = axes.flatten()

for ax, season in zip(axes, ["Autumn", "Spring", "Summer", "Winter"]):

    for surface, stats_dict in {
        "water": stats_water,
        "ice":   stats_ice
    }.items():

        stats = stats_dict.get(season)
        if stats is None:
            continue

        # Mean
        ax.plot(
            stats["beam_position"],
            stats["mean_corr"],
            color=SURFACE_COLORS[surface],
            lw=MEAN_LW,
            label=f"{surface.capitalize()} mean"
        )

        # Median
        ax.plot(
            stats["beam_position"],
            stats["median_corr"],
            color=SURFACE_COLORS[surface],
            lw=MEDIAN_LW,
            ls="--",
            label=f"{surface.capitalize()} median"
        )

        # ---------------------------------------
        # Global (all-conditions) reference
        # ---------------------------------------
        ax.plot(
            stats_global["beam_position"],
            stats_global["mean_corr"],
            color="black",
            lw=GLOBAL_MEAN_LW,
            label="Global mean"
        )

        ax.plot(
            stats_global["beam_position"],
            stats_global["median_corr"],
            color="black",
            lw=GLOBAL_MEDIAN_LW,
            ls="--",
            label="Global median"
        )

    ax.axvspan(NADIR_MIN, NADIR_MAX, color="gray", alpha=0.15)
    ax.set_title(season)
    ax.grid(alpha=0.3)

fig.suptitle("Water + Ice + Global: Correction coefficient vs beam position", y=0.93)
fig.supxlabel("Beam position")
fig.supylabel("Correction coefficient")

# Clean legend (deduplicated)
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="upper center",
    ncol=4,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


# ===============================
# STAGE 2B — Snow-free land + Snow-covered land
#---------------------------------------
# ===============================
df = df_53_61_sh

# ---------------------------------------
# GLOBAL reference: all data, all conditions
# ---------------------------------------
# STAGE 2A — Water + Ice combined
stats_global = (
    compute_stats(df, ["beam_position"])
    .sort_values("beam_position")
)

# Mask nadir region (same logic as before)
stats_global = mask_nadir(stats_global)

SURFACE_COLORS_LAND = {
    "snow_free_land":   "#8c564b",  # brown
    "snow_covered_land": "#9467bd", # purple
}
stats_snowfree  = compute_surface_season_stats(df, "snow-free land")
stats_snowcover = compute_surface_season_stats(df, "snow-covered land")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
axes = axes.flatten()

for ax, season in zip(axes, ["Autumn", "Spring", "Summer", "Winter"]):

    for surface, stats_dict in {
        "snow_free_land":   stats_snowfree,
        "snow_covered_land": stats_snowcover
    }.items():

        stats = stats_dict.get(season)
        if stats is None:
            continue

        ax.plot(
            stats["beam_position"],
            stats["mean_corr"],
            color=SURFACE_COLORS_LAND[surface],
            lw=MEAN_LW,
            label=f"{surface.replace('_',' ').title()} mean"
        )

        ax.plot(
            stats["beam_position"],
            stats["median_corr"],
            color=SURFACE_COLORS_LAND[surface],
            lw=MEDIAN_LW,
            ls="--",
            label=f"{surface.replace('_',' ').title()} median"
        )

        # ---------------------------------------
        # Global (all-conditions) reference
        # ---------------------------------------
        ax.plot(
            stats_global["beam_position"],
            stats_global["mean_corr"],
            color="black",
            lw=GLOBAL_MEAN_LW,
            label="Global mean"
        )

        ax.plot(
            stats_global["beam_position"],
            stats_global["median_corr"],
            color="black",
            lw=GLOBAL_MEDIAN_LW,
            ls="--",
            label="Global median"
        )

    ax.axvspan(NADIR_MIN, NADIR_MAX, color="gray", alpha=0.15)
    ax.set_title(season)
    ax.grid(alpha=0.3)

fig.suptitle("Snow-free vs Snow-covered land: Correction coefficient vs beam position", y=0.93)
fig.supxlabel("Beam position")
fig.supylabel("Correction coefficient")

# ---------------------------------------
# Collect legend entries from ALL axes
# ---------------------------------------
handles_all = []
labels_all = []

for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles_all.extend(h)
    labels_all.extend(l)

# Deduplicate while preserving order
by_label = dict(zip(labels_all, handles_all))

fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="upper center",
    ncol=3,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

#%% Curve fitting and diagnostics
df = df_53_61_nh

plot_polynomial_diagnostics(
    df,
    season="Winter",
    surfaces=[
        "water",
        "ice",
        "snow-free land",
        "snow-covered land",
    ],
    title_prefix="NH 53–61 | "
)