#%%
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.polynomial import Polynomial
import pickle
#%%
# floating variables
df_dir = r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Feb2025'
# r'/home/kkumah/Projects/AVHRR_IR-TB_correction/results/df/Jan2026'
# read all LUTs
all_lut_files = sorted([os.path.join(df_dir, s) for s in os.listdir(df_dir) if s.endswith('.pkl')])

# Define a function to initialize the nested dictionary


# Initialize a nested dictionary
all_lut = {} #defaultdict(nested_dict)

for file_path in all_lut_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    var = parts[0] + '_' + parts[1]
    hemisphere = parts[2]
    season = parts[3]
    # Initialize hierarchy explicitly
    if var not in all_lut:
        all_lut[var] = {}

    if hemisphere not in all_lut[var]:
        all_lut[var][hemisphere] = {}

    # fle_read = pd.read_csv(file_path,engine='python')
    fle_read = pd.read_pickle(file_path)
    all_lut[var][hemisphere][season] = fle_read

# Beam geometry
BEAM_MIN = 0
BEAM_MAX = 408
NADIR_CENTER = 204
NADIR_MIN = 154
NADIR_MAX = 254

DEFAULT_POLY_ORDER = 5

surface_type_mapping = {
    0: 'water',
    1: 'snow-free land',
    2: 'snow-covered land',
    3: 'ice'
}
#%% FUNCTIONS
def nested_dict():
    return defaultdict(dict)
def is_limb(beam):
    return (beam < NADIR_MIN) or (beam > NADIR_MAX)

# — Aggregate by beam
def aggregate_by_beam(df, agg="median"):
    """
    Aggregate correction coefficients per beam.
    """
    return (
        df.groupby("beam_position")["corr_coeff"]
        .agg(agg)
        .reset_index()
    )

#- — Fit polynomial curve
def fit_geometry_curve(stats_df, order=5):
    """
    Fit polynomial correction curve vs scan geometry.
    """
    beams = stats_df["beam_position"].to_numpy()
    corr  = stats_df["corr_coeff"].to_numpy()

    x = beams - NADIR_CENTER
    coeffs = np.polyfit(x, corr, order)

    poly = np.poly1d(coeffs)

    return {
        "coeffs": coeffs,
        "order": order,
        "poly": poly
    }

#— Build GLOBAL geometry (season-aware)
def build_global_geometry(df, agg="median", order=5):
    """
    Build surface-agnostic geometry correction
    for ONE (var, hemisphere, season).
    """
    global_geom = {}

    for lat_bin in df["latitude_bin"].unique():

        df_lat = df[df["latitude_bin"] == lat_bin]

        stats = aggregate_by_beam(df_lat, agg=agg)
        stats = stats[stats["beam_position"].apply(is_limb)]

        if stats.empty:
            continue

        global_geom[lat_bin] = fit_geometry_curve(stats, order=order)

    return global_geom

#— Build SURFACE-SPECIFIC CURVES
def build_surface_geometry(df, agg="median", order=5):
    """
    Build surface-specific geometry curves
    for ONE (var, hemisphere, season).
    """
    surface_geom = defaultdict(dict)

    for (lat_bin, surface), df_sub in df.groupby(
        ["latitude_bin", "surface_type"]
    ):
        stats = aggregate_by_beam(df_sub, agg=agg)
        stats = stats[stats["beam_position"].apply(is_limb)]

        if stats.empty:
            continue

        surface_geom[lat_bin][surface] = fit_geometry_curve(
            stats, order=order
        )

    return surface_geom

#— MASTER BUILDER (11 µm & 12 µm)
def build_all_geometry(all_lut, order=5, agg="median"):
    """
    Build GLOBAL and CURVE geometry libraries.
    """

    global_curve = defaultdict(lambda: defaultdict(dict))
    curve_lib    = defaultdict(lambda: defaultdict(dict))

    for var in all_lut:  # temp_11, temp_12
        for hemisphere in all_lut[var]:
            for season, df in all_lut[var][hemisphere].items():

                # GLOBAL
                global_curve[var][hemisphere][season] = \
                    build_global_geometry(df, agg=agg, order=order)

                # SURFACE CURVES
                curve_lib[var][hemisphere][season] = \
                    build_surface_geometry(df, agg=agg, order=order)

    return global_curve, curve_lib

#%% — RUN & SAVE TO DISK
global_curve, curve_lib = build_all_geometry(
    all_lut,
    order=5,
    agg="median"
)

# curve = curve_lib['temp_11']['NH']['Autumn']['61-75'][0]["poly"]
# corr  = curve(-200)

path_to_sve = r'/ra1/pubdat/AVHRR_CloudSat_proj/AVHRR/ir_correction_LUTs'
with open(os.path.join(path_to_sve, "global_geometry.pkl"), "wb") as f:
    pickle.dump(global_curve, f)

with open(os.path.join(path_to_sve, "surface_geometry.pkl"), "wb") as f:
    pickle.dump(curve_lib, f)


#%% Sanity check plots
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Inputs
# --------------------------------------------------
var = "temp_11"
hemisphere = "NH"
season = "Summer"
lat_bin = "61-75"
surface_code = 2  

N_BEAMS = 409
NADIR_CENTER = N_BEAMS // 2  # 204

# --------------------------------------------------
# Fetch polynomial
# --------------------------------------------------
poly = global_curve[var][hemisphere][season][lat_bin]["poly"]
# curve_lib[var][hemisphere][season][lat_bin][surface_code]["poly"]

# --------------------------------------------------
# Beam positions
# --------------------------------------------------
beam = np.arange(N_BEAMS)
x = beam - NADIR_CENTER   # geometry coordinate

# --------------------------------------------------
# Evaluate correction
# --------------------------------------------------
corr = poly(x)

NADIR_MIN = NADIR_CENTER - 50
NADIR_MAX = NADIR_CENTER + 50

corr_masked = corr.copy()
corr_masked[(beam >= NADIR_MIN) & (beam <= NADIR_MAX)] = np.nan

plt.figure(figsize=(9, 5))
plt.plot(beam, corr_masked, lw=2.5, color="black")

plt.axvspan(
    NADIR_MIN,
    NADIR_MAX,
    color="lightgray",
    alpha=0.4,
    label="Nadir excluded"
)

plt.axvline(NADIR_CENTER, color="gray", ls="--", lw=1)

plt.xlabel("Beam position")
plt.ylabel("Correction coefficient")
plt.title(f"{var} | {hemisphere} {season} | lat {lat_bin} | {surface_type_mapping[surface_code]} (masked)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()