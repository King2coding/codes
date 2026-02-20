#%%
import gc
from CML_Rainfall_Retrieval_Pipeline_modes import *

#%% The Preprocessing and Cleaning Pipeline
# 0) Load your linked dataframe (must contain ID, DateTime, Pmin, Pmax)
df_raw = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20251006.csv')  # or CSV
path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

# 1) Run R0 cleaner (with the new defaults)
cfg = R0AutoConfig(
    semantics="auto",          # let it decide per link
    regularize_grid=True,      # keep exact 15-min grid
)

df_clean, df_sum = clean_minmax_auto(df_raw, cfg)
print(df_sum.head(10))
gc.collect()

#%% The CML-Rainfall-Retrieval Pipeline

# 1) strict 15-min series with Rainlink-style RSL
ts_15 = build_15min_timeseries(df_clean)

# 2) strict past-only baseline and observed attenuation
dfA = rainlink_strict_Aobs(ts_15, wet_thr_db=0.5)

# 3) Leijnse WA + ITU(2005) k–α → *allow true zeros*
df_rate = rainlink_strict_R(dfA, R_min=0.0)

# 4) Prepare data for gridding
df_s5, meta_xy_grid = prepare_inputs_for_gridding(df_rate, ts_15)
gc.collect()
#%% The Gridding Pipeline
df_s5_20250619 = df_s5[df_s5.index.date ==  pd.to_datetime("2025-06-19").date()]
R_da_rl, diag_rl = grid_rain_15min_rainlink_ok(
    df_s5_20250619, 
    meta_xy_grid,
    grid_res_deg=0.03, 
    domain_pad_deg=0.20,
    wet_thr=0.8, 
    dry_thr=0.05,
    ok_model="exponential", 
    ok_range_km=25.0, 
    ok_nugget_frac=0.45,
    min_pts_ok=15, 
    support_k=3, 
    support_radius_km=25.0,
    drizzle_to_zero=0.15,
    n_jobs=20, 
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
    smooth_kernel_px=3,
    smooth_fill_holes=True,
)
print(diag_rl)

#%%
t = pd.Timestamp("2025-06-19 16:15:00")

meta_xy = (
    df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
    .drop_duplicates("ID")
)

R1, d1 = grid_rain_at_time_rainlink(
    df_s5=df_s5[["ID", "R_mm_per_h"]],
    df_meta_for_xy=meta_xy,
    t=t,                      # naive UTC timestamp present in df_s5.index

    # RainLINK-OK parameters:
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    wet_thr=0.5,
    dry_thr=0.0,
    ok_model="exponential",
    ok_range_km=50.0,
    ok_nugget_frac=0.45,
    min_pts_ok=15,
    support_k=3,
    support_radius_km=25.0,
    drizzle_to_zero=0.3,     # you can change from default 0.10 if you like
    n_jobs=20,                 # or >1 if you want parallel
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
    smooth_kernel_px=3,
    smooth_fill_holes=True,
)

print(d1)


#%% Save slices
from pipeline_modes import save_each_time_to_netcdf

out_paths = save_each_time_to_netcdf(
    R_da_rl,
    out_dir="/home/kkumah/Projects/cml-stuff/out_cml_rain_dir",
    base_name="ghana_cml_R",
    engine="netcdf4",
    complevel=5,
    dtype="float32",
    fill_value=-9999.0,     # or np.nan if you prefer
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True
)
print(f"Wrote {len(out_paths)} files. First:\n", out_paths[:3])