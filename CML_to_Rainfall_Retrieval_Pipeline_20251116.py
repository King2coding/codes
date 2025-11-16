# CML_to_Rainfall_Retrieval_Pipeline_20251116.py
# R0 → S1 (Δ-based NLA) → S2 (48h baseline) → S1b (excess NLA)
# → S2b (integrate masks) → WA v2 → k–α → df_s5 → OK+IDW gridding

# %%
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates

from My_program_utils import *

from r0_clean_minmax_auto import clean_minmax_auto, R0AutoConfig
from r0_qc_helpers import masking_report, plot_flags, plot_minmax_basic
from persist_utils import save_r0_outputs

# ---------------------------------------------------------------------------
# 0) Load raw linked data (must contain ID, DateTime, Pmin, Pmax, X/YStart/End)
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(
    r"/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20250929.csv"
)
path_to_put_output = r"/home/kkumah/Projects/cml-stuff/data-cml/outs"

# ---------------------------------------------------------------------------
# 1) R0 cleaner (unified semantics + 15-min grid)
# ---------------------------------------------------------------------------
cfg_r0 = R0AutoConfig(
    semantics="auto",      # auto-detect min/max semantics per link
    regularize_grid=True,  # enforce exact 15-min grid
)

df_clean, df_sum = clean_minmax_auto(df_raw, cfg_r0)
print("R0 summary:")
print(df_sum.head(10))

# QA & sanity checks
rep = masking_report(df_clean)
print("Masking report:")
print(rep.head(15))

# One link quick-look
link0 = df_sum.loc[0, "ID"]
one_link = df_clean[df_clean["ID"] == link0]
plot_minmax_basic(one_link)
plot_flags(df_clean, link0)

# Save R0 outputs
paths_r0 = save_r0_outputs(df_clean, df_sum, cfg_r0, out_dir="outputs")
print("R0 saved:", paths_r0)

# %%
# =========================
# Helpers for S1–S5 pipeline
# =========================

def _utcify_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def _merge_on_id_time(left: pd.DataFrame, right: pd.DataFrame, cols, how="left"):
    """
    Robust merge on (ID, time). Works even if 'right' has duplicate (ID,time).
    Keeps the *last* occurrence per (ID,time) on the right.
    """
    L = _utcify_index(left).reset_index().rename(columns={"index": "time_utc"})
    R = _utcify_index(right).reset_index().rename(columns={"index": "time_utc"})
    R = R[["ID", "time_utc", *cols]].drop_duplicates(["ID", "time_utc"], keep="last")
    out = (
        L.merge(R, on=["ID", "time_utc"], how=how)
         .set_index("time_utc")
         .sort_index()
    )
    return out


# ONE knob for NLA aggressiveness
WETDRY_MODE = "strict"   # "strict" (Rainlink-like) or "relaxed"

# %%
# =====================
# Step 1: Δ-based NLA
# =====================
import importlib
import step1_wetdry_nla_fast2 as s1
import step1b_wetdry_excess_fast as s1b
importlib.reload(s1)
importlib.reload(s1b)

# IMPORTANT: feed R0 output (has 'ID'), not ts_15
df_nla, s1_sum = s1.wetdry_classify_with_mode(
    _utcify_index(df_clean),
    mode=WETDRY_MODE,
    override={"n_jobs": 12},
)
df_nla = _utcify_index(df_nla)
print("Step 1 (Δ-based NLA) done. Columns:", df_nla.columns.tolist())

# %%
# ============================
# Step 2: 48h dry-only baseline
# ============================
import importlib, step2_baseline_dry48
importlib.reload(step2_baseline_dry48)
from step2_baseline_dry48 import compute_dry_baseline_48h, Baseline48Config

# Use only arguments that Baseline48Config actually supports
cfg2 = Baseline48Config(
    window_hours=48,
    fallback_hours=72,
    min_dry_samples=8,          # or 12, if you prefer stricter baseline
    smooth_baseline_samples=0,
)

df_step2, s2_sum = compute_dry_baseline_48h(df_nla, cfg2)
df_step2 = _utcify_index(df_step2)
print(s2_sum.head())
# %%
# ===============================
# Step 1b: Excess-based NLA (A_ex)
# ===============================
df_ex, s1b_sum = s1b.wetdry_from_excess_with_mode(
    df_step2,
    mode=WETDRY_MODE,
    override={"n_jobs": 12},
)
df_ex = _utcify_index(df_ex)
print("Step 1b (excess NLA) done. Columns:", df_ex.columns.tolist())

# %%
# ===============================
# Step 2b: Integrate masks → is_wet_final
# ===============================
import step2b_integrate_masks as s2b
importlib.reload(s2b)
from step2b_integrate_masks import integrate_wetdry_and_excess

df_s12 = integrate_wetdry_and_excess(
    df_step2,
    df_nla,
    df_ex,
    onset_allow_if_excess_pos=(WETDRY_MODE != "strict"),  # strict: OFF, relaxed: ON
)
df_s12 = _utcify_index(df_s12)
assert "is_wet_final" in df_s12.columns
print("Step 2b (integrated wet mask) done. Columns:", df_s12.columns.tolist())

# Attach metadata + excess-per-km if needed
need_meta = ["PathLength", "Frequency", "Polarization"]
if not set(need_meta).issubset(df_s12.columns):
    df_s12 = _merge_on_id_time(df_s12, df_step2, need_meta)

if "A_ex_pool_per_km" not in df_s12.columns and "A_ex_pool_per_km" in df_ex.columns:
    df_s12 = _merge_on_id_time(df_s12, df_ex, ["A_ex_pool_per_km"])

# pick WA source automatically
wa_src = "A_ex_pool_per_km" if "A_ex_pool_per_km" in df_s12.columns else (
         "A_excess_db_per_km" if "A_excess_db_per_km" in df_s12.columns else None)
if wa_src is None:
    raise ValueError(
        "No per-km excess column found. Need 'A_ex_pool_per_km' (preferred) "
        "or 'A_excess_db_per_km'."
    )

print("Wet-antenna γ source:", wa_src)

# %%
# ================================
# Step 2c: Wet-antenna V2 (decay)
# ================================
import step2b_wet_antenna as wa
importlib.reload(wa)
from step2b_wet_antenna import WAConfigV2, apply_wet_antenna_decay

cfg_wa = WAConfigV2(
    gamma_source_col=wa_src,        # from above
    wet_mask_col="is_wet_final",
    wa_db_per_terminal=0.08,
    assume_wet_terminals=2,
    dt_minutes=15,
    tau_on_min=30,
    tau_off_min=60,
    out_raw_col="gamma_raw_db_per_km",
    out_corr_col="gamma_corr_db_per_km",
    floor_zero=True,
)

df_attn = apply_wet_antenna_decay(df_s12, cfg_wa)
df_attn = _utcify_index(df_attn)
print("Step 2c (WA V2) done. Columns:", df_attn.columns.tolist())

# %%
# =============================
# Step 3: k–α (γ → R, gated)
# =============================
import step3_kalpha as s3
importlib.reload(s3)
from step3_kalpha import KAlphaConfigV2, gamma_to_r_gated

# 'lut' must exist in your environment (ITU coeffs table)
cfg_k = KAlphaConfigV2(
    lut=lut,                        # DataFrame with k, α by freq/pol
    pol_col="Polarization",
    freq_col="Frequency",
    gamma_col="gamma_corr_db_per_km",
    gamma_gate_db_per_km=0.02,
    use_wet_mask_col="is_wet_final",
    r_cap_mmph_by_band={6: 60, 8: 80, 19: 120},
)

df_s5, s5_sum = gamma_to_r_gated(df_attn, cfg_k)
df_s5 = _utcify_index(df_s5)
# make index tz-naive UTC for gridding
df_s5.index = df_s5.index.tz_convert("UTC").tz_localize(None)
df_s5.index.name = "time"

print("Step 3 (k–α) done. df_s5 columns:", df_s5.columns.tolist())
print("df_s5 time range:", df_s5.index.min(), "→", df_s5.index.max())

# Quick per-link QA plot (optional)
def _to_utc(ts):
    if ts is None:
        return None
    t = pd.to_datetime(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def plot_rainrate(df_s5, link_id, t0=None, t1=None):
    import matplotlib.pyplot as plt

    d = df_s5[df_s5["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id)
        return

    # Ensure UTC index
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")

    t0u, t1u = _to_utc(t0), _to_utc(t1)
    if t0u is not None:
        d = d[d.index >= t0u]
    if t1u is not None:
        d = d[d.index <= t1u]
    if d.empty:
        print("No rows in requested window.")
        return

    gcol = "gamma_corr_db_per_km" if "gamma_corr_db_per_km" in d.columns else "gamma_raw_db_per_km"

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(d.index, d[gcol], label=gcol)
    ax[0].set_ylabel("γ [dB/km]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(d.index, d["R_mm_per_h"], label="R [mm/h]")
    ax[1].set_ylabel("R [mm/h]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"γ → R — {link_id}")
    plt.tight_layout()
    plt.show()


# Example QA for first link
plot_rainrate(df_s5, df_sum.loc[0, "ID"], t0="2025-06-12", t1="2025-06-14")

# %%
# =========================
# Gridding: OK → IDW combo
# =========================
from sklearn.neighbors import BallTree
from plot_helpers import plot_slice_cartopy_with_links
import step6_grid_ok_pcm as s6pcm
importlib.reload(s6pcm)

# Meta midpoints (one row per ID) from R0 output
meta_xy_grid = (
    df_clean.reset_index()[["ID", "XStart", "YStart", "XEnd", "YEnd"]]
    .drop_duplicates("ID")
)

for c in ["XStart", "YStart", "XEnd", "YEnd"]:
    meta_xy_grid[c] = pd.to_numeric(meta_xy_grid[c], errors="coerce")

# Sanity: all IDs in df_s5 must exist in meta_xy_grid
missing = set(df_s5["ID"].unique()) - set(meta_xy_grid["ID"].unique())
print("IDs missing from meta:", len(missing))
if missing:
    print("Example missing:", list(sorted(missing))[:5])
    raise RuntimeError("Fix meta/id mismatch before gridding.")

# Main gridding call using consensus df_s5
R_da, diag_rl = s6pcm.grid_rain_15min_rainlink_ok(
    df_s5=df_s5[["ID", "R_mm_per_h"]],
    df_meta_for_xy=meta_xy_grid,
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    wet_thr=0.8,
    dry_thr=0.05,
    ok_model="exponential",
    ok_range_km=25.0,
    ok_nugget_frac=0.45,
    min_pts_ok=15,
    support_k=4,              # stricter spatial support
    support_radius_km=25.0,
    drizzle_to_zero=0.30,     # kill sub-0.3 mm/h drizzle
    n_jobs=15,
    parallel_backend_name="processes",
)
print("Gridding diagnostics:")
print(diag_rl)

# Choose a time to plot (peak rain)
mx = []
for t in R_da["time"].values:
    a = R_da.sel(time=np.datetime64(t), method="nearest").values
    mx.append(np.nanmax(a))

imax = int(np.nanargmax(mx))
t_peak = pd.Timestamp(R_da["time"][imax].values)
print("Peak rain at:", t_peak, "max =", mx[imax])

# Plot that slice
plot_slice_cartopy_with_links(
    R_da,
    meta_xy_grid,
    t=t_peak,
    vmin=1.0,
    vmax=15.0,
    nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    cbar_side="right",
    cbar_size="4%",
    cbar_pad=0.05,
)

# %%
# ===============================
# Save each time slice to NetCDF
# ===============================
from pipeline_modes import save_each_time_to_netcdf

out_paths = save_each_time_to_netcdf(
    R_da,
    out_dir="/home/kkumah/Projects/cml-stuff/out_cml_rain_dir",
    base_name="ghana_cml_R_consensus",
    engine="netcdf4",
    complevel=5,
    dtype="float32",
    fill_value=-9999.0,
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True,
)
print(f"Wrote {len(out_paths)} files. First:\n", out_paths[:3])