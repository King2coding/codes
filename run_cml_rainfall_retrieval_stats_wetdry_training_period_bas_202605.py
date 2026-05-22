# run_cml_rainfall_retrieval_stats_wetdry_training_period_bas_202605.py
#
# Multi-day / training-period application script for the revised Ghana CML
# rainfall retrieval workflow prepared for Bas review.
#
# Purpose:
#   Generate CML-only rainfall reference files for the MSG + CML / CML-SAT
#   machine-learning training period using the statistics-based wet/dry-first
#   method. The generated NetCDF files can then be used as the rainfall target
#   for MSG wet/dry classification and wet-only rainfall-intensity training.
#
# This script reuses the Bas-review core functions without changing them.
# It loops over a user-defined date range and, for each output day, uses a
# preceding lookback window for dry-baseline estimation before gridding/saving
# the selected day.

from __future__ import annotations

import gc
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cml_rainfall_retrieval_stats_wetdry_core_bas_202605 import (
    R0AutoConfig,
    cml2metadata_coupling_framework_fast,
    clean_minmax_auto,
    build_15min_timeseries,
    stats_wetdry_Aobs,
    add_spatial_wetdry_support,
    rainlink_strict_R,
    plot_cml_retrieval_diagnostic,
    prepare_inputs_for_gridding,
    grid_rain_15min_rainlink_ok_full_ghana,
    save_15min_grid_and_points_netcdf_for_day,
    extract_datetime_from_filename,
    plot_cml_grid_diagnostics,
    plot_daily_mean_cml_rain_rate,
    plot_daily_equivalent_cml_rainfall,
)


# =============================================================================
# 0. User paths and run controls
# =============================================================================

metadata_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/outs")
metadata_file = metadata_dir / "matched_metadata_kkk_20250527.csv"

raw_cml_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/rsl")

# Use a dedicated output directory for the CML-only reference used by MSG training.
# This avoids overwriting older/API-test products.
output_dir = Path(r"/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref")
output_dir.mkdir(parents=True, exist_ok=True)

diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)

log_dir = output_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# MSG-CML training period. Adjust as needed.
training_start = "2025-06-14"
training_end   = "2025-08-24"

# Baseline/lookback history before each output day.
lookback_hours = 72

# Daily target timestamp used to select the lookback window.
# 23:59 includes all files on that output day up to the last available 15-min slot.
target_hour = 23
target_minute = 59

base_output_name = "ghana_cml_R_15min_bas_stats_ref"
version = "V1"

# If True, days with at least one existing output NetCDF are skipped.
skip_existing_days = True

# Make expensive diagnostics for the peak day/timestamp. Keep False for long runs.
make_daily_diagnostics = False

# Optional: set to a small integer for testing, e.g., 3.
max_days_to_process = None


# =============================================================================
# 1. Helper functions
# =============================================================================

def discover_raw_files(raw_dir: Path) -> list[tuple[datetime, str]]:
    file_names = sorted(os.listdir(raw_dir))
    file_datetimes = [(extract_datetime_from_filename(f), f) for f in file_names]
    file_datetimes = [(dt, f) for dt, f in file_datetimes if dt is not None]
    file_datetimes.sort(key=lambda x: x[0])
    return file_datetimes


def day_has_existing_outputs(out_dir: Path, base_name: str, day: pd.Timestamp) -> bool:
    ymd = day.strftime("%Y%m%d")
    # save_15min_grid_and_points_netcdf_for_day normally includes base_name/version/date/time.
    return any(out_dir.glob(f"{base_name}*{ymd}*.nc"))


def write_day_failure(log_path: Path, day_str: str, exc: BaseException) -> None:
    with open(log_path, "a") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"FAILED DAY: {day_str}\n")
        f.write(f"ERROR: {repr(exc)}\n")
        f.write(traceback.format_exc())
        f.write("\n")


def run_one_output_day(
    *,
    output_day: pd.Timestamp,
    matched_metadata: pd.DataFrame,
    file_datetimes: list[tuple[datetime, str]],
) -> list[str]:
    """
    Run the Bas/statistics wet-dry CML rainfall workflow for one output day.

    The retrieval uses the lookback window for dry-baseline estimation, but
    gridding and NetCDF export are restricted to output_day only.
    """

    latest_dt = datetime(
        int(output_day.year),
        int(output_day.month),
        int(output_day.day),
        target_hour,
        target_minute,
    )

    cutoff_time = latest_dt - timedelta(hours=lookback_hours)
    recent_files = [(dt, f) for dt, f in file_datetimes if cutoff_time <= dt <= latest_dt]
    recent_files.sort(key=lambda x: x[0])

    if not recent_files:
        raise RuntimeError("No raw CML files were found within the requested lookback window.")

    print("First file:", recent_files[0])
    print("Last file: ", recent_files[-1])
    print("Number of files:", len(recent_files))

    # -------------------------------------------------------------------------
    # 2. Couple raw CML files with metadata
    # -------------------------------------------------------------------------
    coupled_frames = []
    for idx, (_, fname) in enumerate(recent_files):
        if idx % 10 == 0:
            print(f"Coupled {idx} / {len(recent_files)} files")

        raw_file = raw_cml_dir / fname
        frame = cml2metadata_coupling_framework_fast(str(raw_file), matched_metadata)
        if frame is not None and not frame.empty:
            coupled_frames.append(frame)

    if not coupled_frames:
        raise RuntimeError("No raw CML files could be coupled to metadata.")

    df_raw = pd.concat(coupled_frames, ignore_index=True)
    print("Raw coupled rows:", len(df_raw))

    # -------------------------------------------------------------------------
    # 3. Clean signal and build strict 15-min link-time table
    # -------------------------------------------------------------------------
    cfg = R0AutoConfig(semantics="auto", regularize_grid=True)
    df_clean, df_summary = clean_minmax_auto(df_raw, cfg)
    print(df_summary.head(10))

    ts_15 = build_15min_timeseries(df_clean)
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Statistics-based wet/dry-first retrieval
    # -------------------------------------------------------------------------
    df_A, wetdry_thresholds = stats_wetdry_Aobs(
        ts_15,
        window_bins=10,
        std_percentile=95.0,
        baseline_win="48H",
        baseline_q=0.50,
        min_dry_bins=8,
        ffill_limit_bins=32,
        require_src_present=True,
    )

    print("Wet/dry rolling-std thresholds by frequency family:")
    for k, v in wetdry_thresholds.items():
        print(f"  {k}: {v:.4f} dB")

    df_A = add_spatial_wetdry_support(
        df_A,
        radius_km=25.0,
        min_neighbors=3,
        min_neighbor_wet_fraction=0.50,
        wet_col="wet_final",
    )

    df_rate = rainlink_strict_R(
        df_A,
        R_min=0.0,
        wet_col="wet_final",
        set_dry_to_zero=True,
    )

    link_rain_csv = output_dir / f"link_level_cml_rainfall_stats_wetdry_{output_day.strftime('%Y%m%d')}.csv"
    df_rate.to_csv(link_rain_csv, index=False)
    print("Saved link-level rainfall:", link_rain_csv)

    # -------------------------------------------------------------------------
    # 5. Prepare selected output day for gridding
    # -------------------------------------------------------------------------
    df_s5, meta_xy_grid = prepare_inputs_for_gridding(df_rate, ts_15)
    df_s5_day = df_s5[df_s5.index.date == output_day.date()]

    if df_s5_day.empty:
        raise RuntimeError(f"No link rainfall data are available for {output_day.date()} after retrieval.")

    # -------------------------------------------------------------------------
    # 6. Gridding and support/confidence layers
    # -------------------------------------------------------------------------
    grid_ds, diag = grid_rain_15min_rainlink_ok_full_ghana(
        df_s5=df_s5_day,
        df_meta_for_xy=meta_xy_grid,
        fixed_extent=(-3.5, 1.5, 4.5, 11.5),
        grid_res_deg=0.03,

        wet_thr=0.8,
        dry_thr=0.05,

        ok_model="exponential",
        ok_range_km=22.0,
        ok_nugget_frac=0.4,
        min_pts_ok=15,

        support_geometry="link_path",
        n_support_points_per_link=5,
        support_point_spacing_km=2.0,
        min_support_points_per_link=3,
        max_support_points_per_link=7,
        use_length_conditioned_support_points=True,

        support_k=2,
        support_radius_km=30.0,
        dry_radius_km=3.0,
        use_dry_constraint=True,

        use_soft_confidence=True,
        confidence_floor=0.03,
        confidence_power=1.1,
        confidence_dry_penalty_weight=0.50,

        drizzle_to_zero=0.10,
        outside_support_fill=np.nan,
        insufficient_training_fill=np.nan,

        smooth_kernel_px=1,
        smooth_fill_holes=False,

        clean_support=True,
        support_closing_iters=1,
        support_fill_holes=True,
        support_max_hole_px=20,

        make_display_field=True,
        display_smooth_kernel_px=3,
        display_smooth_fill_holes=False,
        apply_display_edge_taper=True,
        display_edge_taper_pixels=6,
        display_edge_taper_min_weight=0.05,

        make_coverage_quality=True,
        coverage_quality_med_thr=0.50,
        coverage_quality_high_thr=0.75,

        n_jobs=20,
        parallel_backend_name="processes",
        return_dataset=True,
    )

    print("Gridding counts:", diag.get("counts"))
    print("Coverage-quality counts:", diag.get("coverage_quality_counts"))
    print("Generated timestamps:", grid_ds.sizes.get("time"))

    # -------------------------------------------------------------------------
    # 6b. Optional diagnostics
    # -------------------------------------------------------------------------
    if make_daily_diagnostics and grid_ds.sizes.get("time", 0) > 0:
        rain_max_by_time = grid_ds["R_mm_per_h"].max(dim=("lat", "lon"), skipna=True)
        imax = int(rain_max_by_time.argmax(dim="time").values)
        t_peak = pd.to_datetime(grid_ds["time"].values[imax])

        plot_cml_grid_diagnostics(
            grid_ds,
            meta_xy_grid,
            t_peak,
            rainfall_vmin=0.0,
            rainfall_vmax=12.0,
            save_path=diagnostics_dir / f"grid_diagnostic_peak_{pd.Timestamp(t_peak).strftime('%Y%m%dT%H%M')}.png",
        )

        plot_daily_mean_cml_rain_rate(
            grid_ds,
            meta_xy_grid,
            day=output_day.date(),
            vmin=0.0,
            vmax=4.0,
            save_path=diagnostics_dir / f"daily_mean_rain_rate_{output_day.strftime('%Y%m%d')}.png",
        )

        plot_daily_equivalent_cml_rainfall(
            grid_ds,
            meta_xy_grid,
            day=output_day.date(),
            vmin=0.0,
            vmax=80.0,
            save_path=diagnostics_dir / f"daily_equiv_rainfall_mm_day_{output_day.strftime('%Y%m%d')}.png",
        )

    # -------------------------------------------------------------------------
    # 7. Save one NetCDF per 15-min timestamp for this output day
    # -------------------------------------------------------------------------
    written_files = save_15min_grid_and_points_netcdf_for_day(
        grid_data=grid_ds,
        df_s5=df_s5_day,
        meta_xy=meta_xy_grid,
        out_dir=str(output_dir),
        day=output_day.strftime("%Y-%m-%d"),
        base_name=base_output_name,
        version=version,
        include_display_field=False,
    )

    print(f"Saved {len(written_files)} NetCDF files")
    for f in written_files[:5]:
        print("  ", f)

    if written_files:
        with xr.open_dataset(written_files[0]) as ds_check:
            print("Variables:", list(ds_check.data_vars))
            print("R dtype:", ds_check["R_mm_per_h"].dtype)
            print("Support-confidence dtype:", ds_check["cml_support_confidence"].dtype)
            print("Support-mask values:", np.unique(ds_check["cml_support_mask"].values))
            print("Coverage-quality values:", np.unique(ds_check["cml_coverage_quality"].values))

    # Release memory before next day.
    del df_raw, df_clean, ts_15, df_A, df_rate, df_s5, df_s5_day, grid_ds
    gc.collect()

    return written_files


# =============================================================================
# 2. Main multi-day driver
# =============================================================================

if __name__ == "__main__":
    matched_metadata = pd.read_csv(metadata_file)

    file_datetimes = discover_raw_files(raw_cml_dir)
    if not file_datetimes:
        raise RuntimeError(f"No timestamped raw CML files were found in {raw_cml_dir}")

    days = pd.date_range(training_start, training_end, freq="D")
    if max_days_to_process is not None:
        days = days[:max_days_to_process]

    success_rows = []
    failure_log = log_dir / "failed_training_period_days.txt"

    print("=" * 80)
    print("CML-only Bas/statistics wet-dry reference generation")
    print("Training/reference period:", training_start, "to", training_end)
    print("Number of output days:", len(days))
    print("Output directory:", output_dir)
    print("=" * 80)

    for i, day in enumerate(days, start=1):
        day = pd.Timestamp(day)
        day_str = day.strftime("%Y-%m-%d")

        print("\n" + "#" * 80)
        print(f"Processing day {i}/{len(days)}: {day_str}")
        print("#" * 80)

        if skip_existing_days and day_has_existing_outputs(output_dir, base_output_name, day):
            print(f"Skipping {day_str}: existing NetCDF outputs found.")
            success_rows.append({"day": day_str, "status": "skipped_existing", "n_files": np.nan})
            continue

        try:
            written_files = run_one_output_day(
                output_day=day,
                matched_metadata=matched_metadata,
                file_datetimes=file_datetimes,
            )
            success_rows.append({"day": day_str, "status": "success", "n_files": len(written_files)})

        except Exception as exc:
            print(f"FAILED {day_str}: {exc}")
            write_day_failure(failure_log, day_str, exc)
            success_rows.append({"day": day_str, "status": "failed", "n_files": 0})
            continue

    summary = pd.DataFrame(success_rows)
    summary_file = log_dir / f"training_period_generation_summary_{training_start.replace('-', '')}_{training_end.replace('-', '')}.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("Finished training-period reference generation")
    print(summary["status"].value_counts(dropna=False))
    print("Summary saved:", summary_file)
    print("=" * 80)
