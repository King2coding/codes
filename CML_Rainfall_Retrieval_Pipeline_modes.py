# r0_clean_minmax_auto.py
# R0 — Prior cleaning for 15-min Pmin/Pmax with per-link semantics (RSL dBm vs TL dB)
# Upgrades:
#   - Robust semantics detection using in-range counts
#   - Auto-retry with flipped semantics if first pass nukes a link
#   - src_present flag for QA (distinguish true samples from regularized grid)
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict
import os
import numpy as np
import pandas as pd

from typing import Optional

from pycomlink.processing.wet_antenna import waa_leijnse_2008_from_A_obs
from pycomlink.processing.k_R_relation import calc_R_from_A

import xarray as xr

from joblib import Parallel, delayed, parallel_backend
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter, binary_opening, generate_binary_structure

try:
    from pykrige.ok import OrdinaryKriging
    _PYKRIGE_AVAILABLE = True
except Exception:
    _PYKRIGE_AVAILABLE = False

# For RainLINK-like support mask
try:
    from sklearn.neighbors import BallTree
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')
#%%

_EPS = 1e-12
_KM_PER_DEG = 111.0
_EARTH_R_KM = 6371.0
#-------------------------------------------------------------------
# R0AutoConfig
@dataclass(frozen=True)
class R0AutoConfig:
    # Time
    cadence_minutes: int = 15
    snap_tolerance: str = "2min"        # snap to nearest 15-min if within this tolerance
    regularize_grid: bool = True        # reindex to exact 15-min grid (gaps -> NaN)
    source_tz: str = "Africa/Accra"     # Ghana is UTC year-round

    # Bounds for RSL (dBm) and TL (dB)
    rsl_min_dbm: float = -130.0
    rsl_max_dbm: float = -20.0
    tl_min_db: float = 0.0
    tl_max_db: float = 80.0

    # Within-bin dynamic range (15-min data)
    max_dyn_range_db: float = 12.0

    # Outage heuristics (semantics-aware)
    rsl_outage_floor_dbm: float = -115.0
    tl_outage_high_db: float = 75.0
    outage_min_consec: int = 2

    # Hampel (spike) on Pbar = (Pmin+Pmax)/2  [relaxed]
    hampel_window: int = 7
    hampel_nsigma: float = 6.0

    # Plateau (flag-only)  [relaxed]
    plateau_run_len: int = 20           # 5 h
    plateau_tol_db: float = 0.05

    # Unpaired-jump rule
    unpaired_spread_db: float = 6.0
    unpaired_delta_db: float = 2.0

    # Semantics control
    semantics: str = "auto"             # "auto" | "rsl" | "tl"
    # Auto-retry trigger
    retry_oob_frac: float = 0.50        # if >50% OOB …
    retry_valid_frac: float = 0.05      # … and <5% valid pairs → flip semantics & retry


# ---------- helpers ----------
def _parse_dt(series: pd.Series, cfg: R0AutoConfig) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(cfg.source_tz, nonexistent="shift_forward", ambiguous="NaT")
    return dt.dt.tz_convert("UTC")
#-------------------------------------------------------------------


def _snap_to_grid(dt: pd.Series, minutes: int, tol: pd.Timedelta) -> pd.Series:
    base = dt.dt.floor(f"{minutes}min")
    offs = (dt - base)
    up = offs >= pd.Timedelta(minutes=minutes/2)
    anchor = base.where(~up, base + pd.Timedelta(minutes=minutes))
    diff = (dt - anchor).abs()
    return anchor.where(diff <= tol)  # otherwise NaT

#-------------------------------------------------------------------

def _hampel_mask(x: pd.Series, window: int, nsigma: float) -> pd.Series:
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    return ((x - med).abs() > nsigma * sigma).fillna(False)

#-------------------------------------------------------------------

def _flag_plateaus(x: pd.Series, run_len: int, tol_db: float) -> pd.Series:
    if x.isna().all():
        return pd.Series(False, index=x.index)
    d = x.diff().abs().fillna(0.0) <= tol_db
    gid = (~d).cumsum()
    counts = pd.Series(gid).map(pd.Series(gid).value_counts())
    return (d & (counts.values >= run_len)).reindex_like(x).fillna(False)

#-------------------------------------------------------------------

def _consec_true(mask: pd.Series, min_len: int) -> pd.Series:
    if mask.empty:
        return mask
    gid = (mask != mask.shift(1, fill_value=False)).cumsum()
    run_len = gid.map(gid.value_counts())
    return mask & (run_len >= min_len)
#-------------------------------------------------------------------


def _detect_semantics_robust(pmin: pd.Series, pmax: pd.Series, cfg: R0AutoConfig) -> str:
    """Decide RSL vs TL by counting finite samples that fall inside each range."""
    vals = pd.concat([pmin, pmax], axis=0).dropna().values
    if vals.size == 0:
        return "rsl"
    in_rsl = ((vals >= cfg.rsl_min_dbm) & (vals <= cfg.rsl_max_dbm)).sum()
    in_tl  = ((vals >= cfg.tl_min_db)    & (vals <= cfg.tl_max_db)).sum()
    if in_rsl > in_tl:
        return "rsl"
    if in_tl > in_rsl:
        return "tl"
    # tie-breaker: sign of median
    return "rsl" if np.nanmedian(vals) < 0 else "tl"

# ---------- single-link cleaning ----------
def _clean_one_link(g: pd.DataFrame, semantics: str, cfg: R0AutoConfig) -> Tuple[pd.DataFrame, Dict]:
    """Return cleaned df (after reindex) + metrics computed on src_present only."""
    s = g.copy()

    # Flags container
    flags = {k: np.zeros(len(s), dtype=bool) for k in
             ["OOB", "SWAP", "DYN", "OUTAGE", "UNPAIRED", "SPIKE", "PLAT", "SEM_RSL", "SEM_TL"]}
    flags["SEM_RSL"][:] = (semantics == "rsl")
    flags["SEM_TL"][:] = (semantics == "tl")

    # Bounds by semantics
    if semantics == "rsl":
        bad_min = ~s["Pmin"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
        bad_max = ~s["Pmax"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
    else:
        bad_min = ~s["Pmin"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
        bad_max = ~s["Pmax"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
    oob = bad_min | bad_max
    flags["OOB"] = oob.values
    s.loc[oob, ["Pmin", "Pmax"]] = np.nan

    # Order: Pmax >= Pmin
    need_swap = (s["Pmin"].notna() & s["Pmax"].notna() & (s["Pmin"] > s["Pmax"]))
    s.loc[need_swap, ["Pmin", "Pmax"]] = s.loc[need_swap, ["Pmax", "Pmin"]].values
    flags["SWAP"] = need_swap.values

    # Dynamic range
    spread = s["Pmax"] - s["Pmin"]
    dyn_bad = (spread > cfg.max_dyn_range_db)
    flags["DYN"] = dyn_bad.fillna(False).values
    s.loc[dyn_bad, ["Pmin", "Pmax"]] = np.nan

    # Outage (semantics-aware)
    if semantics == "rsl":
        low_min = s["Pmin"] <= cfg.rsl_outage_floor_dbm
        low_max = s["Pmax"] <= cfg.rsl_outage_floor_dbm
        outage = _consec_true(low_min | low_max, cfg.outage_min_consec)
    else:
        high_min = s["Pmin"] >= cfg.tl_outage_high_db
        high_max = s["Pmax"] >= cfg.tl_outage_high_db
        outage = _consec_true(high_min | high_max, cfg.outage_min_consec)
    flags["OUTAGE"] = outage.values
    s.loc[outage, ["Pmin", "Pmax"]] = np.nan

    # Unpaired jump
    delta_min = s["Pmin"].diff().abs()
    delta_max = s["Pmax"].diff().abs()
    unpaired = (
        (spread > cfg.unpaired_spread_db) &
        (
            ((delta_min > cfg.unpaired_delta_db) & (delta_max <= cfg.unpaired_delta_db)) |
            ((delta_max > cfg.unpaired_delta_db) & (delta_min <= cfg.unpaired_delta_db))
        )
    )
    flags["UNPAIRED"] = unpaired.fillna(False).values
    s.loc[unpaired, ["Pmin", "Pmax"]] = np.nan

    # Hampel spikes
    pbar = (s["Pmin"] + s["Pmax"]) / 2.0
    spike = _hampel_mask(pbar, cfg.hampel_window, cfg.hampel_nsigma)
    flags["SPIKE"] = spike.values
    s.loc[spike, ["Pmin", "Pmax"]] = np.nan

    # Plateau (flag-only)
    pbar_after = (s["Pmin"] + s["Pmax"]) / 2.0
    plat = _flag_plateaus(pbar_after, cfg.plateau_run_len, cfg.plateau_tol_db)
    flags["PLAT"] = plat.values

    # Record source presence BEFORE reindex
    src_present = pd.Series(True, index=s.index, name="src_present")

    # Regularize to exact 15-min grid
    if cfg.regularize_grid and not s.empty:
        full_idx = pd.date_range(s.index.min(), s.index.max(),
                                 freq=f"{cfg.cadence_minutes}min", tz="UTC")
        s = s.reindex(full_idx)

    # Build QC string on the original index, then reindex
    def pack(i: int) -> str:
        labs = [k for k, v in flags.items() if i < len(v) and v[i]]
        return ",".join(labs) if labs else ""

    qc_series = pd.Series([pack(i) for i in range(len(g))], index=g.index)
    s["qc_flags"] = qc_series.reindex(s.index).fillna("")
    s["src_present"] = src_present.reindex(s.index).fillna(False)

    # Convenience fields
    # ------------------------------------------------------------
    # Convenience fields (semantics-aware signal definition)
    # ------------------------------------------------------------
    # Pbar is the mid-point of min/max (either RSL [dBm] or TL [dB])
    s["Pbar"] = (s["Pmin"] + s["Pmax"]) / 2.0
    s["Pspread"] = s["Pmax"] - s["Pmin"]
    s["semantics"] = semantics

    # IMPORTANT:
    # We define a single "signal" time series to baseline against later.
    # - If semantics == "rsl": Pbar is already RSL in dBm (higher = stronger signal).
    # - If semantics == "tl" : Pbar is TL in dB (higher = more loss), so we flip sign.
    #   This makes sig_db behave like received power (higher = stronger signal),
    #   so baseline logic (baseline - signal) produces positive attenuation-like values.
    s["sig_db"] = np.where(s["semantics"] == "rsl", s["Pbar"], -s["Pbar"])

    # Metrics (only over true source samples)
    src = s[s["src_present"]]
    metrics = {
        "n_src": int(src.shape[0]),
        "n_valid_pairs_src": int(src[["Pmin", "Pmax"]].dropna().shape[0]),
        "frac_oob_src": float(src["qc_flags"].str.contains("OOB").mean()) if len(src) else 0.0,
        "frac_valid_src": float(src[["Pmin", "Pmax"]].dropna().shape[0] / max(len(src), 1))
    }
    return s, metrics


# ---------- public API ----------
def clean_minmax_auto(df_raw: pd.DataFrame, cfg: R0AutoConfig = R0AutoConfig()
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean 15-min Pmin/Pmax per link (ID) with robust semantics detection + auto-retry.

    Required columns: ['ID','DateTime','Pmin','Pmax']
    Other columns are passed through untouched.

    Returns
    -------
    df_out : per-timestamp rows with cleaned Pmin/Pmax, Pbar, Pspread, Abar, qc_flags, semantics, src_present
    df_summary : per-ID summary stats (computed on src_present only)
    """
    need = ["ID", "DateTime", "Pmin", "Pmax"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw.copy()

    # Parse & snap to 15-min
    dt = _parse_dt(df["DateTime"], cfg)
    snap = _snap_to_grid(dt, cfg.cadence_minutes, pd.Timedelta(cfg.snap_tolerance))
    df["DateTimeUTC"] = snap
    df = df.dropna(subset=["DateTimeUTC"]).sort_values(["ID", "DateTimeUTC"])
    df = df.drop_duplicates(subset=["ID", "DateTimeUTC"], keep="first")
    df.set_index("DateTimeUTC", inplace=True)
    # Keep a clean, human-readable time stamp column for debugging/exports
    # (prevents scientific notation like 2.025061e+11 in df_clean.head()).
    df["DateTime_str"] = df.index.strftime("%Y%m%d%H%M")

    # Numeric Pmin/Pmax
    df["Pmin"] = pd.to_numeric(df["Pmin"], errors="coerce")
    df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")

    out_frames: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for link_id, g in df.groupby("ID", sort=False):
        # Decide semantics
        sem0 = cfg.semantics if cfg.semantics in ("rsl", "tl") else _detect_semantics_robust(g["Pmin"], g["Pmax"], cfg)

        # First pass
        s0, m0 = _clean_one_link(g, sem0, cfg)

        # Retry logic
        use_retry = (m0["frac_oob_src"] > cfg.retry_oob_frac) and (m0["frac_valid_src"] < cfg.retry_valid_frac)
        if use_retry:
            sem1 = "tl" if sem0 == "rsl" else "rsl"
            s1, m1 = _clean_one_link(g, sem1, cfg)
            # Keep the better (more valid) result
            if m1["n_valid_pairs_src"] > m0["n_valid_pairs_src"]:
                s_keep, m_keep, sem_keep = s1, m1, sem1
            else:
                s_keep, m_keep, sem_keep = s0, m0, sem0
        else:
            s_keep, m_keep, sem_keep = s0, m0, sem0

        # Attach ID
        s_keep["ID"] = link_id

        # Summary (src domain)
        summaries.append({
            "ID": link_id,
            "semantics": sem_keep,
            "n_rows_src": m_keep["n_src"],
            "n_valid_pairs_src": m_keep["n_valid_pairs_src"],
            "frac_oob_src": m_keep["frac_oob_src"],
            "frac_valid_src": m_keep["frac_valid_src"],
        })

        out_frames.append(s_keep)

    df_out = pd.concat(out_frames, axis=0).sort_index()
    df_out["DateTime"] = df_out.index.strftime("%Y%m%d%H%M")  # ensure always present
    # Optional: standardize output time label column name
    # Keeps compatibility with your raw schema while remaining readable.
    if "DateTime" in df_out.columns:
        df_out = df_out.drop(columns=["DateTime"])
    df_out = df_out.rename(columns={"DateTime_str": "DateTime"})
    df_summary = pd.DataFrame(summaries).sort_values(["ID"]).reset_index(drop=True)
    return df_out, df_summary


if __name__ == "__main__":
    pass

# --- pipeline_mode_prime_20260219.py (or pipeline_modes_prime.py) ---
# Step order (prime/operational):
#   R0 clean -> strict 15-min series -> baseline (past-only, 2-pass dry) -> A_obs -> wet mask
#   -> wet-antenna (Leijnse) -> A_rain -> k–α -> R  (HARD wet gate => dry is exactly 0)

# -------------------------------------------------------------------
# 0) Past-only baseline helper
# -------------------------------------------------------------------
def _baseline_q90_past_only(
    rsl_series: pd.Series,
    win: str = "24H",
    q: float = 0.9,
    min_past_bins: int = 8,        # 8 bins = 2 hours @ 15-min
    ffill_limit_bins: int = 32,    # carry forward baseline <= 8 hours
    bfill_limit_bins: int = 4      # tiny backfill at start of record
) -> pd.Series:
    """
    Rolling quantile baseline using ONLY past samples (closed='left').

    Why:
      - operationally stable
      - avoids "peeking" into the future
      - quantile baseline is robust to occasional dips

    Notes:
      - set min_past_bins high enough to reduce noisy baseline early in the day
      - ffill_limit prevents baseline drifting forever across missing data
    """
    base = rsl_series.rolling(window=win, min_periods=min_past_bins, closed="left").quantile(q)
    base = base.ffill(limit=ffill_limit_bins).bfill(limit=bfill_limit_bins)
    return base


# -------------------------------------------------------------------
# 1) STRICT 15-min series builder
# -------------------------------------------------------------------
def build_15min_timeseries(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a strict 15-min, one-row-per-(link,time) table for the PRIME pipeline.

    Key idea (IMPORTANT):
      - We DO NOT assume Pmin/Pmax are always RSL (dBm). Some links are TL (dB).
      - R0 already creates `sig_db` which is semantics-safe:
          * semantics == "rsl" -> sig_db =  Pbar   (higher = stronger)
          * semantics == "tl"  -> sig_db = -Pbar   (higher = stronger)
      - Prime baseline should run on a single, consistent “stronger = larger” series.
        So we baseline `sig_db` (not the old RainLINK RSL formula).

    Inputs expected (at least):
      ['ID','DateTime' (or datetime index), 'Pmin','Pmax','Frequency','PathLength',
       'XStart','YStart','XEnd','YEnd','Polarization']
    Recommended from R0:
      ['src_present','qc_flags','semantics','sig_db']

    Output columns (core):
      ['link_id','time','sig_db','Frequency','PathLength','XStart','YStart','XEnd','YEnd','pol']
    Plus QA passthrough when present:
      ['src_present','qc_flags','semantics']
    """
    r = df_clean.copy()

    # ------------------------------------------------------------
    # 1) Time handling (prefer DatetimeIndex from R0 output)
    # ------------------------------------------------------------
    if isinstance(r.index, pd.DatetimeIndex):
        t = r.index
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        r["time"] = t
    else:
        # fallback to DateTime column (digits-only)
        s = r["DateTime"].astype(str).str.strip()
        s = s.str.replace(r"\.0+$", "", regex=True)
        s = s.str.replace(r"[^0-9]", "", regex=True)

        L = s.str.len()
        t = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

        m12 = L == 12  # YYYYMMDDHHMM
        m14 = L == 14  # YYYYMMDDHHMMSS
        if m12.any():
            t.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", utc=True, errors="coerce")
        if m14.any():
            t.loc[m14] = pd.to_datetime(s[m14], format="%Y%m%d%H%M%S", utc=True, errors="coerce")

        other = ~(m12 | m14)
        if other.any():
            t.loc[other] = pd.to_datetime(s[other], utc=True, errors="coerce")

        r["time"] = t

    # strict 15-min bins (floor)
    r["time15"] = pd.to_datetime(r["time"], utc=True, errors="coerce").dt.floor("15min")

    # ------------------------------------------------------------
    # 2) Link identity (must match gridding/meta exactly)
    # ------------------------------------------------------------
    r["link_id"] = r["ID"].astype(str)

    # ------------------------------------------------------------
    # 3) Numeric conversion for meta + signal pieces
    # ------------------------------------------------------------
    for c in ["Pmin", "Pmax", "Frequency", "PathLength", "XStart", "YStart", "XEnd", "YEnd"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")

    # polarization single char
    r["pol"] = r["Polarization"].astype(str).str.upper().str[0]

    # ------------------------------------------------------------
    # 4) Build the semantics-safe signal used for baseline
    # ------------------------------------------------------------
    if "sig_db" in r.columns:
        # Best case: produced by R0 (already semantics-safe)
        r["sig_db"] = pd.to_numeric(r["sig_db"], errors="coerce")
    else:
        # Fallback: reconstruct from Pmin/Pmax (+ semantics if present)
        r["Pbar"] = (r["Pmin"] + r["Pmax"]) / 2.0
        sem = r["semantics"] if "semantics" in r.columns else "rsl"
        # If sem is missing, default to rsl assumption (negative values typical)
        r["sig_db"] = np.where(sem == "tl", -r["Pbar"], r["Pbar"])

    # ------------------------------------------------------------
    # 5) Aggregate to one value per (link_id, time15)
    # ------------------------------------------------------------
    passthrough = {}
    if "src_present" in r.columns:
        passthrough["src_present"] = "max"  # True if any src in that bin
    if "qc_flags" in r.columns:
        passthrough["qc_flags"] = "first"
    if "semantics" in r.columns:
        passthrough["semantics"] = "first"

    agg_map = {
        "sig_db": "median",
        "Frequency": "median",
        "PathLength": "median",
        "XStart": "median",
        "YStart": "median",
        "XEnd": "median",
        "YEnd": "median",
        "pol": "first",
        **passthrough,
    }

    ts = (
        r.groupby(["link_id", "time15"], as_index=False)
         .agg(agg_map)
         .rename(columns={"time15": "time"})
         .sort_values(["link_id", "time"])
         .reset_index(drop=True)
    )

    # guard
    bad = ts["time"].isna().sum()
    if bad:
        raise ValueError(
            f"{bad} timestamps are NaT after parsing. Example:\n{ts[ts['time'].isna()].head()}"
        )

    return ts

# -------------------------------------------------------------------
# 2) Baseline + A_obs + explicit wet/dry
# -------------------------------------------------------------------
def rainlink_strict_Aobs(
    ts_15: pd.DataFrame,
    wet_thr_db: float = 0.5,
    win: str = "24H",
    q: float = 0.9,
    min_past_bins: int = 8,
    ffill_limit_bins: int = 32,
    two_pass: bool = True,
    require_src_present: bool = True
) -> pd.DataFrame:
    """
    Compute:
      baseline_rsl (past-only q90),
      A_obs_dB = max(0, baseline - RSL),
      wet_rl   = A_obs_dB > wet_thr_db

    two_pass=True (recommended):
      pass-1 wet mask from A1
      pass-2 baseline recomputed using ONLY dry samples (wet masked out)
      final baseline = pass2 where available, else pass1

    require_src_present:
      if True and ts_15 has 'src_present', baseline uses only src_present==True samples.
      This avoids baseline getting influenced by reindexed filler rows.
    """
    out = []
    for lid, g in ts_15.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()
        idx = pd.DatetimeIndex(pd.to_datetime(g["time"], utc=True))

        rsl = pd.Series(g["sig_db"].astype(float).values, index=idx)

        # optionally mask non-source rows
        if require_src_present and ("src_present" in g.columns):
            src_mask = g["src_present"].astype(bool).values
            rsl_src = rsl.mask(~src_mask)
        else:
            rsl_src = rsl

        # pass 1
        base1 = _baseline_q90_past_only(
            rsl_src, win=win, q=q,
            min_past_bins=min_past_bins,
            ffill_limit_bins=ffill_limit_bins
        )
        A1 = np.maximum(0.0, base1.values - rsl.values)
        wet1 = (A1 > wet_thr_db) & np.isfinite(A1)

        if two_pass:
            # recompute baseline using dry-only samples
            rsl_dry = rsl_src.mask(wet1)
            base2 = _baseline_q90_past_only(
                rsl_dry, win=win, q=q,
                min_past_bins=min_past_bins,
                ffill_limit_bins=ffill_limit_bins
            )
            base = base2.fillna(base1)
        else:
            base = base1

        Aobs = np.maximum(0.0, base.values - rsl.values)
        wet = (Aobs > wet_thr_db) & np.isfinite(Aobs)

        g["baseline_rsl"] = base.values
        g["A_obs_dB"] = Aobs
        g["wet_rl"] = wet

        out.append(g)

    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------------
# 3) WA + k–α -> R, with HARD wet gate (dry == 0)
# -------------------------------------------------------------------
def rainlink_strict_R(
    dfA: pd.DataFrame,
    R_min: float = 0.0,
    wet_col: str = "wet_rl",
    set_dry_to_zero: bool = True
) -> pd.DataFrame:
    """
    Convert A_obs -> R using:
      - Leijnse (2008) wet-antenna correction
      - ITU(2005) k–α conversion (via pycomlink calc_R_from_A)

    HARD rule (recommended for prime mode):
      if not wet => A_waa = 0, A_rain = 0, gamma = 0, R = 0

    This is the key fix that prevents the "2–5 mm/h carpet" from tiny positive noise.
    """
    parts = []
    for lid, g in dfA.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()

        L_km = float(pd.to_numeric(g["PathLength"], errors="coerce").iloc[0])
        f_GHz = float(pd.to_numeric(g["Frequency"], errors="coerce").iloc[0])
        pol = str(g["pol"].iloc[0]).upper()[0]  # 'H'/'V'

        A_obs = pd.to_numeric(g["A_obs_dB"], errors="coerce").fillna(0.0).values
        wet = g[wet_col].fillna(False).astype(bool).values if wet_col in g.columns else np.isfinite(A_obs)

        # Wet-antenna ONLY where wet (else force 0)
        waa = waa_leijnse_2008_from_A_obs(A_obs=A_obs, f_Hz=f_GHz * 1e9, pol=pol, L_km=L_km)
        if set_dry_to_zero:
            waa = np.where(wet, waa, 0.0)

        A_rain = np.maximum(0.0, A_obs - waa)
        if set_dry_to_zero:
            A_rain = np.where(wet, A_rain, 0.0)

        # Attenuation to rain rate
        R = calc_R_from_A(
            A=A_rain,
            L_km=L_km,
            f_GHz=f_GHz,
            pol=pol,
            a_b_approximation="ITU_2005",
            R_min=R_min
        )
        if set_dry_to_zero:
            R = np.where(wet, R, 0.0)

        g["A_waa_dB"] = waa
        g["A_rain_dB"] = A_rain
        g["gamma_for_R"] = A_rain / max(L_km, 1e-6)
        g["R_mm_per_h"] = R

        parts.append(g)

    return pd.concat(parts, ignore_index=True)

#-------------------------------------------------------------------
def prepare_inputs_for_gridding(df_rate: pd.DataFrame, ts_15: pd.DataFrame):
    # --- meta (one row per link) ---
    meta = (ts_15.drop_duplicates("link_id")[["link_id","XStart","YStart","XEnd","YEnd"]]
                 .rename(columns={"link_id":"ID"})
                 .copy())
    for c in ["XStart","YStart","XEnd","YEnd"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    # --- rates (time-indexed, tz-naive UTC) ---
    s5 = df_rate[["link_id","time","R_mm_per_h"]].rename(columns={"link_id":"ID"}).copy()
    t = pd.to_datetime(s5["time"], utc=True, errors="coerce")

    ok = t.notna()
    s5 = s5.loc[ok].drop(columns=["time"])
    s5.index = t.loc[ok].dt.tz_convert("UTC").dt.tz_localize(None)
    s5.index.name = "time"
    s5 = s5.sort_index()

    # sanity
    miss = set(s5["ID"].unique()) - set(meta["ID"].unique())
    if miss:
        raise RuntimeError(f"{len(miss)} IDs in rates missing from meta. Example: {list(miss)[:5]}")

    return s5, meta


#-------------------------------------------------------------------
# step6_grid_ok_pcm.py
# ===============================================================
# Gridding 15-min CML rain to lon/lat
# - Keep zeros; map drizzle to 0.0
# - Gate Ordinary Kriging (OK) and fallback to KNN-IDW (hard radius)
# - Wet-only footprint + coverage cleaning (binary opening)
# - Optional link-length weights in IDW
# - Gentle in-mask smoothing (no bleed)
# - RainLINK-style OK variant (wet+dry0 training & strict support)
# - Process/Thread parallelism with controllable KDTree workers
# - Single-time helper and diagnostics
# ===============================================================


# ---------------- geometry ----------------
def _km_factors(lat0_deg: float) -> tuple[float, float]:
    lat0r = np.deg2rad(float(lat0_deg))
    kx = 111.0 * max(0.2, np.cos(lat0r))
    ky = 111.0
    return kx, ky

def _lonlat_to_km(lon, lat, lon0, lat0):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    kx, ky = _km_factors(lat0)
    return (lon - lon0) * kx, (lat - lat0) * ky


# ---------------- masks & smoothing ----------------
def _nearest_distance_mask(
    lon_vec, lat_vec, x_obs, y_obs, max_dist_km: float, workers: int | None = None
) -> np.ndarray:
    """True where nearest observed point within max_dist_km (euclidean in km)."""
    XX, YY = np.meshgrid(lon_vec, lat_vec)
    lon0 = float((lon_vec.min() + lon_vec.max()) / 2.0)
    lat0 = float((lat_vec.min() + lat_vec.max()) / 2.0)
    Pkx, Pky = _lonlat_to_km(XX, YY, lon0, lat0)

    if len(x_obs) == 0:
        return np.zeros_like(XX, dtype=bool)

    Qkx, Qky = _lonlat_to_km(x_obs, y_obs, lon0, lat0)
    tree = cKDTree(np.c_[Qkx.ravel(), Qky.ravel()])
    kw = {} if workers is None else {"workers": int(workers)}
    dmin, _ = tree.query(np.c_[Pkx.ravel(), Pky.ravel()], k=1, **kw)
    return (dmin.reshape(XX.shape) <= float(max_dist_km))


def _clean_wet_mask(mask: np.ndarray) -> np.ndarray:
    st = generate_binary_structure(2, 1)  # 4-neighborhood
    return binary_opening(mask, structure=st)

#-------------------------------------------------------------------

def _smooth_in_mask(Z, mask, kernel_px: int = 3) -> np.ndarray:
    """Box-filter smoothing ONLY where mask is True; no bleed outside."""
    if kernel_px is None or int(kernel_px) <= 1:
        return Z
    Z = np.asarray(Z, float); mask = mask.astype(bool)
    Z0 = np.nan_to_num(Z, nan=0.0); W = mask.astype(float)
    num = uniform_filter(Z0, size=int(kernel_px), mode="nearest")
    den = uniform_filter(W,  size=int(kernel_px), mode="nearest")
    Zs = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)
    return np.where(mask, Zs, np.nan)


# ---------------- interpolators ----------------
def _idw_on_grid_km_weighted(
    xkm, ykm, z, Xkm, Ykm, nnear, power, maxdist_km, w_pts=None, workers: int | None = None
):
    """KNN-IDW in km space with optional per-point weights (e.g., sqrt(PathLength))."""
    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() == 0:
        return np.full_like(Xkm, np.nan, dtype=float)

    pts  = np.c_[xkm[wet], ykm[wet]]
    vals = z[wet].astype(float)
    wp   = np.ones_like(vals) if w_pts is None else np.asarray(w_pts, float)[wet]

    tree = cKDTree(pts)
    k = int(max(1, nnear))
    kw = {"distance_upper_bound": float(maxdist_km)}
    if workers is not None:
        kw["workers"] = int(workers)

    d, idx = tree.query(np.c_[Xkm.ravel(), Ykm.ravel()], k=k, **kw)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]

    valid = np.isfinite(d)
    w_spatial = np.where(valid, 1.0 / np.maximum(d, _EPS) ** float(power), 0.0)

    # pad arrays to handle "no neighbor within radius" (idx == len(vals))
    vals_pad = np.concatenate([vals, [0.0]])
    wp_pad   = np.concatenate([wp,   [0.0]])
    idx = np.where(idx == len(vals), len(vals_pad) - 1, idx)

    num = np.sum(w_spatial * (vals_pad[idx] * wp_pad[idx]), axis=1)
    den = np.sum(w_spatial * wp_pad[idx], axis=1)
    out = np.where(den > 0, num / den, np.nan)
    return out.reshape(Xkm.shape)


def _try_ok_on_grid_km(
    xkm, ykm, z, Xkm, Ykm, *, nlags=10, range_km=25.0, nugget_frac=0.05
):
    """Ordinary Kriging in Euclidean km space with simple exponential variogram."""
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available")

    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() < 3:
        raise RuntimeError("not enough wet points for OK")

    z_w = z[wet]
    var = float(np.nanvar(z_w))
    if var <= 0:
        raise RuntimeError("zero variance – OK ill-posed")

    nugget = max(1e-6, float(nugget_frac) * var)
    sill   = max(1e-6, var)

    ok = OrdinaryKriging(
        xkm[wet], ykm[wet], z_w,
        variogram_model="exponential",
        variogram_parameters={"nugget": nugget, "sill": sill, "range": float(range_km)},
        nlags=int(nlags), enable_plotting=False, coordinates_type="euclidean",
    )
    xvec = np.unique(Xkm[0, :]); yvec = np.unique(Ykm[:, 0])
    zhat, _ = ok.execute("grid", xvec, yvec)
    return np.asarray(zhat, float)


# ---------------- main API (OK-gated + IDW fallback) ----------------
def grid_rain_15min(
    df_s5,
    df_meta_for_xy,
    *,
    grid_res_deg: float = 0.03,
    domain_pad_deg: float = 0.2,
    drizzle_to_zero: float = 0.1,
    use_ok: bool = True,
    min_pts_ok: int = 20,
    nlags: int = 10,
    ok_range_km: float = 25.0,
    ok_nugget_frac: float = 0.05,
    ok_max_train: int | None = None,
    idw_power: float = 2.0,
    idw_nnear: int = 15,
    idw_maxdist_km: float = 25.0,
    max_dist_km_mask: float = 28.0,
    smooth_kernel_px: int = 3,
    n_jobs: int = 1,
    times_sel=None,
    parallel_backend_name: str = "processes",
    kdtree_workers: int | None = 1,
    collocate_bin_km: float = 2.0,
    use_pathlength_weights: bool = True,
    # NEW fill controls (defaults = NaN)
    outside_support_fill=np.nan,           # value outside wet-footprint support
    interior_no_neighbor_fill=np.nan,      # value for interior “holes” (no neighbors within radius)
    insufficient_training_fill=np.nan,     # whole grid when no wet pts or cannot train
):
    """
    df_s5: index=DatetimeIndex (naive UTC), columns: ["ID","R_mm_per_h", ...]
    df_meta_for_xy: columns: ID, XStart, YStart, XEnd, YEnd

    NOTE: We no longer force NaNs to 0.0. Small finite drizzle values are floored to 0.0,
    but unknowns remain NaN unless you override the *_fill parameters.
    """
    df_s5 = df_s5.copy()
    df_s5["R_mm_per_h"] = pd.to_numeric(df_s5["R_mm_per_h"], errors="coerce")

    # Drizzle → 0.0 (only where finite and positive)
    if drizzle_to_zero is not None:
        v = df_s5["R_mm_per_h"].to_numpy(float)
        mask = np.isfinite(v) & (v > 0.0) & (v < float(drizzle_to_zero))
        v[mask] = 0.0
        df_s5["R_mm_per_h"] = v

    # index hygiene
    idx = df_s5.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("df_s5.index must be a DatetimeIndex")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df_s5.index = idx
    df_s5.index.name = "time"

    # time selection
    all_times = df_s5.index.unique().sort_values()
    if times_sel is not None:
        req = pd.to_datetime(np.atleast_1d(times_sel))
        req_naive = []
        for tt in req:
            tt = pd.Timestamp(tt)
            if tt.tzinfo is not None:
                tt = tt.tz_convert("UTC").tz_localize(None)
            req_naive.append(tt)
        times = pd.DatetimeIndex(req_naive).intersection(all_times)
        if len(times) == 0:
            raise ValueError("times_sel did not match any times in df_s5.index.")
    else:
        times = all_times

    # meta → midpoints
    m = df_meta_for_xy.dropna(subset=["XStart","YStart","XEnd","YEnd"]).copy()
    for c in ["XStart","YStart","XEnd","YEnd","PathLength"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon"] = 0.5*(m["XStart"] + m["XEnd"])
    m["lat"] = 0.5*(m["YStart"] + m["YEnd"])

    # grid extent
    lon_min = float(np.nanmin(m["lon"])) - float(domain_pad_deg)
    lon_max = float(np.nanmax(m["lon"])) + float(domain_pad_deg)
    lat_min = float(np.nanmin(m["lat"])) - float(domain_pad_deg)
    lat_max = float(np.nanmax(m["lat"])) + float(domain_pad_deg)

    lon = np.arange(lon_min, lon_max + 1e-9, float(grid_res_deg))
    lat = np.arange(lat_min, lat_max + 1e-9, float(grid_res_deg))

    lon0 = float((lon.min() + lon.max())/2.0)
    lat0 = float((lat.min() + lat.max())/2.0)
    LON, LAT = np.meshgrid(lon, lat)
    Xkm, Ykm = _lonlat_to_km(LON, LAT, lon0, lat0)

    methods_used = []
    n_wet_list  = []

    def _do_time(t):
        g = df_s5.loc[df_s5.index == t, ["ID","R_mm_per_h"]].copy()
        g = g.merge(m[["ID","lon","lat","PathLength"]] if "PathLength" in m.columns else m[["ID","lon","lat"]],
                    on="ID", how="inner")
        g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
        g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
        g["R_mm_per_h"] = pd.to_numeric(g["R_mm_per_h"], errors="coerce")
        g = g.dropna(subset=["lon","lat","R_mm_per_h"])

        x = g["lon"].to_numpy(float); y = g["lat"].to_numpy(float); z = g["R_mm_per_h"].to_numpy(float)
        xkm, ykm = _lonlat_to_km(x, y, lon0, lat0)

        # If no usable points at all → whole grid = insufficient_training_fill
        if len(xkm) == 0:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_info", 0

        # deduplicate near-collocated links (median)
        if len(xkm):
            dfp = pd.DataFrame({"x": xkm, "y": ykm, "z": z})
            bin_km = float(collocate_bin_km)
            bx = np.round(dfp["x"]/bin_km).astype(int)
            by = np.round(dfp["y"]/bin_km).astype(int)
            dfp["_bin"] = list(zip(bx, by))
            gmed = dfp.groupby("_bin", as_index=False).median(numeric_only=True)
            xkm_d = gmed["x"].to_numpy(); ykm_d = gmed["y"].to_numpy(); z_d = gmed["z"].to_numpy()
        else:
            xkm_d = xkm; ykm_d = ykm; z_d = z

        # wet-only footprint (strictly positive)
        wet_pts = np.isfinite(z) & (z > 0.0)
        if wet_pts.sum() < 1:
            # no wet info → whole grid = insufficient_training_fill
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_wet", 0

        x_w, y_w = x[wet_pts], y[wet_pts]
        covmask_wet = _nearest_distance_mask(
            lon, lat, x_w, y_w, max_dist_km=max_dist_km_mask, workers=kdtree_workers
        )
        covmask_wet = _clean_wet_mask(covmask_wet)

        # per-point weights for IDW (sqrt(PathLength) if available)
        w_pts = None
        if use_pathlength_weights and "PathLength" in g.columns:
            w_pts = np.sqrt(np.clip(pd.to_numeric(g["PathLength"], errors="coerce").to_numpy(float), 1.0, None))

        used = "idw_knn"
        Z = np.full_like(LON, np.nan, dtype=float)
        n_wet = int((z_d > 0.0).sum())

        if use_ok and _PYKRIGE_AVAILABLE and (n_wet >= int(min_pts_ok)):
            try:
                Z = _try_ok_on_grid_km(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nlags=int(nlags), range_km=float(ok_range_km), nugget_frac=float(ok_nugget_frac)
                )
                used = "ok_pykrige"
            except Exception:
                Z = _idw_on_grid_km_weighted(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nnear=int(idw_nnear), power=float(idw_power),
                    maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
                )
                used = "idw_knn"
        else:
            Z = _idw_on_grid_km_weighted(
                xkm_d, ykm_d, z_d, Xkm, Ykm,
                nnear=int(idw_nnear), power=float(idw_power),
                maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
            )
            used = "idw_knn"

        # Outside support → fill (default NaN)
        if np.isnan(outside_support_fill):
            Z[~covmask_wet] = np.nan
        else:
            Z[~covmask_wet] = float(outside_support_fill)

        # Smooth ONLY where we have values (no bleed)
        valid_mask = np.isfinite(Z)
        Z = _smooth_in_mask(Z, valid_mask, kernel_px=smooth_kernel_px)

        # Interior “holes” (inside support but still NaN) → fill (default NaN)
        inside_holes = covmask_wet & ~np.isfinite(Z)
        if np.isnan(interior_no_neighbor_fill):
            # leave as NaN
            pass
        else:
            Z[inside_holes] = float(interior_no_neighbor_fill)

        return Z, used, n_wet

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        res = Parallel(n_jobs=int(n_jobs))(delayed(_do_time)(t) for t in times)

    grids    = [r[0] for r in res]
    methods  = [r[1] for r in res]
    n_wet_l  = [r[2] for r in res]

    da = xr.DataArray(
        np.stack(grids),
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="R_mm_per_h",
    )
    da.attrs.update(dict(
        units="mm h-1",
        method="OK→IDW (gated) + cleaned wet-footprint + box-mean smooth",
        grid_res_deg=float(grid_res_deg),
    ))

    counts = {mth: methods.count(mth) for mth in set(methods)}
    first5 = []
    for i in range(min(5, len(times))):
        sl = grids[i]
        first5.append(dict(method=methods[i], n_wet=int(n_wet_l[i]),
                           min=float(np.nanmin(sl)), max=float(np.nanmax(sl))))

    diag = dict(
        n_times=int(len(times)),
        counts=counts,
        first5=first5,
        ok_times=[str(t) for t, mth in zip(times, methods) if mth == "ok_pykrige"],
        pykrige_available=_PYKRIGE_AVAILABLE,
        config=dict(
            grid_res_deg=grid_res_deg, domain_pad_deg=domain_pad_deg,
            drizzle_to_zero=drizzle_to_zero, use_ok=bool(use_ok),
            min_pts_ok=min_pts_ok, nlags=nlags, ok_range_km=ok_range_km,
            ok_nugget_frac=ok_nugget_frac, ok_max_train=ok_max_train,
            idw_power=idw_power, idw_nnear=idw_nnear, idw_maxdist_km=idw_maxdist_km,
            max_dist_km_mask=max_dist_km_mask, smooth_kernel_px=smooth_kernel_px,
            n_jobs=n_jobs, parallel_backend_name=parallel_backend_name,
            kdtree_workers=kdtree_workers, collocate_bin_km=collocate_bin_km,
            use_pathlength_weights=use_pathlength_weights,
            outside_support_fill=outside_support_fill,
            interior_no_neighbor_fill=interior_no_neighbor_fill,
            insufficient_training_fill=insufficient_training_fill,
        ),
        times_used=[str(t) for t in times[:10]],
    )
    return da, diag

#-------------------------------------------------------------------

def grid_rain_at_time(df_s5, df_meta_for_xy, t, **kwargs):
    """Convenience wrapper to grid a single timestamp."""
    return grid_rain_15min(df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs)


# ---------------- RainLINK-like OK variant (wet+dry0 training & strict support) ----------------
def _midpoints(dfm):
    m = dfm.copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon_mid"] = (m["XStart"] + m["XEnd"]) / 2.0
    m["lat_mid"] = (m["YStart"] + m["YEnd"]) / 2.0
    return m
#-------------------------------------------------------------------

def _grid_from_meta(meta_xy, grid_res_deg=0.03, pad_deg=0.20):
    mm = _midpoints(meta_xy)
    x0, x1 = float(mm["lon_mid"].min()), float(mm["lon_mid"].max())
    y0, y1 = float(mm["lat_mid"].min()), float(mm["lat_mid"].max())
    xv = np.arange(x0 - pad_deg, x1 + pad_deg + grid_res_deg, grid_res_deg)
    yv = np.arange(y0 - pad_deg, y1 + pad_deg + grid_res_deg, grid_res_deg)
    lon, lat = np.meshgrid(xv, yv)
    return lon, lat, xv, yv

def _support_mask_wet_haversine(lon_grid, lat_grid, lon_wet, lat_wet, k=2, radius_km=25.0):
    """True where the k-th nearest WET link is within radius_km (haversine)."""
    if not _SKLEARN_AVAILABLE or len(lon_wet) < k:
        return np.zeros_like(lon_grid, dtype=bool)
    tree = BallTree(np.deg2rad(np.c_[lat_wet, lon_wet]), metric="haversine")
    pts  = np.deg2rad(np.c_[lat_grid.ravel(), lon_grid.ravel()])
    d_rad, _ = tree.query(pts, k=int(k))
    d_km = d_rad[:, -1] * _EARTH_R_KM
    return (d_km <= float(radius_km)).reshape(lon_grid.shape)
#-------------------------------------------------------------------

def _estimate_variogram_params(values, range_km=25.0, nugget_frac=0.4):
    """Simple robust parameters; sill from variance of training values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    sill = float(np.nanvar(v)) if v.size else 1e-6
    sill = max(sill, 1e-6)
    return {"sill": sill, "range": float(range_km), "nugget": float(nugget_frac) * sill}

def grid_rain_15min_rainlink_ok(
    df_s5,                    # index: time (naive UTC); cols: ID, R_mm_per_h
    df_meta_for_xy,           # cols: ID, XStart, YStart, XEnd, YEnd
    *,
    grid_res_deg=0.03, domain_pad_deg=0.20,
    wet_thr=0.8, dry_thr=0.05,               # define wet/dry points for training
    ok_model="exponential", ok_range_km=25.0, ok_nugget_frac=0.4,
    min_pts_ok=12,                            # need at least this many (wet+dry0) to run OK
    support_k=2, support_radius_km=25.0,     # mask requires ≥k wet links within radius
    drizzle_to_zero=0.10,                     # floor tiny positives to 0.0 (doesn't touch NaNs)
    times_sel=None,
    # NEW:
    n_jobs: int = 1,
    parallel_backend_name: str = "processes",
    # NEW controls for “no information”
    outside_support_fill=np.nan,             # what to put OUTSIDE strict wet support (np.nan or 0.0)
    insufficient_training_fill=np.nan,
    smooth_kernel_px: int | None = None,
    smooth_fill_holes: bool = True,
):
    """
    RAINLINK-like: OK trained on wet values + dry zeros; apply strict wet support mask.

    outside_support_fill:
        Value for cells outside the wet-link support mask (default NaN).
    insufficient_training_fill:
        Value for the entire grid when there were no usable points or too few to train (default NaN).
    """
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available for RainLINK-style gridding.")

    # 1) grid axes
    LON, LAT, xv, yv = _grid_from_meta(df_meta_for_xy, grid_res_deg, domain_pad_deg)

    # 2) outputs
    all_times = pd.Index(sorted(df_s5.index.unique()))
    times = all_times if times_sel is None else pd.Index(pd.to_datetime(times_sel))
    out = np.full((len(times), LAT.shape[0], LAT.shape[1]), np.nan, float)

    # precompute midpoints by ID
    mid = _midpoints(df_meta_for_xy[["ID","XStart","YStart","XEnd","YEnd"]].drop_duplicates("ID"))
    id2xy = mid.set_index("ID")[["lon_mid","lat_mid"]]

    # diagnostics
    diag = {"counts": {"ok": 0, "fallback": 0}, "wet_counts": [], "train_counts": []}

    def _do_one(it, t):
        # slice points and attach coords
        try:
            pts = (df_s5.loc[t].merge(id2xy, on="ID", how="inner"))
        except KeyError:
            # no rows at this timestamp → whole grid is "no info"
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals = pd.to_numeric(pts["R_mm_per_h"], errors="coerce").values
        lon  = pd.to_numeric(pts["lon_mid"], errors="coerce").values
        lat  = pd.to_numeric(pts["lat_mid"], errors="coerce").values

        good = np.isfinite(vals) & np.isfinite(lon) & np.isfinite(lat)
        if not good.any():
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals, lon, lat = vals[good], lon[good], lat[good]

        # classify wet/dry for training
        wet = vals >= float(wet_thr)
        dry = vals <= float(dry_thr)

        lon_wet, lat_wet = lon[wet], lat[wet]
        # training set = wet values + dry zeros
        lon_tr = np.concatenate([lon[wet], lon[dry]])
        lat_tr = np.concatenate([lat[wet], lat[dry]])
        val_tr = np.concatenate([vals[wet], np.zeros(np.count_nonzero(dry), float)])

        if len(val_tr) < max(3, int(min_pts_ok)):
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

        # OK on geographic coords
        vparam = _estimate_variogram_params(val_tr, range_km=ok_range_km, nugget_frac=ok_nugget_frac)
        try:
            OK = OrdinaryKriging(
                lon_tr, lat_tr, val_tr,
                variogram_model=ok_model,
                variogram_parameters=vparam,
                coordinates_type="geographic",
                enable_plotting=False, verbose=False
            )
            Z, _ = OK.execute("grid", xv, yv)   # (ny, nx)
            Z = np.asarray(Z, float)

            # strict support mask based on WET links only
            if _SKLEARN_AVAILABLE and len(lon_wet) >= max(1, int(support_k)):
                mask = _support_mask_wet_haversine(LON, LAT, lon_wet, lat_wet,
                                                   k=int(support_k), radius_km=float(support_radius_km))
            else:
                # fallback: nearest-distance footprint from wet points in degrees (~approx km)
                mask = _nearest_distance_mask(xv, yv, lon_wet, lat_wet,
                                              max_dist_km=support_radius_km, workers=1)

            # Outside support → fill (NaN by default)
            if np.isnan(outside_support_fill):
                Z = np.where(mask, Z, np.nan)
            else:
                Z = np.where(mask, Z, float(outside_support_fill))
            
            # --- optional smoothing inside strict support (gap-closing) ---
            if smooth_kernel_px is not None and int(smooth_kernel_px) > 1:
                if smooth_fill_holes:
                    # fill holes: overwrite ALL pixels inside support with smoothed field
                    Zs = _smooth_normalized(Z, write_mask=mask, kernel_px=int(smooth_kernel_px))
                    Z = np.where(mask, Zs, Z)  # outside already handled
                else:
                    # no hole fill: smooth only where Z is already finite
                    Zs = _smooth_normalized(Z, write_mask=mask, kernel_px=int(smooth_kernel_px))
                    Z = np.where(mask & np.isfinite(Z), Zs, Z)

            # Drizzle floor to zero (does not touch NaNs)
            if drizzle_to_zero is not None:
                Z = np.where(np.isfinite(Z) & (Z < float(drizzle_to_zero)), 0.0, Z)

            return it, Z, int(wet.sum()), int(len(val_tr)), True
        except Exception:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        results = Parallel(n_jobs=int(n_jobs))(
            delayed(_do_one)(i, t) for i, t in enumerate(times)
        )

    for it, Z, nwet, ntrain, ok_flag in results:
        out[it, :, :] = Z
        diag["wet_counts"].append(nwet)
        diag["train_counts"].append(ntrain)
        diag["counts"]["ok" if ok_flag else "fallback"] += 1

    da = xr.DataArray(
        out, coords={"time": times.tz_localize(None), "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"), name="R_mm_per_h", attrs={"units": "mm h-1"}
    )
    return da, diag
#-------------------------------------------------------------------


def _smooth_normalized(Z, write_mask, kernel_px: int):
    """
    Normalized box smoothing that ignores NaNs:
      Z_smooth = box(Z0) / box(W)
    where Z0=0 at NaNs and W=1 where finite else 0.
    Returns NaN where denominator is 0.
    """
    if kernel_px is None or int(kernel_px) <= 1:
        return Z

    Z = np.asarray(Z, float)
    W = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z, nan=0.0)

    num = uniform_filter(Z0, size=int(kernel_px), mode="nearest")
    den = uniform_filter(W,  size=int(kernel_px), mode="nearest")
    Zs  = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)

    # only write where requested (e.g., strict support)
    out = np.where(write_mask, Zs, np.nan)
    return out
#-------------------------------------------------------------------
def grid_rain_at_time_rainlink(df_s5, df_meta_for_xy, t, **kwargs):
    return grid_rain_15min_rainlink_ok(
        df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs
    )

#-------------------------------------------------------------------
def get_da(Res, name="R_mm_per_h"):
    """Accepts DataArray, Dataset, or dict and returns the rain DataArray."""
    if isinstance(Res, xr.DataArray):
        return Res
    if isinstance(Res, xr.Dataset):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in Dataset vars: {list(Res.data_vars)}")
    if isinstance(Res, dict):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in dict keys: {list(Res.keys())}")
    raise TypeError(f"Unsupported Res type: {type(Res)}")
#-------------------------------------------------------------------

def slice_time(da, t=None):
    """
    Return a 2-D (lat,lon) DataArray and a label timestamp.
    - If 'time' in dims: select nearest t (if t is None -> use first time).
    - If no 'time' dim: just return da and label from attr if present.
    """
    if "time" in da.dims:
        if t is None:
            t_sel = pd.to_datetime(da["time"].values[0]).to_pydatetime()
        else:
            t_sel = pd.Timestamp(t)
            if t_sel.tzinfo is not None:
                t_sel = t_sel.tz_convert("UTC").tz_localize(None)
        out = da.sel(time=np.datetime64(t_sel), method="nearest")
        return out, pd.to_datetime(out["time"].item())
    # 2-D already
    label = pd.to_datetime(da.attrs.get("time", "NaT"))
    return da, label
#-------------------------------------------------------------------

def apply_support_mask(Res, df_s5, meta_xy, t=None, *, k=3, km=35.0, eps=0.1, name="R_mm_per_h"):
    """
    Keep cells with ≥k links having R>eps within km (great-circle).
    df_s5: time-indexed with ['ID','R_mm_per_h'].
    meta_xy: ['ID','XStart','YStart','XEnd','YEnd'] numeric.
    """
    da = get_da(Res, name)
    sl, t_used = slice_time(da, t)

    # points at that time
    pts = (df_s5.loc[pd.Timestamp(t_used)]
           .merge(meta_xy, on="ID", how="inner")
           .assign(lon_mid=lambda d: (pd.to_numeric(d.XStart)+pd.to_numeric(d.XEnd))/2,
                   lat_mid=lambda d: (pd.to_numeric(d.YStart)+pd.to_numeric(d.YEnd))/2))
    wet = (pts["R_mm_per_h"].to_numpy(float) > eps)
    if wet.sum() < max(3, k):
        # nothing to mask—return slice unchanged
        sl2 = sl.copy()
        sl2.attrs["time"] = str(t_used)
        return sl2

    tree = BallTree(np.deg2rad(np.c_[pts["lat_mid"], pts["lon_mid"]]), metric="haversine")
    gx, gy = np.meshgrid(sl["lon"].values, sl["lat"].values)
    q = np.deg2rad(np.c_[gy.ravel(), gx.ravel()])
    neigh = tree.query_radius(q, r=km/6371.0)
    kcnt = np.fromiter((wet[idx].sum() if len(idx) else 0 for idx in neigh), int).reshape(gy.shape)

    Z = sl.values.copy()
    Z[kcnt < k] = np.nan
    out = sl.copy(data=Z)
    out.attrs["time"] = str(t_used)
    return out

#---------------- save to NetCDF with one file per time step, good compression, and chunking ----------------
def save_each_time_to_netcdf(
    data,                           # xr.DataArray OR xr.Dataset (with var_name)
    out_dir,
    base_name="ghana_cml_R",
    *,
    var_name="R_mm_per_h",          # ignored if data is a DataArray
    engine="netcdf4",               # or "h5netcdf"
    complevel=9,
    dtype="float32",
    fill_value=np.nan,              # what to write for NaNs
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True,             # keep a size-1 time dimension in each file
):
    """
    Writes one .nc per timestamp with high compression and correct chunking.
    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Normalize to a DataArray
    if isinstance(data, xr.Dataset):
        if var_name not in data.data_vars and len(data.data_vars) == 1:
            var_name = list(data.data_vars)[0]
        da = data[var_name]
    elif isinstance(data, xr.DataArray):
        var_name = data.name or var_name
        da = data
    else:
        raise TypeError("data must be an xarray DataArray or Dataset")

    # If no time dimension, write a single file and return
    if "time" not in da.dims:
        d2 = da.astype(dtype)
        ds = d2.to_dataset(name=var_name)
        # chunks: match dims exactly
        dims = ds[var_name].dims
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                chunks.append(min(1, sizes[d]))
        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            }
        }
        fn = os.path.join(out_dir, f"{base_name}.nc")
        ds.to_netcdf(fn, engine=engine, encoding=enc)
        return [fn]

    # Otherwise iterate times
    times = pd.to_datetime(da["time"].values)
    out_paths = []

    for t in times:
        # 2-D slice (lat,lon)
        sl = da.sel(time=np.datetime64(t)).astype(dtype)

        # Keep time dim? -> expand to (time,lat,lon) of length 1
        if keep_time_dim:
            sl = sl.expand_dims(time=[np.datetime64(t)])
        ds = sl.to_dataset(name=var_name)

        # Build chunks tuple that matches EXACT dims order
        dims = ds[var_name].dims             # e.g., ('time','lat','lon') or ('lat','lon')
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "time":
                chunks.append(1)
            elif d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                # Unknown dim: just chunk by its full size
                chunks.append(sizes[d])

        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            },
            # coords (don’t compress tiny arrays)
            "lat": {"zlib": False},
            "lon": {"zlib": False},
            "time": {"zlib": False},
        }

        # Nice filename stamp: YYYYmmddTHHMMSS
        t_str = pd.Timestamp(t).strftime("%Y%m%dT%H%M%S")
        fn = os.path.join(out_dir, f"{base_name}_{t_str}.nc")

        # Use unlimited time when present
        unlimited = {"time"} if keep_time_dim else None
        ds.to_netcdf(fn, engine=engine, encoding=enc, unlimited_dims=unlimited)
        out_paths.append(fn)

    return out_paths