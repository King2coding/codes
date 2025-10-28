#%%
# package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import timedelta

from rasterio.enums import Resampling
from joblib import Parallel, delayed

from pyproj import CRS, Transformer
from scipy.interpolate import griddata
import rioxarray  # enables .rio accessor on xarray objects
from rioxarray.merge import merge_arrays

#%%
# define paths
PATH_CML_RAIN = r'/home/kkumah/Projects/cml-stuff/out_cml_rain_dir'
PATH_MSG_BT = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251027_201416'
PATH_MSG_CLM = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm/run_20251027_201601'

#%% Constants
# Which MSG vars to resample (present in your examples)
BT_VARS = ['BT_IR108', 'BT_IR120', 'BT_WV062']   # linear interp
CLM_VARS = ['cloud_mask']                        # nearest interp

# 15-min cadence (CML); adjust if you prefer per-minute
TIME_ROUND = '15min'

#%% Define fuctions

# -----------------------------
# Helpers
# -----------------------------
def _list_files(root, suffix='.nc'):
    if os.path.isdir(root):
        return sorted([os.path.join(r, f)
                       for r, _, fs in os.walk(root)
                       for f in fs if f.endswith(suffix)])
    else:
        # allow a single file
        return [root] if root.endswith(suffix) else []

def round_time_index(ds, freq=TIME_ROUND):
    if 'time' in ds.coords:
        ds = ds.assign_coords(time=ds['time'].dt.round(freq))
        # drop duplicate times after rounding (keep first)
        _, idx = np.unique(ds['time'].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
    return ds

def _find_geos_crs(ds):
    # 1) A data var that names a grid_mapping
    for v in ds.data_vars:
        gm = ds[v].attrs.get("grid_mapping")
        if gm and gm in ds:
            gmv = ds[gm]
            if gmv.attrs.get("grid_mapping_name") == "geostationary":
                wkt = gmv.attrs.get("crs_wkt") or gmv.attrs.get("spatial_ref")
                if wkt:
                    return CRS.from_wkt(wkt)
                # Build from individual attrs if WKT missing
                lon0 = float(gmv.attrs.get("longitude_of_projection_origin", 0.0))
                h    = float(gmv.attrs.get("perspective_point_height", 35785831.0))
                a    = float(gmv.attrs.get("semi_major_axis", 6378137.0))
                rf   = float(gmv.attrs.get("inverse_flattening", 298.257222101))
                sweep= gmv.attrs.get("sweep_angle_axis", "y")
                return CRS.from_proj4(f"+proj=geos +lon_0={lon0} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs")

    # 2) Any variable that already *is* the grid mapping
    for name, var in ds.variables.items():
        if var.attrs.get("grid_mapping_name") == "geostationary":
            wkt = var.attrs.get("crs_wkt") or var.attrs.get("spatial_ref")
            if wkt:
                return CRS.from_wkt(wkt)
            lon0 = float(var.attrs.get("longitude_of_projection_origin", 0.0))
            h    = float(var.attrs.get("perspective_point_height", 35785831.0))
            a    = float(var.attrs.get("semi_major_axis", 6378137.0))
            rf   = float(var.attrs.get("inverse_flattening", 298.257222101))
            sweep= var.attrs.get("sweep_angle_axis", "y")
            return CRS.from_proj4(f"+proj=geos +lon_0={lon0} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs")

    # 3) WKT sitting on coords or globals
    for coord in ("x", "y"):
        if coord in ds.coords:
            wkt = ds.coords[coord].attrs.get("spatial_ref") or ds.coords[coord].attrs.get("crs_wkt")
            if wkt:
                return CRS.from_wkt(wkt)
    for key in ("crs_wkt", "spatial_ref"):
        if key in ds.attrs:
            return CRS.from_wkt(ds.attrs[key])

    # 4) Fallback: MSG SEVIRI default
    return CRS.from_proj4("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +rf=298.257222101 +sweep=y +units=m +no_defs")

def build_transformer_from_geos_attrs(geos_attrs: dict):
    """
    Create pyproj Transformer (GEOS→WGS84) from the 'geostationary' grid_mapping attrs.
    Works with CF-1.5 style attributes present in your netCDF.
    """
    # Required attributes in your file:
    lam = float(geos_attrs.get('longitude_of_projection_origin', 0.0))
    h   = float(geos_attrs.get('perspective_point_height', 35785831.0))
    a   = float(geos_attrs.get('semi_major_axis', 6378137.0))
    rf  = float(geos_attrs.get('inverse_flattening', 298.257222101))
    sweep = geos_attrs.get('sweep_angle_axis', 'y')  # 'x' or 'y'

    # PROJ geostationary CRS string
    # Note: proj uses +h for satellite height above the ellipsoid center.
    proj_geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lam} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs"
    )

    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(proj_geos, wgs84, always_xy=True)
    return transformer

def geos_xy_to_lonlat(ds_msg):
    """Return (lon2d, lat2d) for an MSG dataset with x/y in meters."""
    if "x" not in ds_msg.coords or "y" not in ds_msg.coords:
        raise ValueError("Dataset lacks 'x' and 'y' coordinates (meters in geostationary projection).")

    crs_geos = _find_geos_crs(ds_msg)
    crs_ll   = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_geos, crs_ll, always_xy=True)

    x = ds_msg["x"].values
    y = ds_msg["y"].values
    X, Y = np.meshgrid(x, y)
    lon, lat = transformer.transform(X, Y)
    return lon, lat

def _griddata_resample(src_lon, src_lat, src_field, tgt_lon2d, tgt_lat2d,
                       method='linear', fill_method='nearest'):
    """
    Resample a 2D field (src) defined on irregular lon/lat to the target lon/lat grid.
    Uses scipy.griddata. Returns a 2D array on tgt grid.
    """
    mask = np.isfinite(src_field)
    if mask.sum() < 4:
        # not enough points; just return all-nan
        return np.full_like(tgt_lon2d, np.nan, dtype=float)

    pts = np.column_stack((src_lon[mask].ravel(), src_lat[mask].ravel()))
    vals = src_field[mask].ravel()
    tgt_pts = np.column_stack((tgt_lon2d.ravel(), tgt_lat2d.ravel()))

    out = griddata(pts, vals, tgt_pts, method=method)
    out = out.reshape(tgt_lon2d.shape)

    if np.isnan(out).any() and fill_method:
        # fill holes with a secondary method (usually 'nearest')
        fill = griddata(pts, vals, tgt_pts, method=fill_method).reshape(out.shape)
        nan_idx = np.isnan(out)
        out[nan_idx] = fill[nan_idx]

    return out

def geostationary_to_latlon(ds_msg,
                            target_latitudes,
                            target_longitudes,
                            method="linear",
                            bt_vars=BT_VARS,
                            clm_vars=CLM_VARS,
                            n_jobs=12):
    """
    Warp MSG geostationary 2D fields to a regular lat/lon grid, in parallel.
    - Continuous vars (BT) use `method` with nearest-fill.
    - Categorical vars (cloud_mask) use nearest only.
    Parallelization: per-variable over time with joblib.
    """
    # ---- target grid (compute once) ----
    lat_t = np.asarray(target_latitudes)
    lon_t = np.asarray(target_longitudes)
    Lon_t, Lat_t = np.meshgrid(lon_t, lat_t)

    # ---- source lon/lat (compute once) ----
    src_lon2d, src_lat2d = geos_xy_to_lonlat(ds_msg)

    times = ds_msg["time"].values
    candidate_vars = [v for v in (bt_vars + clm_vars) if v in ds_msg.data_vars]
    if not candidate_vars:
        raise ValueError("No MSG variables found matching BT_VARS/CLM_VARS in this dataset.")

    out = {}

    # helper to resample a single (var, time) slice
    def _resample_one(v, t_i, is_categorical):
        src = ds_msg[v].isel(time=t_i).values  # (y,x)
        if is_categorical:
            return _griddata_resample(src_lon2d, src_lat2d, src,
                                      Lon_t, Lat_t,
                                      method="nearest", fill_method=None).astype("float32")
        else:
            return _griddata_resample(src_lon2d, src_lat2d, src,
                                      Lon_t, Lat_t,
                                      method=method, fill_method="nearest").astype("float32")

    # parallel per variable (each submits a list of time jobs)
    for v in candidate_vars:
        is_cat = v in clm_vars
        # NOTE: use "processes" for best speed with SciPy; change to prefer="threads" if needed.
        frames = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_resample_one)(v, t_i, is_cat) for t_i in range(len(times))
        )
        out[v] = np.stack(frames, axis=0)

    data_vars = {v: (("time", "lat", "lon"), arr) for v, arr in out.items()}
    return xr.Dataset(data_vars=data_vars,
                      coords={"time": times, "lat": lat_t, "lon": lon_t})
#%% Main processing
# 1) Discover files
cml_files = _list_files(PATH_CML_RAIN)
msg_bt_files  = _list_files(PATH_MSG_BT)
msg_clm_files = _list_files(PATH_MSG_CLM)

if not cml_files:
    raise FileNotFoundError(f"No CML files in {PATH_CML_RAIN}")
if not msg_bt_files:
    raise FileNotFoundError(f"No MSG BT files in {PATH_MSG_BT}")
if not msg_clm_files:
    raise FileNotFoundError(f"No MSG CLM files in {PATH_MSG_CLM}")

# 2) Open with xarray (safe concat by coords)
#    Tip: chunk lightly if files are large; here we let xarray decide.
cml = xr.open_mfdataset(cml_files, combine='by_coords')
msg_bt  = xr.open_mfdataset(msg_bt_files, combine='by_coords')
msg_clm = xr.open_mfdataset(msg_clm_files, combine='by_coords')

# 3) Normalize times (to CML cadence) and drop duplicates
cml   = round_time_index(cml, TIME_ROUND)
msg_bt  = round_time_index(msg_bt, TIME_ROUND)
msg_clm = round_time_index(msg_clm, TIME_ROUND)

# --- 5) Time-align FIRST (minute rounding already done on your side) ---
common_times = np.intersect1d(np.intersect1d(cml.time.values, 
                                             msg_bt.time.values),
                                             msg_clm.time.values)
cml     = cml.sel(time=common_times)
msg_bt  = msg_bt.sel(time=common_times)
msg_clm = msg_clm.sel(time=common_times)


# --- 6) Clip CML to Ghana bbox ---
bbox = (-4.0, 1.5, 4.5, 11.5)
cml = cml.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

# --- 7) Attach CRS and warp MSG to lat/lon ---
# (Do this once per MSG dataset)
# Assign geostationary CRS (CF standard)
msg_bt = msg_bt.rio.write_crs("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", inplace=False)
msg_clm = msg_clm.rio.write_crs("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", inplace=False)

# Reproject both to EPSG:4326 (lat/lon)
msg_bt_ll  = msg_bt.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
msg_clm_ll = msg_clm.rio.reproject("EPSG:4326", resampling=Resampling.mode)

# --- 3) Clip warped MSG to Ghana bbox too ---
msg_bt_ll  = msg_bt_ll.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))
msg_clm_ll = msg_clm_ll.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))

# --- 4) Harmonize to Ghana grid resolution ---
res_deg = 0.03
target_lats = np.arange(bbox[2], bbox[3] + 1e-6, res_deg)
target_lons = np.arange(bbox[0], bbox[1] + 1e-6, res_deg)

msg_bt_ll  = msg_bt_ll.interp(y=target_lats, x=target_lons, method="linear")
msg_clm_ll = msg_clm_ll.interp(y=target_lats, x=target_lons, method="linear")
cml_on_msg = cml.interp(lat=target_lats, lon=target_lons, method="linear")

# --- 5) Sanity check alignment ---
assert np.allclose(msg_bt_ll.y, cml_on_msg.lat)
assert np.allclose(msg_bt_ll.x, cml_on_msg.lon)


#%% The Machine Learning Retrieval
#%% Machine Learning Retrieval (XGB, day-wise splits)
import numpy as np
import pandas as pd
import xarray as xr

# ML + metrics
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

# --------------------
# 0) Inputs already in memory
# --------------------
ds_bt   = msg_bt_ll.copy()     # has BT_IR108, BT_IR120, BT_WV062 (dims: time,y,x)
ds_clm  = msg_clm_ll.copy()    # optional cloud mask dataset (unused in features here)
ds_rain = cml_on_msg.copy()    # has R_mm_per_h (dims: time,y,x) — already on same grid

# --------------------
# 1) Build features + target on common grid/time
# --------------------
BT_IR108 = ds_bt["BT_IR108"]
BT_IR120 = ds_bt["BT_IR120"]
BT_WV062 = ds_bt["BT_WV062"]
BT_diff  = BT_IR108 - BT_WV062

R = ds_rain["R_mm_per_h"]

# Ensure coords/order identical (strict join: all must overlap)
BT_IR108, BT_IR120, BT_WV062, BT_diff, R = xr.align(
    BT_IR108, BT_IR120, BT_WV062, BT_diff, R, join="exact"
)

# Harmonize NaNs: valid where all features and target are finite
mask_valid = (
    np.isfinite(BT_IR108) &
    np.isfinite(BT_IR120) &
    np.isfinite(BT_WV062) &
    np.isfinite(R)
)
# If you want to gate by cloud mask presence too:
# mask_valid = mask_valid & np.isfinite(ds_clm["cloud_mask"])

# Stack to samples
feat = (
    xr.concat([BT_IR108, BT_IR120, BT_WV062, BT_diff], dim="feature")
      .transpose("time", "y", "x", "feature")
      .where(mask_valid)
      .stack(sample=("time", "y", "x"))
      .compute()
)  # -> (sample, feature)

tgt = (
    R.where(mask_valid)
     .stack(sample=("time", "y", "x"))
     .compute()
)  # -> (sample,)

# Valid mask over 'sample' (avoid apply_ufunc now that we've computed)
mask = np.isfinite(feat).all("feature") & np.isfinite(tgt)
feat = feat.isel(sample=mask)
tgt  = tgt.isel(sample=mask)

# Numpy arrays for XGB (float32 is ideal)
X = feat.values.astype("float32")        # (n_valid, 4)
y = tgt.values.astype("float32")         # (n_valid,)

# --------------------
# 2) Day-wise grouping for independent splits
# --------------------
# After stacking, 'time' is a coordinate along 'sample'
time_idx = pd.to_datetime(feat["time"].values)  # length == n_valid samples
groups   = time_idx.strftime("%Y-%m-%d").values  # group id per sample

# (Optional) If you want to roughly balance days across month thirds:
# day_of_month = time_idx.day.values
# strata = pd.cut(day_of_month, bins=[0,10,20,31], labels=[0,1,2]).astype(int)
# You can stratify manually at the day level if needed; here we keep it simple.

# --------------------
# 3) Split: 70% / 30% by day, then 10% of 70% for early stopping
# --------------------
rng = 42
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=rng)
train_idx, val30_idx = next(gss.split(X, y, groups=groups))

X_70, y_70, grp_70 = X[train_idx], y[train_idx], groups[train_idx]
X_30, y_30, grp_30 = X[val30_idx], y[val30_idx], groups[val30_idx]

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=123)
tr_idx, in_val_idx = next(gss_inner.split(X_70, y_70, groups=grp_70))

X_tr,  y_tr   = X_70[tr_idx],    y_70[tr_idx]
X_val, y_val  = X_70[in_val_idx],y_70[in_val_idx]

# --------------------
# 4) (Optional) log1p transform for heavy-tailed rain
# --------------------
USE_LOG1P = True
if USE_LOG1P:
    y_tr_t   = np.log1p(y_tr)
    y_val_t  = np.log1p(y_val)
    y_70_t   = np.log1p(y_70)
    invf     = np.expm1
else:
    y_tr_t, y_val_t, y_70_t = y_tr, y_val, y_70
    invf = lambda z: z

# --------------------
# 5) Train XGBoost (mean model)
# --------------------
dtr  = xgb.DMatrix(X_tr,  label=y_tr_t)
dval = xgb.DMatrix(X_val, label=y_val_t)

params_mean = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 10,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "seed": 150,
    "nthread": 0,  # use all cores
}

booster_mean = xgb.train(
    params_mean,
    dtr,
    num_boost_round=500,
    early_stopping_rounds=30,
    evals=[(dtr, "train"), (dval, "val")],
    verbose_eval=50
)

pred_tr_t  = booster_mean.predict(dtr)
pred_val_t = booster_mean.predict(dval)

# Invert transform if used
pred_tr  = invf(pred_tr_t)
pred_val = invf(pred_val_t)

# --------------------
# 6) Metrics
# --------------------
def rel_bias(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return np.nan
    mu_res = np.nanmean(sim[m] - obs[m])
    mu_obs = np.nanmean(obs[m])
    return np.nan if mu_obs == 0 else mu_res / mu_obs

def rmse(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(obs[m], sim[m]))

def corr(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return np.nan
    return np.corrcoef(obs[m], sim[m])[0, 1]

def det_metrics(obs, sim, thr=0.1):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    o = obs[m] >= thr
    s = sim[m] >= thr
    TP = np.sum(s & o)
    FP = np.sum(s & ~o)
    FN = np.sum(~s & o)
    pod = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    far = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan
    db  = (TP + FP) / (TP + FN) if (TP + FN) > 0 else np.nan
    return pod, far, csi, db

print("== Inner validation on 70% ==")
print("RelBias:", rel_bias(y_val, pred_val))
print("RMSE:", rmse(y_val, pred_val))
print("Corr:",  corr(y_val, pred_val))
print("POD, FAR, CSI, DetBias:", det_metrics(y_val, pred_val, thr=0.1))

# --------------------
# 7) Retrain on full 70% and evaluate on independent 30%
# --------------------
d70 = xgb.DMatrix(X_70, label=y_70_t)
booster_final = xgb.train(
    params_mean, d70,
    num_boost_round=int(booster_mean.best_iteration * 1.1)  # small cushion
)

pred_30_t = booster_final.predict(xgb.DMatrix(X_30))
pred_30   = invf(pred_30_t)

print("== Independent 30% ==")
print("RelBias:", rel_bias(y_30, pred_30))
print("RMSE:", rmse(y_30, pred_30))
print("Corr:",  corr(y_30, pred_30))
print("POD, FAR, CSI, DetBias:", det_metrics(y_30, pred_30, thr=0.1))

# --------------------
# 8) (Optional) Feature importances & housekeeping
# --------------------
# Feature names for clarity
booster_final.feature_names = ["BT_IR108", "BT_IR120", "BT_WV062", "BT_diff"]
print("Feature importance (gain):")
print(booster_final.get_score(importance_type="gain"))

# If memory is tight, drop big intermediates
del BT_IR108, BT_IR120, BT_WV062, BT_diff, R