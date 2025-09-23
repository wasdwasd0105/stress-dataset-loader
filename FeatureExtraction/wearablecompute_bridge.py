from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# --- Helper to sanitize floats for JSON ---
def _clean_float(x: Any) -> Any:
    try:
        if isinstance(x, (np.floating,)):
            x = float(x)
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x
    except Exception:
        return x

# ---- Provider monkey-patch (do NOT edit provider file) ----------------------
# Provider imports `trapezoid` but calls `trapz`. Inject alias.
try:
    import scipy.integrate as _sciint
    if not hasattr(_sciint, "trapz"):
        import numpy as _np
        _sciint.trapz = _np.trapz  # provide fallback
except Exception:
    pass

# Provider import
from FeatureExtraction.providers.wearablecompute import wearablecompute as _wc


ArrayLike = Union[np.ndarray, "pd.Series"]
TableLike = "pd.DataFrame"

# -------------------------
# Column inference helpers
# -------------------------

def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def _guess_time_col(df: pd.DataFrame) -> str | None:
    time_candidates = [
        "time", "timestamp", "t", "date", "datetime",
        "time_s", "time_sec", "seconds", "ms", "millis",
    ]
    col = _find_column(df, time_candidates)
    if col:
        return col
    # Datetime-like columns
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    # Monotonic numeric fallback
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.Series(df[c]).dropna()
            if s.index.is_monotonic_increasing:
                return c
    return None



def allowed_input() -> List[str]:
    """
    Return the list of accepted IBI column name candidates for wearablecompute.
    """
    return [
        "ibi", "rr", "rri", "rr_interval", "rr_ms", "rr_s",
        "interbeat_interval", "nn", "nn_interval", "interval",
    ]

def _guess_ibi_col(df: pd.DataFrame) -> str | None:
    candidates = allowed_input()
    col = _find_column(df, candidates)
    if col:
        return col
    # Single numeric column fallback
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return num_cols[0]
    return None


def _guess_eda_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "eda", "gsr", "eda_tonic", "eda_phasic", "electrodermal",
    ]
    col = _find_column(df, candidates)
    if col:
        return col
    # substring heuristic
    for c in df.columns:
        name = str(c).lower()
        if "eda" in name or "gsr" in name:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    return None


def _guess_hr_col(df: pd.DataFrame) -> str | None:
    candidates = ["hr", "heart_rate", "bpm"]
    return _find_column(df, candidates)


def _guess_acc_magnitude(df: pd.DataFrame) -> str | None:
    # If already a magnitude column
    candidates = ["acc", "accel", "acc_mag", "activity", "var"]
    col = _find_column(df, candidates)
    if col:
        return col
    # If x,y,z present, compute magnitude on the fly in export function
    xyz_sets = [
        ("acc_x", "acc_y", "acc_z"),
        ("x", "y", "z"),
    ]
    for xs, ys, zs in xyz_sets:
        if _find_column(df, [xs]) and _find_column(df, [ys]) and _find_column(df, [zs]):
            return "__compute_xyz__"
    return None


# -------------------------
# Public API (similar to HRV bridge)
# -------------------------

def output_json_structure() -> Dict[str, List[str]]:
    """
    Potential features wearablecompute can output (conditionally based on available columns).
    """
    features = [
        # HRV time-domain
        "HRV_Max", "HRV_Min", "HRV_Mean", "HRV_Median", "SDNN", "RMSSD", "NNx", "pNNx",
        # HRV frequency-domain
        "PowerVLF", "PowerLF", "PowerHF", "PowerTotal", "LF/HF",
        "PeakVLF", "PeakLF", "PeakHF", "FractionLF", "FractionHF",
        # EDA peaks
        "EDA_Peaks_Count",
        # Activity bouts
        "Activity_Bouts_Count",
    ]
    return {"wearablecompute": features}


def available_functions() -> Dict[str, List[str]]:
    return output_json_structure()


def _ensure_time_seconds(series: pd.Series) -> np.ndarray:
    s = series
    if np.issubdtype(s.dtype, np.datetime64):
        t0 = s.dropna().iloc[0]
        return (s - t0).dt.total_seconds().to_numpy()
    # numeric: normalize epoch-like ms/seconds
    tnum = pd.to_numeric(s, errors="coerce").to_numpy()
    if np.nanmedian(tnum) > 1e12:  # epoch ms
        return (tnum - np.nanmin(tnum)) / 1000.0
    if np.nanmedian(tnum) > 1e8:  # epoch seconds
        return tnum - np.nanmin(tnum)
    return tnum


def export_json_result(
    df: pd.DataFrame,
    *,
    ibimultiplier: int = 1000,
    x: int = 50,
    fs: int = 4,
) -> str:
    """
    Compute wearablecompute features from a DataFrame. Columns are inferred where possible.
    Returns JSON with structure:
    {
      "provider": {
        "name": "wearablecompute",
        "modules": { "wearablecompute": { "features": { ... } } }
      }
    }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("wearablecompute bridge expects a pandas DataFrame")
    feats: Dict[str, Any] = {}

    # --- HRV metrics (require IBI; time optional) ---
    ibi_col = _guess_ibi_col(df)
    if ibi_col is not None:
        ibi_series = pd.to_numeric(df[ibi_col], errors="coerce")
        time_col = _guess_time_col(df)

        if time_col is not None:
            time_arr = _ensure_time_seconds(df[time_col])
            time_series = pd.Series(time_arr)
            mask = np.isfinite(ibi_series.to_numpy()) & np.isfinite(time_series.to_numpy())
            ibi_series = ibi_series[mask].reset_index(drop=True)
            time_series = time_series[mask].reset_index(drop=True)
        else:
            # synthesize time from IBI (seconds); drop NaNs first
            ibi_drop = ibi_series.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            if ibi_drop.empty:
                time_series = pd.Series([], dtype=float)
                ibi_series = ibi_drop
            else:
                # infer units: ms vs s by median
                med = float(np.nanmedian(ibi_drop)) if ibi_drop.size else float("nan")
                ibi_sec = (ibi_drop.to_numpy() / 1000.0) if (np.isfinite(med) and med > 10.0) else ibi_drop.to_numpy()
                time_series = pd.Series(np.concatenate(([0.0], np.cumsum(ibi_sec)))[: len(ibi_sec)])
                ibi_series = pd.Series(ibi_drop.to_numpy())

        # Remove any remaining non-finite values
        if not time_series.empty and not ibi_series.empty:
            mask2 = np.isfinite(time_series.to_numpy()) & np.isfinite(ibi_series.to_numpy())
            time_series = time_series[mask2].reset_index(drop=True)
            ibi_series = ibi_series[mask2].reset_index(drop=True)

        # Only compute if we have enough samples
        if len(ibi_series) >= 5:
            # time-domain HRV
            try:
                maxHRV, minHRV, meanHRV, medianHRV = _wc.HRV(time_series, ibi_series, ibimultiplier=1000)
                feats.update({
                    "HRV_Max": _clean_float(maxHRV),
                    "HRV_Min": _clean_float(minHRV),
                    "HRV_Mean": _clean_float(meanHRV),
                    "HRV_Median": _clean_float(medianHRV),
                })
            except Exception:
                pass
            try:
                feats["SDNN"] = _clean_float(_wc.SDNN(time_series, ibi_series, ibimultiplier=1000))
            except Exception:
                pass
            try:
                feats["RMSSD"] = _clean_float(_wc.RMSSD(time_series, ibi_series, ibimultiplier=1000))
            except Exception:
                pass
            try:
                nnx, pnnx = _wc.NNx(time_series, ibi_series, ibimultiplier=1000, x=x)
                feats["NNx"] = _clean_float(nnx)
                feats["pNNx"] = _clean_float(pnnx)
            except Exception:
                pass

            # frequency-domain HRV (need more samples for a stable Welch PSD)
            try:
                if len(ibi_series) >= 16:
                    freq = _wc.FrequencyHRV(ibi_series, ibimultiplier=1000, fs=fs)
                    for k in [
                        "PowerVLF", "PowerLF", "PowerHF", "PowerTotal", "LF/HF",
                        "PeakVLF", "PeakLF", "PeakHF", "FractionLF", "FractionHF",
                    ]:
                        if k in freq:
                            feats[k] = _clean_float(freq[k])
            except Exception:
                pass

    # --- EDA peaks (require eda + time) ---
    eda_col = _guess_eda_col(df)
    t_col = _guess_time_col(df)
    if eda_col is not None and t_col is not None:
        try:
            time_for_eda = pd.Series(pd.to_datetime(df[t_col], errors="coerce")) if not np.issubdtype(df[t_col].dtype, np.datetime64) else df[t_col]
            countpeaks, _peakdf = _wc.PeaksEDA(df[eda_col], time_for_eda)
            feats["EDA_Peaks_Count"] = int(countpeaks)
        except Exception:
            pass

    # --- Activity bouts (require acc magnitude + hr + time) ---
    acc_key = _guess_acc_magnitude(df)
    hr_key = _guess_hr_col(df)
    if acc_key is not None and hr_key is not None and t_col is not None:
        try:
            if acc_key == "__compute_xyz__":
                # compute magnitude from x,y,z (prefer acc_x,acc_y,acc_z; else x,y,z)
                cand_sets = [("acc_x", "acc_y", "acc_z"), ("x", "y", "z")]
                ax = ay = az = None
                for xs, ys, zs in cand_sets:
                    ax = _find_column(df, [xs])
                    ay = _find_column(df, [ys])
                    az = _find_column(df, [zs])
                    if ax and ay and az:
                        break
                acc_mag = np.sqrt(pd.to_numeric(df[ax], errors="coerce")**2 +
                                   pd.to_numeric(df[ay], errors="coerce")**2 +
                                   pd.to_numeric(df[az], errors="coerce")**2)
            else:
                acc_mag = pd.to_numeric(df[acc_key], errors="coerce")
            # ensure same length
            n = min(len(acc_mag), len(df[hr_key]), len(df[t_col]))
            acc_mag = acc_mag.iloc[:n]
            hr_ser = pd.to_numeric(df[hr_key], errors="coerce").iloc[:n]
            time_ser = df[t_col].iloc[:n]
            countbouts, _ret = _wc.exercisepts(acc_mag.reset_index(drop=True), hr_ser.reset_index(drop=True), time_ser.reset_index(drop=True))
            feats["Activity_Bouts_Count"] = int(countbouts)
        except Exception:
            pass

    # Sanitize metrics for JSON (no NaN/Inf)
    feats = {k: _clean_float(v) for k, v in feats.items()}
    result = {
        "provider": {
            "name": "wearablecompute",
            "modules": {"wearablecompute": {"features": feats}}
        }
    }
    return json.dumps(result, ensure_ascii=False)
