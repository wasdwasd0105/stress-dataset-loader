from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Sequence, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as ss

# Provider import
from FeatureExtraction.providers.Heart_Rate_Variability import BIL_HRV as _hrv_provider



# ---- Provider monkey-patch (do NOT edit provider file) ----------------------
# Provider imports `trapezoid` but calls `trapz`. Inject alias.
try:
    from scipy.integrate import trapezoid as _scipy_trapezoid
    if getattr(_hrv_provider, "trapz", None) is None:
        setattr(_hrv_provider, "trapz", _scipy_trapezoid)
except Exception:
    # Fallback using numpy
    import numpy as _np
    def _np_trapz(y, x):
        return _np.trapz(y, x)
    if getattr(_hrv_provider, "trapz", None) is None:
        setattr(_hrv_provider, "trapz", _np_trapz)


ArrayLike = Union[np.ndarray, "pd.Series"]
TableLike = "pd.DataFrame"


# -------------------------
# Internal utilities
# -------------------------
def _to_bool_flag(v: Union[str, bool]) -> str:
    """
    Normalize various truthy/falsy inputs to the strings "true"/"false"
    because the provider expects string flags.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return "true"
    if s in {"false", "f", "0", "no", "n"}:
        return "false"
    return "false"


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """
    Return the first column name from candidates that exists in df.columns, or None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None

def allowed_input() -> List[str]:
    """
    Return the list of accepted IBI column name candidates.
    """
    return ["IBI", "ibi", "RR", "rr", "ECG", "ecg"]

def _guess_ibi_col(df: pd.DataFrame) -> str | None:
    """
    Guess the IBI column from common names. If none match and there is exactly
    one numeric column, use that as IBI.
    """
    ibi_candidates = allowed_input()
    col = _find_column(df, ibi_candidates)
    if col:
        return col
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # If exactly one numeric column, treat it as IBI
    if len(num_cols) == 1:
        return num_cols[0]
    return None


def _guess_time_col(df: pd.DataFrame) -> str | None:
    """
    Guess the time column from common names.
    """
    time_candidates = ["time", "timestamp", "t", "Time", "Timestamp"]
    return _find_column(df, time_candidates)


def _coerce_time_ibi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['time','IBI'] in SECONDS.
    - If no explicit 'IBI' column, try common aliases or (if only one numeric column) use it.
    - If no explicit 'time' column, create it from the cumulative sum of IBI.
    - If IBI appears to be in ms (median > 10), convert to seconds.
    - If time is datetime-like or epoch-ms, convert to seconds offset from the first sample.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("Empty DataFrame provided.")

    ibi_col = _guess_ibi_col(df)
    if ibi_col is None:
        raise ValueError("Could not infer IBI column (IBI/RR/RRI/interval). "
                         "Provide a single numeric column or a named IBI-like column.")

    time_col = _guess_time_col(df)  # may be None

    # IBI values
    ibi_raw = pd.to_numeric(df[ibi_col], errors="coerce").to_numpy()
    ibi_raw = ibi_raw[~np.isnan(ibi_raw)]
    if ibi_raw.size == 0:
        raise ValueError("IBI column contains no numeric values.")

    # Unit inference: if median > 10 assume ms; otherwise seconds
    med = float(np.nanmedian(ibi_raw))
    ibi_sec = ibi_raw / 1000.0 if med > 10.0 else ibi_raw

    # Time vector
    if time_col is None:
        time_sec = np.cumsum(ibi_sec)
        # Start at 0s; align length with IBI count
        time_sec = np.concatenate(([0.0], time_sec))[:ibi_sec.size]
    else:
        tc = df[time_col]
        if np.issubdtype(tc.dtype, np.datetime64):
            t0 = tc.dropna().iloc[0]
            time_sec = (tc - t0).dt.total_seconds().to_numpy()
        else:
            tnum = pd.to_numeric(tc, errors="coerce").to_numpy()
            # If likely epoch ms (very large), normalize and convert to seconds
            if np.nanmedian(tnum) > 1e8:  # ~3 years in seconds; treat as epoch secs/ms
                if np.nanmedian(tnum) > 1e12:  # epoch ms
                    time_sec = (tnum - np.nanmin(tnum)) / 1000.0
                else:  # epoch seconds
                    time_sec = tnum - np.nanmin(tnum)
            else:
                # Otherwise assume the units are already seconds
                time_sec = tnum
        # Align lengths in case of mismatch
        n = min(len(time_sec), len(ibi_sec))
        time_sec = time_sec[:n]
        ibi_sec = ibi_sec[:n]

    out = pd.DataFrame({"time": time_sec, "IBI": ibi_sec})
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("Coercion to ['time','IBI'] resulted in empty data after cleaning.")
    return out


def _write_temp_ibi_csv(df: pd.DataFrame) -> str:
    """
    Provider reads from path; write a temp CSV with required columns ['time','IBI'].
    Assumes IBI is in seconds (provider multiplies by 1000 internally).
    """
    df_coerced = _coerce_time_ibi_dataframe(df)
    tmp = tempfile.NamedTemporaryFile(prefix="hrv_", suffix=".csv", delete=False)
    df_coerced.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _metric_keys() -> List[str]:
    """
    Canonical HRV metrics exposed by provider.hrv().
    """
    return [
        "MeanRR", "MeanHR", "MinHR", "MaxHR", "SDNN", "RMSSD", "NNx", "pNNx",
        "PowerVLF", "PowerLF", "PowerHF", "PowerTotal", "LF/HF",
        "PeakVLF", "PeakLF", "PeakHF", "FractionLF", "FractionHF",
    ]


# -------------------------
# Public API (match user's structure)
# -------------------------
def output_json_structure() -> Dict[str, List[str]]:
    """
    Return a dict: {module_name: [feature_names...]} for available providers.
    For HRV bridge, the module is 'BIL_HRV' and values are metric names.
    """
    return {"BIL_HRV": _metric_keys()}


def available_functions() -> Dict[str, List[str]]:
    return output_json_structure()


def export_json_result(
    data: pd.DataFrame,
    *,
    complete_sequence: Union[str, bool] = "false",
    threshold: float = 0.1,
    x: int = 50,
    correction: Union[str, bool] = "false",
    fs: int = 4,
) -> str:
    """
    Compute HRV on a DataFrame (will coerce columns to ['time','IBI'] if needed).

    Structure:
    {
      "provider": {
        "name": "Heart_Rate_Variability",
        "modules": {
          "BIL_HRV": { "features": { ... } }
        }
      }
    }
    """
    # Validate input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("For HRV, `data` must be a pandas DataFrame with ['time','IBI'].")

    # Prepare file path for provider
    temp_path: str | None = None
    file_path = _write_temp_ibi_csv(data)
    temp_path = file_path

    try:
        # Normalize flags for provider
        complete_sequence_s = _to_bool_flag(complete_sequence)
        correction_s = _to_bool_flag(correction)

        metrics: Dict[str, Any] = _hrv_provider.hrv(
            file=file_path,
            complete_sequence=complete_sequence_s,
            threshold=threshold,
            x=x,
            correction=correction_s,
            fs=fs,
        )

        # Ensure plain JSON types
        def _clean(v: Any) -> Any:
            if isinstance(v, (np.generic,)):
                return v.item()
            return v

        metrics = {k: _clean(v) for k, v in metrics.items()}

        result = {
            "provider": {
                "name": "Heart_Rate_Variability",
                "modules": {
                    "BIL_HRV": {
                        "features": metrics
                    }
                }
            }
        }
        return json.dumps(result, ensure_ascii=False)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

