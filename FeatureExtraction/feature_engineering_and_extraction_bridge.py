from __future__ import annotations

import json
from typing import Dict, List, Sequence, Union, Any

import numpy as np
import pandas as pd


from FeatureExtraction.providers.feature_engineering_and_extraction import statistical_features as _sf
from FeatureExtraction.providers.feature_engineering_and_extraction import time_domain_features as _tdf
from FeatureExtraction.providers.feature_engineering_and_extraction import freq_domian_features as _fdf
import scipy.stats as ss


ArrayLike = Union[np.ndarray, "pd.Series"]
TableLike = "pd.DataFrame"


# -------------------------
# Internal utilities
# -------------------------
def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    """Convert Series/array to 1D float array and drop NaNs."""
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _compute_mode(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    if ss is None:
        # Simple fallback mode
        vals, counts = np.unique(arr, return_counts=True)
        return float(vals[np.argmax(counts)])
    try:
        m = ss.mode(arr, keepdims=False)
        val = getattr(m, "mode", m)
        if hasattr(val, "__len__"):
            return float(val[0])
        return float(val)
    except Exception:
        vals, counts = np.unique(arr, return_counts=True)
        return float(vals[np.argmax(counts)])


_FEATURE_REGISTRY: Dict[str, List[str]] = {
    "statistical_features": ["mean", "median", "std", "range", "mode", "skewness", "kurtosis"],
    # minimal TD features from time_domain_features.py
    "time_domain_features": ["rms", "energy", "average_power"],
    # comprehensive names used when freq provider is available (we'll fill dynamically)
    # "freq_domain_features": [
    #     # Time_feature (11)
    #     "t_mean", "t_std", "t_sqrt_amp", "t_rms", "t_peak",
    #     "t_skew", "t_kurtosis", "t_crest_factor", "t_clearance_factor",
    #     "t_shape_factor", "t_impulse_factor",
    #     # Fre_feature (12)
    #     "f_mean", "f_var", "f_skewness", "f_kurtosis", "f_central_freq",
    #     "f_rms_freq_dev", "f_root_moment2", "f_root_moment4_over_moment2",
    #     "f_spectral_variation", "f_frequency_index", "f_third_central_moment",
    #     "f_fourth_central_moment",
    # ],
}


# -------------------------
# Public API
# -------------------------
def output_json_structure() -> Dict[str, List[str]]:
    """
    Return a dict: {module_name: [feature_names...]} for available providers.
    """
    available = {"statistical_features": _FEATURE_REGISTRY["statistical_features"]}
    if _tdf is not None:
        available["time_domain_features"] = _FEATURE_REGISTRY["time_domain_features"]
    if _fdf is not None:
        pass
        #available["freq_domain_features"] = _FEATURE_REGISTRY["freq_domain_features"]
    return available


def available_functions() -> Dict[str, List[str]]:
    return output_json_structure()


def print_summarize(data: Union[ArrayLike, TableLike]) -> None:
    """
    Print a human-readable summary (delegates to provider's Features.summarize()).

    - If `data` is a 1D array/Series: prints one summary.
    - If `data` is a DataFrame: prints a summary per numeric column.
    """
    if pd is not None and isinstance(data, pd.DataFrame):
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            print("[INFO] No numeric columns to summarize.")
            return
        for col in cols:
            arr = _as_1d_float_array(data[col])
            print(f"\n=== Summary for column: {col} ===")
            _sf.Features(arr).summarize()
        return

    # Single vector
    arr = _as_1d_float_array(data)  # type: ignore[arg-type]
    _sf.Features(arr).summarize()


def _extract_time_domain_basic(data: ArrayLike) -> Dict[str, float]:
    arr = _as_1d_float_array(data)
    if arr.size == 0 or _tdf is None:
        return {}
    try:
        return {
            "rms": float(_tdf.root_mean_square(arr)),
            "energy": float(_tdf.energy(arr)),
            "average_power": float(_tdf.average_power(arr)),
        }
    except Exception:
        return {}


def _extract_freq_domain(data: ArrayLike, fs: float = 25600.0) -> Dict[str, float]:
    arr = _as_1d_float_array(data)
    if arr.size == 0 or _fdf is None:
        return {}
    try:
        # will not work due to Fea_Extra errors
        fx = _fdf.Fea_Extra(arr, Fs=fs)
        t_vec = fx.Time_feature(arr)
        f_vec = fx.Fre_feature(arr)
        t_names = _FEATURE_REGISTRY["freq_domain_features"][:11]
        f_names = _FEATURE_REGISTRY["freq_domain_features"][11:]
        out: Dict[str, float] = {}
        for name, val in zip(t_names, t_vec):
            out[name] = float(val)
        for name, val in zip(f_names, f_vec):
            out[name] = float(val)
        return out
    except Exception:
        # if provider has syntax/import issues, silently skip
        return {}


def _extract_all_vector(data: ArrayLike) -> Dict[str, float]:
    """
    Extract all canonical features for a 1D vector.
    """
    arr = _as_1d_float_array(data)
    if arr.size == 0:
        return {name: float("nan") for name in _FEATURE_REGISTRY["statistical_features"]}

    # Use numpy for classic stats; provider for skew/kurtosis to mirror summarize()
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=0))
    rng = float(np.max(arr) - np.min(arr))
    mode = _compute_mode(arr)

    feats = _sf.Features(arr)
    skewness = float(feats.get_skewness(bias=False))
    kurtosis = float(feats.get_kurtosis(bias=False))

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "range": rng,
        "mode": mode,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def export_json_result(data: Union[ArrayLike, TableLike]) -> str:
    """
    Return all extraction results in JSON format using a multi-module structure.

    Structure:
    {
      "<column or vector>": {
        "provider": {
          "name": "feature_engineering_and_extraction",
          "modules": {
            "statistical_features": {"features": {...}},
            "time_domain_features": {"features": {...}},
          }
        }
      }
    }
    Only available modules are included.
    """
    def _build_modules(vec: ArrayLike) -> Dict[str, Dict[str, Dict[str, float]]]:
        modules: Dict[str, Dict[str, Dict[str, float]]] = {}
        # statistical
        modules["statistical_features"] = {"features": _extract_all_vector(vec)}
        # time-domain (optional)
        td = _extract_time_domain_basic(vec)
        if td:
            modules["time_domain_features"] = {"features": td}
        # frequency-domain (optional)
        fd = _extract_freq_domain(vec)
        if fd:
            modules["freq_domain_features"] = {"features": fd}
        return modules

    # DataFrame case
    if pd is not None and isinstance(data, pd.DataFrame):
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
        result: Dict[str, Any] = {}
        for col in cols:
            result[col] = {
                "provider": {
                    "name": "feature_engineering_and_extraction",
                    "modules": _build_modules(data[col])
                }
            }
        return json.dumps(result, ensure_ascii=False)

    # Single vector case
    result = {
        "provider": {
            "name": "feature_engineering_and_extraction",
            "modules": _build_modules(data)  # type: ignore[arg-type]
        }
    }
    return json.dumps(result, ensure_ascii=False)



# -------------------------
# Allowed input columns placeholder
# -------------------------
from typing import List

def allowed_input() -> List[str]:
    """
    Return an empty list; placeholder for allowed input column candidates.
    """
    return []
