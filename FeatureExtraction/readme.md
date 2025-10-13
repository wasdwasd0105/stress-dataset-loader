

# Feature Extraction Bridge

This provides a **bridge layer** that unifies access to multiple feature extraction providers.  
The bridge makes it easy to compute and export statistical, time-domain, and other features  
from biosignal datasets in a consistent JSON-friendly format.

## Overview

Instead of calling each provider separately, the bridge exposes **3 general functions** that serve  
as a unified API across providers:

1. **`export_json_result(data)`**  
   - Input: a NumPy array, Pandas Series, or DataFrame.  
   - Output: a JSON string with all extracted features, grouped by provider and module.  
   - Automatically includes available providers (e.g., `statistical_features`, `time_domain_features`, `freq_domain_features`).  
   - Produces well‑structured, indented JSON suitable for downstream processing or storage.

   Example output for a single column:
   ```json
   {
     "ECG": {
       "provider": {
         "name": "feature_engineering_and_extraction",
         "modules": {
           "statistical_features": {
             "features": {
               "mean": 0.01,
               "median": 0.00,
               "std": 0.25,
               "range": 1.5,
               "mode": -0.1,
               "skewness": 1.2,
               "kurtosis": 4.5
             }
           },
           "time_domain_features": {
             "features": {
               "rms": 0.24,
               "energy": 12.5,
               "average_power": 0.03
             }
           },
           "freq_domain_features": {
             "features": {
               "t_mean": 0.01,
               "t_std": 0.25,
               "...": "..."
             }
           }
         }
       }
     }
   }
   ```

2. **`allowed_input()`**  
   - Returns a list of candidate column names (or an empty list if none defined).  
   - Used to document what input column names are acceptable for IBI, EDA, HR, or other signals  
     depending on the provider bridge implementation.
   - Returns an empty list if the provider support all columns

   Example:
   ```python
   from FeatureExtraction.feature_engineering_and_extraction_bridge import allowed_input
   print(allowed_input())
   # ["IBI", "ibi", "RR", "rr", "ECG", "ecg"]
   ```

3. **`output_json_structure()`**  
   - Returns the structure of available modules and their feature names.  
   - Helpful to see what features can be expected in the JSON export without running the computations.

   Example:
   ```python
   from FeatureExtraction.feature_engineering_and_extraction_bridge import output_json_structure
   print(output_json_structure())
   # {
   #   "statistical_features": ["mean", "median", "std", "range", "mode", "skewness", "kurtosis"],
   #   "time_domain_features": ["rms", "energy", "average_power"],
   #   "freq_domain_features": ["t_mean", "t_std", ..., "f_fourth_central_moment"]
   # }
   ```

## Supported Providers

- **Statistical Features & Time-Domain Features (`feature_engineering_and_extraction`)**
  - Provides mean, median, std, range, mode, skewness, and kurtosis.
  - Provides RMS, energy, and average power.

- **Heart Rate Variability (HRV)** (`Heart_Rate_Variability`)
  -  HRV metrics (e.g., IBI column candidates, RR intervals, etc.).

- **WearableCompute** (`wearablecompute`)
  -  Wearable compute metrics with allowed input candidates.

    
## Usage in CLI (developing)

The main CLI (`DatasetLoaderCli.py`) integrates the bridge. You can use:

- `-feature-extraction` → specify choose the feature extraction provider
- `-stats` → prints human-readable summaries  
- `-stats-json` → prints structured JSON with features  
- `-stats-cols` → specify which columns to summarize


Example:
```bash
(base) wasdwasd0105@Mac stress-dataset-loader % python DatasetLoaderCli.py -dataset ADARP -example-case -feature-extraction Heart_Rate_Variability -stats-json -stats-cols HR
Dataset: ADARP
Loader : EmpaticaE4Loader
Case   : Part 112C/A01d53_200210-194142
Data   : /Users/wasdwasd0105/GitHub/stress-dataset-loader/datasets_lite/ADARP
=== Empatica E4 Summary ===
Case: Part 112C/A01d53_200210-194142
Signals: ['TEMP', 'HR', 'ACC_x', 'ACC_y', 'ACC_z', 'IBI', 'EDA', 'BVP']
Duration (s): 120171.71875
Start: 1581363702.0
End:   1581483873.71875
Sampling rate estimates (Hz):
  ACC_x: 32
  ACC_y: 32
  ACC_z: 32
  BVP: 64
  EDA: 4
  HR: 1
  IBI: 1.42216
  TEMP: 4
{
  "provider": {
    "name": "Heart_Rate_Variability",
    "modules": {
      "BIL_HRV": {
        "features": {
          "MeanRR": 96.4,
          "MeanHR": 643.0,
          "MinHR": 306.1,
          "MaxHR": 1135.0,
          "SDNN": 17.9,
          "RMSSD": 0.5,
          "NNx": 0.0,
          "pNNx": 0.0,
          "PowerVLF": 61.31,
          "PowerLF": 23.98,
          "PowerHF": 0.81,
          "PowerTotal": 86.09,
          "LF/HF": 29.69,
          "PeakVLF": 0.02,
          "PeakLF": 0.05,
          "PeakHF": 0.16,
          "FractionLF": 96.74,
          "FractionHF": 3.26
        }
      }
    }
  }
}
[INFO] No plotting requested.
(base) wasdwasd0105@Mac stress-dataset-loader % 
```


## UI coming soon