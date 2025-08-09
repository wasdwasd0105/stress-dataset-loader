import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, List


class PropofolLoader:
    def __init__(self, subject_id: str, path: str = "."):
        """
        Initializes the loader for a specific subject.
        Small event files (LOC, ROC) are loaded on initialization.
        Feature data is loaded on demand.

        :param subject_id: The subject ID (e.g., "5" or "S5").
        :param path: The base directory where subject data is stored.
        """
        self.subject_id = f"S{subject_id}" if not subject_id.startswith("S") else subject_id
        self.path = path
        self.data: Dict[str, Union[pd.Series, float, pd.DataFrame]] = {}
        self.metadata: Dict[str, str] = {}
        self._subject_path = self.path

        # Load small, essential metadata and event files upfront.
        self._load_scalar_events()
        self._extract_metadata()

        self.features_list = ['eda_tonic', 'mu_amp', 'sigma_amp', 'events', 'muHR', 'sigmaHR',
                              'HF', 'muPR', 'sigmaPR', 'HFnu', 'muRR', 'sigmaRR', 'LF', 'pow_tot',
                              't_EDA_tonic', 'LFnu', 'ratio', 'LOC', 'ROC', 't_EDA', 't_HRV']

    def load_features(self, features: Union[str, List[str]]) -> None:
        """
        Loads one or more specified time-series features. It flexibly handles CSV files
        with either one (value), two (time, value) columns, or a single row of values.

        :param features: A single feature name (str) or a list of feature names (list[str]).
        """
        if isinstance(features, str):
            features = [features]  # Allow passing a single feature name as a string

        for feature in features:
            # Skip loading if the data is already in memory
            if feature in self.data:
                print(f"Feature '{feature}' is already loaded.")
                continue

            feature_file = os.path.join(self._subject_path, f"{self.subject_id}_{feature}.csv")

            if os.path.exists(feature_file):
                print(f"Loading data for feature: {feature} from {feature_file}...")
                try:
                    # Load the CSV file without assuming a data type initially.
                    df = pd.read_csv(feature_file, header=None)

                    # Case 1: CSV is a single row with many comma-separated values.
                    if df.shape[0] == 1 and df.shape[1] > 1:
                        print(
                            f"Diagnostic: Found 1 row, {df.shape[1]} columns. Stacking into a single column of values.")
                        values = df.stack().reset_index(drop=True)
                        self.data[feature] = pd.to_numeric(values, errors='coerce')
                        self.data[f"{feature}_time"] = pd.Series(range(len(values)))

                    # Case 2: CSV has multiple rows and two or more columns.
                    elif df.shape[1] >= 2:
                        print(
                            f"Diagnostic: Found {df.shape[0]} rows, {df.shape[1]} columns. Assuming Time (col 0), Value (col 1).")
                        self.data[f"{feature}_time"] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                        self.data[feature] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

                    # Case 3: CSV has multiple rows and exactly one column.
                    elif df.shape[1] == 1:
                        print(
                            f"Diagnostic: Found {df.shape[0]} rows, 1 column. Assuming values and generating time index.")
                        values = df.iloc[:, 0]
                        self.data[feature] = pd.to_numeric(values, errors='coerce')
                        self.data[f"{feature}_time"] = pd.Series(range(len(values)))

                    else:
                        print(
                            f"Warning: Feature file '{feature_file}' is empty or has an unsupported format. Skipping.")

                except Exception as e:
                    print(f"Error reading or processing file {feature_file}: {e}")
            else:
                print(f"Warning: Data file not found for feature '{feature}' at {feature_file}")

    def load_all(self) -> None:
        """
        Loads all available features for the subject.
        Useful if you know you will need all the data.
        """
        all_features = [
            "muRR", "sigmaRR", "muHR", "sigmaHR", "pow_tot", "LF", "HF", "ratio", "LFnu", "HFnu",
            "eda_tonic", "muPR", "sigmaPR", "mu_amp", "sigma_amp"
        ]
        print("Loading all features...")
        self.load_features(all_features)

    def _load_scalar_events(self) -> None:
        """Loads scalar event data like LOC, ROC, and other event markers."""
        for name in ["LOC", "ROC"]:
            file = os.path.join(self._subject_path, f"{self.subject_id}_{name}.csv")
            if os.path.exists(file):
                self.data[name] = float(pd.read_csv(file, header=None).squeeze())

        file = os.path.join(self._subject_path, f"{self.subject_id}_events.csv")
        if os.path.exists(file):
            self.data["events"] = pd.read_csv(file, header=None).squeeze().values

    def _extract_metadata(self) -> None:
        """A placeholder for extracting metadata."""
        self.metadata = {"Subject": self.subject_id}

    def get_data(self) -> Dict:
        """Returns the currently loaded data."""
        return self.data

    def plot(self, feature_name: Union[str, List[str]], max_points: int = 20000, title: Optional[str] = None) -> None:
        """
        Plot one or multiple features over the full available time range.
        Accepts a single feature name (str) or a list of names (List[str]).
        """
        # If a list is provided, plot each feature separately.
        if isinstance(feature_name, list):
            for f in feature_name:
                self._plot_one(str(f), max_points=max_points, title=title)
            return

        # Otherwise, plot the single requested feature.
        self._plot_one(str(feature_name), max_points=max_points, title=title)

    def _plot_one(self, feature_name: str, max_points: int = 20000, title: Optional[str] = None) -> None:
        if feature_name not in self.data or f"{feature_name}_time" not in self.data:
            self.load_features(feature_name)

        time = self.data.get(f"{feature_name}_time")
        values = self.data.get(feature_name)

        if time is None or values is None:
            print(f"Error: Could not load or find data for feature '{feature_name}'.")
            return

        combined = pd.DataFrame({'time': pd.to_numeric(time, errors='coerce'),
                                 'values': pd.to_numeric(values, errors='coerce')}).dropna()

        if combined.empty:
            print(f"Error: No valid data points found for '{feature_name}'.")
            return

        combined = combined.sort_values('time')
        # Per-channel downsample if very long
        if len(combined) > max_points:
            step = int(np.ceil(len(combined) / max_points))
            combined = combined.iloc[::step]

        plt.figure(figsize=(12, 6))
        plt.plot(combined['time'], combined['values'], label=feature_name)

        # Plot event markers if they exist
        # if "LOC" in self.data:
        #     plt.axvline(self.data["LOC"], color='red', linestyle='--', label='LOC')
        # if "ROC" in self.data:
        #     plt.axvline(self.data["ROC"], color='green', linestyle='--', label='ROC')
        # for ev in self.data.get("events", []):
        #     plt.axvline(ev, color='purple', linestyle=':', alpha=0.5)

        plt.title(title or f"Feature: {feature_name} for {self.subject_id}")
        plt.xlabel("Time (s)")
        plt.ylabel(feature_name)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Combine all currently loaded features into a single time-indexed DataFrame.
        Uses outer-join on time to align features with different time bases.
        """
        frames = []
        for key in list(self.data.keys()):
            if key.endswith("_time"):
                feat = key[:-5]
                if feat in self.data and isinstance(self.data[feat], pd.Series):
                    t = pd.Series(self.data[key]).astype(float)
                    v = self.data[feat]
                    df = pd.DataFrame({"time": t, feat: v}).dropna()
                    if not df.empty:
                        df = df.sort_values("time").set_index("time")
                        frames.append(df[[feat]])
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1).sort_index()
        self.signal_df = combined
        return combined

    def get_metadata(self) -> Dict[str, any]:
        """
        Return basic metadata including subject, available features, and time span (if any).
        """
        meta: Dict[str, any] = {"Subject": self.subject_id}
        # Features currently loaded
        feats = sorted([k for k in self.data.keys() if not k.endswith("_time") and isinstance(self.data[k], pd.Series)])
        meta["features_loaded"] = feats
        # Duration if we can build a combined frame
        try:
            df = self.signal_df if hasattr(self, "signal_df") and self.signal_df is not None else self.to_dataframe()
        except Exception:
            df = None
        if df is not None and not df.empty:
            idx = df.index.values
            meta["start"] = float(idx.min())
            meta["end"] = float(idx.max())
            meta["duration_sec"] = float(idx.max() - idx.min())
        # Events
        if "LOC" in self.data:
            meta["LOC"] = float(self.data["LOC"])
        if "ROC" in self.data:
            meta["ROC"] = float(self.data["ROC"])
        return meta

    def print_summary(self) -> None:
        meta = self.get_metadata()
        print("=== Propofol Subject Summary ===")
        print(f"Subject: {meta.get('Subject')}")
        # if "duration_sec" in meta:
        #     print(f"Time range (s): {meta.get('start')} — {meta.get('end')}")
        #     print(f"Duration (s)  : {meta.get('duration_sec')}")
        # feats = meta.get("features_loaded", [])
        print(f"Features loaded: {self.features_list}")
        if "LOC" in meta or "ROC" in meta:
            print(f"LOC: {meta.get('LOC', 'N/A')}, ROC: {meta.get('ROC', 'N/A')}")

    def plot_all(self, max_points: int = 20000) -> None:
        """
        Plot all currently loaded features; if none loaded yet, automatically load the full set.
        """

        print("Error: The Dataset is too large use plot all!")
        # If no features are loaded, try loading all known features
        # if not any(k for k in self.data.keys() if not k.endswith("_time")):
        #     self.load_all()
        # feats = sorted([k for k in self.data.keys() if not k.endswith("_time")])
        # for f in feats:
        #     self.plot(f, max_points=max_points)


loader = PropofolLoader("S9", path="../datasets_lite/propofol-anesthesia-dynamics/Data")
loader.print_summary()
loader.plot(["sigmaHR", "muHR"])
