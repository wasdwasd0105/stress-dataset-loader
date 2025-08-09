from __future__ import annotations

import os
import zipfile
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EmpaticaE4Loader:
    """
    Unified-style Empatica E4 loader (keeps your original parsing behavior).

    Public API:
      - E4Loader(case, path)
      - to_dataframe() -> pd.DataFrame         # time-indexed table (seconds float by default)
      - get_metadata() -> Dict[str, any]
      - print_summary() -> None
      - plot(features: List[str], start=None, end=None, max_points=20000, title=None) -> None
      - plot_all(**kwargs) -> None
    """

    def __init__(self, case: str, path: str, prefer_datetime_index: bool = False):
        """
        Parameters:
        - case: subfolder like 'S9/midterm_2' or zip file like 'S15_E4.zip'
        - path: base path like './e4/wearable-exam-stress/data'
        - prefer_datetime_index: if True, combined index will be pandas datetime;
                                 otherwise it's seconds (float).
        """
        self.case = case
        self.prefer_datetime_index = prefer_datetime_index

        # Resolve root folder (unzipping if needed)
        if case.endswith(".zip"):
            zip_path = os.path.join(path, case)
            extract_folder = os.path.join(path, case[:-4])  # remove .zip
            if not os.path.exists(extract_folder):
                print(f"[INFO] Unzipping {zip_path} to {extract_folder} ...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                print(f"[INFO] Extraction complete.")
            self.root = extract_folder
        else:
            self.root = os.path.join(path, case)

        self.signals: Dict[str, pd.DataFrame] = {}
        self.signal_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, any] = {}

        self._load_all_signals()
        self._build_combined_dataframe()
        self._compute_metadata()

    # ---------- Unified API ----------

    def to_dataframe(self) -> pd.DataFrame:
        if self.signal_df is None:
            self._build_combined_dataframe()
        return self.signal_df

    def get_metadata(self) -> Dict[str, any]:
        return self.metadata

    def print_summary(self) -> None:
        print("=== Empatica E4 Summary ===")
        print(f"Case: {self.case}")
        cols = list(self.signal_df.columns) if self.signal_df is not None else []
        print(f"Signals: {cols}")
        dur = self.metadata.get("duration_sec")
        print(f"Duration (s): {dur if dur is not None else 'N/A'}")
        if "start" in self.metadata and "end" in self.metadata:
            print(f"Start: {self.metadata['start']}")
            print(f"End:   {self.metadata['end']}")
        if "fs_estimate" in self.metadata:
            print("Sampling rate estimates (Hz):")
            for k, v in sorted(self.metadata["fs_estimate"].items()):
                if v is not None:
                    print(f"  {k}: {v:.6g}")

    def plot(
        self,
        features: List[str],
        start: Optional[float] = None,
        end: Optional[float] = None,
        max_points: int = 20000,
        title: Optional[str] = None,
    ) -> None:
        """
        One figure per feature.
        Downsamples per-channel using its non-NaN rows to keep sparse signals (e.g., HR) intact.
        Time window is in seconds relative to the start of the combined index; if the index is
        datetime (prefer_datetime_index=True), start/end are offsets (seconds) from first timestamp.
        """
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data loaded.")

        # Build a windowed view without destroying sparsity
        view = df
        if start is not None or end is not None:
            if self.prefer_datetime_index and not np.issubdtype(df.index.dtype, np.floating):
                base = df.index.min()
                s = base if start is None else base + pd.to_timedelta(float(start), unit="s")
                e = df.index.max() if end is None else base + pd.to_timedelta(float(end), unit="s")
                view = df.loc[s:e]
            else:
                s = 0.0 if start is None else float(start)
                e = df.index.max() if end is None else float(end)
                view = df.loc[s:e]

        available = set(view.columns)
        missing = [c for c in features if c not in available]
        if missing:
            raise ValueError(f"Signals not found: {missing}\nAvailable: {sorted(available)}")

        for ch in features:
            series = view[ch].dropna()
            plt.figure(figsize=(12, 4))
            if series.empty:
                # show a helpful empty panel
                ax = plt.gca()
                ax.set_title(title or f"{self.case} — {ch}")
                ax.set_xlabel("Time (s)" if np.issubdtype(view.index.dtype, np.floating) else "Time")
                ax.set_ylabel(ch)
                ax.text(0.5, 0.5, "No data in window", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.show()
                continue

            # per-channel downsampling
            if len(series) > max_points:
                step = int(np.ceil(len(series) / max_points))
                series = series.iloc[::step]

            plt.plot(series.index.values, series.values, linewidth=0.8)
            plt.xlabel("Time (s)" if np.issubdtype(series.index.dtype, np.floating) else "Time")
            plt.ylabel(ch)
            plt.title(title or f"{self.case} — {ch}")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()

    def plot_all(self, **kwargs) -> None:
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data loaded.")
        self.plot(features=list(df.columns), **kwargs)

    # ---------- Your original behavior (unchanged parsing) ----------

    def _parse_start_time(self, line: str) -> float:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                return datetime.strptime(line.strip(), fmt).timestamp()
            except ValueError:
                continue
        try:
            return float(line)
        except ValueError:
            raise ValueError(f"Unrecognized start time format: {line}")

    def _load_all_signals(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Case path not found: {self.root}")

        for file in os.listdir(self.root):
            if not file.endswith(".csv"):
                continue
            # Skip tag and response
            if file.lower().startswith("tags") or file.lower() == "response.csv":
                continue

            signal_name = file[:-4]  # drop .csv
            full_path = os.path.join(self.root, file)
            self.signals[signal_name] = self._load_signal(full_path, signal_name)

    def _load_signal(self, filepath: str, signal_name: str) -> pd.DataFrame:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline()
                has_three_column_header = ("," in header_line and "time" in header_line.lower())

            # Newer format: index, value, timestamp (headered)
            if has_three_column_header:
                df = pd.read_csv(filepath, skiprows=1, header=None,
                                 names=["index", signal_name, "timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df[["timestamp", signal_name]]

            # Otherwise: Empatica standard layout
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()

                if signal_name.upper() == "IBI":
                    start_time = self._parse_start_time(line1.split(",")[0])
                    df = pd.read_csv(filepath, skiprows=1, header=None,
                                     names=["time", "duration"], sep=",|\\s+", engine="python")
                    df["timestamp"] = start_time + df["time"]
                    return df[["timestamp", "duration"]]

                elif signal_name.upper() == "ACC":
                    start_time = self._parse_start_time(line1.split(",")[0])
                    sampling_rates = [float(s) for s in line2.split(",") if s.strip()]
                    sampling_rate = sampling_rates[0] if sampling_rates else 32.0
                    df = pd.read_csv(filepath, skiprows=2, header=None, names=["x", "y", "z"])
                    df["timestamp"] = [start_time + i / sampling_rate for i in range(len(df))]
                    return df

                else:
                    start_time = self._parse_start_time(line1)
                    sampling_rate = float(line2)
                    df = pd.read_csv(filepath, skiprows=2, header=None, names=[signal_name])
                    df["timestamp"] = [start_time + i / sampling_rate for i in range(len(df))]
                    return df

        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {e}")

    # ---------- Build combined DF + metadata ----------

    def _build_combined_dataframe(self) -> None:
        """
        Align all signals on a common time index.
        - If prefer_datetime_index=True and any timestamp column is datetime, keep datetime index.
        - Otherwise, convert all timestamps to seconds (float, UNIX epoch).
        """
        aligned = []
        use_datetime = self.prefer_datetime_index

        # Decide index type: if any signal gives datetime, and user prefers datetime, keep datetime
        if self.prefer_datetime_index:
            for df in self.signals.values():
                if "timestamp" in df.columns and np.issubdtype(df["timestamp"].dtype, np.datetime64):
                    use_datetime = True
                    break

        for name, df in self.signals.items():
            if "timestamp" not in df.columns:
                continue
            temp = df.copy()

            # Normalize index type
            ts = temp["timestamp"]
            if use_datetime:
                if not np.issubdtype(ts.dtype, np.datetime64):
                    ts = pd.to_datetime(ts, unit="s", errors="coerce")
                temp = temp.set_index(ts)
            else:
                # seconds float
                if np.issubdtype(ts.dtype, np.datetime64):
                    idx = ts.astype("int64") / 1_000_000_000.0
                else:
                    idx = ts.astype(float)
                temp = temp.set_index(idx)

            # Rename data columns to avoid collisions
            value_cols = [c for c in temp.columns if c != "timestamp"]
            temp = temp[value_cols]
            temp.columns = [f"{name}_{c}" if name.upper() == "ACC" else name for c in temp.columns]

            aligned.append(temp)

        if not aligned:
            self.signal_df = pd.DataFrame()
            return

        combined = pd.concat(aligned, axis=1).sort_index()
        combined = combined.loc[:, ~combined.columns.duplicated()]
        self.signal_df = combined

    def _compute_metadata(self) -> None:
        df = self.signal_df
        meta: Dict[str, any] = {"case": self.case, "signals": list(df.columns) if df is not None else []}

        if df is not None and not df.empty:
            # Start/end/duration
            if np.issubdtype(df.index.dtype, np.floating):
                start = float(df.index.min())
                end = float(df.index.max())
                duration = end - start
                meta.update({
                    "start": start,
                    "end": end,
                    "duration_sec": duration,
                    "index_type": "seconds",
                })
            else:
                start = df.index.min()
                end = df.index.max()
                duration = (end - start).total_seconds()
                meta.update({
                    "start": start,
                    "end": end,
                    "duration_sec": float(duration),
                    "index_type": "datetime",
                })

            # per-signal fs estimates (median 1 / median Δt on non-NaN points)
            fs_est = {}
            for col in df.columns:
                series = df[col].dropna()
                if len(series) >= 3:
                    t = series.index
                    if np.issubdtype(t.dtype, np.floating):
                        dt = np.diff(t.values)
                    else:
                        dt = np.diff(t.view("int64")) / 1_000_000_000.0
                    dt = dt[dt > 0]
                    fs_est[col] = (1.0 / np.median(dt)) if len(dt) else None
                else:
                    fs_est[col] = None
            meta["fs_estimate"] = fs_est

        self.metadata = meta


# Test below

# loader = EmpaticaE4Loader(case="Part 112C/A01d53_200210-194142", path="../datasets_lite/ADARP")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])

# loader = EmpaticaE4Loader(case="13-E4-Drv13/Right.zip", path="../datasets_lite/AffectiveROAD/Database/E4")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])


# loader = EmpaticaE4Loader(case="D1_1/ID_8/round_4/phase1", path="../datasets_lite/EmoPairCompete")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])

# loader = EmpaticaE4Loader(case="S10/midterm_2", path="../datasets_lite/wearable-exam-stress/data")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])

# loader = EmpaticaE4Loader(case="STRESS/S17", path="../datasets_lite/wearable-device-dataset/Wearable_Dataset/")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])

# loader = EmpaticaE4Loader(case="S16/S16_E4_Data.zip", path="../datasets_lite/WESAD")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])

# loader = EmpaticaE4Loader(case="S15/S15_E4.zip", path="../datasets_lite/ppg-dalia")
# loader.print_summary()
# loader.plot_all()
# loader.plot(['BVP', 'EDA', 'HR'])
