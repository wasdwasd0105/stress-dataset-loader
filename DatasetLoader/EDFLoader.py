from __future__ import annotations

import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt


class EDFLoader:
    """
    Unified EDF loader with case/path signature.
    """

    def __init__(self, case: str, path: str):
        """
        :param case: EDF filename (with or without .edf extension)
        :param path: Directory containing the EDF file
        """
        if not case.lower().endswith(".edf"):
            case += ".edf"
        self.file_path = os.path.join(path, case)

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"EDF file not found: {self.file_path}")

        self.reader: Optional[pyedflib.EdfReader] = None
        self.signal_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, any] = {}

        # Internal
        self._fs_map: Dict[str, float] = {}
        self._labels: List[str] = []

        # self.load()

    # -------------------- lifecycle --------------------

    def load(self) -> None:
        # Open, extract metadata, then close immediately so we don't keep the file handle open.
        with pyedflib.EdfReader(self.file_path) as reader:
            self._extract_metadata_from(reader)

    # -------------------- unified API --------------------

    def to_dataframe(self) -> pd.DataFrame:
        with pyedflib.EdfReader(self.file_path) as reader:
            n_signals = reader.signals_in_file
            labels = reader.getSignalLabels()
            fs_list = reader.getSampleFrequencies()

            series_list = []
            for i in range(n_signals):
                label = labels[i]
                fs = float(fs_list[i])
                self._fs_map[label] = fs

                sig = reader.readSignal(i).astype(float, copy=False)
                t = np.arange(len(sig), dtype=float) / fs
                series_list.append(pd.Series(sig, index=t, name=label))

        df = pd.concat(series_list, axis=1).sort_index()
        self.signal_df = df

        if not df.empty:
            self.metadata["duration_sec"] = float(df.index.max() - df.index.min())

        return df

    def get_metadata(self) -> Dict[str, any]:
        return self.metadata

    def print_summary(self) -> None:
        print("=== EDF File Summary ===")
        print(f"File: {os.path.basename(self.file_path)}")
        for k, v in self.metadata.items():
            if k == "fs_map":
                continue
            print(f"{k}: {v}")
        if "fs_map" in self.metadata:
            print("Sampling rates (Hz):")
            for ch, fs in self.metadata["fs_map"].items():
                print(f"  {ch}: {fs}")

    def plot(self, features: List[str], max_points: int = 20000, title: Optional[str] = None) -> None:
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data to plot.")

        missing = [ch for ch in features if ch not in df.columns]
        if missing:
            raise ValueError(f"Channels not found: {missing}\nAvailable: {list(df.columns)}")

        for ch in features:
            series = df[ch].dropna()
            plt.figure(figsize=(12, 3.5))
            if series.empty:
                ax = plt.gca()
                ax.set_title(title or f"{os.path.basename(self.file_path)} — {ch}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ch)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.show()
                continue

            if len(series) > max_points:
                step = int(np.ceil(len(series) / max_points))
                series = series.iloc[::step]

            plt.plot(series.index.values, series.values, linewidth=0.9)
            plt.xlabel("Time (s)")
            plt.ylabel(ch)
            plt.title(title or f"{os.path.basename(self.file_path)} — {ch}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def plot_all(self, **kwargs) -> None:
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No data to plot.")
        self.plot(features=list(df.columns), **kwargs)

    # -------------------- internals --------------------

    def _extract_metadata_from(self, reader) -> None:
        labels = reader.getSignalLabels()
        fs_list = reader.getSampleFrequencies()
        self._labels = list(labels)
        self.metadata = {
            "n_channels": reader.signals_in_file,
            "signal_labels": self._labels,
            "sample_frequencies": list(map(float, fs_list)),
            "duration_sec": float(reader.file_duration),
            "start_datetime": reader.getStartdatetime(),
            "max_sampling_rate": float(max(fs_list)) if len(fs_list) else None,
            "fs_map": {labels[i]: float(fs_list[i]) for i in range(len(labels))},
        }

# loader = EDFLoader(case="LK27/scientisst_forearm.edf", path="../datasets_lite/scientisst-move-biosignals/")
# loader.load()
# df = loader.to_dataframe()
# loader.print_summary()
# loader.plot_all()
# loader.plot(["eda:gel", "emg"])