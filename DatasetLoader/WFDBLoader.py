import numpy as np
import wfdb
import pandas as pd
import os
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt



class WFDBLoader:
    def __init__(self, record_name: str, path: Optional[str] = None):
        self.record_name = record_name
        self.path = path
        self.record = None
        self.annotation = None
        self.signal_df = None
        self.metadata = {}
        self.record_path = os.path.join(self.path, self.record_name) if self.path else self.record_name
        self._load_record()

    def _load_record(self) -> None:
        #print(self.record_path)
        self.record = wfdb.rdrecord(self.record_path)
        self._extract_metadata()

    def load_annotations(self, annotator: str = 'atr') -> None:
        self.annotation = wfdb.rdann(self.record_path, annotator)

    def to_dataframe(self) -> pd.DataFrame:
        if self.record is None:
            raise ValueError("No record loaded. Call load_record() first.")
        data = self.record.p_signal if self.record.p_signal is not None else self.record.d_signal
        self.signal_df = pd.DataFrame(data, columns=self.record.sig_name)
        if hasattr(self.record, 'fs') and self.record.fs:
            self.signal_df['time'] = self.signal_df.index / self.record.fs
            self.signal_df.set_index('time', inplace=True)
        return self.signal_df

    def get_labels(self) -> pd.Series:
        if self.annotation is None:
            raise ValueError("No annotations loaded. Call load_annotations() first.")
        return pd.Series(self.annotation.symbol, index=self.annotation.sample, name="annotation")

    def get_metadata(self) -> Dict[str, any]:
        return self.metadata

    def print_summary(self) -> None:
        print("=== WFDB Record Summary ===")
        print(f"Record Name: {self.record_name}")
        if self.record:
            print(f"Sampling Frequency: {self.metadata.get('fs', 'N/A')} Hz")
            print(f"Signal Names: {self.metadata.get('sig_names', [])}")
            print(f"Duration: {self.metadata.get('duration', 'N/A')} seconds")
        if self.annotation:
            print(f"Number of Annotations: {len(self.annotation.sample)}")

    def _extract_metadata(self) -> None:
        if self.record:
            self.metadata = {
                "fs": self.record.fs,
                "sig_names": self.record.sig_name,
                "units": self.record.units,
                "n_sig": self.record.n_sig,
                "length": len(self.record.p_signal) if self.record.p_signal is not None else len(self.record.d_signal),
                "duration": len(self.record.p_signal) / self.record.fs if self.record.p_signal is not None else None
            }


    def plot_all(self, time_range: Optional[tuple] = None, title: Optional[str] = None) -> None:
        """
        Plot the record and annotation using wfdb's built-in plot function.
        :param time_range: Optional (start_time, end_time) in seconds.
        :param title: Optional custom plot title.
        """
        if self.record is None:
            raise ValueError("No record loaded. Call load_record() first.")

        wfdb.plot_wfdb(
            record=self.record,
            annotation=self.annotation,
            title=title or f"WFDB Record: {self.record_name}",
            time_units='seconds',
            figsize=(10, 4),
            plot_sym=True,
            return_fig=False
        )

    def plot(
        self,
        features: [str],
        start: Optional[float] = None,
        end: Optional[float] = None,
        annotate: bool = True,
        max_points: int = 20000,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot each requested signal in its own figure.
        - features: list of signal names (must match self.record.sig_name)
        - start, end: optional time window in seconds
        - annotate: overlay annotation markers if loaded
        - max_points: cap total plotted points (per-channel) for responsiveness
        """
        if self.record is None:
            raise ValueError("No record loaded. Call load_record() first.")

        # Prepare dataframe if needed
        if self.signal_df is None:
            self.to_dataframe()

        fs = self.metadata.get("fs", None)
        if fs is None:
            raise ValueError("Sampling frequency unknown; cannot plot.")

        # Validate feature names
        available = set(self.signal_df.columns)
        missing = [c for c in features if c not in available]
        if missing:
            raise ValueError(f"Signals not found: {missing}\nAvailable: {sorted(available)}")

        # Apply time window
        df = self.signal_df
        if start is not None or end is not None:
            s = 0 if start is None else float(start)
            e = df.index.max() if end is None else float(end)
            df = df.loc[s:e]

        # Downsample for performance
        if len(df) > max_points:
            step = int(np.ceil(len(df) / max_points))
            df = df.iloc[::step]

        for ch in features:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index.values, df[ch].values, linewidth=0.8)

            # Unit label
            unit = ""
            try:
                idx = self.record.sig_name.index(ch)
                unit = f" ({self.record.units[idx]})" if self.record.units else ""
            except Exception:
                pass

            plt.ylabel(f"{ch}{unit}")
            plt.xlabel("Time (s)")
            plt.title(title or f"WFDB: {self.record_name} — {ch}")

            # Overlay annotations if requested
            if annotate and (self.annotation is not None):
                ann_t = np.asarray(self.annotation.sample, dtype=float) / fs
                if start is not None or end is not None:
                    s = df.index.min()
                    e = df.index.max()
                    mask = (ann_t >= s) & (ann_t <= e)
                    ann_t = ann_t[mask]
                    ann_sym = np.array(self.annotation.symbol)[mask]
                else:
                    ann_sym = np.array(self.annotation.symbol)

                if ann_t.size:
                    ymin, ymax = np.nanmin(df[ch].values), np.nanmax(df[ch].values)
                    plt.vlines(ann_t, ymin, ymax, linestyles="dotted", alpha=0.3)
                    idxs = np.searchsorted(df.index.values, ann_t)
                    idxs = np.clip(idxs, 0, len(df) - 1)
                    plt.scatter(df.index.values[idxs], df[ch].values[idxs], s=10, alpha=0.7)

            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()

# loader = WFDBLoader("1121", path="../datasets_lite/autonomic-aging-cardiovascular")
# df = loader.to_dataframe()
# meta = loader.get_metadata()
# loader.print_summary()
# loader.plot_all(title="Autonomic Aging: Record 1121")
# loader.plot(features=["ECG", "NIBP"])
