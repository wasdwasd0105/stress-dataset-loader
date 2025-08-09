from __future__ import annotations

import os
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CardioRespiratoryLoader:
    """
    Unified cardiorespiratory test loader.
    """

    def __init__(self, path: str, id: Optional[Union[str, int]] = None, id_test: Optional[Union[str, int]] = None):
        if (id is None and id_test is None) or (id is not None and id_test is not None):
            raise ValueError("Specify exactly one of `id` or `id_test`.")

        self.path = path
        self.id = str(id) if id is not None else None
        self.id_test = str(id_test) if id_test is not None else None
        self.by = "ID" if self.id is not None else "ID_test"
        self.identifier = self.id or self.id_test

        self.subject_info: pd.DataFrame | None = None
        self.test_measure: pd.DataFrame | None = None
        self.info_rows: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None

        self.signal_df: pd.DataFrame | None = None
        self.metadata: Dict[str, any] = {}

        self._load_all()
        self._filter()
        self._build_dataframe()
        self._compute_metadata()

    # ---------- Unified API ----------

    def to_dataframe(self) -> pd.DataFrame:
        if self.signal_df is None:
            self._build_dataframe()
        return self.signal_df

    def get_metadata(self) -> Dict[str, any]:
        return self.metadata

    def print_summary(self) -> None:
        print("=== CardioRespiratory Summary ===")
        print(f"{self.by}: {self.identifier}")
        print(f"Samples: {self.metadata.get('n_samples', 'N/A')}")
        print(f"Duration (s): {self.metadata.get('duration_sec', 'N/A')}")
        if self.metadata.get("start") is not None:
            print(f"Time range: {self.metadata['start']} — {self.metadata['end']}")
        print(f"Variables: {self.metadata.get('variables', [])}")
        if self.info_rows is not None and not self.info_rows.empty:
            print("Participant Info:")
            print(self.info_rows.drop(columns=[self.by], errors="ignore").to_string(index=False))

    def plot(
        self,
        features: List[str],
        max_points: int = 20000,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot full dataset for selected features.
        One figure per variable.
        """
        df = self.to_dataframe()
        if df is None or df.empty:
            raise ValueError("No data available.")

        missing = [v for v in features if v not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for ch in features:
            series = df[ch].dropna()
            plt.figure(figsize=(12, 3.5))
            if series.empty:
                ax = plt.gca()
                ax.set_title(title or f"{self.by}={self.identifier} — {ch}")
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
            plt.title(title or f"{self.by}={self.identifier} — {ch}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def plot_all(self, **kwargs) -> None:
        df = self.to_dataframe()
        if df is None or df.empty:
            raise ValueError("No data available.")
        self.plot(features=list(df.columns), **kwargs)

    # ---------- Internals ----------

    def _load_all(self):
        subject_path = os.path.join(self.path, "subject-info.csv")
        measure_path = os.path.join(self.path, "test_measure.csv")
        self.subject_info = pd.read_csv(subject_path)
        self.test_measure = pd.read_csv(measure_path)

    def _filter(self):
        self.subject_info[self.by] = self.subject_info[self.by].astype(str)
        self.test_measure[self.by] = self.test_measure[self.by].astype(str)

        self.info_rows = self.subject_info[self.subject_info[self.by] == self.identifier]
        self.test_data = self.test_measure[self.test_measure[self.by] == self.identifier]

        if self.info_rows.empty:
            print(f"[Warning] No subject info found for {self.by} = {self.identifier}")
        if self.test_data.empty:
            print(f"[Warning] No test data found for {self.by} = {self.identifier}")

    def _build_dataframe(self):
        if self.test_data is None or self.test_data.empty:
            self.signal_df = pd.DataFrame()
            return

        df = self.test_data.copy()

        time_col = None
        for cand in ["time", "Time", "timestamp", "Timestamp"]:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            raise ValueError("No time column found (expected one of: time, Time, timestamp, Timestamp).")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for c in [self.by, "ID", "ID_test"]:
            if c in numeric_cols:
                numeric_cols.remove(c)
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)

        out = df[[time_col] + numeric_cols].copy()
        out = out.sort_values(time_col)
        out.set_index(time_col, inplace=True)
        out.index = out.index.astype(float)
        self.signal_df = out

    def _compute_metadata(self):
        df = self.signal_df if self.signal_df is not None else pd.DataFrame()
        meta: Dict[str, any] = {
            "path": self.path,
            "by": self.by,
            "identifier": self.identifier,
            "variables": list(df.columns) if not df.empty else [],
            "n_samples": int(len(df)),
        }

        if not df.empty:
            start = float(df.index.min())
            end = float(df.index.max())
            meta.update({
                "start": start,
                "end": end,
                "duration_sec": end - start,
                "index_type": "seconds",
            })

        self.metadata = meta


# # test data 1
# loader = CardioRespiratoryLoader(id_test="119_15", path="../datasets_lite/treadmill-exercise-cardioresp")
# loader.print_summary()
# df_info = loader.get_metadata()
# df = loader.to_dataframe()
# loader.plot(features=["RR", "VO2"])
#
# # test data 2
# loader2 = CardioRespiratoryLoader(id="15", path="../datasets_lite/actes-cycloergometer-exercise")
# loader2.print_summary()
# loader2.plot_all()

