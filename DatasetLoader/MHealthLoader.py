import pandas as pd
from typing import Optional, Dict
import matplotlib.pyplot as plt

# ONLY works on https://archive.ics.uci.edu/dataset/319/mhealth+dataset !!

class MHealthLoader:

    def __init__(self, case: str, path: str):
        import os
        self.file_path = os.path.join(path, case)
        self.data = None
        self.columns = [
            "chest_acc_x", "chest_acc_y", "chest_acc_z",
            "ecg_lead1", "ecg_lead2",
            "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
            "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
            "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
            "arm_acc_x", "arm_acc_y", "arm_acc_z",
            "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
            "arm_mag_x", "arm_mag_y", "arm_mag_z",
            "label"
        ]
        self.load()

    def load(self) -> None:
        try:
            self.data = pd.read_csv(
                self.file_path,
                sep=r"\s+",
                header=None,
                names=self.columns,
                engine="python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {e}")

    def to_dataframe(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load() first.")
        return self.data

    def get_metadata(self) -> Dict[str, any]:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load() first.")
        return {
            "n_samples": self.data.shape[0],
            "n_features": self.data.shape[1] - 1,
            "activity_labels": sorted(self.data["label"].unique()),
            "missing_values": int(self.data.isnull().sum().sum()),
        }

    def print_summary(self) -> None:
        meta = self.get_metadata()
        print("=== mHealth Dataset Summary ===")
        print(f"File: {self.file_path}")
        print(f"Total samples       : {meta['n_samples']}")
        print(f"Number of features  : {meta['n_features']}")
        features_list = [col for col in self.columns if col != "label"]
        print(f"Features list       : {features_list}")


    def plot(self, channels: Optional[list] = None, activity_label: Optional[int] = None) -> None:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load() first.")

        df = self.data.copy()

        # Filter by activity label
        if activity_label is not None:
            df = df[df["label"] == activity_label]

        # Select columns to plot
        all_sensor_cols = [col for col in self.columns if col != "label"]
        if channels:
            missing = [ch for ch in channels if ch not in all_sensor_cols]
            if missing:
                raise ValueError(f"Channels not found in data: {missing}")
            plot_cols = channels
        else:
            plot_cols = all_sensor_cols

        df[plot_cols].plot(subplots=True, figsize=(12, 2 * len(plot_cols)),
                           title=f"mHealth Sensor Plot{' - Activity ' + str(activity_label) if activity_label is not None else ''}")
        plt.tight_layout()
        plt.show()

    def plot_all(self) -> None:
        """Plot all channels for all samples."""
        self.plot(channels=None, activity_label=None)


# test
# if __name__ == "__main__":
#     loader = MHealthLoader(case="mHealth_subject10.log", path="../datasets_lite/MHEALTHDATASET")
#     loader.load()
#     loader.plot(channels=["chest_acc_x", "ecg_lead1"])
#     loader.print_summary()
#     loader.plot_all()