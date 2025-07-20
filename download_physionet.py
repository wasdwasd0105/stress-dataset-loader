import os
import subprocess

# Dataset config: name → (base_url, list of files) or (full_url, output_dir) for mirrored folders
datasets = {
    # https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/  // N = 1121
    "1-autonomic-aging-cardiovascular": (
        "https://physionet.org/files/autonomic-aging-cardiovascular/1.0.0/",
        ["1121.dat", "1121.hea"]
    ),

    # https://physionet.org/content/propofol-anesthesia-dynamics/1.0/  // N = 9
    "2-propofol-anesthesia-dynamics": (
        "https://physionet.org/files/propofol-anesthesia-dynamics/1.0/Data/",
        [
            "S9_HF.csv", "S9_HFnu.csv", "S9_LF.csv", "S9_LFnu.csv", "S9_LOC.csv", "S9_ROC.csv",
            "S9_eda_tonic.csv", "S9_events.csv", "S9_muHR.csv", "S9_muPR.csv", "S9_muRR.csv",
            "S9_mu_amp.csv", "S9_pow_tot.csv", "S9_ratio.csv", "S9_sigmaHR.csv", "S9_sigmaPR.csv",
            "S9_sigmaRR.csv", "S9_sigma_amp.csv", "S9_t_EDA.csv", "S9_t_EDA_tonic.csv", "S9_t_HRV.csv"
        ]
    ),

    # https://physionet.org/content/wearable-device-dataset/1.0.0/  // N = 36
    # ???



    # # https://physionet.org/content/scientisst-move-biosignals/1.0.1/ // N = 17
    # "9-scientisst-move-biosignals": (
    #     "https://physionet.org/files/scientisst-move-biosignals/1.0.1/ME93",
    #     ["empatica.edf", "scientisst_chest.edf"]
    # ),

    #  https://physionet.org/content/wrist/1.0.0/ // N = 8
    "10-wrist": (
        "https://physionet.org/content/wrist/1.0.0/",
        [
            "s8_run.atr", "s8_run.dat", "s8_run.hea", "s8_walk.atr", "s8_walk.dat", "s8_walk.hea"
        ]
    ),

    # https://physionet.org/content/wearable-exercise-frailty/1.0.0/ // N = 80
    # ???
    # "wearable-exercise-frailty": (
    #     "https://physionet.org/content/wearable-exercise-frailty/1.0.0/",
    #     [

    #     ]
    # ),

    # https://physionet.org/content/treadmill-exercise-cardioresp/1.0.1/ // N = 992
    "12-treadmill-exercise-cardioresp": (
        "https://physionet.org/files/treadmill-exercise-cardioresp/1.0.1/",
        [
            "subject-info.csv", "test_measure.csv"
        ]
    ),

    # https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/ // N = 18
    "13-actes-cycloergometer-exercise": (
        "https://physionet.org/files/actes-cycloergometer-exercise/1.0.0/",
        [
            "subject-info.csv", "test_measure.csv"
        ]
    ),

    # https://physionet.org/content/nstdb/1.0.0/ // N = 15
    # "nstdb": (
    #     "https://physionet.org/files/nstdb/1.0.0/",
    #     [
    #     ]
    # ),

}


datasets_folder = {
    # https://physionet.org/content/wearable-exam-stress/1.0.0/ // N = 10
    "4-wearable-exam-stress-S10": (
        "https://physionet.org/files/wearable-exam-stress/1.0.0/data/S10/",
        "wearable-exam-stress/data/S10"
    ),
}


# zip file 
# 5. https://zenodo.org/records/6640290 // N = 11
# 6. https://www.media.mit.edu/tools/affectiveroad/ // N = 10 
# 7. https://github.com/DTUComputeStatisticsAndDataAnalysis/EmoPairCompete/tree/main   // N = 28 -> https://zenodo.org/records/11151714
# 8. https://github.com/WJMatthew/WESAD // N = 15 -> https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
# 14. https://archive.ics.uci.edu/dataset/495/ppg+dalia // N = 15 -> https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip
# 16. https://archive.ics.uci.edu/dataset/319/mhealth+dataset // N = 10




def download_with_wget_directory(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "wget", "-r", "-np", "-nH",
        "--cut-dirs=5", "-P", output_dir,
        "-R", "index.html*",
        url
    ]
    print(f"\n📂 Mirroring directory: {url}")
    subprocess.run(command)  # stdout/stderr default to show full progress

def download_files_wget(file_list, base_url, folder):
    os.makedirs(folder, exist_ok=True)
    total = len(file_list)

    for i, filename in enumerate(file_list, start=1):
        file_url = base_url + filename
        print(f"\n📄 [{i}/{total}] Downloading {filename}...")
        command = ["wget", "-c", "-P", folder, file_url]
        subprocess.run(command)  # shows full wget output (progress bar, ETA, etc.)

# Run all downloads
# for dataset_name, config in datasets.items():
#     print(f"\n=== 🔽 Processing dataset: {dataset_name} ===")
#     base_url, files = config
#     download_files_wget(files, base_url, dataset_name)


for dataset_name, config in datasets_folder.items():
    print(f"\n=== 🔽 Processing dataset: {dataset_name} ===")
    base_url, out_dir = config
    download_with_wget_directory(base_url, out_dir)
