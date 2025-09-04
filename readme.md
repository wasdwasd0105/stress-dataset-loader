# Stress Dataset Loader

A flexible Python tool for loading, exploring, and plotting various stress-related and physiological datasets.

## Features

- **Multiple dataset formats** via loaders:
  - `WFDBLoader`
  - `EmpaticaE4Loader`
  - `EDFLoader`
  - `PropofolLoader`
  - `MHealthLoader`
  - `CardioRespiratoryLoader`
- **Unified command-line interface** for:
  - Listing available datasets
  - Printing dataset metadata
  - Loading specific cases
  - Plotting selected features or all features
- **Metadata-driven loading** via `./metadata/<dataset>.json`



## Usage For DatasetLoaderUI (Jupyter Dataset Loader UI)

provide a Jupyter UI for for exploring, loading, and visualizing various biosignals.

**usage:**  

1. Open the notebook in VScode.
2. Run all cells.
3. Select a dataset from the dropdown.
4. Choose an example case or enter a manual case ID.
5. Select one feature, or enable "Plot all features".
6. Click Plot to visualize the signals.



## Usage For DatasetLoaderCli (Dataset Loader Command-Line Interface)

Run DatasetLoaderCli.py with the following options:

**usage:**  
```bash
DatasetLoaderCli.py [-h] [-dataset [DATASET]] [-case CASE | -example-case] [-plot PLOT] [-plot-all]
```

Flexible Dataset Loader: pick dataset, case, loader, and plot features.

**options:**
```text
-h, --help            Show this help message and exit.
-dataset [DATASET]    Dataset name.
                      Use `-dataset` with no value to list all datasets.
                      Use `-dataset <name>` to print dataset metadata summary.
-case CASE            Case identifier (record/subject/file/etc. per dataset).
                      Loads from `datasets/` folder.
-example-case         Load example case from `metadata["example_case"]` in `datasets_lite/`.
-plot PLOT            Comma-separated list of features to plot.
-plot-all             Plot all available features.
```

### Examples
1.	List available datasets:

    python DatasetLoaderCli.py -dataset

2.	Show metadata for a dataset:
    
    python DatasetLoaderCli.py -dataset autonomic-aging-cardiovascular

3.	Load an example case and plot all features:

    python DatasetLoaderCli.py -dataset autonomic-aging-cardiovascular -example-case -plot-all

4.	Load a specific case and plot selected features:

    python ./DatasetLoaderCli.py -dataset ADARP -example-case -plot EDA,HR
