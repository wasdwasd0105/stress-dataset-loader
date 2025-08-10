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



## Usage

Run main.py with the following options:

**usage:**  
```bash
main.py [-h] [-dataset [DATASET]] [-case CASE | -example-case] [-plot PLOT] [-plot-all]
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



## Examples
1.	List available datasets:

    python main.py -dataset

2.	Show metadata for a dataset:
    
    python main.py -dataset autonomic-aging-cardiovascular

3.	Load an example case and plot all features:

    python main.py -dataset autonomic-aging-cardiovascular -example-case -plot-all

4.	Load a specific case and plot selected features:

    python ./main.py -dataset ADARP -example-case -plot EDA,HR
