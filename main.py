#!/usr/bin/env python3
from __future__ import annotations

import argparse

import json
import os
import sys
from typing import List, Any, Dict

#imports for all loaders
from DatasetLoader.WFDBLoader import WFDBLoader
from DatasetLoader.EmpaticaE4Loader import EmpaticaE4Loader
from DatasetLoader.EDFLoader import EDFLoader
from DatasetLoader.PropofolLoader import PropofolLoader
from DatasetLoader.MHealthLoader import MHealthLoader
from DatasetLoader.CardioRespiratoryLoader import CardioRespiratoryLoader


LOADER_CLASSES = {
    "WFDBLoader": WFDBLoader,
    "EmpaticaE4Loader": EmpaticaE4Loader,
    "EDFLoader": EDFLoader,
    "PropofolLoader": PropofolLoader,
    "MHealthLoader": MHealthLoader,
    "CardioRespiratoryLoader": CardioRespiratoryLoader,
}

DATASET_DIR = "datasets"
DATASET_LITE_DIR = "datasets_lite"
METADATA_DIR = "metadata"
LOADERS_DIR = "DatasetLoader"


def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def load_json_metadata(dataset: str) -> Dict[str, Any]:
    meta_path = os.path.join(METADATA_DIR, f"{dataset}.json")
    if not os.path.isfile(meta_path):
        die(f"Metadata JSON not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)




def build_data_path(base_folder: str, dataset: str, entry_point: str) -> str:
    # base_folder = dataset/ or dataset_lite/
    base_root = os.path.abspath(os.path.join(base_folder, dataset))
    # entry_point may be "./" or nested
    ep = entry_point if entry_point else "./"
    full = os.path.abspath(os.path.join(base_root, ep))
    return full


def init_loader(loader_name: str, data_path: str, case: str):
    """
    Normalize constructor differences across loaders.
    """
    LoaderClass = LOADER_CLASSES.get(loader_name)
    if LoaderClass is None:
        die(f"Unknown loader '{loader_name}'. Allowed: {', '.join(LOADER_CLASSES.keys())}")

    # WFDBLoader(record_name, path)
    if loader_name == "WFDBLoader":
        return LoaderClass(case, path=data_path)

    # EmpaticaE4Loader(case, path)
    if loader_name == "EmpaticaE4Loader":
        return LoaderClass(case=case, path=data_path)

    # EDFLoader(case, path)
    if loader_name == "EDFLoader":
        return LoaderClass(case=case, path=data_path)

    # PropofolLoader(subject_id, path)
    if loader_name == "PropofolLoader":
        # case can be "S9" or "9"; the class handles prefix internally
        return LoaderClass(case, path=data_path)

    # MHealthLoader(case, path)
    if loader_name == "MHealthLoader":
        return LoaderClass(case=case, path=data_path)

    # CardioRespiratoryLoader(path, id=... or id_test=...)
    if loader_name == "CardioRespiratoryLoader":
        # Heuristic: if case contains an underscore like "119_15", treat as ID_test; else ID
        if "_" in case:
            return LoaderClass(path=data_path, id_test=case)
        else:
            return LoaderClass(path=data_path, id=case)

    # Fallback (shouldn’t hit)
    return die(f"Unknown loader '{loader_name}'. Allowed: {', '.join(LOADER_CLASSES.keys())}")


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(
        description="Load biosignal datasets by name, inspect metadata or a case, and plot features.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # List all available datasets (from ./metadata/*.json)\n"
            "  python main.py -dataset\n\n"
            "  # Show metadata summary for a dataset\n"
            "  python main.py -dataset autonomic-aging-cardiovascular\n\n"
            "  # Print summary for the example case from metadata (uses datasets_lite/)\n"
            "  python main.py -dataset autonomic-aging-cardiovascular -example-case\n\n"
            "  # Print summary for a specific case (uses datasets/)\n"
            "  python main.py -dataset autonomic-aging-cardiovascular -case 1121\n\n"
            "  # Plot specific features for a case\n"
            "  python main.py -dataset autonomic-aging-cardiovascular -case 1121 -plot ECG,RESP\n\n"
            "  # Plot all features for a case\n"
            "  python main.py -dataset autonomic-aging-cardiovascular -case 1121 -plot-all\n"
        ),
    )
    p.add_argument(
        "-dataset",
        nargs="?",
        const="__LIST__",
        required=False,
        help=(
            "Dataset name (metadata JSON at ./metadata/<name>.json).\n"
            "  - Use '-dataset' with no value to list available datasets.\n"
            "  - Use '-dataset <name>' alone to print that dataset's metadata summary."
        ),
    )
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-case",
        help="Case identifier for the selected dataset (loads from ./datasets/)",
    )
    group.add_argument(
        "-example-case",
        action="store_true",
        help="Use metadata['example_case'] and load from ./datasets_lite/",
    )
    p.add_argument(
        "-plot",
        help="Comma-separated feature list to plot (e.g., '-plot ECG,ABP'). Requires -case or -example-case.",
    )
    p.add_argument(
        "-plot-all",
        dest="plot_all",
        action="store_true",
        help="Plot all available features for the selected case. Requires -case or -example-case.",
    )

    # If no arguments at all, show help and exit
    if argv is None and len(sys.argv) == 1:
        p.print_help()
        return 0

    args = p.parse_args(argv)

    # If only -dataset is provided (no -case, no -example-case, no -plot), or if -dataset is provided with no value, list available datasets
    if (args.dataset in (None, "__LIST__")) and (args.case is None) and (not args.example_case) and (args.plot is None) and (not getattr(args, "plot_all", False)):
        print("=== Available Datasets ===")
        try:
            files = os.listdir(METADATA_DIR)
        except FileNotFoundError:
            files = []
        datasets = sorted(f[:-5] for f in files if f.endswith(".json"))
        for ds in datasets:
            print(ds)
        return 0

    if args.dataset in (None, "__LIST__"):
        # Already handled listing above; if we reach here with a missing dataset but other flags provided, error out.
        die("Please provide a dataset name after -dataset, or run '-dataset' alone to list available datasets.")
    meta = load_json_metadata(args.dataset)

    # Print metadata summary if only -dataset is provided, no case/example-case/plot flags
    if args.case is None and not args.example_case and args.plot is None and not getattr(args, "plot_all", False):
        print("=== Dataset Metadata Summary ===")
        print(f"Name           : {meta.get('name', 'N/A')}")
        print(f"Description    : {meta.get('notes', 'N/A')}")
        print(f"Data Loader    : {meta.get('loader', 'N/A')}")
        print(f"Record Count   : {meta.get('record_count', 'N/A')}")
        print(f"Format         : {meta.get('format', 'N/A')}")
        print(f"Example Case   : {meta.get('example_case', 'N/A')}")
        structure = meta.get('structure', {})
        if isinstance(structure, dict) and 'sensor_features' in structure:
            print(f"Dataset Features: {', '.join(structure['sensor_features'])}")
        else:
            print("Dataset Features: N/A")
        return 0

    loader_name = meta.get("loader")
    if not loader_name:
        die(f"'loader' missing in metadata for dataset '{args.dataset}'")

    # Choose base folder + case
    if args.example_case:
        case = meta.get("example_case")
        if not case:
            die(f"'example_case' missing in metadata for dataset '{args.dataset}'")
        base_folder = DATASET_LITE_DIR
    else:
        case = args.case
        base_folder = DATASET_DIR

    entry_point = meta.get("entry_point", "./")
    data_path = build_data_path(base_folder, args.dataset, entry_point)

    print(f"Dataset: {args.dataset}")
    print(f"Loader : {loader_name}")
    print(f"Case   : {case}")
    print(f"Data   : {data_path}")

    # If a case or example_case was provided but no plotting flags, load and print summary then exit
    if (args.case or args.example_case) and not (args.plot or getattr(args, "plot_all", False)):
        loader = init_loader(loader_name, data_path, case)
        if hasattr(loader, "print_summary"):
            loader.print_summary()
        return 0

    # Create loader
    loader = init_loader(loader_name, data_path, case)

    # Print summary if available
    if hasattr(loader, "print_summary"):
        loader.print_summary()

    # Plotting
    if getattr(args, "plot_all", False):
        if not hasattr(loader, "plot_all"):
            die(f"Loader '{loader_name}' has no plot_all() method.")
        loader.plot_all()
    elif args.plot:
        features = [s.strip() for s in args.plot.split(",") if s.strip()]
        if not features:
            die("No features parsed from -plot argument.")
        if not hasattr(loader, "plot"):
            die(f"Loader '{loader_name}' has no plot() method.")
        #print(features)
        loader.plot(features)
    else:
        print("[INFO] No plotting requested.")

    return 0


if __name__ == "__main__":
    sys.exit(main())