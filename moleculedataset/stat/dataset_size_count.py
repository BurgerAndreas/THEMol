# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Analyze and generate a summary of dataset sizes across molecular subsets.

This script processes multiple dataset folders sequentially, calculating the 
number of entries, unique molecules, constraints, and total steps, and merges 
the results into a single summary CSV.
"""

import argparse
import glob
import os
from typing import Any, Dict, Optional

import pandas as pd


def _find_csv_in_folder(folder: str) -> str:
    """Locate a single CSV file within a dataset folder.

    Args:
        folder: Path to the directory to search.

    Returns:
        The path to the first CSV file found (sorted by name).

    Raises:
        FileNotFoundError: If no CSV files are found in the folder.
    """
    candidates = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {folder}")
    return candidates[0]


def _format_num(num: int) -> str:
    """Format an integer with commas.

    Args:
        num: The integer to format.

    Returns:
        A comma-separated string representation of the integer.
    """
    return f"{num:,}"


def _analyze_subset(folder: str, name: str, level: str) -> Optional[Dict[str, Any]]:
    """Analyze a single dataset folder to extract size metrics.

    Args:
        folder: Path to the dataset directory.
        name: Name of the subset.
        level: Level of theory for the subset.

    Returns:
        A dictionary containing the summary metrics for the subset, or None if failed.
    """
    try:
        csv_path = _find_csv_in_folder(folder)
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)

        entries = len(df)
        metrics = []

        if name in ["TorsionScan", "TorsionScanRelax"]:
            if "mapped_isomeric_smiles" in df.columns:
                num_molecules = df["mapped_isomeric_smiles"].nunique()
                metrics.append(f"{_format_num(num_molecules)} molecules")
            if "num_constraints" in df.columns:
                num_constraints = df["num_constraints"].sum()
                metrics.append(f"{_format_num(num_constraints)} constraints")

        if name == "Hessian Relax":
            if "num_steps" in df.columns:
                num_steps = df["num_steps"].sum()
                metrics.append(f"{_format_num(num_steps)} steps")
        elif name == "TorsionScanRelax":
            if "num_total_steps" in df.columns:
                num_steps = df["num_total_steps"].sum()
                metrics.append(f"{_format_num(num_steps)} steps")

        supp_metrics = f"({', '.join(metrics)})" if metrics else ""

        return {
            "Subset": name,
            "Level of Theory": level,
            "Entries": _format_num(entries),
            "Supplementary Metrics": supp_metrics
        }
    except Exception as e:
        print(f"Failed to process {folder}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve base data directory from environment or relative path
    base_dir = os.environ.get("MOLECULE_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    parser = argparse.ArgumentParser(description="Analyze dataset sizes and generate a summary CSV.")
    parser.add_argument("--data_dir",
                        default=base_dir,
                        help="Base directory containing dataset subsets.")
    parser.add_argument("--output",
                        default=os.path.join(file_dir, "dataset_size_count.csv"),
                        help="Path to save the summary CSV.")
    return parser.parse_args()


def main() -> None:
    """Main execution flow for analyzing dataset sizes."""
    args = parse_args()

    subsets = [
        {"name": "Hessian", "folder": "Hessian", "level": "B3LYP-D3(BJ)/DZVP"},
        {"name": "Hessian Relax", "folder": "HessianRelax", "level": "B3LYP-D3(BJ)/DZVP"},
        {"name": "TorsionScan", "folder": "TorsionScan", "level": "B3LYP-D3(BJ)/DZVP"},
        {"name": "TorsionScanRelax", "folder": "TorsionScanRelax", "level": "B3LYP-D3(BJ)/DZVP"},
        {"name": "MBIS", "folder": "MBIS", "level": "PBE0/def2-TZVPD(or DZVP for I atoms)"}
    ]

    results = []
    for subset in subsets:
        folder_path = os.path.join(args.data_dir, subset["folder"])
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}. Skipping.")
            continue

        res = _analyze_subset(folder_path, subset["name"], subset["level"])
        if res:
            results.append(res)

    if not results:
        print("No analysis data was successfully collected.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_merged = pd.DataFrame(results)
    
    df_merged.to_csv(args.output, index=False)

    print(f"Dataset size summary successfully saved to {args.output}")
    print("\nSummary:")
    print(df_merged.to_string(index=False))


if __name__ == "__main__":
    main()
