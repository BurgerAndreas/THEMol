# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Analyze and generate histograms for relaxation steps across molecular datasets.

This script processes multiple dataset folders sequentially, calculating histograms 
for relaxation steps (or average steps per constraint) and merging the results 
into a single summary CSV.
"""

import argparse
import glob
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


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


def _get_bin_label(value: float, bin_size: int) -> str:
    """Generate a fixed-format label for a histogram bin.

    Args:
        value: The numerical value to bin.
        bin_size: The width of each bin.

    Returns:
        A string label like '5 < n <= 10' or 'n <= 0'.
    """
    if value <= 0:
        return "n <= 0"

    # Round to nearest integer for consistent binning
    ival = int(round(value))
    if ival == 0:
        return "n <= 0"

    upper = ((ival - 1) // bin_size + 1) * bin_size
    lower = upper - bin_size
    return f"{lower} < n <= {upper}"


def _get_bin_sort_key(label: str) -> int:
    """Extract a sortable integer key from a bin label.

    Args:
        label: The bin label string.

    Returns:
        The upper bound of the bin as an integer for sorting.
    """
    if "n <= 0" in label:
        return -1
    try:
        return int(label.split("<=")[1].strip())
    except (IndexError, ValueError):
        return 999999


def _analyze_dataset_steps(folder: str, bin_size: int) -> Optional[Tuple[str, Counter, int]]:
    """Analyze relaxation steps for a single dataset folder.

    Args:
        folder: Path to the dataset directory.
        bin_size: Histogram bin width.

    Returns:
        A tuple of (dataset_name, bin_counts, total_count) if successful, else None.
    """
    name = os.path.basename(os.path.normpath(folder))
    try:
        csv_path = _find_csv_in_folder(folder)
        df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)

        # Select appropriate logic based on dataset type (TorsionScan vs Standard Relax)
        if "num_total_steps" in df.columns and "num_constraints" in df.columns:
            # For TorsionScanRelax, we analyze average steps per constraint
            values = (df["num_total_steps"] / df["num_constraints"].replace(0, float('nan'))).dropna()
        elif "num_steps" in df.columns:
            # For HessianRelax, we analyze total steps directly
            values = df["num_steps"].dropna()
        else:
            print(f"No recognized step columns in {csv_path}. Skipping.")
            return None

        counts = Counter()
        # Track progress at the row level for better granularity
        for val in tqdm(values, desc=f"Analyzing {name}", leave=False):
            counts[_get_bin_label(val, bin_size)] += 1

        return name, counts, len(values)
    except Exception as e:
        print(f"Failed to process {folder}: {e}")
        return None


def _format_percentage(count: int, total: int) -> str:
    """Format a frequency count as a percentage string.

    Args:
        count: The bin frequency.
        total: The total number of records.

    Returns:
        A string formatted as 'count(percentage%)'.
    """
    if total <= 0:
        return "0(0.0000%)"
    pct = (count / total) * 100
    return f"{count}({pct:.4f}%)"


def _generate_merged_table_rows(all_labels: List[str], dataset_names: List[str],
                                dataset_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare merged rows for the summary histogram table.

    Args:
        all_labels: Sorted list of all bin labels found across datasets.
        dataset_names: Sorted list of dataset names.
        dataset_data: Mapping of dataset names to their respective counts and totals.

    Returns:
        A list of dictionaries representing the merged table rows.
    """
    merged_rows = []
    for label in all_labels:
        row = {"range": label}
        for name in dataset_names:
            total = dataset_data[name]["total"]
            count = dataset_data[name]["counts"].get(label, 0)
            row[f"{name}_count"] = count
            row[f"{name}_pct"] = _format_percentage(count, total)
        merged_rows.append(row)

    # Append the final summary 'at total' row
    total_row = {"range": "at total"}
    for name in dataset_names:
        total = dataset_data[name]["total"]
        total_row[f"{name}_count"] = total
        total_row[f"{name}_pct"] = "100.000%"
    merged_rows.append(total_row)

    return merged_rows


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve base data directory from environment or relative path
    base_dir = os.environ.get("MOLECULE_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    default_dirs = [os.path.join(base_dir, d) for d in ["HessianRelax", "TorsionScanRelax"]]

    parser = argparse.ArgumentParser(description="Analyze relaxation step distributions across datasets.")
    parser.add_argument("--relax_dirs",
                        nargs="+",
                        default=default_dirs,
                        help="One or more dataset directories to analyze.")
    parser.add_argument("--bin-size", type=int, default=5, help="Size of histogram bins (default: 5).")
    parser.add_argument("--output",
                        default=os.path.join(file_dir, "relax_steps_histogram.csv"),
                        help="Path to save the merged summary CSV.")
    return parser.parse_args()


def main() -> None:
    """Main execution flow for analyzing and merging dataset histograms."""
    args = parse_args()

    # Filter for valid existing directories
    valid_dirs = [d for d in args.relax_dirs if os.path.exists(d)]
    if not valid_dirs:
        print("No valid dataset directories found.")
        return

    print(f"Processing {len(valid_dirs)} datasets sequentially...")

    # Collect analysis results
    results = []
    for d in valid_dirs:
        res = _analyze_dataset_steps(d, args.bin_size)
        if res:
            results.append(res)

    if not results:
        print("No analysis data was successfully collected.")
        return

    # Aggregate and sort metadata
    dataset_names = sorted([res[0] for res in results])
    dataset_data = {name: {"counts": counts, "total": total} for name, counts, total in results}

    # Determine all unique bin labels and sort them logically
    unique_labels = {label for _, counts, _ in results for label in counts.keys()}
    sorted_labels = sorted(list(unique_labels), key=_get_bin_sort_key)

    # Generate merged table and save to CSV
    merged_rows = _generate_merged_table_rows(sorted_labels, dataset_names, dataset_data)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_merged = pd.DataFrame(merged_rows)
    df_merged.to_csv(args.output, index=False)

    print(f"Merged histogram successfully saved to {args.output}")
    print(f"\nSummary of Relaxation Steps Histogram (bin_size={args.bin_size}):")
    print(df_merged.to_string(index=False))


if __name__ == "__main__":
    main()
