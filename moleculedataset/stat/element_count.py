# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Calculate element frequencies and molecular size distributions from SMILES CSVs.

This script processes one or more dataset folders containing SMILES strings, calculates 
atom-level element counts, molecule-level element presence, and molecular size 
(non-hydrogen atom count) histograms. Results are saved as CSV files.
"""

import argparse
import csv
import glob
import itertools
import multiprocessing as mp
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from bytemol.core import Molecule


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


def _iter_smiles_from_csv(csv_path: str, smiles_column: str) -> Iterable[str]:
    """Yield SMILES strings from a specific column in a CSV file.

    Args:
        csv_path: Path to the input CSV file.
        smiles_column: Name of the column containing SMILES strings.

    Yields:
        SMILES strings found in the CSV.
    """
    with open(csv_path, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            value = row.get(smiles_column, "")
            if value:
                yield value


def _process_smiles_chunk(chunk: List[str]) -> Tuple[Counter, Counter, Counter, int, int, int]:
    """Process a chunk of SMILES to compute element and size statistics.

    Args:
        chunk: A list of SMILES strings to process.

    Returns:
        A tuple containing:
            - atom_counts: Frequency of each element (excluding H).
            - molecule_counts: Count of molecules containing each element (excluding H).
            - size_counts: Frequency of molecular sizes (non-H atom count).
            - total_atoms: Sum of all non-H atoms in the chunk.
            - total_molecules: Number of successfully processed molecules.
            - failed: Number of SMILES strings that failed to parse.
    """
    atom_counts = Counter()
    molecule_counts = Counter()
    size_counts = Counter()
    total_atoms = 0
    total_molecules = 0
    failed = 0
    for smi in chunk:
        try:
            # Load molecule using bytemol; nconfs=0 for efficiency
            mol = Molecule.from_mapped_smiles(smi, nconfs=0)
            rkmol = mol.get_rkmol()
            # Extract symbols for all non-hydrogen atoms
            elements = [atom.GetSymbol() for atom in rkmol.GetAtoms() if atom.GetSymbol() != "H"]
            size = len(elements)

            total_atoms += size
            total_molecules += 1
            atom_counts.update(elements)
            size_counts[size] += 1
            # For molecule-level counts, use a set to count each element once per molecule
            for element in set(elements):
                molecule_counts[element] += 1
        except Exception:
            failed += 1
    return atom_counts, molecule_counts, size_counts, total_atoms, total_molecules, failed


def _chunk_iterable(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive chunks from an iterable.

    Args:
        iterable: The input iterable.
        size: Maximum size of each chunk.

    Yields:
        Lists of elements from the iterable.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _calculate_dataset_stats(smiles_list: Iterable[str],
                             workers: int = 1) -> Tuple[Counter, Counter, Counter, int, int, int]:
    """Calculate aggregate statistics for a list of SMILES using multiprocessing.

    Args:
        smiles_list: Iterable of SMILES strings.
        workers: Number of parallel worker processes to use.

    Returns:
        Aggregated statistics (same format as _process_smiles_chunk).
    """
    atom_counts: Counter = Counter()
    molecule_counts: Counter = Counter()
    size_counts: Counter = Counter()
    total_atoms = 0
    total_molecules = 0
    failed = 0

    chunk_size = 1000
    chunks = _chunk_iterable(smiles_list, chunk_size)

    # Use 'fork' context for multiprocessing efficiency on Linux
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for res in tqdm(
                pool.imap_unordered(_process_smiles_chunk, chunks),
                desc="Analyzing dataset",
                unit="chunk",
        ):
            a_counts, m_counts, s_counts, a_total, m_total, fail = res
            atom_counts.update(a_counts)
            molecule_counts.update(m_counts)
            size_counts.update(s_counts)
            total_atoms += a_total
            total_molecules += m_total
            failed += fail

    return atom_counts, molecule_counts, size_counts, total_atoms, total_molecules, failed


def _format_percentage(count: int, total: int) -> str:
    """Format a count as 'count(percentage%)'.

    Args:
        count: The numerator.
        total: The denominator.

    Returns:
        A formatted string with the count and percentage to 4 decimal places.
    """
    if total <= 0:
        return "0(0.0000%)"
    pct = 100.0 * count / total
    return f"{count}({pct:.4f}%)"


def _generate_table_rows(
    row_keys: List[Any],
    dataset_names: List[str],
    totals: Dict[str, int],
    counts: Dict[str, Counter],
) -> List[List[str]]:
    """Prepare data rows for a statistics table.

    Args:
        row_keys: The values to appear in the first column (e.g., elements or sizes).
        dataset_names: Names of the datasets (columns).
        totals: Mapping of dataset names to their respective total counts.
        counts: Mapping of dataset names to their frequency counters.

    Returns:
        A list of rows, where each row is a list of strings.
    """
    rows = []
    for key in row_keys:
        row = [str(key)]
        for name in dataset_names:
            row.append(_format_percentage(counts[name].get(key, 0), totals[name]))
        rows.append(row)
    return rows


def _save_stats_to_csv(
    output_csv: str,
    row_keys: List[Any],
    dataset_names: List[str],
    totals: Dict[str, int],
    counts: Dict[str, Counter],
    first_col_header: str = "element",
) -> None:
    """Save statistics to a CSV file.

    Args:
        output_csv: Path to save the CSV.
        row_keys: Keys for the rows (e.g., element symbols).
        dataset_names: Names of the datasets.
        totals: Total counts for normalization.
        counts: Frequency data.
        first_col_header: Header label for the first column.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([first_col_header] + dataset_names)
        writer.writerows(_generate_table_rows(row_keys, dataset_names, totals, counts))


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.environ.get("MOLECULE_DATA_DIR")
    if not base_dir:
        # Fallback to relative path if environment variable is not set
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    default_inputs = [
        os.path.join(base_dir, "Hessian"),
        os.path.join(base_dir, "HessianRelax"),
        os.path.join(base_dir, "TorsionScan"),
        os.path.join(base_dir, "TorsionScanRelax"),
        os.path.join(base_dir, "MBIS"),
    ]

    parser = argparse.ArgumentParser(description="Analyze molecular datasets for element and size distribution.")
    parser.add_argument("--input_csv",
                        nargs="+",
                        default=default_inputs,
                        help="Paths to dataset folders containing CSV files.")
    parser.add_argument("--smiles-column",
                        default="mapped_isomeric_smiles",
                        help="Name of the SMILES column in the CSV files.")
    parser.add_argument("--output-dir", default=file_dir, help="Directory to save the resulting CSV files.")
    parser.add_argument("--workers",
                        type=int,
                        default=os.cpu_count() or 1,
                        help="Number of parallel workers for processing.")
    return parser.parse_args()


def main() -> None:
    """Main execution flow for analyzing datasets."""
    args = parse_args()

    print(f"We are using {args.workers} workers to analyze the datasets:")
    for folder in args.input_csv:
        print(folder)

    # Identify dataset folders and their primary CSV files
    dataset_dirs = args.input_csv
    dataset_names: List[str] = []
    csv_paths: Dict[str, str] = {}
    for folder in dataset_dirs:
        name = os.path.basename(os.path.normpath(folder))
        dataset_names.append(name)
        csv_paths[name] = _find_csv_in_folder(folder)

    # Initialize storage for aggregated statistics
    atom_counts: Dict[str, Counter] = {}
    molecule_counts: Dict[str, Counter] = {}
    size_counts: Dict[str, Counter] = {}
    atom_totals: Dict[str, int] = {}
    molecule_totals: Dict[str, int] = {}
    failed_counts: Dict[str, int] = {}

    final_dataset_names: List[str] = []

    # Process each dataset
    for name in dataset_names:
        smiles_iter = _iter_smiles_from_csv(csv_paths[name], args.smiles_column)
        all_smiles = list(smiles_iter)

        # Standard analysis
        final_dataset_names.append(name)
        a_counts, m_counts, s_counts, a_total, m_total, failed = _calculate_dataset_stats(all_smiles,
                                                                                          workers=args.workers)
        atom_counts[name] = a_counts
        molecule_counts[name] = m_counts
        size_counts[name] = s_counts
        atom_totals[name] = a_total
        molecule_totals[name] = m_total
        failed_counts[name] = failed

        # Optional analysis for unique SMILES if duplicates are found
        unique_smiles = list(set(all_smiles))
        if len(unique_smiles) < len(all_smiles):
            unique_name = f"{name} (unique)"
            final_dataset_names.append(unique_name)
            print(f"Dataset {name} contains duplicates. Analyzing {len(unique_smiles)} unique entries.")
            a_counts, m_counts, s_counts, a_total, m_total, failed = _calculate_dataset_stats(unique_smiles,
                                                                                              workers=args.workers)
            atom_counts[unique_name] = a_counts
            molecule_counts[unique_name] = m_counts
            size_counts[unique_name] = s_counts
            atom_totals[unique_name] = a_total
            molecule_totals[unique_name] = m_total
            failed_counts[unique_name] = failed

    dataset_names = final_dataset_names

    # Determine unique elements across all datasets and sort by frequency
    elements = set()
    for name in dataset_names:
        elements.update(atom_counts[name].keys())

    atom_elements = sorted(elements, key=lambda e: -sum(atom_counts[n].get(e, 0) for n in dataset_names))
    molecule_elements = sorted(elements, key=lambda e: -sum(molecule_counts[n].get(e, 0) for n in dataset_names))

    # Determine unique molecular sizes across all datasets
    all_sizes = set()
    for name in dataset_names:
        all_sizes.update(size_counts[name].keys())
    sorted_sizes = sorted(list(all_sizes))

    # Save final results to CSV files
    atom_csv = os.path.join(args.output_dir, "element_counts_atom.csv")
    molecule_csv = os.path.join(args.output_dir, "element_counts_molecule.csv")
    size_csv = os.path.join(args.output_dir, "molecular_size_histogram.csv")

    _save_stats_to_csv(atom_csv, atom_elements, dataset_names, atom_totals, atom_counts)
    _save_stats_to_csv(molecule_csv, molecule_elements, dataset_names, molecule_totals, molecule_counts)
    _save_stats_to_csv(size_csv,
                       sorted_sizes,
                       dataset_names,
                       molecule_totals,
                       size_counts,
                       first_col_header="#non-hydrogen atoms")

    # Report failures
    for name in dataset_names:
        if failed_counts.get(name):
            print(f"{name}: Failed to process {failed_counts[name]} SMILES strings.")


if __name__ == "__main__":
    main()
