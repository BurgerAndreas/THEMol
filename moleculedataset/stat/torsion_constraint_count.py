# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Analyze and count in-ring torsions from molecular dataset CSVs.

This script identifies torsions that reside within ring systems using bytemol/RDKit 
and generates statistics on the number of constraints associated with these torsions.
"""

import argparse
import ast
import os
from collections import Counter
from multiprocessing import Pool
from typing import List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from bytemol.core import Molecule


def _is_torsion_in_ring(args: Tuple[str, Union[str, List[int]]]) -> bool:
    """Check if a specific torsion (defined by 4 atom indices) is within a ring.

    Args:
        args: A tuple of (SMILES string, torsion atom indices).
              Indices can be a string representation of a list or a list of integers.

    Returns:
        True if the bond between the central two atoms of the torsion is in a ring, 
        False otherwise.
    """
    smiles, indices_val = args

    try:
        # Parse indices if they are provided as a string
        if isinstance(indices_val, str):
            indices = ast.literal_eval(indices_val)
        else:
            indices = indices_val

        # Torsions must involve exactly 4 atoms
        if not isinstance(indices, (list, tuple)) or len(indices) != 4:
            return False

        # Load molecule using bytemol (mapped SMILES preserve atom ordering)
        bmol = Molecule.from_mapped_smiles(smiles)
        mol = bmol.to_rkmol()

        if mol is None:
            return False

        # Basic bounds check
        if max(indices) >= mol.GetNumAtoms():
            return False

        # The 'torsion' is defined by atoms at indices [1] and [2] (the central bond)
        v_idx = indices[1]
        w_idx = indices[2]

        bond = mol.GetBondBetweenAtoms(v_idx, w_idx)
        if bond is None:
            return False

        return bond.IsInRing()

    except Exception:
        # Catch-all for parsing or processing errors
        return False


def main() -> None:
    """Main execution flow for analyzing in-ring torsion statistics."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.environ.get("MOLECULE_DATA_DIR")
    if not base_dir:
        # Fallback to local data directory
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    parser = argparse.ArgumentParser(description="Analyze in-ring torsions from CSV dataset.")
    parser.add_argument("--data_dir",
                        default=os.path.join(base_dir, "TorsionScan"),
                        help="Directory containing torsion_dataset.csv.")
    parser.add_argument("--output",
                        default=os.path.join(file_dir, "TorsionScan_constraints_histogram.csv"),
                        help="Path to save the resulting statistics CSV.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Number of parallel worker processes.")

    args = parser.parse_args()

    data_dir = args.data_dir
    csv_path = os.path.join(data_dir, "torsion_dataset.csv")

    if not os.path.exists(csv_path):
        print(f"Dataset CSV not found: {csv_path}")
        return

    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Validate required columns
    required_cols = ['mapped_isomeric_smiles', 'torsion_indices', 'num_constraints']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column in CSV: {col}")
            return

    print(f"Analyzing {len(df)} records using {args.workers} workers...")

    # Multiprocessing tasks: (SMILES, torsion_indices)
    tasks = list(zip(df['mapped_isomeric_smiles'], df['torsion_indices']))

    # Execute ring check in parallel
    in_ring_flags = []
    with Pool(args.workers) as pool:
        # imap with chunksize for optimal performance on many small tasks
        for res in tqdm(pool.imap(_is_torsion_in_ring, tasks, chunksize=1000), total=len(tasks), desc="Checking rings"):
            in_ring_flags.append(res)

    # Filter and calculate statistics
    df['in_ring'] = in_ring_flags
    df_in_ring = df[df['in_ring']]
    df_out_ring = df[~df['in_ring']]

    total_in_ring = len(df_in_ring)
    total_out_ring = len(df_out_ring)

    print(f"Found {total_in_ring} in-ring torsions and {total_out_ring} out-of-ring torsions.")

    # Aggregate counts of constraints
    in_ring_counter = Counter(df_in_ring['num_constraints'])
    out_ring_counter = Counter(df_out_ring['num_constraints'])

    all_constraints = set(in_ring_counter.keys()) | set(out_ring_counter.keys())

    stats_data = []
    for count in sorted(all_constraints):
        in_freq = in_ring_counter.get(count, 0)
        out_freq = out_ring_counter.get(count, 0)

        in_percentage = (in_freq / total_in_ring * 100) if total_in_ring > 0 else 0
        out_percentage = (out_freq / total_out_ring * 100) if total_out_ring > 0 else 0

        stats_data.append({
            "num_constraints": count,
            "in_ring_count": in_freq,
            "in_ring_percentage": f"{in_percentage:.3f}%",
            "out_of_ring_count": out_freq,
            "out_of_ring_percentage": f"{out_percentage:.3f}%"
        })

    stats_df = pd.DataFrame(stats_data)

    # Add summary total row
    total_row = pd.DataFrame([{
        "num_constraints": "at total",
        "in_ring_count": total_in_ring,
        "in_ring_percentage": "100.000%" if total_in_ring > 0 else "0.000%",
        "out_of_ring_count": total_out_ring,
        "out_of_ring_percentage": "100.000%" if total_out_ring > 0 else "0.000%"
    }])

    if not stats_df.empty:
        stats_df = pd.concat([stats_df, total_row], ignore_index=True)
    else:
        stats_df = total_row

    # Organize columns for final output
    cols = ["num_constraints", "in_ring_count", "in_ring_percentage", "out_of_ring_count", "out_of_ring_percentage"]
    stats_df = stats_df[cols]

    stats_df.to_csv(args.output, index=False)
    print(f"Statistics successfully saved to {args.output}")
    print("\nSummary of Torsion Constraints:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
