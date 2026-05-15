# Statistical Scripts

This directory contains utility scripts to calculate various statistical insights and distributions across the dataset. The results are saved as CSV files and provide an overview of the chemical space and structural complexity covered by the datasets.

## Available Scripts

1. **`element_count.py`**
   - **Description**: Analyzes the atomic element frequencies and molecular size distributions across all dataset subsets.
   - **Outputs**:
     - `element_counts_atom.csv`: Total count and percentage of each non-hydrogen element in the dataset.
     - `element_counts_molecule.csv`: Count and percentage of molecules containing at least one atom of a specific element.
     - `molecular_size_histogram.csv`: Histogram of molecular sizes (measured by the number of non-hydrogen atoms).
   - **Usage**:
     ```bash
     python element_count.py --workers 4
     ```
2. **`step_count.py`**
   - **Description**: Generates histograms of the number of relaxation steps required to reach convergence  for the `HessianRelax` and `TorsionScanRelax` datasets.
   - **Outputs**:
     - `relax_steps_histogram.csv`: A unified histogram showing the distribution of relaxation steps across the relevant subsets.
   - **Usage**:
     ```bash
     python step_count.py --bin-size 5
     ```
3. **`torsion_constraint_count.py`**
   - **Description**: Distinguishes between in-ring and out-of-ring torsions within the `TorsionScan` dataset and calculates the distribution of constraints applied to each type.
   - **Outputs**:
     - `TorsionScan_constraints_histogram.csv`: A statistical summary of the constraint counts for in-ring vs. out-of-ring torsions.
   - **Usage**:
     ```bash
     python torsion_constraint_count.py --workers 4
     ```
4. **`dataset_size_count.py`**
   - **Description**: Analyzes dataset sizes and generates a summary of the number of entries, unique molecules, constraints, and total steps across molecular subsets.
   - **Outputs**:
     - `dataset_size_count.csv`: A summary table detailing the entries and supplementary metrics for each subset and level of theory.
   - **Usage**:
     ```bash
     python dataset_size_count.py
     ```

## Environment Configuration

By default, these scripts will look for data directories inside the path specified by the `MOLECULE_DATA_DIR` environment variable. If the variable is not set, they will fall back to a relative `../../data` directory.

To run these scripts on a specific data directory, simply export the variable first:

```bash
export MOLECULE_DATA_DIR=/path/to/your/data_directory
python element_count.py
```

