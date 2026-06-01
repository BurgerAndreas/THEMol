#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run one PySCF single-point energy, force, and Hessian calculation.

The input is one molecule group from a THEMol Hessian HDF5 file. The sample
number is zero-based by default and indexes the sorted top-level HDF5 groups.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.data.elements import ELEMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate PySCF energy, force, and Hessian for one THEMol HDF5 sample."
    )
    parser.add_argument("h5_file", help="Input THEMol Hessian HDF5 file.")
    parser.add_argument("sample_number", type=int, help="Sample index within the HDF5 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path. Defaults to results/hessians/{no_df,df}/<input-stem>_sample_<index>_pyscf[_df].h5.",
    )
    parser.add_argument("--one-based", action="store_true", help="Interpret sample_number as one-based.")
    parser.add_argument("--xc", default="wb97m-v", help="PySCF DFT functional.")
    parser.add_argument("--basis", default="def2-tzvpd", help="PySCF basis set.")
    parser.add_argument("--unit", default="Angstrom", choices=["Angstrom", "Bohr"], help="Input coordinate unit.")
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        help="Override PySCF spin, equal to N_alpha - N_beta. Closed-shell singlets use 0.",
    )
    parser.add_argument("--grid-level", type=int, default=4, help="PySCF DFT integration grid level.")
    parser.add_argument(
        "--nlc-grid-level",
        type=int,
        default=3,
        help="PySCF nonlocal-correlation grid level, used when the mean-field object exposes nlcgrids.",
    )
    parser.add_argument("--conv-tol", type=float, default=1e-10, help="SCF energy convergence tolerance.")
    parser.add_argument("--max-cycle", type=int, default=100, help="Maximum SCF cycles.")
    parser.add_argument("--verbose", type=int, default=3, help="PySCF verbosity level.")
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=0,
        help="PySCF memory limit in MB. Use 0 to keep PySCF's default.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "1")),
        help="Number of BLAS/OpenMP threads to ask PySCF to use.",
    )
    parser.add_argument(
        "--density-fit",
        action="store_true",
        help="Use density fitting. This may change numerical results and is off by default.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5 file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and print the selected sample metadata without running PySCF.",
    )
    return parser.parse_args()


def read_h5_string(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    value = group[name].asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def infer_charge_and_spin(smiles: str | None, atomic_numbers: np.ndarray) -> tuple[int, int, str]:
    """Infer charge/spin from SMILES when possible, otherwise use electron parity."""
    charge = 0
    radical_electrons = 0
    source = "electron_parity"

    if smiles:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is not None:
                charge = int(Chem.GetFormalCharge(mol))
                radical_electrons = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
                source = "rdkit_mapped_smiles"
        except Exception as exc:  # noqa: BLE001 - inference fallback should not stop the calculation.
            print(f"WARNING: RDKit charge/spin inference failed: {exc}", file=sys.stderr)

    nelec = int(np.sum(atomic_numbers) - charge)
    if radical_electrons:
        spin = radical_electrons
        if (nelec - spin) % 2 != 0:
            spin = nelec % 2
            source += "_parity_adjusted"
    else:
        spin = nelec % 2

    return charge, spin, source


def select_sample(h5_file: Path, sample_number: int, one_based: bool) -> dict[str, Any]:
    sample_index = sample_number - 1 if one_based else sample_number
    if sample_index < 0:
        raise ValueError("sample_number selects a negative zero-based index")

    with h5py.File(h5_file, "r") as handle:
        keys = sorted(handle.keys())
        if sample_index >= len(keys):
            raise IndexError(f"sample index {sample_index} out of range for {len(keys)} samples")

        key = keys[sample_index]
        group = handle[key]
        atomic_numbers = group["atomic_numbers"][()].reshape(-1).astype(np.int32)
        coords = group["coords"][()].astype(np.float64)
        reference_hessian = group["hessian"][()].astype(np.float64)
        mapped_isomeric_smiles = read_h5_string(group, "mapped_isomeric_smiles")
        mapped_nonisomeric_smiles = read_h5_string(group, "mapped_nonisomeric_smiles")

    return {
        "sample_index": sample_index,
        "sample_key": key,
        "atomic_numbers": atomic_numbers,
        "coords": coords,
        "reference_hessian": reference_hessian,
        "mapped_isomeric_smiles": mapped_isomeric_smiles,
        "mapped_nonisomeric_smiles": mapped_nonisomeric_smiles,
    }


def build_molecule(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    basis: str,
    unit: str,
    charge: int,
    spin: int,
    verbose: int,
) -> gto.Mole:
    atoms = [(ELEMENTS[int(z)], tuple(map(float, xyz))) for z, xyz in zip(atomic_numbers, coords)]
    return gto.M(
        atom=atoms,
        basis=basis,
        unit=unit,
        charge=charge,
        spin=spin,
        verbose=verbose,
    )


def configure_mean_field(mol: gto.Mole, args: argparse.Namespace):
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = args.xc
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle

    if args.max_memory_mb > 0:
        mf.max_memory = args.max_memory_mb

    if hasattr(mf, "grids"):
        mf.grids.level = args.grid_level
    if hasattr(mf, "nlcgrids"):
        mf.nlcgrids.level = args.nlc_grid_level

    if args.density_fit:
        mf = mf.density_fit()

    return mf


def flatten_hessian(hessian_4d: np.ndarray) -> np.ndarray:
    """Convert PySCF Hessian axes (atom_i, atom_j, xyz_i, xyz_j) to 3N x 3N."""
    if hessian_4d.ndim != 4 or hessian_4d.shape[2:] != (3, 3):
        raise ValueError(f"Unexpected PySCF Hessian shape: {hessian_4d.shape}")
    return hessian_4d.transpose(0, 2, 1, 3).reshape(3 * hessian_4d.shape[0], 3 * hessian_4d.shape[1])


def write_results(
    output: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin: int,
    charge_spin_source: str,
    energy: float,
    gradient: np.ndarray,
    hessian_4d: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    hessian_2d = flatten_hessian(hessian_4d).astype(np.float64)
    force = (-gradient).astype(np.float64)

    with h5py.File(output, "w") as handle:
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["program"] = "calculate_pyscf_sample.py"
        handle.attrs["pyscf_version"] = pyscf.__version__
        handle.attrs["python_version"] = platform.python_version()
        handle.attrs["command"] = " ".join(sys.argv)
        handle.attrs["input_h5_file"] = str(Path(args.h5_file).resolve())
        handle.attrs["sample_index_zero_based"] = int(sample["sample_index"])
        handle.attrs["sample_key"] = sample["sample_key"]

        calc = handle.create_group(sample["sample_key"])
        calc.attrs["xc"] = args.xc
        calc.attrs["basis"] = args.basis
        calc.attrs["unit"] = args.unit
        calc.attrs["charge"] = charge
        calc.attrs["spin"] = spin
        calc.attrs["charge_spin_source"] = charge_spin_source
        calc.attrs["grid_level"] = args.grid_level
        calc.attrs["nlc_grid_level"] = args.nlc_grid_level
        calc.attrs["conv_tol"] = args.conv_tol
        calc.attrs["max_cycle"] = args.max_cycle
        calc.attrs["verbose"] = args.verbose
        calc.attrs["threads"] = args.threads
        calc.attrs["max_memory_mb"] = args.max_memory_mb
        calc.attrs["density_fit"] = bool(args.density_fit)
        calc.attrs["energy_unit"] = "Hartree"
        calc.attrs["gradient_unit"] = "Hartree/Bohr"
        calc.attrs["force_unit"] = "Hartree/Bohr"
        calc.attrs["hessian_unit"] = "Hartree/Bohr^2"

        for name in ("mapped_isomeric_smiles", "mapped_nonisomeric_smiles"):
            value = sample[name]
            if value is not None:
                calc.create_dataset(name, data=value, dtype=string_dtype)

        calc.create_dataset("atomic_numbers", data=sample["atomic_numbers"].reshape(-1, 1), dtype="i4")
        calc.create_dataset("coords", data=sample["coords"], dtype="f8")
        calc.create_dataset("energy", data=np.array(energy, dtype=np.float64))
        calc.create_dataset("gradient", data=gradient.astype(np.float64), dtype="f8")
        calc.create_dataset("force", data=force, dtype="f8")
        calc.create_dataset("hessian", data=hessian_2d, dtype="f8")
        calc.create_dataset("hessian_4d", data=hessian_4d.astype(np.float64), dtype="f8")
        calc.create_dataset("reference_hessian", data=sample["reference_hessian"], dtype="f8")

        metadata = {
            "input_h5_file": str(Path(args.h5_file).resolve()),
            "sample_index_zero_based": int(sample["sample_index"]),
            "sample_key": sample["sample_key"],
            "xc": args.xc,
            "basis": args.basis,
            "charge": charge,
            "spin": spin,
            "charge_spin_source": charge_spin_source,
        }
        calc.create_dataset("metadata_json", data=json.dumps(metadata, sort_keys=True), dtype=string_dtype)


def main() -> None:
    args = parse_args()
    h5_file = Path(args.h5_file)
    if not h5_file.exists():
        raise FileNotFoundError(h5_file)

    if args.threads > 0:
        lib.num_threads(args.threads)

    sample = select_sample(h5_file, args.sample_number, args.one_based)
    inferred_charge, inferred_spin, source = infer_charge_and_spin(
        sample["mapped_nonisomeric_smiles"],
        sample["atomic_numbers"],
    )
    charge = args.charge if args.charge is not None else inferred_charge
    spin = args.spin if args.spin is not None else inferred_spin
    if args.charge is not None or args.spin is not None:
        source = "command_line_override"

    if args.output:
        output = Path(args.output)
    else:
        output_dir = Path("results/hessians/df" if args.density_fit else "results/hessians/no_df")
        suffix = "pyscf_df" if args.density_fit else "pyscf"
        output = output_dir / f"{h5_file.stem}_sample_{sample['sample_index']}_{suffix}.h5"

    print(f"Input file: {h5_file}")
    print(f"Sample index: {sample['sample_index']} key={sample['sample_key']}")
    print(f"Atoms: {len(sample['atomic_numbers'])}")
    print(f"SMILES: {sample['mapped_nonisomeric_smiles']}")
    print(f"Charge/spin: {charge}/{spin} ({source})")
    print(f"Output: {output}")

    if args.dry_run:
        print("Dry run requested; stopping before PySCF calculation.")
        return

    mol = build_molecule(
        sample["atomic_numbers"],
        sample["coords"],
        args.basis,
        args.unit,
        charge,
        spin,
        args.verbose,
    )
    mf = configure_mean_field(mol, args)

    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("SCF did not converge; no gradient/Hessian written")

    gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
    hessian_4d = np.asarray(mf.Hessian().kernel(), dtype=np.float64)

    write_results(output, sample, args, charge, spin, source, energy, gradient, hessian_4d)
    print(f"Energy (Hartree): {energy:.16f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Force shape: {gradient.shape}")
    print(f"Hessian shape: {flatten_hessian(hessian_4d).shape}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
