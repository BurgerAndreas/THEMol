#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run a finite-difference CCSD(T)/CBS Hessian for one THEMol sample.

The CBS estimate uses a two-point cc-pVXZ extrapolation:

* Hartree-Fock energy/gradient from the high-cardinal basis.
* CCSD(T) correlation energy/gradient extrapolated as X^-alpha.

Each displaced CBS gradient is cached in the work directory, so interrupted jobs
can be resumed by rerunning with the same --work-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyscf
from pyscf import cc, gto, lib, scf
from pyscf.data import elements
from pyscf.data.elements import ELEMENTS
from pyscf.grad import ccsd_t as ccsd_t_grad

from calculate_pyscf_sample import (
    flatten_hessian,
    infer_charge_and_spin,
    read_h5_string,
    select_sample,
)

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class GradientResult:
    energy_total: float
    energy_hf: float
    gradient_total: np.ndarray
    gradient_hf: np.ndarray


def frozen_core_orbitals(atomic_numbers: np.ndarray) -> int:
    """Return the standard chemical-core frozen orbital count for a closed-shell molecule."""
    return int(sum(elements.chemcore_atm[int(z)] for z in atomic_numbers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate a finite-difference CCSD(T)/CBS Hessian for one THEMol HDF5 sample."
    )
    parser.add_argument("h5_file", help="Input THEMol Hessian HDF5 file.")
    parser.add_argument("sample_number", type=int, help="Sample index within the HDF5 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path. Defaults to results/hessians/ccsd_t_cbs/<input-stem>_sample_<index>_ccsd_t_cbs.h5.",
    )
    parser.add_argument("--one-based", action="store_true", help="Interpret sample_number as one-based.")
    parser.add_argument("--basis-low", default="cc-pVTZ", help="Low-cardinal basis for CBS extrapolation.")
    parser.add_argument("--basis-high", default="cc-pVQZ", help="High-cardinal basis for CBS extrapolation.")
    parser.add_argument("--cardinal-low", type=float, default=3.0, help="Cardinal number for --basis-low.")
    parser.add_argument("--cardinal-high", type=float, default=4.0, help="Cardinal number for --basis-high.")
    parser.add_argument(
        "--cbs-exponent",
        type=float,
        default=3.0,
        help="Power-law exponent for CCSD(T) correlation CBS extrapolation.",
    )
    parser.add_argument(
        "--step-bohr",
        type=float,
        default=0.005,
        help="Central finite-difference displacement in Bohr.",
    )
    parser.add_argument("--unit", default="Angstrom", choices=["Angstrom"], help="Input coordinate unit.")
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        help="Override PySCF spin, equal to N_alpha - N_beta. Only closed-shell spin=0 is supported.",
    )
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10, help="RHF energy convergence tolerance.")
    parser.add_argument("--scf-max-cycle", type=int, default=100, help="Maximum RHF cycles.")
    parser.add_argument("--cc-conv-tol", type=float, default=1e-8, help="CCSD amplitude convergence tolerance.")
    parser.add_argument("--cc-max-cycle", type=int, default=100, help="Maximum CCSD cycles.")
    parser.add_argument("--verbose", type=int, default=4, help="PySCF verbosity level.")
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
        "--no-frozen-core",
        action="store_true",
        help="Use all electrons instead of frozen-core CCSD(T). Frozen core is the default.",
    )
    parser.add_argument(
        "--work-dir",
        help="Directory for cached displaced-gradient files. Defaults to ccsd_t_cbs_work/<input-stem>_sample_<index>.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5 file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and print selected sample metadata without running CCSD(T).",
    )
    return parser.parse_args()


def build_molecule(
    atomic_numbers: np.ndarray,
    coords_angstrom: np.ndarray,
    basis: str,
    charge: int,
    spin: int,
    verbose: int,
) -> gto.Mole:
    atoms = [(ELEMENTS[int(z)], tuple(map(float, xyz))) for z, xyz in zip(atomic_numbers, coords_angstrom)]
    return gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        verbose=verbose,
    )


def run_ccsd_t_gradient(
    atomic_numbers: np.ndarray,
    coords_angstrom: np.ndarray,
    basis: str,
    args: argparse.Namespace,
    charge: int,
    spin: int,
) -> GradientResult:
    if spin != 0:
        raise ValueError("Only closed-shell spin=0 samples are supported by this RCCSD(T)/CBS driver")

    mol = build_molecule(atomic_numbers, coords_angstrom, basis, charge, spin, args.verbose)
    if args.max_memory_mb > 0:
        mol.max_memory = args.max_memory_mb

    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle
    if args.max_memory_mb > 0:
        mf.max_memory = args.max_memory_mb

    energy_hf = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for basis {basis}")

    gradient_hf = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)

    frozen = 0 if args.no_frozen_core else frozen_core_orbitals(atomic_numbers)
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = args.cc_conv_tol
    mycc.max_cycle = args.cc_max_cycle
    if args.max_memory_mb > 0:
        mycc.max_memory = args.max_memory_mb
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError(f"CCSD did not converge for basis {basis}")

    triples_energy = float(mycc.ccsd_t())
    gradient_total = np.asarray(ccsd_t_grad.Gradients(mycc).kernel(), dtype=np.float64)
    energy_total = float(energy_hf + mycc.e_corr + triples_energy)

    return GradientResult(
        energy_total=energy_total,
        energy_hf=energy_hf,
        gradient_total=gradient_total,
        gradient_hf=gradient_hf,
    )


def extrapolate_correlation(low: np.ndarray | float, high: np.ndarray | float, x_low: float, x_high: float, alpha: float):
    denominator = x_high**alpha - x_low**alpha
    return (np.asarray(high) * x_high**alpha - np.asarray(low) * x_low**alpha) / denominator


def cbs_from_basis_results(low: GradientResult, high: GradientResult, args: argparse.Namespace) -> GradientResult:
    low_corr_energy = low.energy_total - low.energy_hf
    high_corr_energy = high.energy_total - high.energy_hf
    low_corr_gradient = low.gradient_total - low.gradient_hf
    high_corr_gradient = high.gradient_total - high.gradient_hf

    corr_energy_cbs = extrapolate_correlation(
        low_corr_energy,
        high_corr_energy,
        args.cardinal_low,
        args.cardinal_high,
        args.cbs_exponent,
    )
    corr_gradient_cbs = extrapolate_correlation(
        low_corr_gradient,
        high_corr_gradient,
        args.cardinal_low,
        args.cardinal_high,
        args.cbs_exponent,
    )

    return GradientResult(
        energy_total=float(high.energy_hf + corr_energy_cbs),
        energy_hf=high.energy_hf,
        gradient_total=np.asarray(high.gradient_hf + corr_gradient_cbs, dtype=np.float64),
        gradient_hf=high.gradient_hf,
    )


def cache_path(work_dir: Path, label: str) -> Path:
    return work_dir / f"{label}.npz"


def load_cached_cbs_gradient(path: Path) -> GradientResult | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        return GradientResult(
            energy_total=float(data["energy_total"]),
            energy_hf=float(data["energy_hf"]),
            gradient_total=np.asarray(data["gradient_total"], dtype=np.float64),
            gradient_hf=np.asarray(data["gradient_hf"], dtype=np.float64),
        )


def save_cbs_gradient(path: Path, result: GradientResult) -> None:
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        energy_total=np.array(result.energy_total, dtype=np.float64),
        energy_hf=np.array(result.energy_hf, dtype=np.float64),
        gradient_total=result.gradient_total.astype(np.float64),
        gradient_hf=result.gradient_hf.astype(np.float64),
    )
    tmp_path.replace(path)


def calculate_cbs_gradient(
    atomic_numbers: np.ndarray,
    coords_angstrom: np.ndarray,
    label: str,
    args: argparse.Namespace,
    charge: int,
    spin: int,
    work_dir: Path,
) -> GradientResult:
    path = cache_path(work_dir, label)
    cached = load_cached_cbs_gradient(path)
    if cached is not None:
        print(f"Reusing cached CBS gradient: {path}", flush=True)
        return cached

    print(f"Calculating CBS gradient: {label}", flush=True)
    low = run_ccsd_t_gradient(atomic_numbers, coords_angstrom, args.basis_low, args, charge, spin)
    high = run_ccsd_t_gradient(atomic_numbers, coords_angstrom, args.basis_high, args, charge, spin)
    cbs = cbs_from_basis_results(low, high, args)
    save_cbs_gradient(path, cbs)
    return cbs


def hessian_from_finite_difference(
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin: int,
    work_dir: Path,
) -> tuple[float, np.ndarray, np.ndarray]:
    atomic_numbers = sample["atomic_numbers"]
    coords = sample["coords"].astype(np.float64)
    natoms = len(atomic_numbers)
    step_angstrom = args.step_bohr * BOHR_TO_ANGSTROM

    reference = calculate_cbs_gradient(atomic_numbers, coords, "reference", args, charge, spin, work_dir)
    hessian_4d = np.zeros((natoms, natoms, 3, 3), dtype=np.float64)

    for disp_atom in range(natoms):
        for disp_axis in range(3):
            coord_label = f"a{disp_atom:03d}_x{disp_axis}"
            plus_coords = coords.copy()
            minus_coords = coords.copy()
            plus_coords[disp_atom, disp_axis] += step_angstrom
            minus_coords[disp_atom, disp_axis] -= step_angstrom

            plus = calculate_cbs_gradient(
                atomic_numbers,
                plus_coords,
                f"{coord_label}_plus",
                args,
                charge,
                spin,
                work_dir,
            )
            minus = calculate_cbs_gradient(
                atomic_numbers,
                minus_coords,
                f"{coord_label}_minus",
                args,
                charge,
                spin,
                work_dir,
            )
            hessian_4d[:, disp_atom, :, disp_axis] = (plus.gradient_total - minus.gradient_total) / (
                2.0 * args.step_bohr
            )

    hessian_2d = flatten_hessian(hessian_4d)
    hessian_2d = (hessian_2d + hessian_2d.T) / 2.0
    hessian_4d = hessian_2d.reshape(natoms, 3, natoms, 3).transpose(0, 2, 1, 3)
    return reference.energy_total, reference.gradient_total, hessian_4d


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
    work_dir: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    hessian_2d = flatten_hessian(hessian_4d).astype(np.float64)
    force = (-gradient).astype(np.float64)

    with h5py.File(output, "w") as handle:
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["program"] = "calculate_ccsd_t_cbs_sample.py"
        handle.attrs["pyscf_version"] = pyscf.__version__
        handle.attrs["python_version"] = platform.python_version()
        handle.attrs["command"] = " ".join(sys.argv)
        handle.attrs["input_h5_file"] = str(Path(args.h5_file).resolve())
        handle.attrs["sample_index_zero_based"] = int(sample["sample_index"])
        handle.attrs["sample_key"] = sample["sample_key"]

        calc = handle.create_group(sample["sample_key"])
        calc.attrs["method"] = "CCSD(T)/CBS"
        calc.attrs["basis_low"] = args.basis_low
        calc.attrs["basis_high"] = args.basis_high
        calc.attrs["cardinal_low"] = args.cardinal_low
        calc.attrs["cardinal_high"] = args.cardinal_high
        calc.attrs["cbs_exponent"] = args.cbs_exponent
        calc.attrs["unit"] = args.unit
        calc.attrs["charge"] = charge
        calc.attrs["spin"] = spin
        calc.attrs["charge_spin_source"] = charge_spin_source
        calc.attrs["scf_conv_tol"] = args.scf_conv_tol
        calc.attrs["scf_max_cycle"] = args.scf_max_cycle
        calc.attrs["cc_conv_tol"] = args.cc_conv_tol
        calc.attrs["cc_max_cycle"] = args.cc_max_cycle
        calc.attrs["verbose"] = args.verbose
        calc.attrs["threads"] = args.threads
        calc.attrs["max_memory_mb"] = args.max_memory_mb
        calc.attrs["frozen_core"] = not args.no_frozen_core
        calc.attrs["step_bohr"] = args.step_bohr
        calc.attrs["work_dir"] = str(work_dir)
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
            "method": "CCSD(T)/CBS",
            "basis_low": args.basis_low,
            "basis_high": args.basis_high,
            "charge": charge,
            "spin": spin,
            "charge_spin_source": charge_spin_source,
            "frozen_core": not args.no_frozen_core,
            "step_bohr": args.step_bohr,
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
        output = Path("results/hessians/ccsd_t_cbs") / f"{h5_file.stem}_sample_{sample['sample_index']}_ccsd_t_cbs.h5"

    work_dir = Path(args.work_dir) if args.work_dir else Path("ccsd_t_cbs_work") / f"{h5_file.stem}_sample_{sample['sample_index']}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {h5_file}")
    print(f"Sample index: {sample['sample_index']} key={sample['sample_key']}")
    print(f"Atoms: {len(sample['atomic_numbers'])}")
    print(f"SMILES: {sample['mapped_nonisomeric_smiles']}")
    print(f"Charge/spin: {charge}/{spin} ({source})")
    print(f"CBS bases: {args.basis_low}/{args.basis_high}")
    print(f"Frozen core: {not args.no_frozen_core}")
    print(f"Finite-difference step: {args.step_bohr} Bohr")
    print(f"Output: {output}")
    print(f"Work dir: {work_dir}")

    if spin != 0:
        raise ValueError("Only closed-shell spin=0 samples are supported by this RCCSD(T)/CBS driver")

    if args.dry_run:
        print("Dry run requested; stopping before PySCF calculation.")
        return

    energy, gradient, hessian_4d = hessian_from_finite_difference(sample, args, charge, spin, work_dir)
    write_results(output, sample, args, charge, spin, source, energy, gradient, hessian_4d, work_dir)
    print(f"Energy (Hartree): {energy:.16f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Force shape: {gradient.shape}")
    print(f"Hessian shape: {flatten_hessian(hessian_4d).shape}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
