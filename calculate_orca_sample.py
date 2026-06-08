#!/usr/bin/env python3
"""Run one ORCA energy, force, and analytic Hessian calculation for a THEMol sample."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate ORCA energy, force, and analytic Hessian for one THEMol HDF5 sample."
    )
    parser.add_argument("h5_file", help="Input THEMol Hessian HDF5 file.")
    parser.add_argument("sample_number", type=int, help="Sample index within the HDF5 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path. Defaults to results/hessians/orca_wb97m_d4_def2_tzvpd/<input-stem>_sample_<index>_orca_wb97m_d4_def2_tzvpd.h5.",
    )
    parser.add_argument("--one-based", action="store_true", help="Interpret sample_number as one-based.")
    parser.add_argument("--functional", default="WB97M-D4", help="ORCA functional keyword.")
    parser.add_argument("--basis", default="def2-TZVPD", help="ORCA basis keyword.")
    parser.add_argument("--grid", default="DefGrid3", help="ORCA integration grid keyword.")
    parser.add_argument("--scf", default="TightSCF", help="ORCA SCF convergence keyword.")
    parser.add_argument(
        "--scf-max-iter",
        type=int,
        help="Write an ORCA %%scf block with this MaxIter value.",
    )
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        help="Override spin as N_alpha - N_beta. Multiplicity is spin + 1.",
    )
    parser.add_argument("--nprocs", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")))
    parser.add_argument(
        "--maxcore-mb",
        type=int,
        default=0,
        help="ORCA %%maxcore value per process in MB. Defaults to 75%% of node memory divided by nprocs.",
    )
    parser.add_argument(
        "--orca-command",
        default=os.environ.get("ORCA_COMMAND") or str(Path(os.environ.get("EBROOTORCA", "")) / "orca"),
        help="ORCA executable path. Defaults to $ORCA_COMMAND, then $EBROOTORCA/orca.",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for ORCA input/output. Defaults to orca_work/<input-stem>_sample_<index>.",
    )
    parser.add_argument(
        "--no-rijcosx",
        action="store_true",
        help="Add NoRIJCOSX to test the integral-direct calculation instead of ORCA's hybrid default.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5 file.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Do not delete the ORCA work directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the ORCA input and stop before running ORCA.",
    )
    return parser.parse_args()


def read_h5_string(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    value = group[name].asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def symbol_from_atomic_number(atomic_number: int) -> str:
    try:
        from rdkit import Chem

        return Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number))
    except Exception as exc:  # noqa: BLE001 - fail with context if RDKit is unavailable.
        raise RuntimeError("RDKit is required to convert atomic numbers to symbols") from exc


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


def default_maxcore_mb(nprocs: int) -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total_mb = int(line.split()[1]) / 1024
                return max(512, int(total_mb * 0.75 / max(1, nprocs)))
    return 2048


def write_orca_input(
    input_path: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin: int,
    maxcore_mb: int,
) -> None:
    multiplicity = spin + 1
    if multiplicity < 1:
        raise ValueError(f"Invalid spin {spin}; ORCA multiplicity would be {multiplicity}")

    keywords = [args.functional, args.basis, args.scf, args.grid, "EnGrad", "Freq"]
    if args.no_rijcosx:
        keywords.append("NoRIJCOSX")

    lines = [
        "# Generated by calculate_orca_sample.py",
        f"# sample_index_zero_based={sample['sample_index']} sample_key={sample['sample_key']}",
        f"# smiles={sample['mapped_nonisomeric_smiles']}",
        "! " + " ".join(keywords),
        "",
        f"%pal nprocs {args.nprocs} end",
        f"%maxcore {maxcore_mb}",
    ]
    if args.scf_max_iter is not None:
        lines.extend(
            [
                "%scf",
                f"  MaxIter {args.scf_max_iter}",
                "end",
            ]
        )
    lines.extend(
        [
            "",
            f"* xyz {charge} {multiplicity}",
        ]
    )
    for atomic_number, xyz in zip(sample["atomic_numbers"], sample["coords"]):
        symbol = symbol_from_atomic_number(int(atomic_number))
        lines.append(f"  {symbol:<2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    lines.append("*")
    lines.append("")

    input_path.write_text("\n".join(lines), encoding="utf-8")


def run_orca(orca_command: str, input_path: Path, output_path: Path) -> None:
    command = [orca_command, input_path.name]
    with output_path.open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"ORCA failed with exit code {completed.returncode}; see {output_path}")


def validate_orca_command(orca_command: str) -> str:
    if os.sep in orca_command:
        if Path(orca_command).exists():
            return orca_command
    else:
        resolved = shutil.which(orca_command)
        if resolved:
            return resolved
    raise FileNotFoundError(f"ORCA executable not found: {orca_command}")


def parse_engrad(path: Path) -> tuple[float, np.ndarray]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)

    if len(values) < 2:
        raise ValueError(f"Could not parse ORCA engrad file: {path}")

    natoms = int(values[0])
    energy = float(values[1])
    gradient_values = np.array([float(value) for value in values[2 : 2 + 3 * natoms]], dtype=np.float64)
    if gradient_values.size != 3 * natoms:
        raise ValueError(f"Expected {3 * natoms} gradient values in {path}, got {gradient_values.size}")
    return energy, gradient_values.reshape(natoms, 3)


def parse_hessian(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip().lower() == "$hessian")
    except StopIteration as exc:
        raise ValueError(f"No $hessian section found in {path}") from exc

    size = int(lines[start + 1].strip())
    hessian = np.zeros((size, size), dtype=np.float64)
    row_counts = np.zeros(size, dtype=np.int32)
    index = start + 2

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("$"):
            break

        columns = [int(value) for value in stripped.split()]
        index += 1
        for _ in range(size):
            if index >= len(lines):
                raise ValueError(f"Unexpected end of $hessian section in {path}")
            row_parts = lines[index].split()
            index += 1
            if not row_parts:
                continue
            row = int(row_parts[0])
            values = [float(value) for value in row_parts[1:]]
            if len(values) != len(columns):
                raise ValueError(f"Malformed Hessian row {row} in {path}")
            hessian[row, columns] = values
            row_counts[row] += len(values)

    if not np.all(row_counts == size):
        raise ValueError(f"Incomplete Hessian in {path}; parsed columns per row: {row_counts.tolist()}")
    return hessian


def write_results(
    output: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin: int,
    charge_spin_source: str,
    energy: float,
    gradient: np.ndarray,
    hessian: np.ndarray,
    work_dir: Path,
    maxcore_mb: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    force = (-gradient).astype(np.float64)

    with h5py.File(output, "w") as handle:
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["program"] = "calculate_orca_sample.py"
        handle.attrs["python_version"] = platform.python_version()
        handle.attrs["command"] = " ".join(sys.argv)
        handle.attrs["input_h5_file"] = str(Path(args.h5_file).resolve())
        handle.attrs["sample_index_zero_based"] = int(sample["sample_index"])
        handle.attrs["sample_key"] = sample["sample_key"]

        calc = handle.create_group(sample["sample_key"])
        calc.attrs["functional"] = args.functional
        calc.attrs["basis"] = args.basis
        calc.attrs["grid"] = args.grid
        calc.attrs["scf"] = args.scf
        calc.attrs["charge"] = charge
        calc.attrs["spin"] = spin
        calc.attrs["multiplicity"] = spin + 1
        calc.attrs["charge_spin_source"] = charge_spin_source
        calc.attrs["nprocs"] = args.nprocs
        calc.attrs["maxcore_mb"] = maxcore_mb
        calc.attrs["rijcosx"] = not args.no_rijcosx
        calc.attrs["energy_unit"] = "Hartree"
        calc.attrs["gradient_unit"] = "Hartree/Bohr"
        calc.attrs["force_unit"] = "Hartree/Bohr"
        calc.attrs["hessian_unit"] = "Hartree/Bohr^2"
        calc.attrs["work_dir"] = str(work_dir)

        for name in ("mapped_isomeric_smiles", "mapped_nonisomeric_smiles"):
            value = sample[name]
            if value is not None:
                calc.create_dataset(name, data=value, dtype=string_dtype)

        calc.create_dataset("atomic_numbers", data=sample["atomic_numbers"].reshape(-1, 1), dtype="i4")
        calc.create_dataset("coords", data=sample["coords"], dtype="f8")
        calc.create_dataset("energy", data=np.array(energy, dtype=np.float64))
        calc.create_dataset("gradient", data=gradient.astype(np.float64), dtype="f8")
        calc.create_dataset("force", data=force, dtype="f8")
        calc.create_dataset("hessian", data=hessian.astype(np.float64), dtype="f8")
        calc.create_dataset("reference_hessian", data=sample["reference_hessian"], dtype="f8")

        metadata = {
            "input_h5_file": str(Path(args.h5_file).resolve()),
            "sample_index_zero_based": int(sample["sample_index"]),
            "sample_key": sample["sample_key"],
            "functional": args.functional,
            "basis": args.basis,
            "charge": charge,
            "spin": spin,
            "charge_spin_source": charge_spin_source,
            "rijcosx": not args.no_rijcosx,
        }
        calc.create_dataset("metadata_json", data=json.dumps(metadata, sort_keys=True), dtype=string_dtype)


def main() -> None:
    args = parse_args()
    h5_file = Path(args.h5_file)
    if not h5_file.exists():
        raise FileNotFoundError(h5_file)

    sample = select_sample(h5_file, args.sample_number, args.one_based)
    inferred_charge, inferred_spin, source = infer_charge_and_spin(
        sample["mapped_nonisomeric_smiles"],
        sample["atomic_numbers"],
    )
    charge = args.charge if args.charge is not None else inferred_charge
    spin = args.spin if args.spin is not None else inferred_spin
    if args.charge is not None or args.spin is not None:
        source = "command_line_override"

    output = (
        Path(args.output)
        if args.output
        else Path("results/hessians/orca_wb97m_d4_def2_tzvpd")
        / f"{h5_file.stem}_sample_{sample['sample_index']}_orca_wb97m_d4_def2_tzvpd.h5"
    )
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("orca_work") / f"{h5_file.stem}_sample_{sample['sample_index']}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    basename = "orca"
    input_path = work_dir / f"{basename}.inp"
    output_path = work_dir / f"{basename}.out"
    engrad_path = work_dir / f"{basename}.engrad"
    hessian_path = work_dir / f"{basename}.hess"
    maxcore_mb = args.maxcore_mb if args.maxcore_mb > 0 else default_maxcore_mb(args.nprocs)

    print(f"Input file: {h5_file}")
    print(f"Sample index: {sample['sample_index']} key={sample['sample_key']}")
    print(f"Atoms: {len(sample['atomic_numbers'])}")
    print(f"SMILES: {sample['mapped_nonisomeric_smiles']}")
    print(f"Charge/spin/multiplicity: {charge}/{spin}/{spin + 1} ({source})")
    print(f"ORCA: {args.orca_command}")
    print(f"Work dir: {work_dir}")
    print(f"Output HDF5: {output}")

    write_orca_input(input_path, sample, args, charge, spin, maxcore_mb)
    if args.dry_run:
        print(f"Dry run requested; wrote {input_path} and stopped before ORCA.")
        return

    orca_command = validate_orca_command(args.orca_command)
    run_orca(str(orca_command), input_path, output_path)
    energy, gradient = parse_engrad(engrad_path)
    hessian = parse_hessian(hessian_path)
    write_results(output, sample, args, charge, spin, source, energy, gradient, hessian, work_dir, maxcore_mb)

    print(f"Energy (Hartree): {energy:.16f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Force shape: {gradient.shape}")
    print(f"Hessian shape: {hessian.shape}")
    print(f"Wrote {output}")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
