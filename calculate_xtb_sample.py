#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run one xTB energy, force, and numerical Hessian calculation."""

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
        description="Calculate xTB energy, force, and numerical Hessian for one THEMol HDF5 sample."
    )
    parser.add_argument("h5_file", help="Input THEMol Hessian HDF5 file.")
    parser.add_argument("sample_number", type=int, help="Sample index within the HDF5 file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path. Defaults to results/hessians/<method>/<input-stem>_sample_<index>_<method>.h5.",
    )
    parser.add_argument("--one-based", action="store_true", help="Interpret sample_number as one-based.")
    parser.add_argument("--gfn", type=int, default=2, choices=[0, 1, 2], help="xTB GFN method.")
    parser.add_argument(
        "--gxtb",
        action="store_true",
        help="Run g-xTB via a modified xtb binary using the --gxtb flag.",
    )
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        help="Override spin as N_alpha - N_beta. Passed to xTB as --uhf.",
    )
    parser.add_argument("--parallel", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")))
    parser.add_argument(
        "--xtb-command",
        default=os.environ.get("XTB_COMMAND", "xtb"),
        help="xTB executable path. Defaults to $XTB_COMMAND, then xtb on PATH.",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for xTB input/output. Defaults to xtb_work/<input-stem>_sample_<index>.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5 file.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Do not delete the xTB work directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the xTB XYZ input and stop before running xTB.",
    )
    return parser.parse_args()


def read_h5_string(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    value = group[name].asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


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


def write_xyz(path: Path, sample: dict[str, Any]) -> None:
    lines = [
        str(len(sample["atomic_numbers"])),
        f"sample_index_zero_based={sample['sample_index']} sample_key={sample['sample_key']}",
    ]
    for atomic_number, xyz in zip(sample["atomic_numbers"], sample["coords"]):
        symbol = symbol_from_atomic_number(int(atomic_number))
        lines.append(f"{symbol:<2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_xtb_command(xtb_command: str) -> str:
    if os.sep in xtb_command:
        if Path(xtb_command).exists():
            return xtb_command
    else:
        resolved = shutil.which(xtb_command)
        if resolved:
            return resolved
    raise FileNotFoundError(f"xTB executable not found: {xtb_command}")


def xtb_base_command(xtb_command: str, xyz_path: Path, args: argparse.Namespace, charge: int, spin: int) -> list[str]:
    command = [
        xtb_command,
        xyz_path.name,
        "--chrg",
        str(charge),
        "--uhf",
        str(spin),
    ]
    if args.gxtb:
        command.append("--gxtb")
    else:
        command.extend(["--gfn", str(args.gfn)])
    if args.parallel > 0:
        command.extend(["--parallel", str(args.parallel)])
    return command


def run_xtb(
    xtb_command: str,
    xyz_path: Path,
    args: argparse.Namespace,
    charge: int,
    spin: int,
    mode: str,
    output_path: Path,
) -> None:
    command = xtb_base_command(xtb_command, xyz_path, args, charge, spin)
    command.append(mode)
    with output_path.open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command,
            cwd=xyz_path.parent,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"xTB {mode} failed with exit code {completed.returncode}; see {output_path}")


def parse_gradient(path: Path, natoms: int) -> tuple[float, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    cycle_indices = [index for index, line in enumerate(lines) if line.lower().startswith("cycle")]
    if not cycle_indices:
        raise ValueError(f"No gradient cycle found in {path}")

    cycle_index = cycle_indices[-1]
    header = lines[cycle_index].replace("=", " ").split()
    try:
        energy_index = header.index("energy") + 1
    except ValueError as exc:
        raise ValueError(f"No energy value found in gradient header: {lines[cycle_index]}") from exc
    energy = parse_float(header[energy_index])

    gradient_lines = lines[cycle_index + 1 + natoms : cycle_index + 1 + 2 * natoms]
    if len(gradient_lines) != natoms:
        raise ValueError(f"Expected {natoms} gradient lines in {path}, got {len(gradient_lines)}")

    gradient = np.array(
        [[parse_float(value) for value in line.split()[:3]] for line in gradient_lines],
        dtype=np.float64,
    )
    if gradient.shape != (natoms, 3):
        raise ValueError(f"Unexpected gradient shape parsed from {path}: {gradient.shape}")
    return energy, gradient


def method_name(args: argparse.Namespace) -> str:
    return "g-xTB" if args.gxtb else f"GFN{args.gfn}-xTB"


def parse_hessian(path: Path, size: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip().lower() == "$hessian")
    except StopIteration:
        start = 0

    values: list[float] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            break
        values.extend(parse_float(value) for value in stripped.split())

    expected = size * size
    if len(values) != expected:
        raise ValueError(f"Expected {expected} Hessian values in {path}, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(size, size)


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
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    force = (-gradient).astype(np.float64)

    with h5py.File(output, "w") as handle:
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["program"] = "calculate_xtb_sample.py"
        handle.attrs["python_version"] = platform.python_version()
        handle.attrs["command"] = " ".join(sys.argv)
        handle.attrs["input_h5_file"] = str(Path(args.h5_file).resolve())
        handle.attrs["sample_index_zero_based"] = int(sample["sample_index"])
        handle.attrs["sample_key"] = sample["sample_key"]

        calc = handle.create_group(sample["sample_key"])
        calc.attrs["method"] = method_name(args)
        calc.attrs["charge"] = charge
        calc.attrs["spin"] = spin
        calc.attrs["charge_spin_source"] = charge_spin_source
        calc.attrs["parallel"] = args.parallel
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
            "method": method_name(args),
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
        method = "gxtb" if args.gxtb else f"xtb_gfn{args.gfn}"
        output = Path("results/hessians") / method / f"{h5_file.stem}_sample_{sample['sample_index']}_{method}.h5"
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("xtb_work") / f"{h5_file.stem}_sample_{sample['sample_index']}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = work_dir / "molecule.xyz"
    grad_output_path = work_dir / "xtb_grad.out"
    hess_output_path = work_dir / "xtb_hess.out"
    gradient_path = work_dir / "gradient"
    hessian_path = work_dir / "hessian"

    print(f"Input file: {h5_file}")
    print(f"Sample index: {sample['sample_index']} key={sample['sample_key']}")
    print(f"Atoms: {len(sample['atomic_numbers'])}")
    print(f"SMILES: {sample['mapped_nonisomeric_smiles']}")
    print(f"Charge/spin: {charge}/{spin} ({source})")
    print(f"xTB: {args.xtb_command}")
    print(f"Work dir: {work_dir}")
    print(f"Output HDF5: {output}")

    write_xyz(xyz_path, sample)
    if args.dry_run:
        print(f"Dry run requested; wrote {xyz_path} and stopped before xTB.")
        return

    xtb_command = validate_xtb_command(args.xtb_command)
    run_xtb(xtb_command, xyz_path, args, charge, spin, "--grad", grad_output_path)
    energy, gradient = parse_gradient(gradient_path, len(sample["atomic_numbers"]))
    run_xtb(xtb_command, xyz_path, args, charge, spin, "--hess", hess_output_path)
    hessian = parse_hessian(hessian_path, size=3 * len(sample["atomic_numbers"]))
    write_results(output, sample, args, charge, spin, source, energy, gradient, hessian, work_dir)

    print(f"Energy (Hartree): {energy:.16f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Force shape: {gradient.shape}")
    print(f"Hessian shape: {hessian.shape}")
    print(f"Wrote {output}")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
