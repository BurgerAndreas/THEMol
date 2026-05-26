#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run UMA energy, force, and autograd Hessian calculations."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate UMA energy, force, and Hessian for one or more THEMol HDF5 samples."
    )
    parser.add_argument("h5_file", help="Input THEMol Hessian HDF5 file.")
    parser.add_argument(
        "sample_numbers",
        type=int,
        nargs="+",
        help="Sample indices within the HDF5 file. The UMA model is loaded once and reused.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 path for a single sample. Use --output-dir for multiple samples.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for per-sample output HDF5 files. Defaults to the current directory.",
    )
    parser.add_argument("--one-based", action="store_true", help="Interpret sample_number as one-based.")
    parser.add_argument("--model", default="uma-s-1p2", help="UMA model name passed to fairchem.")
    parser.add_argument("--task-name", default="omol", help="FAIRChem task/domain name.")
    parser.add_argument("--device", default="cuda", help="Torch device for UMA inference.")
    parser.add_argument("--seed", type=int, help="Optional seed passed to get_predict_unit.")
    parser.add_argument(
        "--molecule-cell-size",
        type=float,
        default=120.0,
        help="Vacuum cell size used by AtomicData.from_ase for molecular inputs.",
    )
    parser.add_argument("--charge", type=int, help="Override molecular charge.")
    parser.add_argument(
        "--spin-multiplicity",
        type=int,
        help="Override UMA spin multiplicity. Singlets use 1, doublets 2, triplets 3.",
    )
    parser.add_argument(
        "--hessian-loop",
        action="store_true",
        help="Use FAIRChem's lower-memory Hessian loop instead of vmap.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output HDF5 file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and print the selected sample metadata without running UMA.",
    )
    return parser.parse_args()


def read_h5_string(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    value = group[name].asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def infer_charge_and_spin_difference(smiles: str | None, atomic_numbers: np.ndarray) -> tuple[int, int, str]:
    """Infer charge and N_alpha - N_beta from SMILES when possible, otherwise use electron parity."""
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
        spin_difference = radical_electrons
        if (nelec - spin_difference) % 2 != 0:
            spin_difference = nelec % 2
            source += "_parity_adjusted"
    else:
        spin_difference = nelec % 2

    return charge, spin_difference, source


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


def build_predictor(args: argparse.Namespace):
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings

    settings = InferenceSettings(
        predict_untrained_forces={args.task_name},
        predict_untrained_hessian={args.task_name},
        hessian_vmap=not args.hessian_loop,
    )
    kwargs: dict[str, Any] = {
        "device": args.device,
        "inference_settings": settings,
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return pretrained_mlip.get_predict_unit(args.model, **kwargs)


def calculate_uma(
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin_multiplicity: int,
    predictor: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    from ase import Atoms
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    atoms = Atoms(numbers=sample["atomic_numbers"].reshape(-1), positions=sample["coords"])
    atoms.info.update({"charge": int(charge), "spin": int(spin_multiplicity)})

    data = AtomicData.from_ase(
        atoms,
        task_name=args.task_name,
        r_data_keys=["spin", "charge"],
        molecule_cell_size=args.molecule_cell_size,
    )
    batch = atomicdata_list_to_batch([data])
    preds = predictor.predict(batch)

    missing = {"energy", "forces", "hessian"} - set(preds)
    if missing:
        raise RuntimeError(f"UMA prediction did not include required keys: {sorted(missing)}")

    energy = float(preds["energy"].detach().cpu().numpy().reshape(-1)[0])
    forces = np.asarray(preds["forces"].detach().cpu().numpy(), dtype=np.float64)
    hessian = np.asarray(preds["hessian"].detach().cpu().numpy(), dtype=np.float64).squeeze()

    natoms = len(sample["atomic_numbers"])
    expected_force_shape = (natoms, 3)
    expected_hessian_shape = (3 * natoms, 3 * natoms)
    if forces.shape != expected_force_shape:
        raise ValueError(f"Unexpected UMA force shape: {forces.shape}; expected {expected_force_shape}")
    if hessian.shape != expected_hessian_shape:
        raise ValueError(f"Unexpected UMA Hessian shape: {hessian.shape}; expected {expected_hessian_shape}")

    return energy, forces, hessian


def write_results(
    output: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    charge: int,
    spin_difference: int,
    spin_multiplicity: int,
    charge_spin_source: str,
    energy: float,
    forces: np.ndarray,
    hessian: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    gradient = (-forces).astype(np.float64)
    hessian = ((hessian + hessian.T) / 2.0).astype(np.float64)

    with h5py.File(output, "w") as handle:
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["program"] = "calculate_uma_sample.py"
        handle.attrs["python_version"] = platform.python_version()
        handle.attrs["command"] = " ".join(sys.argv)
        handle.attrs["input_h5_file"] = str(Path(args.h5_file).resolve())
        handle.attrs["sample_index_zero_based"] = int(sample["sample_index"])
        handle.attrs["sample_key"] = sample["sample_key"]

        calc = handle.create_group(sample["sample_key"])
        calc.attrs["model"] = args.model
        calc.attrs["task_name"] = args.task_name
        calc.attrs["device"] = args.device
        calc.attrs["charge"] = charge
        calc.attrs["spin"] = spin_multiplicity
        calc.attrs["spin_multiplicity"] = spin_multiplicity
        calc.attrs["spin_difference"] = spin_difference
        calc.attrs["charge_spin_source"] = charge_spin_source
        calc.attrs["molecule_cell_size"] = args.molecule_cell_size
        calc.attrs["hessian_vmap"] = not args.hessian_loop
        calc.attrs["energy_unit"] = "eV"
        calc.attrs["gradient_unit"] = "eV/Angstrom"
        calc.attrs["force_unit"] = "eV/Angstrom"
        calc.attrs["hessian_unit"] = "eV/Angstrom^2"

        for name in ("mapped_isomeric_smiles", "mapped_nonisomeric_smiles"):
            value = sample[name]
            if value is not None:
                calc.create_dataset(name, data=value, dtype=string_dtype)

        calc.create_dataset("atomic_numbers", data=sample["atomic_numbers"].reshape(-1, 1), dtype="i4")
        calc.create_dataset("coords", data=sample["coords"], dtype="f8")
        calc.create_dataset("energy", data=np.array(energy, dtype=np.float64))
        calc.create_dataset("gradient", data=gradient, dtype="f8")
        calc.create_dataset("force", data=forces.astype(np.float64), dtype="f8")
        calc.create_dataset("hessian", data=hessian, dtype="f8")
        calc.create_dataset("reference_hessian", data=sample["reference_hessian"], dtype="f8")

        metadata = {
            "input_h5_file": str(Path(args.h5_file).resolve()),
            "sample_index_zero_based": int(sample["sample_index"]),
            "sample_key": sample["sample_key"],
            "model": args.model,
            "task_name": args.task_name,
            "charge": charge,
            "spin_difference": spin_difference,
            "spin_multiplicity": spin_multiplicity,
            "charge_spin_source": charge_spin_source,
        }
        calc.create_dataset("metadata_json", data=json.dumps(metadata, sort_keys=True), dtype=string_dtype)


def sanitized_model_name(model: str) -> str:
    return model.replace("-", "_").replace(".", "p")


def output_path_for_sample(args: argparse.Namespace, h5_file: Path, sample_index: int) -> Path:
    if args.output:
        if len(args.sample_numbers) != 1:
            raise ValueError("--output can only be used when calculating one sample")
        return Path(args.output)

    filename = f"{h5_file.stem}_sample_{sample_index}_{sanitized_model_name(args.model)}.h5"
    if args.output_dir:
        return Path(args.output_dir) / filename
    return Path(f"uma_{filename}")


def sample_charge_and_spin(sample: dict[str, Any], args: argparse.Namespace) -> tuple[int, int, int, str]:
    inferred_charge, inferred_spin_difference, source = infer_charge_and_spin_difference(
        sample["mapped_nonisomeric_smiles"],
        sample["atomic_numbers"],
    )
    charge = args.charge if args.charge is not None else inferred_charge
    spin_difference = inferred_spin_difference
    spin_multiplicity = args.spin_multiplicity if args.spin_multiplicity is not None else spin_difference + 1
    if args.charge is not None or args.spin_multiplicity is not None:
        source = "command_line_override"
    return charge, spin_difference, spin_multiplicity, source


def main() -> None:
    args = parse_args()
    h5_file = Path(args.h5_file)
    if not h5_file.exists():
        raise FileNotFoundError(h5_file)

    if args.output and len(args.sample_numbers) != 1:
        raise ValueError("--output can only be used when calculating one sample")

    samples = [select_sample(h5_file, sample_number, args.one_based) for sample_number in args.sample_numbers]
    sample_settings = [sample_charge_and_spin(sample, args) for sample in samples]

    print(f"Input file: {h5_file}")
    print(f"Sample count: {len(samples)}")
    print(f"Sample indices: {', '.join(str(sample['sample_index']) for sample in samples)}")
    print(f"UMA model/task/device: {args.model}/{args.task_name}/{args.device}")
    print(f"Hessian mode: {'loop' if args.hessian_loop else 'vmap'}")
    print(f"Output directory: {args.output_dir or Path.cwd()}")

    if args.dry_run:
        for sample, (charge, _spin_difference, spin_multiplicity, source) in zip(samples, sample_settings):
            output = output_path_for_sample(args, h5_file, int(sample["sample_index"]))
            print(
                f"Dry run sample {sample['sample_index']} key={sample['sample_key']} "
                f"atoms={len(sample['atomic_numbers'])} charge/spin={charge}/{spin_multiplicity} "
                f"source={source} output={output}"
            )
        print("Dry run requested; stopping before UMA model load.")
        return

    predictor = build_predictor(args)
    for index, (sample, settings) in enumerate(zip(samples, sample_settings), start=1):
        charge, spin_difference, spin_multiplicity, source = settings
        output = output_path_for_sample(args, h5_file, int(sample["sample_index"]))

        print(f"[{index}/{len(samples)}] Sample index: {sample['sample_index']} key={sample['sample_key']}")
        print(f"Atoms: {len(sample['atomic_numbers'])}")
        print(f"SMILES: {sample['mapped_nonisomeric_smiles']}")
        print(f"Charge/spin multiplicity: {charge}/{spin_multiplicity} ({source})")
        print(f"Output: {output}")

        energy, forces, hessian = calculate_uma(sample, args, charge, spin_multiplicity, predictor)
        write_results(output, sample, args, charge, spin_difference, spin_multiplicity, source, energy, forces, hessian)

        print(f"Energy (eV): {energy:.16f}")
        print(f"Force shape: {forces.shape}")
        print(f"Gradient shape: {forces.shape}")
        print(f"Hessian shape: {hessian.shape}")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
