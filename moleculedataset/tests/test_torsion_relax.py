# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
from ast import literal_eval

import h5py
import numpy as np
import pandas as pd
import pytest

from moleculedataset.utils.uuid import deterministic_uuid


def _read_h5_string(dataset):
    value = dataset.asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _parse_torsion_indices(value):
    parsed = literal_eval(value)
    assert isinstance(parsed, (list, tuple)), f"Invalid torsion_indices value: {value}"
    return [int(index) for index in parsed]


def _sorted_constraint_keys(group):
    return sorted((key for key in group.keys() if key.startswith("constraint ")), key=lambda key: int(key.split()[1]))


@pytest.fixture
def datafolder():
    base_dir = os.environ.get("MOLECULE_DATA_DIR")
    if not base_dir:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    return os.path.join(base_dir, "TorsionScanRelax")


def test_torsion_relax_xyz_dataset_structure(datafolder):
    if not datafolder:
        pytest.skip("No datafolder provided.")

    dataset_csv = os.path.join(datafolder, "torsion_relax_dataset.csv")
    if not os.path.exists(dataset_csv):
        pytest.fail(f"torsion_relax_dataset.csv not found in {datafolder}")

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False, na_filter=False)
    if len(df) == 0:
        pytest.fail(f"{dataset_csv} is empty")

    required_columns = [
        "uuid", "mapped_nonisomeric_smiles", "mapped_isomeric_smiles", "torsion_indices", "h5_file", "num_constraints",
        "num_total_steps"
    ]
    for col in required_columns:
        if col not in df.columns:
            pytest.fail(f"Column '{col}' not found in {dataset_csv}. Columns: {df.columns}")

    # Prepare tasks for multiprocessing
    tasks = []
    grouped = df.groupby("h5_file")
    for h5_file, group in grouped:
        h5_path = os.path.join(datafolder, h5_file)
        sample_size = min(20, len(group))
        sampled_rows = group.sample(n=sample_size, random_state=0)
        rows = sampled_rows.to_dict("records")
        tasks.append((h5_path, rows))

    # Process files in parallel
    num_cpus = os.cpu_count() or 2
    num_workers = max(1, num_cpus // 2)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_check_h5_file, tasks)

    # Check results for any failures
    for success, message in results:
        if not success:
            pytest.fail(message)


def _check_h5_file(task_data):
    h5_path, rows = task_data
    if not os.path.exists(h5_path):
        return False, f"HDF5 file {h5_path} not found."

    try:
        with h5py.File(h5_path, "r") as f:
            for row in rows:
                uuid = str(row["uuid"])
                if uuid not in f:
                    return False, f"UUID {uuid} not found in {h5_path}"

                grp = f[uuid]

                if "mapped_nonisomeric_smiles" not in grp:
                    return False, f"mapped_nonisomeric_smiles missing for {uuid}"
                dset = grp["mapped_nonisomeric_smiles"]
                string_info = h5py.check_string_dtype(dset.dtype)
                if string_info is None:
                    return False, f"mapped_nonisomeric_smiles not string for {uuid}"
                if string_info.encoding != "utf-8":
                    return False, f"mapped_nonisomeric_smiles not utf-8 for {uuid}"
                if _read_h5_string(dset) != row["mapped_nonisomeric_smiles"]:
                    return False, f"mapped_nonisomeric_smiles mismatch for {uuid}"

                if "mapped_isomeric_smiles" not in grp:
                    return False, f"mapped_isomeric_smiles missing for {uuid}"
                dset = grp["mapped_isomeric_smiles"]
                string_info = h5py.check_string_dtype(dset.dtype)
                if string_info is None:
                    return False, f"mapped_isomeric_smiles not string for {uuid}"
                if string_info.encoding != "utf-8":
                    return False, f"mapped_isomeric_smiles not utf-8 for {uuid}"
                smiles_str = _read_h5_string(dset)
                if smiles_str != row["mapped_isomeric_smiles"]:
                    return False, f"mapped_isomeric_smiles mismatch for {uuid}"

                if "atomic_numbers" not in grp:
                    return False, f"atomic_numbers missing for {uuid}"
                atnums = grp["atomic_numbers"]
                if atnums.dtype != np.int32:
                    return False, f"atomic_numbers dtype mismatch for {uuid}: {atnums.dtype}"
                if atnums.ndim != 2:
                    return False, f"atomic_numbers ndim mismatch for {uuid}: {atnums.shape}"
                if atnums.shape[1] != 1:
                    return False, f"atomic_numbers shape mismatch for {uuid}: {atnums.shape}"
                N = atnums.shape[0]

                if "torsion_atom_indices" not in grp:
                    return False, f"torsion_atom_indices missing for {uuid}"
                indices = grp["torsion_atom_indices"]
                if indices.dtype != np.int32:
                    return False, f"torsion_atom_indices dtype mismatch for {uuid}: {indices.dtype}"
                if indices.ndim != 1:
                    return False, f"torsion_atom_indices ndim mismatch for {uuid}: {indices.shape}"
                if indices.shape[0] != 4:
                    return False, f"torsion_atom_indices shape mismatch for {uuid}: {indices.shape}"
                if indices.size != 4:
                    return False, f"torsion_atom_indices size mismatch for {uuid}: {indices.shape}"
                indices_list = indices[:].tolist()
                if indices_list != _parse_torsion_indices(row["torsion_indices"]):
                    return False, f"torsion_indices mismatch for {uuid}"

                indices_str = "[" + ",".join(str(x) for x in indices_list) + "]"

                expected_uuid = deterministic_uuid(smiles_str + indices_str)
                if uuid != expected_uuid:
                    return False, f"UUID mismatch for {uuid}: expected {expected_uuid}, got {uuid}"

                constraint_keys = _sorted_constraint_keys(grp)
                expected_num_constraints = int(row["num_constraints"])
                if len(constraint_keys) != expected_num_constraints:
                    return False, (
                        f"num_constraints mismatch for {uuid}: {len(constraint_keys)} vs {expected_num_constraints}")
                if constraint_keys != [f"constraint {i}" for i in range(expected_num_constraints)]:
                    return False, f"constraint key mismatch for {uuid}"

                total_steps = 0

                for step_key in constraint_keys:
                    step_grp = grp[step_key]

                    if "coords" not in step_grp:
                        return False, f"coords missing for {uuid}/{step_key}"
                    coords = step_grp["coords"]
                    if coords.dtype != np.float64:
                        return False, f"coords dtype mismatch for {uuid}/{step_key}: {coords.dtype}"
                    if coords.ndim != 3:
                        return False, f"coords ndim mismatch for {uuid}/{step_key}: {coords.ndim}"
                    num_steps = coords.shape[0]
                    if coords.shape != (num_steps, N, 3):
                        return False, f"coords shape mismatch for {uuid}/{step_key}: {coords.shape} vs ({num_steps}, {N}, 3)"
                    total_steps += num_steps

                    if "energy" not in step_grp:
                        return False, f"energy missing for {uuid}/{step_key}"
                    energy = step_grp["energy"]
                    if energy.dtype != np.float64:
                        return False, f"energy dtype mismatch for {uuid}/{step_key}: {energy.dtype}"
                    if energy.ndim != 1:
                        return False, f"energy ndim mismatch for {uuid}/{step_key}: {energy.ndim}"
                    if energy.shape != (num_steps,):
                        return False, f"energy shape mismatch for {uuid}/{step_key}: {energy.shape} vs ({num_steps},)"

                    if "forces" not in step_grp:
                        return False, f"forces missing for {uuid}/{step_key}"
                    forces = step_grp["forces"]
                    if forces.dtype != np.float64:
                        return False, f"forces dtype mismatch for {uuid}/{step_key}: {forces.dtype}"
                    if forces.ndim != 3:
                        return False, f"forces ndim mismatch for {uuid}/{step_key}: {forces.ndim}"
                    if forces.shape != (num_steps, N, 3):
                        return False, f"forces shape mismatch for {uuid}/{step_key}: {forces.shape} vs ({num_steps}, {N}, 3)"

                expected_total_steps = int(row["num_total_steps"])
                if total_steps != expected_total_steps:
                    return False, f"num_total_steps mismatch for {uuid}: {total_steps} vs {expected_total_steps}"
    except Exception as e:
        return False, f"Exception occurred while processing {h5_path}: {str(e)}"

    return True, ""
