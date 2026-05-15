# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

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
    return os.path.join(base_dir, "TorsionScan")


def test_torsion_dataset_structure(datafolder):
    if not datafolder:
        pytest.skip("No datafolder provided.")

    dataset_csv = os.path.join(datafolder, "torsion_dataset.csv")
    if not os.path.exists(dataset_csv):
        pytest.fail(f"torsion_dataset.csv not found in {datafolder}")

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False, na_filter=False)
    if len(df) == 0:
        pytest.fail(f"{dataset_csv} is empty")

    required_columns = [
        "uuid", "mapped_nonisomeric_smiles", "mapped_isomeric_smiles", "torsion_indices", "h5_file", "num_constraints"
    ]
    for col in required_columns:
        if col not in df.columns:
            pytest.fail(f"Column '{col}' not found in {dataset_csv}. Columns: {df.columns}")

    grouped = df.groupby("h5_file")
    for h5_file, group in grouped:
        h5_path = os.path.join(datafolder, h5_file)
        if not os.path.exists(h5_path):
            pytest.fail(f"HDF5 file {h5_path} not found.")

        with h5py.File(h5_path, "r") as f:
            sample_size = min(20, len(group))
            sampled_rows = group.sample(n=sample_size, random_state=0)

            for row in sampled_rows.itertuples(index=False):
                uuid = row.uuid
                if uuid not in f:
                    pytest.fail(f"UUID {uuid} not found in {h5_path}")

                grp = f[uuid]

                assert "mapped_nonisomeric_smiles" in grp, f"mapped_nonisomeric_smiles missing for {uuid}"
                dset = grp["mapped_nonisomeric_smiles"]
                string_info = h5py.check_string_dtype(dset.dtype)
                assert string_info is not None, f"mapped_nonisomeric_smiles not string for {uuid}"
                assert string_info.encoding == "utf-8", f"mapped_nonisomeric_smiles not utf-8 for {uuid}"
                assert _read_h5_string(dset) == row.mapped_nonisomeric_smiles, (
                    f"mapped_nonisomeric_smiles mismatch for {uuid}")

                assert "mapped_isomeric_smiles" in grp, f"mapped_isomeric_smiles missing for {uuid}"
                dset = grp["mapped_isomeric_smiles"]
                string_info = h5py.check_string_dtype(dset.dtype)
                assert string_info is not None, f"mapped_isomeric_smiles not string for {uuid}"
                assert string_info.encoding == "utf-8", f"mapped_isomeric_smiles not utf-8 for {uuid}"
                smiles_str = _read_h5_string(dset)
                assert smiles_str == row.mapped_isomeric_smiles, f"mapped_isomeric_smiles mismatch for {uuid}"

                assert "atomic_numbers" in grp, f"atomic_numbers missing for {uuid}"
                atnums = grp["atomic_numbers"]
                assert atnums.dtype == np.int32, f"atomic_numbers dtype mismatch for {uuid}: {atnums.dtype}"
                assert atnums.ndim == 2, f"atomic_numbers ndim mismatch for {uuid}: {atnums.shape}"
                assert atnums.shape[1] == 1, f"atomic_numbers shape mismatch for {uuid}: {atnums.shape}"
                N = atnums.shape[0]

                assert "torsion_atom_indices" in grp, f"torsion_atom_indices missing for {uuid}"
                indices = grp["torsion_atom_indices"]
                assert indices.dtype == np.int32, f"torsion_atom_indices dtype mismatch for {uuid}: {indices.dtype}"
                assert indices.ndim == 1, f"torsion_atom_indices ndim mismatch for {uuid}: {indices.shape}"
                assert indices.shape[0] == 4, f"torsion_atom_indices shape mismatch for {uuid}: {indices.shape}"
                assert indices.size == 4, f"torsion_atom_indices size mismatch for {uuid}: {indices.shape}"
                indices_list = indices[:].tolist()
                assert indices_list == _parse_torsion_indices(row.torsion_indices), (
                    f"torsion_indices mismatch for {uuid}")

                indices_str = "[" + ",".join(str(x) for x in indices_list) + "]"

                expected_uuid = deterministic_uuid(smiles_str + indices_str)
                assert uuid == expected_uuid, f"UUID mismatch for {uuid}: expected {expected_uuid}, got {uuid}"

                constraint_keys = _sorted_constraint_keys(grp)
                expected_num_constraints = int(row.num_constraints)
                assert len(constraint_keys) == expected_num_constraints, (
                    f"num_constraints mismatch for {uuid}: {len(constraint_keys)} vs {expected_num_constraints}")
                assert constraint_keys == [f"constraint {i}" for i in range(expected_num_constraints)], (
                    f"constraint key mismatch for {uuid}")

                for step_key in constraint_keys:
                    step_grp = grp[step_key]

                    assert "coords" in step_grp, f"coords missing for {uuid}/{step_key}"
                    coords = step_grp["coords"]
                    assert coords.dtype == np.float64, f"coords dtype mismatch for {uuid}/{step_key}: {coords.dtype}"
                    assert coords.shape == (
                        N, 3), f"coords shape mismatch for {uuid}/{step_key}: {coords.shape} vs ({N}, 3)"

                    assert "energy" in step_grp, f"energy missing for {uuid}/{step_key}"
                    energy = step_grp["energy"]
                    assert energy.dtype == np.float64, f"energy dtype mismatch for {uuid}/{step_key}: {energy.dtype}"
                    assert energy.size == 1, f"energy size mismatch for {uuid}/{step_key}: {energy.shape}"

                    assert "forces" in step_grp, f"forces missing for {uuid}/{step_key}"
                    forces = step_grp["forces"]
                    assert forces.dtype == np.float64, f"forces dtype mismatch for {uuid}/{step_key}: {forces.dtype}"
                    assert forces.shape == (
                        N, 3), f"forces shape mismatch for {uuid}/{step_key}: {forces.shape} vs ({N}, 3)"
