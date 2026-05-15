# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os

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


def _sorted_step_keys(group, prefix):
    return sorted((key for key in group.keys() if key.startswith(prefix)), key=lambda key: int(key.split()[1]))


@pytest.fixture
def datafolder():
    base_dir = os.environ.get("MOLECULE_DATA_DIR")
    if not base_dir:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    return os.path.join(base_dir, "HessianRelax")


def test_relax_dataset_structure(datafolder):
    if not datafolder:
        pytest.skip("No datafolder provided.")

    dataset_csv = os.path.join(datafolder, "relax_dataset.csv")
    if not os.path.exists(dataset_csv):
        pytest.fail(f"relax_dataset.csv not found in {datafolder}")

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False, na_filter=False)
    if len(df) == 0:
        pytest.fail(f"{dataset_csv} is empty")

    required_columns = ["uuid", "mapped_nonisomeric_smiles", "mapped_isomeric_smiles", "num_steps", "h5_file"]
    for col in required_columns:
        if col not in df.columns:
            pytest.fail(f"Column '{col}' not found in {dataset_csv}. Columns: {df.columns}")

    grouped = df.groupby("h5_file")
    for h5_file, group in grouped:
        h5_path = os.path.join(datafolder, h5_file)
        if not os.path.exists(h5_path):
            pytest.fail(f"HDF5 file {h5_path} not found.")

        with h5py.File(h5_path, "r") as f:
            sample_size = min(100, len(group))
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

                expected_uuid = deterministic_uuid(smiles_str)
                assert uuid == expected_uuid, f"UUID mismatch for {uuid}: expected {expected_uuid}, got {uuid}"

                step_keys = _sorted_step_keys(grp, "step ")
                expected_num_steps = int(row.num_steps)
                assert len(step_keys) == expected_num_steps, (
                    f"num_steps mismatch for {uuid}: {len(step_keys)} vs {expected_num_steps}")
                assert step_keys == [f"step {i}" for i in range(expected_num_steps)], f"step key mismatch for {uuid}"

                for step_key in step_keys:
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
