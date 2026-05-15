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


@pytest.fixture
def datafolder():
    base_dir = os.environ.get("MOLECULE_DATA_DIR")
    if not base_dir:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    return os.path.join(base_dir, "Hessian")


def test_hessian_dataset_structure(datafolder):
    if not datafolder:
        pytest.skip("No datafolder provided.")

    dataset_csv = os.path.join(datafolder, "hessian_dataset.csv")
    if not os.path.exists(dataset_csv):
        pytest.fail(f"hessian_dataset.csv not found in {datafolder}")

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False, na_filter=False)
    if len(df) == 0:
        pytest.fail(f"{dataset_csv} is empty")

    required_columns = ["uuid", "mapped_nonisomeric_smiles", "mapped_isomeric_smiles", "h5_file"]
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

                # Verify UUID
                expected_uuid = deterministic_uuid(smiles_str)
                assert uuid == expected_uuid, f"UUID mismatch for {uuid}: expected {expected_uuid}, got {uuid}"

                assert "coords" in grp, f"coords missing for {uuid}"
                coords = grp["coords"]
                assert coords.dtype == np.float64, f"coords dtype mismatch for {uuid}: {coords.dtype}"
                assert coords.shape == (N, 3), f"coords shape mismatch for {uuid}: {coords.shape} vs ({N}, 3)"

                assert "hessian" in grp, f"hessian missing for {uuid}"
                hessian = grp["hessian"]
                assert hessian.dtype == np.float64, f"hessian dtype mismatch for {uuid}: {hessian.dtype}"
                assert hessian.shape == (3 * N, 3 * N), (
                    f"hessian shape mismatch for {uuid}: {hessian.shape} vs ({3 * N}, {3 * N})")
