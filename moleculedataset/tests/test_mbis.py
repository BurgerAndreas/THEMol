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
    return os.path.join(base_dir, "MBIS")


def test_mbis_dataset_structure(datafolder):
    """
    Test the structure of the final HDF5 files in the datafolder.
    
    Args:
        datafolder (str): Path to the folder containing mbis_dataset.csv and hdf5 files.
    """
    if not datafolder:
        pytest.skip("No datafolder provided.")

    dataset_csv = os.path.join(datafolder, "mbis_dataset.csv")
    if not os.path.exists(dataset_csv):
        pytest.fail(f"mbis_dataset.csv not found in {datafolder}")

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False, na_filter=False)
    if len(df) == 0:
        pytest.fail(f"{dataset_csv} is empty")

    # Check if required columns exist
    required_columns = ["uuid", "mapped_nonisomeric_smiles", "mapped_isomeric_smiles", "h5_file"]
    for col in required_columns:
        if col not in df.columns:
            pytest.fail(f"Column '{col}' not found in {dataset_csv}. Columns: {df.columns}")

    # Group by h5_file and sample a subset of rows from each shard.
    grouped = df.groupby('h5_file')

    for h5_file, group in grouped:
        h5_path = os.path.join(datafolder, h5_file)
        if not os.path.exists(h5_path):
            pytest.fail(f"HDF5 file {h5_path} not found.")

        with h5py.File(h5_path, 'r') as f:
            sample_size = min(100, len(group))
            sampled_rows = group.sample(n=sample_size, random_state=0)

            for row in sampled_rows.itertuples(index=False):
                uuid = row.uuid
                if uuid not in f:
                    pytest.fail(f"UUID {uuid} not found in {h5_path}")

                grp = f[uuid]

                # Check datasets and shapes/dtypes

                # 1. mapped_nonisomeric_smiles
                assert 'mapped_nonisomeric_smiles' in grp, f"mapped_nonisomeric_smiles missing for {uuid}"
                dset = grp['mapped_nonisomeric_smiles']
                # h5py string dtype check
                string_info = h5py.check_string_dtype(dset.dtype)
                assert string_info is not None, f"mapped_nonisomeric_smiles not string for {uuid}"
                assert string_info.encoding == "utf-8", f"mapped_nonisomeric_smiles not utf-8 for {uuid}"
                assert _read_h5_string(dset) == row.mapped_nonisomeric_smiles, (
                    f"mapped_nonisomeric_smiles mismatch for {uuid}")

                # 2. mapped_isomeric_smiles
                assert 'mapped_isomeric_smiles' in grp, f"mapped_isomeric_smiles missing for {uuid}"
                dset = grp['mapped_isomeric_smiles']
                string_info = h5py.check_string_dtype(dset.dtype)
                assert string_info is not None, f"mapped_isomeric_smiles not string for {uuid}"
                assert string_info.encoding == "utf-8", f"mapped_isomeric_smiles not utf-8 for {uuid}"
                smiles_str = _read_h5_string(dset)
                assert smiles_str == row.mapped_isomeric_smiles, f"mapped_isomeric_smiles mismatch for {uuid}"

                # 3. atomic_numbers: (N, 1) int32
                assert 'atomic_numbers' in grp, f"atomic_numbers missing for {uuid}"
                atnums = grp['atomic_numbers']
                assert atnums.dtype == np.int32, f"atomic_numbers dtype mismatch for {uuid}: {atnums.dtype}"
                assert atnums.ndim == 2 and atnums.shape[
                    1] == 1, f"atomic_numbers shape mismatch for {uuid}: {atnums.shape}"
                N = atnums.shape[0]

                expected_uuid = deterministic_uuid(smiles_str)
                assert uuid == expected_uuid, f"UUID mismatch for {uuid}: expected {expected_uuid}, got {uuid}"

                # 4. coords: (N, 3) float64
                assert 'coords' in grp, f"coords missing for {uuid}"
                coords = grp['coords']
                assert coords.dtype == np.float64, f"coords dtype mismatch for {uuid}: {coords.dtype}"
                assert coords.shape == (N, 3), f"coords shape mismatch for {uuid}: {coords.shape} vs ({N}, 3)"

                # 5. mbis_info/atomic_volumes: (N, 1) float64
                assert 'mbis_info' in grp, f"mbis_info group missing for {uuid}"
                mbis_grp = grp['mbis_info']

                assert 'atomic_volumes' in mbis_grp, f"atomic_volumes missing for {uuid}"
                vols = mbis_grp['atomic_volumes']
                assert vols.dtype == np.float64, f"atomic_volumes dtype mismatch for {uuid}: {vols.dtype}"
                assert vols.shape == (N, 1), f"atomic_volumes shape mismatch for {uuid}: {vols.shape} vs ({N}, 1)"

                # 6. mbis_info/atomic_charge: (N, 1) float64
                assert 'atomic_charge' in mbis_grp, f"atomic_charge missing for {uuid}"
                charges = mbis_grp['atomic_charge']
                assert charges.dtype == np.float64, f"atomic_charge dtype mismatch for {uuid}: {charges.dtype}"
                assert charges.shape == (N, 1), f"atomic_charge shape mismatch for {uuid}: {charges.shape} vs ({N}, 1)"

                # 7. mbis_info/atomic_dipole: (N, 3) float64
                assert 'atomic_dipole' in mbis_grp, f"atomic_dipole missing for {uuid}"
                dipoles = mbis_grp['atomic_dipole']
                assert dipoles.dtype == np.float64, f"atomic_dipole dtype mismatch for {uuid}: {dipoles.dtype}"
                assert dipoles.shape == (N, 3), f"atomic_dipole shape mismatch for {uuid}: {dipoles.shape} vs ({N}, 3)"

                # 8. mbis_info/atomic_quadrupole: (N, 3, 3) float64
                assert 'atomic_quadrupole' in mbis_grp, f"atomic_quadrupole missing for {uuid}"
                quads = mbis_grp['atomic_quadrupole']
                assert quads.dtype == np.float64, f"atomic_quadrupole dtype mismatch for {uuid}: {quads.dtype}"
                assert quads.shape == (N, 3,
                                       3), f"atomic_quadrupole shape mismatch for {uuid}: {quads.shape} vs ({N}, 3, 3)"

                # 9. parameters: (M, 3) float64
                assert 'parameters' in grp, f"parameters missing for {uuid}"
                params = grp['parameters']
                assert params.dtype == np.float64, f"parameters dtype mismatch for {uuid}: {params.dtype}"
                # M is variable, but second dimension must be 3
                assert params.ndim == 2 and params.shape[1] == 3, \
                    f"parameters shape mismatch for {uuid}: {params.shape}"
