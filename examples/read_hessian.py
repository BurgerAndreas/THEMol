# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os

import h5py
import pandas as pd


def read_hessian(data_dir):
    csv_path = os.path.join(data_dir, "hessian_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded Hessian CSV with {len(df)} entries.")

    sample_row = df.iloc[0]
    uuid = sample_row["uuid"]
    h5_filename = sample_row["h5_file"]
    h5_path = os.path.join(data_dir, h5_filename)

    with h5py.File(h5_path, "r") as f:
        if uuid in f:
            grp = f[uuid]
            smiles = grp["mapped_nonisomeric_smiles"][()].decode("utf-8")
            atomic_numbers = grp["atomic_numbers"][:]
            coords = grp["coords"][:]
            hessian = grp["hessian"][:]

            print("--- Example Hessian Data ---")
            print(f"UUID: {uuid}")
            print(f"SMILES: {smiles}")
            print(f"Number of atoms: {len(atomic_numbers)}")
            print(f"Coords shape: {coords.shape}")
            print(f"Hessian shape: {hessian.shape}")
        else:
            print(f"UUID {uuid} not found in {h5_filename}")


if __name__ == "__main__":
    base_dir = os.environ.get("MOLECULE_DATA_DIR", "../data")
    DATA_DIR = os.path.join(base_dir, "Hessian")
    read_hessian(DATA_DIR)
