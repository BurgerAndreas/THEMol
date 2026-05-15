# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os

import h5py
import pandas as pd


def read_hessian_relax(data_dir):
    csv_path = os.path.join(data_dir, "relax_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded Hessian Relax CSV with {len(df)} entries.")

    sample_row = df.iloc[0]
    uuid = sample_row["uuid"]
    h5_filename = sample_row["h5_file"]
    h5_path = os.path.join(data_dir, h5_filename)

    with h5py.File(h5_path, "r") as f:
        if uuid in f:
            grp = f[uuid]
            smiles = grp["mapped_nonisomeric_smiles"][()].decode("utf-8")
            atomic_numbers = grp["atomic_numbers"][:]

            print("--- Example Hessian Relax Data ---")
            print(f"UUID: {uuid}")
            print(f"SMILES: {smiles}")
            print(f"Number of atoms: {len(atomic_numbers)}")

            step_keys = [k for k in grp.keys() if k.startswith("step ")]
            print(f"Found {len(step_keys)} relaxation steps.")

            if step_keys:
                step_grp = grp[step_keys[0]]
                coords = step_grp["coords"][:]
                energy = step_grp["energy"][()]
                forces = step_grp["forces"][:]
                print(
                    f"'{step_keys[0]}' -> Coords shape: {coords.shape}, Forces shape: {forces.shape}, Energy: {energy}")
        else:
            print(f"UUID {uuid} not found in {h5_filename}")


if __name__ == "__main__":
    base_dir = os.environ.get("MOLECULE_DATA_DIR", "../data")
    DATA_DIR = os.path.join(base_dir, "HessianRelax")
    read_hessian_relax(DATA_DIR)
