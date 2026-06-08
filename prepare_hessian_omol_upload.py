#!/usr/bin/env python3
"""Build the Hugging Face upload tree for recalculated THEMol Hessians."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from pyscf.data.elements import ELEMENTS


REPO_ID = "andreasburger/Hessian-OMol"
SOURCE_HESSIAN_FILE = "hessian_0.h5"
SOURCE_DIR = Path("results/hessians")
UPLOAD_DIR = Path("hf_upload/hessian_omol_recalculated")
SAMPLE_RE = re.compile(r"hessian_0_sample_(\d+)_")


@dataclass(frozen=True)
class Method:
    method_id: str
    display_name: str
    category: str
    level_of_theory: str
    local_dir: Path
    pattern: str
    upload_dir: Path


METHODS = [
    Method(
        "g_xtb",
        "g-xTB",
        "semiempirical",
        "g-xTB numerical Hessians",
        SOURCE_DIR / "gxtb",
        "hessian_0_sample_*_gxtb.h5",
        Path("hessians/semiempirical/g_xtb"),
    ),
    Method(
        "gfn2_xtb",
        "GFN2-xTB",
        "semiempirical",
        "GFN2-xTB numerical Hessians",
        SOURCE_DIR / "xtb_gfn2",
        "hessian_0_sample_*_xtb_gfn2.h5",
        Path("hessians/semiempirical/gfn2_xtb"),
    ),
    Method(
        "orca_wb97m_d4_def2_qzvppd",
        "ORCA wB97M-D4/def2-QZVPPD",
        "dft",
        "wB97M-D4/def2-QZVPPD analytic Hessians with RIJCOSX",
        SOURCE_DIR / "orca_wb97m_d4_def2_qzvppd",
        "hessian_0_sample_*_orca_wb97m_d4_def2_qzvppd.h5",
        Path("hessians/orca/wb97m_d4_def2_qzvppd"),
    ),
    Method(
        "orca_wb97m_d4_def2_tzvpd",
        "ORCA wB97M-D4/def2-TZVPD",
        "dft",
        "wB97M-D4/def2-TZVPD analytic Hessians with RIJCOSX",
        SOURCE_DIR / "orca_wb97m_d4_def2_tzvpd",
        "hessian_0_sample_*_orca_wb97m_d4_def2_tzvpd.h5",
        Path("hessians/orca/wb97m_d4_def2_tzvpd"),
    ),
    Method(
        "pyscf_ccsd_t_cbs",
        "PySCF RCCSD(T)/CBS",
        "ab_initio",
        "frozen-core RCCSD(T)/CBS using cc-pVTZ/cc-pVQZ finite-difference Hessians from analytic gradients",
        SOURCE_DIR / "ccsd_t_cbs",
        "hessian_0_sample_*_ccsd_t_cbs.h5",
        Path("hessians/pyscf/ccsd_t_cbs"),
    ),
    Method(
        "pyscf_wb97m_v_def2_tzvpd",
        "PySCF wB97M-V/def2-TZVPD",
        "dft",
        "wB97M-V/def2-TZVPD analytic Hessians",
        SOURCE_DIR / "no_df",
        "hessian_0_sample_*_pyscf.h5",
        Path("hessians/pyscf/wb97m_v_def2_tzvpd"),
    ),
    Method(
        "pyscf_wb97m_v_def2_tzvpd_density_fit",
        "PySCF wB97M-V/def2-TZVPD with density fitting",
        "dft",
        "wB97M-V/def2-TZVPD analytic Hessians with density fitting",
        SOURCE_DIR / "df",
        "hessian_0_sample_*_pyscf_df.h5",
        Path("hessians/pyscf/wb97m_v_def2_tzvpd_density_fit"),
    ),
    Method(
        "uma_s_1p2_omol",
        "UMA-S-1.2 OMol autograd Hessian",
        "machine_learning",
        "UMA-S-1.2 OMol autograd Hessians",
        SOURCE_DIR / "uma",
        "hessian_0_sample_*_uma_s_1p2.h5",
        Path("hessians/ml/uma_s_1p2_omol"),
    ),
]


def read_h5_string(group: h5py.Group, name: str) -> str:
    if name not in group:
        return ""
    value = group[name].asstr()[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scalar_float(group: h5py.Group, name: str) -> str:
    if name not in group:
        return ""
    value = np.asarray(group[name][()])
    if value.shape == ():
        return repr(float(value))
    return ""


def shape_text(group: h5py.Group, name: str) -> str:
    if name not in group:
        return ""
    return "x".join(str(dim) for dim in group[name].shape)


def formula(atomic_numbers: np.ndarray) -> str:
    symbols = [ELEMENTS[int(z)] for z in atomic_numbers.reshape(-1)]
    counts = Counter(symbols)
    ordered = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "") for symbol in ordered)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_from_path(path: Path) -> int:
    match = SAMPLE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse sample id from {path}")
    return int(match.group(1))


def first_group(handle: h5py.File) -> h5py.Group:
    keys = list(handle.keys())
    if len(keys) != 1:
        raise ValueError(f"Expected one molecule group, found {keys}")
    return handle[keys[0]]


def clear_generated_tree() -> None:
    for child in ("hessians", "metadata"):
        path = UPLOAD_DIR / child
        if path.exists():
            shutil.rmtree(path)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def row_for_file(method: Method, source: Path, dest: Path) -> dict[str, Any]:
    sample_id = sample_id_from_path(source)
    with h5py.File(source, "r") as handle:
        group = first_group(handle)
        atomic_numbers = np.asarray(group["atomic_numbers"][()]).reshape(-1)
        atom_count = int(len(atomic_numbers))
        row = {
            "sample_id": sample_id,
            "source_hessian_file": SOURCE_HESSIAN_FILE,
            "sample_key": group.name.strip("/"),
            "method_id": method.method_id,
            "method_display_name": method.display_name,
            "category": method.category,
            "level_of_theory": method.level_of_theory,
            "atom_count": atom_count,
            "formula": formula(atomic_numbers),
            "mapped_nonisomeric_smiles": read_h5_string(group, "mapped_nonisomeric_smiles"),
            "charge": group.attrs.get("charge", ""),
            "spin": group.attrs.get("spin", ""),
            "energy": scalar_float(group, "energy"),
            "hessian_shape": shape_text(group, "hessian"),
            "hessian_4d_shape": shape_text(group, "hessian_4d"),
            "reference_hessian_shape": shape_text(group, "reference_hessian"),
            "file_size_bytes": dest.stat().st_size,
            "sha256": sha256(dest),
            "path": dest.relative_to(UPLOAD_DIR).as_posix(),
            "original_path": source.as_posix(),
        }
    return row


def build_tree() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clear_generated_tree()
    rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for method in METHODS:
        files = sorted(method.local_dir.glob(method.pattern), key=lambda p: sample_id_from_path(p))
        if not files:
            continue
        dest_dir = UPLOAD_DIR / method.upload_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        method_file_rows = []
        for source in files:
            sample_id = sample_id_from_path(source)
            dest = dest_dir / f"hessian_0_sample_{sample_id:05d}.h5"
            shutil.copy2(source, dest)
            row = row_for_file(method, source, dest)
            rows.append(row)
            method_file_rows.append(row)

        atom_counts = [int(row["atom_count"]) for row in method_file_rows]
        method_rows.append(
            {
                "method_id": method.method_id,
                "method_display_name": method.display_name,
                "category": method.category,
                "hessian_count": len(method_file_rows),
                "sample_count": len({int(row["sample_id"]) for row in method_file_rows}),
                "atom_count_min": min(atom_counts),
                "atom_count_max": max(atom_counts),
                "total_size_bytes": sum(int(row["file_size_bytes"]) for row in method_file_rows),
                "directory": method.upload_dir.as_posix(),
            }
        )

    rows.sort(key=lambda row: (row["method_id"], int(row["sample_id"])))
    method_rows.sort(key=lambda row: row["method_id"])
    return rows, method_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(rows: list[dict[str, Any]], method_rows: list[dict[str, Any]]) -> None:
    metadata_dir = UPLOAD_DIR / "metadata"
    write_csv(metadata_dir / "manifest.csv", rows)
    write_csv(metadata_dir / "method_summary.csv", method_rows)

    sample_ids = [int(row["sample_id"]) for row in rows]
    atom_counts = [int(row["atom_count"]) for row in rows]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "source_directory": SOURCE_DIR.as_posix(),
        "included_hdf5_files": len(rows),
        "excluded_files": ["results/hessians/xtb_gfn2/test_sample_47783_xtb_gfn2.h5"],
        "total_size_bytes": sum(int(row["file_size_bytes"]) for row in rows),
        "unique_samples": len(set(sample_ids)),
        "sample_id_min": min(sample_ids),
        "sample_id_max": max(sample_ids),
        "atom_count_min": min(atom_counts),
        "atom_count_max": max(atom_counts),
        "methods": method_rows,
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    total_mb = summary["total_size_bytes"] / 1_000_000
    methods_text = "\n".join(
        f"- `{row['method_id']}`: {row['hessian_count']} files, {row['method_display_name']}"
        for row in method_rows
    )
    readme = f"""---
pretty_name: Hessian-OMol
license: other
tags:
- chemistry
- molecular-hessians
- hdf5
- omol
- quantum-chemistry
---

# Hessian-OMol

This dataset contains recalculated Hessians for selected molecules from the THEMol Hessian data, using the `hessian_0.h5` source shard. The files here are the Hessians calculated so far for the comparison study in `THEMol`; the full original THEMol source HDF5 shards are not included in this upload.

## Contents

- HDF5 Hessian files: {summary['included_hdf5_files']}
- Unique source samples: {summary['unique_samples']}
- Atom count range: {summary['atom_count_min']} to {summary['atom_count_max']} atoms
- Total uploaded HDF5 size: {total_mb:.2f} MB
- Manifest: `metadata/manifest.csv`
- Method summary: `metadata/method_summary.csv`

## File Layout

Files are organized by calculation engine and method:

```text
hessians/
  ml/uma_s_1p2_omol/
  orca/wb97m_d4_def2_qzvppd/
  orca/wb97m_d4_def2_tzvpd/
  pyscf/ccsd_t_cbs/
  pyscf/wb97m_v_def2_tzvpd/
  pyscf/wb97m_v_def2_tzvpd_density_fit/
  semiempirical/g_xtb/
  semiempirical/gfn2_xtb/
metadata/
  manifest.csv
  method_summary.csv
  summary.json
```

Each Hessian file is named `hessian_0_sample_<zero-padded sample id>.h5`. The `path` and `original_path` columns in `metadata/manifest.csv` map uploaded names back to the local calculation outputs.

## HDF5 Schema

Each `.h5` file contains one molecule group keyed by the original THEMol sample key. Within that group, the common datasets are:

- `atomic_numbers`: atomic numbers, shape `(N, 1)`
- `coords`: coordinates, shape `(N, 3)`
- `energy`: scalar energy from the method in that file
- `force`: force array, shape `(N, 3)`
- `gradient`: gradient array, shape `(N, 3)`
- `hessian`: flattened Cartesian Hessian, shape `(3N, 3N)`
- `hessian_4d`: Hessian as atom/axis blocks, shape `(N, N, 3, 3)` when present
- `reference_hessian`: source THEMol Hessian copied from `hessian_0.h5` when present
- `metadata_json`: calculation metadata, including source sample id and method details

## Methods

{methods_text}

## Notes

The calculations were run on samples from `data/hessians/hessian_0.h5` in the local THEMol workflow. The original source THEMol shard is referenced in metadata but is not uploaded here.
"""
    (UPLOAD_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    rows, method_rows = build_tree()
    write_metadata(rows, method_rows)
    (UPLOAD_DIR / ".generated_by_hessian_omol_upload").write_text(
        f"Generated by {Path(__file__).name} at {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    print(f"Staged {len(rows)} HDF5 files in {UPLOAD_DIR}")
    for row in method_rows:
        print(f"{row['method_id']}: {row['hessian_count']} files")


if __name__ == "__main__":
    main()
