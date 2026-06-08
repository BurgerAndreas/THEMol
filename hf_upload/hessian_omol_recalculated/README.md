---
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

- HDF5 Hessian files: 201
- Unique source samples: 57
- Atom count range: 3 to 50 atoms
- Total uploaded HDF5 size: 21.02 MB
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

- `g_xtb`: 40 files, g-xTB
- `gfn2_xtb`: 10 files, GFN2-xTB
- `orca_wb97m_d4_def2_qzvppd`: 37 files, ORCA wB97M-D4/def2-QZVPPD
- `orca_wb97m_d4_def2_tzvpd`: 45 files, ORCA wB97M-D4/def2-TZVPD
- `pyscf_ccsd_t_cbs`: 2 files, PySCF RCCSD(T)/CBS
- `pyscf_wb97m_v_def2_tzvpd`: 28 files, PySCF wB97M-V/def2-TZVPD
- `pyscf_wb97m_v_def2_tzvpd_density_fit`: 9 files, PySCF wB97M-V/def2-TZVPD with density fitting
- `uma_s_1p2_omol`: 30 files, UMA-S-1.2 OMol autograd Hessian

## Notes

The calculations were run on samples from `data/hessians/hessian_0.h5` in the local THEMol workflow. The original source THEMol shard is referenced in metadata but is not uploaded here.
