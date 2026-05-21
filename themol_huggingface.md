---
license: cc-by-nc-4.0
language:
- en
pretty_name: THEMol
size_categories:
- 1B<n<10B
tags:
- chemistry
- molecular-dynamics
- density-functional-theory
- medicine
- electrolytes
configs:
- config_name: hessian
  data_files:
  - split: full
    path: "Hessian/hessian_dataset.csv"
- config_name: hessianrelax
  data_files:
  - split: full
    path: "HessianRelax/relax_dataset.csv"
- config_name: torsionscan
  data_files:
  - split: full
    path: "TorsionScan/torsion_dataset.csv"
- config_name: torsionscanrelax
  data_files:
  - split: full
    path: "TorsionScanRelax/torsion_relax_dataset.csv"
- config_name: mbis
  data_files:
  - split: full
    path: "MBIS/mbis_dataset.csv"
---

# THEMol: Torsion, Hessian, Energy of Molecules

## Dataset Summary

THEMol is an open-source collection of quantum mechanical properties tailored for organic molecules. It provides large-scale density functional theory (DFT) data for exploring intramolecular potential energy surfaces, including optimized geometries, structural relaxation trajectories, torsion scans, constrained torsion relaxation trajectories, Hessian matrices, and MBIS-derived atomic properties.

The dataset contains five tailored subsets and over three billion DFT calculations for molecules with up to 50 non-hydrogen atoms. Its chemical space spans twelve essential elements and diverse molecular architectures relevant to drug discovery, electrolytes, ionic liquids, and broader molecular modeling applications.

See the [paper](https://arxiv.org/abs/2605.14973) and [GitHub repository](https://github.com/ByteDance-Seed/THEMol) for more details. The GitHub repository provides data readers, validation utilities, and statistical scripts.

## Dataset Details

### Dataset Description

- **Repository** [https://github.com/ByteDance-Seed/THEMol](https://github.com/ByteDance-Seed/THEMol)
- **Paper** [https://arxiv.org/abs/2605.14973](https://arxiv.org/abs/2605.14973)
- **Data format** CSV index files plus HDF5 data files.
- **Computation scale** more than three billion DFT calculations across five subsets.
- **Molecular scope** organic molecules with up to 50 non-hydrogen atoms.
- **Element coverage** H, C, N, O, S, F, Cl, Br, Si, B, P, and I.

### Subsets

| Subset | Level of Theory | Entries | Supplementary Metrics | Description |
| :--- | :--- | ---: | :--- | :--- |
| Hessian | B3LYP-D3(BJ)/DZVP | 3,102,537 | - | Optimized molecular geometries and corresponding Hessian matrices. |
| Hessian Relax | B3LYP-D3(BJ)/DZVP | 4,811,722 | 281,123,880 relaxation steps | Complete structural relaxation trajectories for the Hessian subset. |
| TorsionScan | B3LYP-D3(BJ)/DZVP | 4,192,791 | 2,436,985 molecules; 93,994,576 constraints | Comprehensive in-ring and non-ring torsional scans after constrained optimization. |
| TorsionScan Relax | B3LYP-D3(BJ)/DZVP | 4,914,677 | 3,090,560 molecules; 110,235,160 constraints; 2,993,685,868 steps | Complete constrained structural relaxation trajectories for the TorsionScan subset. |
| MBIS | PBE0/def2-TZVPD, or DZVP for I atoms | 3,082,151 | - | Atomic properties and model parameters from Minimal Basis Iterative Stockholder (MBIS) partitioning. |

## Dataset Structure

The dataset is organized by subset. Each subset contains one CSV index file and HDF5 files referenced by the `h5_file` column.

```text
/
├── Hessian/
│   ├── hessian_dataset.csv
│   └── *.h5
├── HessianRelax/
│   ├── relax_dataset.csv
│   └── *.h5
├── TorsionScan/
│   ├── torsion_dataset.csv
│   └── *.h5
├── TorsionScanRelax/
│   ├── torsion_relax_dataset.csv
│   └── *.h5
└── MBIS/
    ├── mbis_dataset.csv
    └── *.h5
```

### Data Fields

All HDF5 files are keyed by `uuid`. The CSV index files contain the UUID, mapped SMILES strings, and the HDF5 file location needed to retrieve each molecular record.

#### Hessian

**CSV columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `h5_file`

**HDF5 structure**:

```text
/<uuid>/
  mapped_nonisomeric_smiles  utf-8 string
  mapped_isomeric_smiles     utf-8 string
  atomic_numbers             (N, 1) int32
  coords                     (N, 3) float64
  hessian                    (3N, 3N) float64
```

#### Hessian Relax

**CSV columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `num_steps`, `h5_file`

**HDF5 structure**:

```text
/<uuid>/
  mapped_nonisomeric_smiles  utf-8 string
  mapped_isomeric_smiles     utf-8 string
  atomic_numbers             (N, 1) int32
  step 0/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
  ...
  step k/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
```

#### TorsionScan

**CSV columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `torsion_indices`, `h5_file`, `num_constraints`

**HDF5 structure**:

```text
/<uuid>/
  mapped_nonisomeric_smiles  utf-8 string
  mapped_isomeric_smiles     utf-8 string
  atomic_numbers             (N, 1) int32
  torsion_atom_indices       (4,) int32
  constraint 0/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
  constraint 1/
    ...
```

#### TorsionScan Relax

**CSV columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `torsion_indices`, `h5_file`, `num_constraints`, `num_total_steps`

**HDF5 structure**:

```text
/<uuid>/
  mapped_nonisomeric_smiles  utf-8 string
  mapped_isomeric_smiles     utf-8 string
  atomic_numbers             (N, 1) int32
  torsion_atom_indices       (4,) int32
  constraint 0/
    energy                   (M,) float64
    coords                   (M, N, 3) float64
    forces                   (M, N, 3) float64
  constraint 1/
    ...
```
Here `M` is the number of steps.

#### MBIS

**CSV columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `h5_file`

**HDF5 structure**:

```text
/<uuid>/
  mapped_nonisomeric_smiles  utf-8 string
  mapped_isomeric_smiles     utf-8 string
  atomic_numbers             (N, 1) int32
  coords                     (N, 3) float64
  mbis_info/
    atomic_volumes           (N, 1) float64
    atomic_charge            (N, 1) float64
    atomic_dipole            (N, 3) float64
    atomic_quadrupole        (N, 3, 3) float64
  parameters                 (M, 3) float64
```
Here `M` is the number of MBIS Slater functions.

For `parameters`, each row contains the 0-based parent atom index, the Slater-function charge population `N_i`, and the inverse width `1/sigma_i`.

### Units

| Quantity | Unit |
| :--- | :--- |
| Coordinates | Å |
| Energy | kcal mol^-1 |
| Force | kcal mol^-1 Å^-1 |
| Hessian | kcal mol^-1 Å^-2 |
| Charge | e |
| Dipole | e Å |
| Quadrupole | e Å^2 |
| Volume | Å^3 |
| 1/sigma_i | Å^-1 |

## Licenses

[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use THEMol in your research or applications, please cite:

```bibtex
@misc{THEMol,
      title={THEMol dataset: Torsion, Hessian, and Energy of Molecules}, 
      author={Jiashu Liang and Tianze Zheng and Yu Xia and Xingyuan Xu and Xu Han and Zhi Wang and Siyuan Liu and Ailun Wang and Yu Liu and Shiqian Tan and Dongfei Liu and Zhichen Pu and Yuanheng Wang and Qiming Sun and Xiaojie Wu and Wen Yan},
      year={2026},
      eprint={2605.14973},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2605.14973}, 
}
```
