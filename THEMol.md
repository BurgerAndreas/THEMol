# THEMol: Torsion, Hessian, Energy of Molecules

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
</p>

We are extremely delighted to release **THEMol** (Torsion, Hessian, and Energy of Molecules). THEMol is an open-source collection of quantum mechanical properties tailored for organic molecules, providing an unprecedented exploration of the intramolecular potential energy surface of organic molecules with up to 50 heavy atoms. 

This repository provides statistical scripts and reading examples to easily access, validate, and process the THEMol dataset.

## News
[2026/05/15]🔥 We release the THEMol dataset and the corresponding open-source codebase.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Data Download](#data-download)
  - [Validation](#validation)
- [Features](#features)
- [License](#license)
- [Citation](#citation)
- [About ByteDance Seed Team](#about-bytedance-seed-team)

## Introduction

We introduce an open-source collection of quantum mechanical properties tailored for organic molecules, THEMol (Torsion, Hessian, Energy of Molecules). Comprising five tailored subsets and in total over three billion DFT calculations, THEMol provides an unprecedented exploration of the intramolecular potential energy surface of organic molecules, with up to 50 heavy atoms. The chemical space sampling is comprehensive, spanning twelve essential elements and diverse molecular architectures relevant to drug discovery, electrolytes, ionic liquids, and beyond.

The dataset also features exhaustive conformational sampling, including comprehensive in-ring and out-of-ring torsional scans. Furthermore, it contains an extensive library of Hessian matrices, computed at relaxed geometries, to capture critical second-derivative information of the potential energy landscape. Additionally, we supply electron density-derived atomic multipoles computed via the Minimal Basis Iterative Stockholder (MBIS) partition scheme. Organized into five distinct subsets, the data encompasses optimized geometries, relaxation trajectories, and derived molecular properties. We anticipate that this massive and diverse dataset will significantly empower the development of highly accurate and transferable molecular potentials.

The dataset consists of 5 subsets. Detailed information about the data format, units, and structure of each subset can be found in [moleculedataset/README.md](moleculedataset/README.md).

| Subset | Description |
| :--- | :--- |
| Hessian | Optimized molecular geometries and their corresponding Hessian matrices. |
| Hessian Relax | Complete structural relaxation trajectories for the Hessian subset. |
| TorsionScan | Comprehensive in-ring and non-ring torsional scans after constrained optimization. |
| TorsionScan Relax | Complete constrained structural relaxation trajectories for the TorsionScan subset. |
| MBIS | Atomic properties and model parameters derived from the Minimal Basis Iterative Stockholder (MBIS) method at the optimized geometry. |


## Getting started

### Prerequisites
* Python version >= 3.11

### Python Dependencies
All required Python packages are listed in requirements.txt. To install them, run:
```bash
pip install -r requirements.txt
```

### Environment Setup
After cloning the repository and installing dependencies, initialize the project and configure your Python path by running:
```bash
bash after_clone.sh
source set_pythonpath.sh
```

### Data Download

The dataset is hosted on Hugging Face at `ByteDance-Seed/THEMol`. Use
`--local-dir` to choose the target download directory. The examples below use
`data/` under the repository root, which is also the default location used by the
validation tests.

To download the whole dataset, using `data/` as the target directory:
```bash
hf download ByteDance-Seed/THEMol \
  --repo-type dataset \
  --local-dir data
```

To download only a subset, such as `Hessian`, add an `--include` pattern:
```bash
hf download ByteDance-Seed/THEMol \
  --repo-type dataset \
  --include "Hessian/*" \
  --local-dir data
```

### Validation

There are two kinds of validation tests in `moleculedataset/tests`.

To validate a downloaded dataset against the SHA256 reference file in
`moleculedataset/tests/sha256_ref.csv`, set `MOLECULEDATASET_DATA_DIR` to the
directory containing the downloaded data. By default, this checks all dataset
subsets listed in the reference file:
```bash
export MOLECULEDATASET_DATA_DIR=/path/to/your/data_directory
pytest moleculedataset/tests/test_sha256_ref.py
```

To validate only one subset, set `CHECK_DATASET` to the subset name:
```bash
export CHECK_DATASET=Hessian
```

If you only want a faster existence + file size check, you can skip SHA256:
```bash
export SKIP_SHA256=1
```

The other tests, such as `test_hessian.py`, `test_mbis.py`, `test_relax.py`,
`test_torsion.py`, and `test_torsion_relax.py`, validate the self-consistency
of each dataset type. They are useful if you modify the files or add new data in
the same format. These tests check that each dataset CSV and its referenced HDF5
files agree on structure, UUIDs, metadata, and sampled records. Set
`MOLECULE_DATA_DIR` to the dataset root before running them:
```bash
export MOLECULE_DATA_DIR=/path/to/your/data_directory
pytest moleculedataset/tests
```

## Features
- **Reading Examples**: Clean code examples to correctly parse and read the H5 data files. You can find them in the [`examples/`](examples/README.md) directory.
- **Statistical Scripts**: Handy scripts for analyzing and extracting statistical insights from the dataset. You can find them in the [`moleculedataset/stat/`](moleculedataset/stat/README.md) directory.
- **Validation Suite**: `pytest`-based validation to ensure the integrity of your downloaded dataset.

## License
- **Code License**: The code in this repository is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
- **Data License**: The data (H5 files) is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation
If you find THEMol useful for your research and applications, feel free to give us a star ⭐ or cite us using:

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

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
