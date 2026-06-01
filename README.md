# Benchmarking THEMol Hessians

The goal of this repo is to recalculate some of the .h5 Hessians here from the THEMol dataset (B3LYP-D3BJ/dzvp) at different levels of theory and compare. 

As "ground truth" we use analytic Hessians at ωB97M-V/def2-TZVPD level of theory from PySCF. Additionally we compare ORCA, UMA autograd Hessians, and GFN2-xTB and g-xTB numerical Hessians.

All calculations were done on th Trillium cluster of the Digital Research Alliance of Canada.
Each job used a full node with 192 cores, 749G memory, and 2x AMD EPYC 9655 (Zen 5) @ 2.6 GHz CPUs.	


## PySCF ωB97M-V/def2-TZVPD
```bash
for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch calculate_pyscf_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/no_df/hessian_0_sample_${s}_pyscf.h5"; done

for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch calculate_pyscf_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/df/hessian_0_sample_${s}_pyscf_df.h5" --density-fit; done
```

We used samples from `data/hessians/hessian_0.h5` from the THEMol `Hessian` dataset.

Level of theory for the PySCF recalculations: `ωB97M-V/def2-TZVPD`
- Functional: `wb97m-v`
- Basis: `def2-tzvpd`
- Grid level: `4`
- NLC grid level: `3`
- SCF tolerance: `1e-10`
Which is the same level of theory and basis set as OMol25 (they used Orca)

| sample | atoms | no DF | DF |
| ---: | ---: | ---: | ---: |
| `47783` | 3 | `00:02:21` | `00:02:10` |
| `3860` | 5 | `00:02:18` | `00:02:15` |
| `220` | 8 | `00:25:37` | `00:24:50` |
| `1229` | 10 | `00:45:44` | `00:44:55` |
| `316` | 12 | `02:40:25` | `02:40:20` |
| `35` | 15 | `03:05:18` | `03:05:14` |
| `67` | 18 | `05:46:47` | `05:33:15` |
| `26` | 20 | `09:29:39` | `09:27:57` |
| `33` | 24 | `18:36:25` | `18:52:01` |
| `11` | 28 | `>24:00:00` timeout | `>24:00:00` timeout |


```bash
uv run python plot_pyscf_comparisons.py
```


## ORCA ωB97M-D4/def2-TZVPD
```bash
./submit_orca_readme_samples.sh data/hessians/hessian_0.h5 results/hessians/orca_wb97m_d4_def2_tzvpd
```

Level of theory for the ORCA recalculations: `ωB97M-D4/def2-TZVPD` with analytic Hessians and `RIJCOSX`.

| sample | atoms | ORCA |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:44` |
| `3860` | 5 | `00:00:39` |
| `220` | 8 | `00:01:30` |
| `1229` | 10 | `00:01:30` |
| `316` | 12 | `00:03:12` |
| `35` | 15 | `00:03:30` |
| `67` | 18 | `00:04:02` |
| `26` | 20 | `00:04:46` failed; missing `orca.hess` |
| `33` | 24 | `00:11:47` |
| `11` | 28 | `00:14:05` |

## GFN2-xTB

```bash
for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch --time=01:00:00 calculate_xtb_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/xtb_gfn2/hessian_0_sample_${s}_xtb_gfn2.h5"; done
```

| sample | atoms | GFN2-xTB |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:15` |
| `3860` | 5 | `00:00:12` |
| `220` | 8 | `00:00:15` |
| `1229` | 10 | `00:00:12` |
| `316` | 12 | `00:00:16` |
| `35` | 15 | `00:00:15` |
| `67` | 18 | `00:00:12` |
| `26` | 20 | `00:00:15` |
| `33` | 24 | `00:00:16` |
| `11` | 28 | `00:00:18` |

## g-xTB

g-xTB requires the modified xTB binary with `--gxtb` support. On the cluster, use the user-local static Linux x86_64 binary installed at `$HOME/software/gxtb/xtb-6.7.1/bin/xtb`.

Run the 10 Hessian samples inside one 2-hour Slurm allocation:

```bash
G_XTB_COMMAND="$HOME/software/gxtb/xtb-6.7.1/bin/xtb" sbatch calculate_gxtb_readme_samples.sbatch data/hessians/hessian_0.h5 results/hessians/gxtb
```

The full 10-sample g-xTB batch completed in `00:00:40`.

| sample | atoms | g-xTB |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:13` |
| `3860` | 5 | `00:00:01` |
| `220` | 8 | `00:00:02` |
| `1229` | 10 | `00:00:03` |
| `316` | 12 | `00:00:03` |
| `35` | 15 | `00:00:02` |
| `67` | 18 | `00:00:03` |
| `26` | 20 | `00:00:02` |
| `33` | 24 | `00:00:03` |
| `11` | 28 | `00:00:04` |

## UMA

UMA Hessians are computed with FAIRChem's inference-time autograd Hessian support for the `omol` task.
We use a separate UMA virtual environment and pin `numpy`/`h5py` to the same versions used by the PySCF, ORCA, and xTB environment so HDF5 reads keep the same ABI.

```bash
uv venv .venv-uma --python 3.11
uv pip install --python .venv-uma/bin/python "fairchem-core>=2" huggingface-hub rdkit
uv pip install --python .venv-uma/bin/python --reinstall "numpy~=1.26.2" "h5py~=3.10.0"
uv run --no-project --python .venv-uma/bin/python hf auth login
uv run --no-project --python .venv-uma/bin/python python -c "from fairchem.core.calculate.pretrained_mlip import get_reference_energies, pretrained_checkpoint_path_from_name; print(pretrained_checkpoint_path_from_name('uma-s-1p2')); get_reference_energies('uma-s-1p2', 'atom_refs'); get_reference_energies('uma-s-1p2', 'form_elem_refs')"
```

```bash
sbatch calculate_uma_sample.sbatch data/hessians/hessian_0.h5
```

The same 10 README samples can also be run in one already-provisioned UMA environment:

```bash
HF_HUB_OFFLINE=1 uv run --no-project --python .venv-uma/bin/python python calculate_uma_sample.py data/hessians/hessian_0.h5 47783 3860 220 1229 316 35 67 26 33 11 --output-dir results/hessians/uma --model uma-s-1p2 --task-name omol --device cuda --overwrite
```

This writes `results/hessians/uma/hessian_0_sample_*_uma_s_1p2.h5`, which `plot_pyscf_comparisons.py` includes by default via `--uma-dir results/hessians/uma --uma-pattern 'hessian_0_sample_*_uma_s_1p2.h5'`.
The full 10-sample UMA batch completed in `00:00:04.3` after loading the predictor in `00:00:13.5`.

If the vmap Hessian path runs out of GPU memory, use the lower-memory loop implementation:

```bash
sbatch calculate_uma_sample.sbatch data/hessians/hessian_0.h5 -- --hessian-loop
```

| sample | atoms | UMA |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:02.6` |
| `3860` | 5 | `00:00:00.2` |
| `220` | 8 | `00:00:00.1` |
| `1229` | 10 | `00:00:00.1` |
| `316` | 12 | `00:00:00.1` |
| `35` | 15 | `00:00:00.1` |
| `67` | 18 | `00:00:00.1` |
| `26` | 20 | `00:00:00.2` |
| `33` | 24 | `00:00:00.2` |
| `11` | 28 | `00:00:00.2` |

