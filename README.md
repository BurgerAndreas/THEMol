# Benchmarking THEMol Hessians

The goal of this repo is to recalculate some of the .h5 Hessians here from the THEMol dataset (B3LYP-D3BJ/dzvp) at different levels of theory and compare. 

As "ground truth" we use analytic Hessians at ωB97M-V/def2-TZVPD level of theory from PySCF. Additionally we compare ORCA, UMA autograd Hessians, and GFN2-xTB and g-xTB numerical Hessians.

## PySCF wB97
```bash
for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch calculate_pyscf_sample.sbatch hessian_0.h5 "$s" "results/no_df/hessian_0_sample_${s}_pyscf.h5"; done

for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch calculate_pyscf_sample.sbatch hessian_0.h5 "$s" "results/df/hessian_0_sample_${s}_pyscf_df.h5" --density-fit; done
```

We used samples from `hessian_0.h5` from the THEMol `Hessian` dataset.

Level of theory for the PySCF recalculations: `ωB97M-V/def2-TZVPD`
- Functional: `wb97m-v`
- Basis: `def2-tzvpd`
- Grid level: `4`
- NLC grid level: `3`
- SCF tolerance: `1e-10`


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


## ORCA wB97
```bash
./submit_orca_readme_samples.sh hessian_0.h5 results/orca_wb97m_d4_def2_tzvpd
```

## GFN2-xTB

```bash
for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch --time=01:00:00 calculate_xtb_sample.sbatch hessian_0.h5 "$s" "results/xtb_gfn2/hessian_0_sample_${s}_xtb_gfn2.h5"; done
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
G_XTB_COMMAND="$HOME/software/gxtb/xtb-6.7.1/bin/xtb" sbatch calculate_gxtb_readme_samples.sbatch hessian_0.h5 results/gxtb
```

## UMA

UMA Hessians are computed with FAIRChem's inference-time autograd Hessian support for the `omol` task. 
We use a separate UMA virtual environment so the PySCF, ORCA, and xTB environment can use an older `numpy` and `h5py` versions.

```bash
uv venv .venv-uma --python 3.11
uv pip install --python .venv-uma/bin/python -e . fairchem-core huggingface-hub
uv run --no-project --python .venv-uma/bin/python hf auth login
```

```bash
sbatch calculate_uma_sample.sbatch hessian_0.h5
```

If the vmap Hessian path runs out of GPU memory, use the lower-memory loop implementation:

```bash
sbatch calculate_uma_sample.sbatch hessian_0.h5 -- --hessian-loop
```

