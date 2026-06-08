# Benchmarking THEMol Hessians

The goal of this repo is to recalculate some of the .h5 Hessians here from the THEMol dataset (B3LYP-D3BJ/dzvp) at different levels of theory and compare. 

As "ground truth" we use analytic Hessians at ωB97M-D4/def2-QZVPPD level of theory from ORCA. Additionally we compare other DFT methods from ORCA, PySCF, UMA autograd Hessians, and GFN2-xTB and g-xTB numerical Hessians.

All calculations were done on th Trillium cluster of the Digital Research Alliance of Canada.
Each job used a full node with 192 cores, 749G memory, and 2x AMD EPYC 9655 (Zen 5) @ 2.6 GHz CPUs.	

You can find the Hessians here: https://huggingface.co/datasets/andreasburger/Hessian-OMol

## Results

On 10 Hessians with 3-28 atoms:

| Method | H MAE | EigVals MAE | \|cos first EigVec\| |
|---|---:|---:|---:|
| ωB97M-V/def2-TZVPD vs GFN2-xTB | 0.465 | 0.742 | 0.742 |
| ωB97M-V/def2-TZVPD vs g-xTB | 0.402 | 0.542 | 0.774 |
| ωB97M-V/def2-TZVPD vs B3LYP-D3BJ/dzvp | 0.151 | 0.214 | 0.747 |
| ωB97M-V/def2-TZVPD vs wB97M-D4/def2-TZVPD | 0.021 | 0.022 | 0.990 |
| ωB97M-V/def2-TZVPD vs UMA | 0.092 | 0.032 | 0.986 |

For comparison, here are the results we reported for our Hessian Interatomic Potentials (HIP) paper, where the methods were all trained on ωB97x/6-31G(d):

| Model | Hessian Method | Hessian Trained | H MAE ↓ eV/Å² | EigVals λ MAE ↓ eV/Å² | \|cos first EigVec v₁\| ↑ | First EigVal λ₁ ↓ eV/Å² | Time ↓ ms |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaNet (E-F) |  |  | 0.502 | 1.190 | 0.903 | 0.245 | 728.6 |
| LEFTNet-DF (E-F) | AD |  | 1.650 | 2.247 | 0.505 | 1.362 | 331.4 |
| LEFTNet-CF (E-F) | AD |  | 0.364 | 1.011 | 0.947 | 0.130 | 1047.9 |
| EquiformerV2 (E-F) | AD |  | 2.254 | 4.199 | 0.279 | 1.372 | 564.9 |
| AlphaNet |  | ✓ | 0.390 | 0.790 | 0.899 | 0.244 | 747.0 |
| LEFTNet-DF | AD | ✓ | 0.200 | 0.172 | 0.937 | 0.136 | 332.2 |
| LEFTNet-CF | AD | ✓ | 0.153 | 0.226 | 0.951 | 0.127 | 1051.6 |
| EquiformerV2 | AD | ✓ | 0.077 | 0.070 | 0.916 | 0.104 | 562.3 |
| HIP-EquiformerV2 | HIP |  | 0.030 | 0.063 | 0.982 | 0.031 | 38.5 |
| HIP-EquiformerV2* | HIP | ✓ | 0.020 | 0.041 | 0.982 | 0.031 | 31.4 |

It seems UMA-S-1.2 fitted ωB97M-V/def2-TZVPD train split as accurately as HIP fitted the ωB97x/6-31G(d) test split. Let's assume ωB97M-V/def2-TZVPD and ωB97x/6-31G(d) are equally easy to fit, and ignore that UMA was trained on vastly more samples than HIP (500 Mio vs 1.7 Mio). This suggests that one could train HIP on UMA autograd Hessians instead of ωB97M-V/def2-TZVPD Hessians, without being bottlenecked by the UMA accuracy. 



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

Expanded 46-sample ORCA comparison jobs:

```bash
mkdir -p results/hessians/orca_wb97m_d4_def2_tzvpd && for s in 9719 5921 5335 2850 3155 318 245 56 92 77 26 37 31 39 16 25 12 27 9 38 40 0 17 88 81 4 78 19 13 29 2 59 231 294 160 609 798; do sbatch calculate_orca_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/orca_wb97m_d4_def2_tzvpd/hessian_0_sample_${s}_orca_wb97m_d4_def2_tzvpd.h5" --basis def2-TZVPD --scf "TightSCF SlowConv" --scf-max-iter 500; done
```

Level of theory for the ORCA recalculations: `ωB97M-D4/def2-TZVPD` with analytic Hessians and `RIJCOSX`.

| sample | atoms | ORCA |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:44` |
| `3860` | 5 | `00:00:39` |
| `9719` | 5 | `00:00:59` |
| `5921` | 6 | `00:01:19` |
| `5335` | 7 | `00:01:11` |
| `220` | 8 | `00:01:30` |
| `2850` | 8 | `00:01:22` |
| `1229` | 10 | `00:01:30` |
| `3155` | 10 | `00:01:47` |
| `316` | 12 | `00:03:12` |
| `318` | 12 | `00:02:16` |
| `245` | 14 | `00:02:59` |
| `35` | 15 | `00:03:30` |
| `56` | 15 | `00:02:43` |
| `92` | 16 | `00:06:53` |
| `67` | 18 | `00:04:02` |
| `77` | 18 | `00:06:49` |
| `26` | 20 | `00:08:17` after retry |
| `37` | 20 | `00:08:28` |
| `31` | 22 | `00:05:28` |
| `33` | 24 | `00:11:47` |
| `39` | 24 | `00:12:27` |
| `16` | 25 | `00:11:11` |
| `25` | 26 | `00:12:00` |
| `11` | 28 | `00:14:05` |
| `12` | 28 | `00:16:14` |
| `27` | 30 | `00:14:53` |
| `9` | 32 | failed; PROPINT segmentation fault |
| `38` | 32 | `00:23:25` |
| `40` | 34 | `00:25:36` |
| `0` | 35 | `00:25:08` |
| `17` | 36 | `00:34:51` |
| `88` | 36 | `00:34:50` |
| `81` | 38 | `00:36:22` |
| `4` | 40 | `00:30:22` |
| `78` | 40 | `00:26:19` |
| `19` | 42 | `00:40:04` |
| `13` | 44 | `00:46:04` |
| `29` | 44 | `00:35:33` |
| `2` | 45 | `00:50:45` |
| `59` | 46 | `00:49:05` |
| `231` | 48 | `00:51:41` |
| `294` | 48 | `00:58:06` |
| `160` | 49 | `00:39:24` |
| `609` | 50 | `01:09:58` |
| `798` | 50 | `01:16:41` |

## ORCA ωB97M-D4/def2-QZVPPD
```bash
mkdir -p results/hessians/orca_wb97m_d4_def2_qzvppd && for s in 47783 3860 220 1229 316 35 67 26 33 11; do sbatch calculate_orca_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/orca_wb97m_d4_def2_qzvppd/hessian_0_sample_${s}_orca_wb97m_d4_def2_qzvppd.h5" --basis def2-QZVPPD; done
```

Level of theory for the ORCA recalculations: `ωB97M-D4/def2-QZVPPD` with analytic Hessians and `RIJCOSX`.

Additional larger-sample scaling jobs:

```bash
mkdir -p results/hessians/orca_wb97m_d4_def2_qzvppd && for s in 9 17 4 13 231 609; do sbatch calculate_orca_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/orca_wb97m_d4_def2_qzvppd/hessian_0_sample_${s}_orca_wb97m_d4_def2_qzvppd.h5" --basis def2-QZVPPD; done
```

Additional 30-sample ORCA comparison jobs:

```bash
mkdir -p results/hessians/orca_wb97m_d4_def2_qzvppd && for s in 9719 5921 5335 2850 3155 318 245 56 92 77 37 31 39 16 25 12 27 38 40 0 88 81 78 19 29 2 59 294 160 798; do sbatch calculate_orca_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/orca_wb97m_d4_def2_qzvppd/hessian_0_sample_${s}_orca_wb97m_d4_def2_qzvppd.h5" --basis def2-QZVPPD; done
```

| sample | atoms | ORCA |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:54` |
| `3860` | 5 | `00:00:49` |
| `9719` | 5 | `00:01:11` |
| `5921` | 6 | `00:01:50` |
| `5335` | 7 | `00:01:37` |
| `220` | 8 | `00:03:29` |
| `2850` | 8 | `00:02:21` |
| `1229` | 10 | `00:02:57` |
| `3155` | 10 | `00:04:10` |
| `316` | 12 | `00:09:31` |
| `318` | 12 | `00:05:41` |
| `245` | 14 | `00:07:51` |
| `35` | 15 | `00:08:35` |
| `56` | 15 | `00:07:19` |
| `92` | 16 | `00:16:00` |
| `67` | 18 | `00:12:01` |
| `77` | 18 | `00:17:17` |
| `26` | 20 | `00:19:37` |
| `37` | 20 | `00:22:39` |
| `31` | 22 | `00:18:53` |
| `33` | 24 | `01:08:38` |
| `39` | 24 | `00:28:41` |
| `16` | 25 | `00:28:09` |
| `25` | 26 | failed; CP-SCF response non-convergence after retry |
| `11` | 28 | `00:55:28` |
| `12` | 28 | failed; SCF non-convergence after retry |
| `27` | 30 | `00:38:04` |
| `9` | 32 | `00:51:58` |
| `38` | 32 | `01:23:11` |
| `40` | 34 | `01:48:16` |
| `0` | 35 | `01:09:03` |
| `17` | 36 | failed; SCF non-convergence after retry |
| `88` | 36 | `01:45:45` |
| `81` | 38 | `02:43:33` |
| `4` | 40 | `02:00:09` |
| `78` | 40 | `01:34:05` |
| `19` | 42 | failed; CP-SCF response non-convergence after retry |
| `13` | 44 | failed; SCF non-convergence after retry |
| `29` | 44 | `02:18:39` |
| `2` | 45 | failed; SCF non-convergence after retry |
| `59` | 46 | `22:33:54` after retry |
| `231` | 48 | failed; SCF non-convergence after retry |
| `294` | 48 | `06:10:36` |
| `160` | 49 | `02:39:43` |
| `609` | 50 | timeout after retry (`23:30:00`) |
| `798` | 50 | timeout after retry (`23:30:00`) |

## PySCF CCSD(T)/CBS
```bash
./submit_ccsd_t_cbs_small_samples.sh data/hessians/hessian_0.h5 results/hessians/ccsd_t_cbs
```

Level of theory for the PySCF spot checks: frozen-core `RCCSD(T)/CBS` using finite-difference Hessians from analytic gradients. The CBS estimate uses `cc-pVTZ/cc-pVQZ` with the Hartree-Fock contribution from `cc-pVQZ` and the CCSD(T) correlation contribution extrapolated as `X^-3`.

Partial CBS gradients are cached in `ccsd_t_cbs_work/<input-stem>_sample_<sample>`, so timed-out jobs can be resumed by resubmitting the same sample.

| sample | atoms | PySCF CCSD(T)/CBS |
| ---: | ---: | --- |
| `47783` | 3 | `01:58:15` |
| `3860` | 5 | `02:06:45`; resume job completed in `00:00:18` |

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

Additional 30-sample g-xTB comparison jobs:

```bash
mkdir -p results/hessians/gxtb && for s in 9719 5921 5335 2850 3155 318 245 56 92 77 37 31 39 16 25 12 27 38 40 0 88 81 78 19 29 2 59 294 160 798; do sbatch --export=ALL,G_XTB_COMMAND="$HOME/software/gxtb/xtb-6.7.1/bin/xtb" calculate_gxtb_sample.sbatch data/hessians/hessian_0.h5 "$s" "results/hessians/gxtb/hessian_0_sample_${s}_gxtb.h5"; done
```

The full 10-sample g-xTB batch completed in `00:00:40`.

| sample | atoms | g-xTB |
| ---: | ---: | ---: |
| `47783` | 3 | `00:00:13` |
| `3860` | 5 | `00:00:01` |
| `9719` | 5 | `00:00:17` |
| `5921` | 6 | `00:00:16` |
| `5335` | 7 | `00:00:17` |
| `220` | 8 | `00:00:02` |
| `2850` | 8 | `00:00:17` |
| `1229` | 10 | `00:00:03` |
| `3155` | 10 | `00:00:16` |
| `316` | 12 | `00:00:03` |
| `318` | 12 | `00:00:15` |
| `245` | 14 | `00:00:15` |
| `35` | 15 | `00:00:02` |
| `56` | 15 | `00:00:16` |
| `92` | 16 | `00:00:17` |
| `67` | 18 | `00:00:03` |
| `77` | 18 | `00:00:18` |
| `26` | 20 | `00:00:02` |
| `37` | 20 | `00:00:18` |
| `31` | 22 | `00:00:15` |
| `33` | 24 | `00:00:03` |
| `39` | 24 | `00:00:17` |
| `16` | 25 | `00:00:19` |
| `25` | 26 | `00:00:16` |
| `11` | 28 | `00:00:04` |
| `12` | 28 | `00:00:16` |
| `27` | 30 | `00:00:16` |
| `38` | 32 | `00:00:19` |
| `40` | 34 | `00:00:17` |
| `0` | 35 | `00:00:17` |
| `88` | 36 | `00:00:17` |
| `81` | 38 | `00:00:18` |
| `78` | 40 | `00:00:18` |
| `19` | 42 | `00:00:18` |
| `29` | 44 | `00:00:19` |
| `2` | 45 | `00:00:18` |
| `59` | 46 | `00:00:19` |
| `294` | 48 | `00:00:20` |
| `160` | 49 | `00:00:19` |
| `798` | 50 | `00:00:20` |

## UMA (uma_s_1p2)

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
salloc --account=rrg-aspuru --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-node=1

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 uv run --no-project --python .venv-uma/bin/python python calculate_uma_sample.py data/hessians/hessian_0.h5 47783 3860 9719 5921 5335 220 2850 1229 3155 316 318 245 35 56 92 67 77 26 37 31 33 39 16 25 11 12 27 9 38 40 0 17 88 81 4 78 19 13 29 2 59 231 294 160 609 798 --output-dir results/hessians/uma --model uma-s-1p2 --task-name omol --device cuda --overwrite --hessian-loop
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

