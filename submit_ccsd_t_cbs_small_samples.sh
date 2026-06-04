#!/bin/bash
set -euo pipefail

H5_FILE="${1:-data/hessians/hessian_0.h5}"
OUTPUT_DIR="${2:-results/hessians/ccsd_t_cbs}"
H5_STEM="$(basename "${H5_FILE%.h5}")"
SAMPLES=(47783 3860)

mkdir -p "${OUTPUT_DIR}" logs/hessians/ccsd_t_cbs ccsd_t_cbs_work

for sample in "${SAMPLES[@]}"; do
    sbatch calculate_ccsd_t_cbs_sample.sbatch \
        "${H5_FILE}" \
        "${sample}" \
        "${OUTPUT_DIR}/${H5_STEM}_sample_${sample}_ccsd_t_cbs.h5"
done
