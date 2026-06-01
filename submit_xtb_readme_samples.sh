#!/bin/bash
set -euo pipefail

H5_FILE="${1:-data/hessians/hessian_0.h5}"
OUTPUT_DIR="${2:-results/hessians/xtb_gfn2}"
H5_STEM="$(basename "${H5_FILE%.h5}")"
SAMPLES=(47783 3860 220 1229 316 35 67 26 33 11)

mkdir -p "${OUTPUT_DIR}"

for sample in "${SAMPLES[@]}"; do
    sbatch calculate_xtb_sample.sbatch \
        "${H5_FILE}" \
        "${sample}" \
        "${OUTPUT_DIR}/${H5_STEM}_sample_${sample}_xtb_gfn2.h5"
done
