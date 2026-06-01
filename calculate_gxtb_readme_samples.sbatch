#!/bin/bash
#SBATCH --job-name=gxtb-readme-hessians
#SBATCH --account=rrg-aspuru
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/hessians/gxtb/gxtb_readme_hessians_%j.out
#SBATCH --error=logs/hessians/gxtb/gxtb_readme_hessians_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <hessian_file.h5> [output_dir] [extra calculate_xtb_sample.py args...]" >&2
    exit 2
fi

H5_FILE="$1"
OUTPUT_DIR="${2:-results/hessians/gxtb}"
EXTRA_ARGS=("${@:3}")
H5_STEM="$(basename "${H5_FILE%.h5}")"
SAMPLES=(47783 3860 220 1229 316 35 67 26 33 11)

cd "${SLURM_SUBMIT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${G_XTB_COMMAND:-}" ]]; then
    DEFAULT_G_XTB_COMMAND="${HOME}/software/gxtb/xtb-6.7.1/bin/xtb"
    if [[ -x "${DEFAULT_G_XTB_COMMAND}" ]]; then
        G_XTB_COMMAND="${DEFAULT_G_XTB_COMMAND}"
    else
        echo "Set G_XTB_COMMAND to the modified xtb binary that supports --gxtb." >&2
        echo "Expected default binary was not executable: ${DEFAULT_G_XTB_COMMAND}" >&2
        exit 2
    fi
fi

THREADS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export OMP_STACKSIZE=4G

echo "Running on host: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "g-xTB command: ${G_XTB_COMMAND}"
echo "g-xTB parallel threads: ${THREADS}"
echo "Input HDF5: ${H5_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Samples: ${SAMPLES[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

for sample in "${SAMPLES[@]}"; do
    output_file="${OUTPUT_DIR}/${H5_STEM}_sample_${sample}_gxtb.h5"
    work_dir="${G_XTB_WORK_DIR:-gxtb_work}/${H5_STEM}_sample_${sample}_${SLURM_JOB_ID}"

    echo "=== Starting sample ${sample} at $(date -Is) ==="
    echo "Output HDF5: ${output_file}"
    echo "Work dir: ${work_dir}"

    uv run --no-project --python .venv/bin/python python calculate_xtb_sample.py \
        "${H5_FILE}" \
        "${sample}" \
        --output "${output_file}" \
        --gxtb \
        --parallel "${THREADS}" \
        --xtb-command "${G_XTB_COMMAND}" \
        --work-dir "${work_dir}" \
        --keep-work-dir \
        --overwrite \
        "${EXTRA_ARGS[@]}"

    echo "=== Finished sample ${sample} at $(date -Is) ==="
done
