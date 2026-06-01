#!/bin/bash
#SBATCH --job-name=gxtb-hessian
#SBATCH --account=rrg-aspuru
#SBATCH --time=23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/hessians/gxtb/gxtb_hessian_%j.out
#SBATCH --error=logs/hessians/gxtb/gxtb_hessian_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 <hessian_file.h5> <zero_based_sample_number> [output.h5] [extra calculate_xtb_sample.py args...]" >&2
    exit 2
fi

if [[ -z "${G_XTB_COMMAND:-}" ]]; then
    echo "Set G_XTB_COMMAND to the modified xtb binary that supports --gxtb." >&2
    echo "The cluster xtb/6.7.1 module does not currently expose g-xTB." >&2
    exit 2
fi

H5_FILE="$1"
SAMPLE_NUMBER="$2"
OUTPUT_FILE="${3:-results/hessians/gxtb/$(basename "${H5_FILE%.h5}")_sample_${SAMPLE_NUMBER}_gxtb.h5}"
EXTRA_ARGS=("${@:4}")

cd "${SLURM_SUBMIT_DIR}"

THREADS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export OMP_STACKSIZE=4G

WORK_DIR="${G_XTB_WORK_DIR:-gxtb_work/hessian_$(basename "${H5_FILE%.h5}")_sample_${SAMPLE_NUMBER}_${SLURM_JOB_ID}}"

echo "Running on host: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "g-xTB command: ${G_XTB_COMMAND}"
echo "g-xTB parallel threads: ${THREADS}"
echo "Input HDF5: ${H5_FILE}"
echo "Sample number: ${SAMPLE_NUMBER}"
echo "Output HDF5: ${OUTPUT_FILE}"
echo "Work dir: ${WORK_DIR}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

uv run --no-project --python .venv/bin/python python calculate_xtb_sample.py \
    "${H5_FILE}" \
    "${SAMPLE_NUMBER}" \
    --output "${OUTPUT_FILE}" \
    --gxtb \
    --parallel "${THREADS}" \
    --xtb-command "${G_XTB_COMMAND}" \
    --work-dir "${WORK_DIR}" \
    --keep-work-dir \
    --overwrite \
    "${EXTRA_ARGS[@]}"
