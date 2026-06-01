#!/bin/bash
#SBATCH --job-name=uma-hessian
#SBATCH --account=rrg-aspuru
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/hessians/uma/uma_hessian_%j.out
#SBATCH --error=logs/hessians/uma/uma_hessian_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <hessian_file.h5> [zero_based_sample_numbers...] [-- extra calculate_uma_sample.py args...]" >&2
    echo "Defaults to the README benchmark samples when no sample numbers are supplied." >&2
    exit 2
fi

H5_FILE="$1"
shift

SAMPLE_NUMBERS=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    SAMPLE_NUMBERS+=("$1")
    shift
done

if [[ ${#SAMPLE_NUMBERS[@]} -eq 0 ]]; then
    read -r -a SAMPLE_NUMBERS <<< "${UMA_SAMPLE_NUMBERS:-47783 3860 220 1229 316 35 67 26 33 11}"
fi

cd "${SLURM_SUBMIT_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

UMA_MODEL="${UMA_MODEL:-uma-s-1p2}"
UMA_TASK_NAME="${UMA_TASK_NAME:-omol}"
UMA_DEVICE="${UMA_DEVICE:-cuda}"
UMA_OUTPUT_DIR="${UMA_OUTPUT_DIR:-results/hessians/uma}"
UMA_PYTHON="${UMA_PYTHON:-.venv-uma/bin/python}"

if [[ ! -x "${UMA_PYTHON}" ]]; then
    echo "UMA Python environment not found or not executable: ${UMA_PYTHON}" >&2
    echo "Create it with the README UMA setup commands, or set UMA_PYTHON." >&2
    exit 2
fi

echo "Running on host: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Threads: ${THREADS}"
echo "Input HDF5: ${H5_FILE}"
echo "Sample numbers: ${SAMPLE_NUMBERS[*]}"
echo "Output directory: ${UMA_OUTPUT_DIR}"
echo "UMA model: ${UMA_MODEL}"
echo "UMA task: ${UMA_TASK_NAME}"
echo "UMA device: ${UMA_DEVICE}"
echo "UMA Python: ${UMA_PYTHON}"
echo "HF Hub offline: ${HF_HUB_OFFLINE}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

uv run --no-project --python "${UMA_PYTHON}" python calculate_uma_sample.py \
    "${H5_FILE}" \
    "${SAMPLE_NUMBERS[@]}" \
    --output-dir "${UMA_OUTPUT_DIR}" \
    --model "${UMA_MODEL}" \
    --task-name "${UMA_TASK_NAME}" \
    --device "${UMA_DEVICE}" \
    --overwrite \
    "${EXTRA_ARGS[@]}"
