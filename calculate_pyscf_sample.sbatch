#!/bin/bash
#SBATCH --job-name=pyscf-hessian
#SBATCH --account=rrg-aspuru
#SBATCH --time=23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=logs/hessians/pyscf/pyscf_hessian_%j.out
#SBATCH --error=logs/hessians/pyscf/pyscf_hessian_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 <hessian_file.h5> <zero_based_sample_number> [output.h5] [extra calculate_pyscf_sample.py args...]" >&2
    exit 2
fi

H5_FILE="$1"
SAMPLE_NUMBER="$2"
EXTRA_ARGS=("${@:4}")
OUTPUT_DIR="results/hessians/no_df"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${arg}" == "--density-fit" ]]; then
        OUTPUT_DIR="results/hessians/df"
        break
    fi
done
OUTPUT_SUFFIX="pyscf"
if [[ "${OUTPUT_DIR}" == "results/hessians/df" ]]; then
    OUTPUT_SUFFIX="pyscf_df"
fi
OUTPUT_FILE="${3:-${OUTPUT_DIR}/$(basename "${H5_FILE%.h5}")_sample_${SAMPLE_NUMBER}_${OUTPUT_SUFFIX}.h5}"

cd "${SLURM_SUBMIT_DIR}"

THREADS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

# Use most of the allocated node memory while leaving headroom for the OS and HDF5.
MAX_MEMORY_MB="${PYSCF_MAX_MEMORY_MB:-$(awk '/MemTotal/ {printf "%d", ($2 / 1024) * 0.80}' /proc/meminfo)}"

echo "Running on host: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Threads: ${THREADS}"
echo "PySCF max memory MB: ${MAX_MEMORY_MB}"
echo "Input HDF5: ${H5_FILE}"
echo "Sample number: ${SAMPLE_NUMBER}"
echo "Output HDF5: ${OUTPUT_FILE}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

uv run --no-project --python .venv/bin/python python calculate_pyscf_sample.py \
    "${H5_FILE}" \
    "${SAMPLE_NUMBER}" \
    --output "${OUTPUT_FILE}" \
    --xc wb97m-v \
    --basis def2-tzvpd \
    --grid-level 4 \
    --nlc-grid-level 3 \
    --conv-tol 1e-10 \
    --max-cycle 100 \
    --threads "${THREADS}" \
    --max-memory-mb "${MAX_MEMORY_MB}" \
    --overwrite \
    "${EXTRA_ARGS[@]}"
