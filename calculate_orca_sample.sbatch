#!/bin/bash
#SBATCH --job-name=orca-hessian
#SBATCH --account=rrg-aspuru
#SBATCH --time=23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --output=orca_hessian_%j.out
#SBATCH --error=orca_hessian_%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 <hessian_file.h5> <zero_based_sample_number> [output.h5] [extra calculate_orca_sample.py args...]" >&2
    exit 2
fi

H5_FILE="$1"
SAMPLE_NUMBER="$2"
OUTPUT_FILE="${3:-orca_$(basename "${H5_FILE%.h5}")_sample_${SAMPLE_NUMBER}.h5}"
EXTRA_ARGS=("${@:4}")

cd "${SLURM_SUBMIT_DIR}"

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.1

THREADS="${SLURM_NTASKS:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ORCA %maxcore is per MPI process. Use most of the node while leaving headroom.
MAXCORE_MB="${ORCA_MAXCORE_MB:-$(awk -v n="${THREADS}" '/MemTotal/ {printf "%d", (($2 / 1024) * 0.50) / n}' /proc/meminfo)}"
WORK_DIR="${ORCA_WORK_DIR:-orca_work/hessian_$(basename "${H5_FILE%.h5}")_sample_${SAMPLE_NUMBER}_${SLURM_JOB_ID}}"

echo "Running on host: $(hostname)"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "ORCA root: ${EBROOTORCA}"
echo "ORCA nprocs: ${THREADS}"
echo "ORCA maxcore MB per process: ${MAXCORE_MB}"
echo "Input HDF5: ${H5_FILE}"
echo "Sample number: ${SAMPLE_NUMBER}"
echo "Output HDF5: ${OUTPUT_FILE}"
echo "Work dir: ${WORK_DIR}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

uv run --no-project --python .venv/bin/python python calculate_orca_sample.py \
    "${H5_FILE}" \
    "${SAMPLE_NUMBER}" \
    --output "${OUTPUT_FILE}" \
    --functional WB97M-D4 \
    --basis def2-TZVPD \
    --grid DefGrid3 \
    --scf TightSCF \
    --nprocs "${THREADS}" \
    --maxcore-mb "${MAXCORE_MB}" \
    --orca-command "${EBROOTORCA}/orca" \
    --work-dir "${WORK_DIR}" \
    --keep-work-dir \
    --overwrite \
    "${EXTRA_ARGS[@]}"
