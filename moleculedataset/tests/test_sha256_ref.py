import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest


HASH_CHUNK_SIZE = 8192 * 1024
REFERENCE_CSV = Path(__file__).with_name("sha256_ref.csv")
DATASET_DIR_ENV = "MOLECULEDATASET_DATA_DIR"
SKIP_SHA256_ENV = "SKIP_SHA256"
CHECK_DATASET_ENV = "CHECK_DATASET"
SUPPORTED_DATASETS = {
    "Hessian",
    "HessianRelax",
    "MBIS",
    "TorsionScan",
    "TorsionScanRelax",
}

cpu_count = os.cpu_count() or 1
WORKERS = max(1, cpu_count // 2)

def get_file_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_reference_rows():
    if not REFERENCE_CSV.exists():
        pytest.fail(f"Reference file not found: {REFERENCE_CSV}")

    with REFERENCE_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            relative_path = row.get("relative_path")
            sha256_value = row.get("sha256")
            if not relative_path or not sha256_value:
                continue
            rows.append(
                {
                    "relative_path": relative_path,
                    "size": int(row["size"]) if row.get("size") else None,
                    "sha256": sha256_value,
                }
            )
    return rows


def get_target_dataset_dir() -> Path:
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    dataset_dir = os.environ.get(DATASET_DIR_ENV, default_dir)
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.exists():
        pytest.fail(f"Dataset directory does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        pytest.fail(f"Dataset path is not a directory: {dataset_path}")
    return dataset_path


def format_examples(paths):
    return ", ".join(paths[:5])


def should_skip_sha256_check() -> bool:
    return os.environ.get(SKIP_SHA256_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def get_requested_datasets():
    dataset_value = os.environ.get(CHECK_DATASET_ENV, "").strip()
    if not dataset_value:
        return None

    requested_datasets = {dataset.strip() for dataset in dataset_value.split(",") if dataset.strip()}
    unsupported_datasets = sorted(requested_datasets - SUPPORTED_DATASETS)
    if unsupported_datasets:
        pytest.fail(
            f"Unsupported {CHECK_DATASET_ENV} value(s): {unsupported_datasets}. "
            f"Supported values: {sorted(SUPPORTED_DATASETS)}"
        )
    return requested_datasets

def filter_reference_rows(reference_rows, requested_datasets):
    if not requested_datasets:
        return reference_rows

    filtered_rows = [
        row  for row in reference_rows if row["relative_path"].split("/", 1)[0] in requested_datasets
    ]
    return filtered_rows


@pytest.mark.serial
def test_downloaded_dataset_matches_sha256_reference():
    dataset_dir = get_target_dataset_dir()
    reference_rows = load_reference_rows()
    reference_rows = filter_reference_rows(reference_rows, get_requested_datasets())
    skip_sha256_check = should_skip_sha256_check()

    missing_files = []
    size_mismatches = []
    hash_mismatches = []
    hash_tasks = []

    for row in reference_rows:
        relative_path = row["relative_path"]
        target_file = dataset_dir / Path(relative_path)

        if not target_file.exists():
            missing_files.append(relative_path)
            continue
        if not target_file.is_file():
            missing_files.append(relative_path)
            continue

        expected_size = row["size"]
        actual_size = target_file.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            size_mismatches.append(
                f"{relative_path} (expected {expected_size}, got {actual_size})"
            )
            continue

        if not skip_sha256_check:
            hash_tasks.append((relative_path, target_file, row["sha256"]))

    if hash_tasks:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_relative_path = {
                executor.submit(get_file_sha256, target_file): (relative_path, expected_sha256)
                for relative_path, target_file, expected_sha256 in hash_tasks
            }
            for future in as_completed(future_to_relative_path):
                relative_path, expected_sha256 = future_to_relative_path[future]
                actual_sha256 = future.result()
                if actual_sha256 != expected_sha256:
                    hash_mismatches.append(relative_path)

    error_messages = []
    if missing_files:
        error_messages.append(
            f"Missing files: {len(missing_files)}. Examples: {format_examples(missing_files)}"
        )
    if size_mismatches:
        error_messages.append(
            f"Size mismatches: {len(size_mismatches)}. Examples: {format_examples(size_mismatches)}"
        )
    if hash_mismatches:
        error_messages.append(
            f"SHA256 mismatches: {len(hash_mismatches)}. Examples: {format_examples(hash_mismatches)}"
        )
    elif skip_sha256_check:
        error_messages.append(
            f"SHA256 check skipped because {SKIP_SHA256_ENV} is enabled."
        )

    if skip_sha256_check:
        assert not missing_files and not size_mismatches, "\n".join(error_messages)
    else:
        assert not error_messages, "\n".join(error_messages)
