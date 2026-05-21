#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Plot scalar comparisons for PySCF Hessian benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from pyscf.data.elements import MASSES


sns.set_theme(context="poster", style="whitegrid", font_scale=0.6)
plt.rcParams["axes.linewidth"] *= 0.2
plt.rcParams["grid.linewidth"] *= 0.5
plt.rcParams["legend.fontsize"] *= 0.7

BOHR2ANG = 0.529177210903
ANG2BOHR = 1.0 / BOHR2ANG
HARTREE_TO_EV = 27.211386245988
KCAL_MOL_TO_EV = 1.0 / 23.060548867
PYSCF_PLOT_FACTORS = {
    "energy": HARTREE_TO_EV,
    "force_rms": HARTREE_TO_EV / BOHR2ANG,
    "hessian_rms": HARTREE_TO_EV / (BOHR2ANG**2),
}
THEMOL_PLOT_FACTORS = {
    "energy": KCAL_MOL_TO_EV,
    "force_rms": KCAL_MOL_TO_EV,
    "hessian_rms": KCAL_MOL_TO_EV,
}


@dataclass(frozen=True)
class ResultRecord:
    sample: int
    path: Path
    key: str
    atoms: int
    energy: float | None
    force_rms: float | None
    hessian_rms: float | None
    reference_energy: float | None
    reference_force_rms: float | None
    reference_hessian_rms: float | None


@dataclass(frozen=True)
class ModeRecord:
    sample: int
    atoms: int
    eigvals: np.ndarray
    eigvecs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make DF/no-DF and no-DF/THEMol scalar scatter plots from PySCF output HDF5 files."
    )
    parser.add_argument("--no-df-dir", default="results/no_df", help="Directory with no-density-fitting outputs.")
    parser.add_argument("--df-dir", default="results/df", help="Directory with density-fitting outputs.")
    parser.add_argument("--output-dir", default="plots/pyscf_comparisons", help="Directory for plots and CSV summary.")
    parser.add_argument("--pattern", default="*.h5", help="Glob pattern for result HDF5 files.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument(
        "--rot-thresh",
        type=float,
        default=1e-6,
        help="Norm threshold for dropping near-zero rotational vectors in the Eckart projector.",
    )
    parser.add_argument(
        "--negative-eigval-threshold",
        type=float,
        default=-1e-8,
        help="Threshold below which Eckart-projected eigenvalues are counted as negative.",
    )
    return parser.parse_args()


def sample_from_path(path: Path) -> int | None:
    match = re.search(r"_sample_(\d+)", path.name)
    return int(match.group(1)) if match else None


def first_group(handle: h5py.File) -> h5py.Group:
    keys = list(handle.keys())
    if len(keys) != 1:
        raise ValueError(f"{handle.filename} should contain exactly one top-level sample group; found {len(keys)}")
    return handle[keys[0]]


def scalar_dataset(group: h5py.Group, name: str) -> float | None:
    if name not in group:
        return None
    data = np.asarray(group[name][()])
    if data.size != 1:
        return None
    return float(data.reshape(-1)[0])


def rms_dataset(group: h5py.Group, name: str) -> float | None:
    if name not in group:
        return None
    data = np.asarray(group[name][()], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(data))))


def inertia_tensor(coords3d: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Adapted from frequency_analysis.py."""
    x, y, z = coords3d.T
    squares = np.sum(coords3d**2 * masses[:, None], axis=0)
    i_xx = squares[1] + squares[2]
    i_yy = squares[0] + squares[2]
    i_zz = squares[0] + squares[1]
    i_xy = -np.sum(masses * x * y)
    i_xz = -np.sum(masses * x * z)
    i_yz = -np.sum(masses * y * z)
    return np.array(((i_xx, i_xy, i_xz), (i_xy, i_yy, i_yz), (i_xz, i_yz, i_zz)))


def get_trans_rot_vectors(cart_coords: np.ndarray, masses: np.ndarray, rot_thresh: float) -> np.ndarray:
    """Mass-weighted translation/rotation vectors adapted from frequency_analysis.py."""
    coords3d = np.reshape(cart_coords, (-1, 3))
    total_mass = masses.sum()
    com = np.sum(coords3d * masses[:, None], axis=0) / total_mass
    coords3d_centered = coords3d - com[None, :]

    _, inertia_vecs = np.linalg.eigh(inertia_tensor(coords3d, masses))
    inertia_vecs = inertia_vecs.T

    masses3d = np.repeat(masses, 3)
    sqrt_masses = np.sqrt(masses3d)
    natoms = len(masses)

    trans_vecs = []
    for vec in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        trans_vec = sqrt_masses * np.tile(vec, natoms)
        trans_vecs.append(trans_vec / np.linalg.norm(trans_vec))

    rot_vecs = np.zeros((3, cart_coords.size))
    for atom_idx in range(masses.size):
        p_vec = inertia_vecs.dot(coords3d_centered[atom_idx])
        for xyz_idx in range(3):
            rot_vecs[0, 3 * atom_idx + xyz_idx] = (
                inertia_vecs[2, xyz_idx] * p_vec[1] - inertia_vecs[1, xyz_idx] * p_vec[2]
            )
            rot_vecs[1, 3 * atom_idx + xyz_idx] = (
                inertia_vecs[2, xyz_idx] * p_vec[0] - inertia_vecs[0, xyz_idx] * p_vec[2]
            )
            rot_vecs[2, 3 * atom_idx + xyz_idx] = (
                inertia_vecs[0, xyz_idx] * p_vec[1] - inertia_vecs[1, xyz_idx] * p_vec[0]
            )
    rot_vecs *= sqrt_masses[None, :]
    rot_vecs = rot_vecs[np.linalg.norm(rot_vecs, axis=1) > rot_thresh]

    tr_vecs = np.concatenate((trans_vecs, rot_vecs), axis=0)
    return np.linalg.qr(tr_vecs.T)[0].T


def get_trans_rot_projector(cart_coords: np.ndarray, masses: np.ndarray, rot_thresh: float) -> np.ndarray:
    tr_vecs = get_trans_rot_vectors(cart_coords, masses=masses, rot_thresh=rot_thresh)
    u_matrix, singular_values, _ = np.linalg.svd(tr_vecs.T)
    return u_matrix[:, singular_values.size :].T


def eckart_projected_modes(
    hessian: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    rot_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted eigensystem of mass-weighted, Eckart-projected Hessian."""
    coords_bohr = np.asarray(coords_angstrom, dtype=np.float64).reshape(-1) * ANG2BOHR
    hessian = np.asarray(hessian, dtype=np.float64)
    masses = np.array([MASSES[int(z)] for z in atomic_numbers], dtype=np.float64)
    masses3d = np.repeat(masses, 3)
    mw_hessian = hessian / np.sqrt(np.outer(masses3d, masses3d))
    projector = get_trans_rot_projector(coords_bohr, masses=masses, rot_thresh=rot_thresh)
    projected_hessian = projector.dot(mw_hessian).dot(projector.T)
    projected_hessian = (projected_hessian + projected_hessian.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(projected_hessian)
    sorted_indices = np.argsort(eigvals)
    return eigvals[sorted_indices], eigvecs[:, sorted_indices]


def read_reference_from_original(group: h5py.Group, result_file: h5py.File) -> tuple[float | None, float | None, float | None]:
    energy = scalar_dataset(group, "reference_energy")
    force_rms = rms_dataset(group, "reference_force")
    hessian_rms = rms_dataset(group, "reference_hessian")
    if energy is not None or force_rms is not None or hessian_rms is not None:
        return energy, force_rms, hessian_rms

    input_h5_file = result_file.attrs.get("input_h5_file")
    sample_key = result_file.attrs.get("sample_key")
    if not input_h5_file or not sample_key:
        return None, None, None

    input_path = Path(str(input_h5_file))
    if not input_path.exists():
        return None, None, None

    with h5py.File(input_path, "r") as original:
        if str(sample_key) not in original:
            return None, None, None
        original_group = original[str(sample_key)]
        energy = scalar_dataset(original_group, "energy")
        if "force" in original_group:
            force_rms = rms_dataset(original_group, "force")
        elif "gradient" in original_group:
            gradient_rms = rms_dataset(original_group, "gradient")
            force_rms = gradient_rms
        else:
            force_rms = None
        hessian_rms = rms_dataset(original_group, "hessian")
    return energy, force_rms, hessian_rms


def read_result(path: Path) -> ResultRecord:
    sample = sample_from_path(path)
    if sample is None:
        raise ValueError(f"Could not parse sample number from {path}")

    with h5py.File(path, "r") as handle:
        group = first_group(handle)
        atomic_numbers = np.asarray(group["atomic_numbers"][()]).reshape(-1)
        reference_energy, reference_force_rms, reference_hessian_rms = read_reference_from_original(group, handle)
        return ResultRecord(
            sample=sample,
            path=path,
            key=group.name.lstrip("/"),
            atoms=int(len(atomic_numbers)),
            energy=scalar_dataset(group, "energy"),
            force_rms=rms_dataset(group, "force"),
            hessian_rms=rms_dataset(group, "hessian"),
            reference_energy=reference_energy,
            reference_force_rms=reference_force_rms,
            reference_hessian_rms=reference_hessian_rms,
        )


def read_records(directory: Path, pattern: str) -> dict[int, ResultRecord]:
    records: dict[int, ResultRecord] = {}
    for path in sorted(directory.glob(pattern)):
        record = read_result(path)
        records[record.sample] = record
    return records


def read_modes(path: Path, hessian_name: str, hessian_factor: float, rot_thresh: float) -> ModeRecord:
    sample = sample_from_path(path)
    if sample is None:
        raise ValueError(f"Could not parse sample number from {path}")

    with h5py.File(path, "r") as handle:
        group = first_group(handle)
        atomic_numbers = np.asarray(group["atomic_numbers"][()]).reshape(-1).astype(int)
        coords = np.asarray(group["coords"][()], dtype=np.float64)
        if hessian_name not in group:
            raise ValueError(f"{hessian_name} not found in {path}")
        hessian = np.asarray(group[hessian_name][()], dtype=np.float64) * hessian_factor
        eigvals, eigvecs = eckart_projected_modes(hessian, coords, atomic_numbers, rot_thresh)
        return ModeRecord(sample=sample, atoms=int(len(atomic_numbers)), eigvals=eigvals, eigvecs=eigvecs)


def read_mode_records(
    directory: Path,
    pattern: str,
    hessian_name: str,
    hessian_factor: float,
    rot_thresh: float,
) -> dict[int, ModeRecord]:
    records: dict[int, ModeRecord] = {}
    for path in sorted(directory.glob(pattern)):
        record = read_modes(path, hessian_name=hessian_name, hessian_factor=hessian_factor, rot_thresh=rot_thresh)
        records[record.sample] = record
    return records


def add_identity_line(ax, xs: list[float], ys: list[float]) -> None:
    lo = min(xs + ys)
    hi = max(xs + ys)
    if math.isclose(lo, hi):
        pad = abs(lo) * 0.01 if lo else 1.0
        lo -= pad
        hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="0.35", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def scatter_plot(
    rows: list[tuple[int, int, float, float]],
    x_label: str,
    y_label: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    if not rows:
        return

    atoms = [row[1] for row in rows]
    xs = [row[2] for row in rows]
    ys = [row[3] for row in rows]

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    points = ax.scatter(xs, ys, c=atoms, s=56, cmap="viridis", edgecolors="black", linewidths=0.5)
    add_identity_line(ax, xs, ys)
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Atom count")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def metric_by_atoms_plot(
    rows: list[dict[str, float | int | str]],
    metric: str,
    y_label: str,
    title: str,
    output: Path,
    dpi: int,
    y_tick_format: str | None = None,
) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    markers = ("o", "^", "s", "D", "P", "X")
    comparisons = list(dict.fromkeys(str(row["comparison"]) for row in rows))
    for idx, comparison in enumerate(comparisons):
        subset = [row for row in rows if row["comparison"] == comparison]
        if not subset:
            continue
        xs = [float(row["atoms"]) for row in subset]
        ys = [float(row[metric]) for row in subset]
        ax.scatter(xs, ys, s=58, marker=markers[idx % len(markers)], label=comparison, linewidths=0.5)
    ax.set_xlabel("Atom count")
    ax.set_ylabel(y_label)
    if y_tick_format is not None:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(y_tick_format))
    atom_counts = sorted({int(row["atoms"]) for row in rows})
    ax.set_xticks(atom_counts)
    ax.set_xticklabels([str(atom_count) for atom_count in atom_counts])
    ax.legend(loc="best", edgecolor="none")
    ax.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def negative_count_plot(rows: list[dict[str, float | int | str]], output: Path, dpi: int) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    all_counts = []
    for comparison, marker in (("ωB97M-V/def2-TZVPD w/wo Dens.Fit", "o"), ("ωB97M-V/def2-TZVPD vs B3LYP-D3BJ/dzvp", "^")):
        subset = [row for row in rows if row["comparison"] == comparison]
        if not subset:
            continue
        xs = [int(row["no_df_negative_eigenvalues"]) for row in subset]
        ys = [int(row["comparison_negative_eigenvalues"]) for row in subset]
        all_counts.extend(xs)
        all_counts.extend(ys)
        ax.scatter(xs, ys, s=58, marker=marker, label=comparison, edgecolors="black", linewidths=0.5)
    if all_counts:
        lo = min(all_counts)
        hi = max(all_counts)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="0.35", label="same count")
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlabel("No-DF negative eigenvalue count")
    ax.set_ylabel("Comparison negative eigenvalue count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def make_pair_rows(
    left: dict[int, ResultRecord],
    right: dict[int, ResultRecord],
    left_attr: str,
    right_attr: str,
    left_factor: float = 1.0,
    right_factor: float = 1.0,
) -> list[tuple[int, int, float, float]]:
    rows = []
    for sample in sorted(set(left) & set(right)):
        left_value = getattr(left[sample], left_attr)
        right_value = getattr(right[sample], right_attr)
        if left_value is not None and right_value is not None:
            rows.append((sample, left[sample].atoms, float(left_value) * left_factor, float(right_value) * right_factor))
    return rows


def make_reference_rows(
    records: dict[int, ResultRecord],
    calculated_attr: str,
    reference_attr: str,
    calculated_factor: float = 1.0,
    reference_factor: float = 1.0,
) -> list[tuple[int, int, float, float]]:
    rows = []
    for sample in sorted(records):
        calculated_value = getattr(records[sample], calculated_attr)
        reference_value = getattr(records[sample], reference_attr)
        if calculated_value is not None and reference_value is not None:
            rows.append(
                (
                    sample,
                    records[sample].atoms,
                    float(calculated_value) * calculated_factor,
                    float(reference_value) * reference_factor,
                )
            )
    return rows


def make_absolute_error_rows(
    left: dict[int, ResultRecord],
    right: dict[int, ResultRecord],
    left_attr: str,
    right_attr: str,
    comparison: str,
    left_factor: float = 1.0,
    right_factor: float = 1.0,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for sample in sorted(set(left) & set(right)):
        left_value = getattr(left[sample], left_attr)
        right_value = getattr(right[sample], right_attr)
        if left_value is not None and right_value is not None:
            rows.append(
                {
                    "sample": sample,
                    "atoms": left[sample].atoms,
                    "comparison": comparison,
                    "absolute_error": abs(float(left_value) * left_factor - float(right_value) * right_factor),
                }
            )
    return rows


def hessian_matrix(record: ResultRecord, name: str, factor: float) -> np.ndarray | None:
    with h5py.File(record.path, "r") as handle:
        group = first_group(handle)
        if name not in group:
            return None
        return np.asarray(group[name][()], dtype=np.float64) * factor


def hessian_mae_rows(
    left: dict[int, ResultRecord],
    right: dict[int, ResultRecord],
    comparison: str,
    left_hessian_name: str,
    right_hessian_name: str,
    left_factor: float,
    right_factor: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for sample in sorted(set(left) & set(right)):
        left_hessian = hessian_matrix(left[sample], left_hessian_name, left_factor)
        right_hessian = hessian_matrix(right[sample], right_hessian_name, right_factor)
        if left_hessian is None or right_hessian is None or left_hessian.shape != right_hessian.shape:
            continue
        rows.append(
            {
                "sample": sample,
                "atoms": left[sample].atoms,
                "comparison": comparison,
                "hessian_mae": float(np.mean(np.abs(left_hessian - right_hessian))),
            }
        )
    return rows


def eckart_summary_rows(
    no_df_modes: dict[int, ModeRecord],
    comparison_modes: dict[int, ModeRecord],
    comparison_label: str,
    negative_threshold: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for sample in sorted(set(no_df_modes) & set(comparison_modes)):
        no_df_record = no_df_modes[sample]
        comparison_record = comparison_modes[sample]
        mode_count = min(no_df_record.eigvals.size, comparison_record.eigvals.size)
        if mode_count == 0:
            continue
        no_df_eigvals = no_df_record.eigvals[:mode_count]
        comparison_eigvals = comparison_record.eigvals[:mode_count]
        first_cosine = float(np.dot(no_df_record.eigvecs[:, 0], comparison_record.eigvecs[:, 0]))
        rows.append(
            {
                "sample": sample,
                "atoms": no_df_record.atoms,
                "comparison": comparison_label,
                "mode_count": mode_count,
                "eigenvalue_mae": float(np.mean(np.abs(no_df_eigvals - comparison_eigvals))),
                "no_df_negative_eigenvalues": int(np.sum(no_df_record.eigvals < negative_threshold)),
                "comparison_negative_eigenvalues": int(np.sum(comparison_record.eigvals < negative_threshold)),
                "negative_eigenvalue_count_delta": int(
                    np.sum(comparison_record.eigvals < negative_threshold)
                    - np.sum(no_df_record.eigvals < negative_threshold)
                ),
                "first_eigenvector_cosine": first_cosine,
                "first_eigenvector_abs_cosine": abs(first_cosine),
            }
        )
    return rows


def write_summary_csv(output: Path, no_df: dict[int, ResultRecord], df: dict[int, ResultRecord]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "atoms",
                "no_df_energy_hartree",
                "df_energy_hartree",
                "themol_energy_kcal_mol",
                "no_df_force_rms_hartree_per_bohr",
                "df_force_rms_hartree_per_bohr",
                "themol_force_rms_kcal_mol_per_angstrom",
                "no_df_hessian_rms_hartree_per_bohr2",
                "df_hessian_rms_hartree_per_bohr2",
                "themol_hessian_rms_kcal_mol_per_angstrom2",
                "no_df_file",
                "df_file",
            ]
        )
        for sample in sorted(set(no_df) | set(df)):
            no_df_record = no_df.get(sample)
            df_record = df.get(sample)
            reference = no_df_record or df_record
            writer.writerow(
                [
                    sample,
                    reference.atoms if reference else "",
                    no_df_record.energy if no_df_record else "",
                    df_record.energy if df_record else "",
                    reference.reference_energy if reference else "",
                    no_df_record.force_rms if no_df_record else "",
                    df_record.force_rms if df_record else "",
                    reference.reference_force_rms if reference else "",
                    no_df_record.hessian_rms if no_df_record else "",
                    df_record.hessian_rms if df_record else "",
                    reference.reference_hessian_rms if reference else "",
                    no_df_record.path if no_df_record else "",
                    df_record.path if df_record else "",
                ]
            )


def write_eckart_summary_csv(output: Path, rows: list[dict[str, float | int | str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "atoms",
                "comparison",
                "mode_count",
                "eigenvalue_mae",
                "no_df_negative_eigenvalues",
                "comparison_negative_eigenvalues",
                "negative_eigenvalue_count_delta",
                "first_eigenvector_cosine",
                "first_eigenvector_abs_cosine",
            ]
        )
        for row in sorted(rows, key=lambda item: (str(item["comparison"]), int(item["sample"]))):
            writer.writerow(
                [
                    row["sample"],
                    row["atoms"],
                    row["comparison"],
                    row["mode_count"],
                    row["eigenvalue_mae"],
                    row["no_df_negative_eigenvalues"],
                    row["comparison_negative_eigenvalues"],
                    row["negative_eigenvalue_count_delta"],
                    row["first_eigenvector_cosine"],
                    row["first_eigenvector_abs_cosine"],
                ]
            )


def main() -> None:
    args = parse_args()
    no_df_dir = Path(args.no_df_dir)
    df_dir = Path(args.df_dir)
    output_dir = Path(args.output_dir)

    no_df = read_records(no_df_dir, args.pattern)
    df = read_records(df_dir, args.pattern)
    if not no_df:
        raise SystemExit(f"No no-DF result files found in {no_df_dir}")
    if not df:
        raise SystemExit(f"No DF result files found in {df_dir}")

    no_df_modes = read_mode_records(
        no_df_dir,
        args.pattern,
        hessian_name="hessian",
        hessian_factor=PYSCF_PLOT_FACTORS["hessian_rms"],
        rot_thresh=args.rot_thresh,
    )
    df_modes = read_mode_records(
        df_dir,
        args.pattern,
        hessian_name="hessian",
        hessian_factor=PYSCF_PLOT_FACTORS["hessian_rms"],
        rot_thresh=args.rot_thresh,
    )
    themol_modes = read_mode_records(
        no_df_dir,
        args.pattern,
        hessian_name="reference_hessian",
        hessian_factor=THEMOL_PLOT_FACTORS["hessian_rms"],
        rot_thresh=args.rot_thresh,
    )

    plot_specs = [
        (
            make_reference_rows(
                no_df,
                "energy",
                "reference_energy",
                PYSCF_PLOT_FACTORS["energy"],
                THEMOL_PLOT_FACTORS["energy"],
            ),
            "No-DF energy (eV)",
            "THEMol energy (eV)",
            "THEMol vs No-DF Energy",
            output_dir / "themol_vs_no_df_energy.png",
        ),
        (
            make_reference_rows(
                no_df,
                "force_rms",
                "reference_force_rms",
                PYSCF_PLOT_FACTORS["force_rms"],
                THEMOL_PLOT_FACTORS["force_rms"],
            ),
            "No-DF force RMS (eV/A)",
            "THEMol force RMS (eV/A)",
            "THEMol vs No-DF Force RMS",
            output_dir / "themol_vs_no_df_force_rms.png",
        ),
        (
            make_reference_rows(
                no_df,
                "hessian_rms",
                "reference_hessian_rms",
                PYSCF_PLOT_FACTORS["hessian_rms"],
                THEMOL_PLOT_FACTORS["hessian_rms"],
            ),
            "No-DF Hessian RMS (eV/A^2)",
            "THEMol Hessian RMS (eV/A^2)",
            "THEMol vs No-DF Hessian RMS",
            output_dir / "themol_vs_no_df_hessian_rms.png",
        ),
    ]

    written = []
    skipped = []
    for rows, x_label, y_label, title, output in plot_specs:
        if rows:
            scatter_plot(rows, x_label, y_label, title, output, args.dpi)
            written.append((output, len(rows)))
        else:
            skipped.append(output.name)

    df_error_plot_specs = [
        ("energy", "Energy MAE (eV)", "Energy MAE", output_dir / "df_vs_no_df_energy.png", "{x:.1e}"),
        (
            "force_rms",
            "Force RMS MAE (eV/A)",
            "Force RMS MAE",
            output_dir / "df_vs_no_df_force_rms.png",
            None,
        ),
    ]
    for metric, y_label, title, output, y_tick_format in df_error_plot_specs:
        rows = make_absolute_error_rows(
            no_df,
            df,
            metric,
            metric,
            "ωB97M-V/def2-TZVPD w/wo Dens.Fit",
            PYSCF_PLOT_FACTORS[metric],
            PYSCF_PLOT_FACTORS[metric],
        )
        if rows:
            metric_by_atoms_plot(rows, "absolute_error", y_label, title, output, args.dpi, y_tick_format)
            written.append((output, len(rows)))
        else:
            skipped.append(output.name)

    hessian_mae_by_atoms_rows = hessian_mae_rows(
        no_df,
        df,
        "ωB97M-V/def2-TZVPD w/wo Dens.Fit",
        "hessian",
        "hessian",
        PYSCF_PLOT_FACTORS["hessian_rms"],
        PYSCF_PLOT_FACTORS["hessian_rms"],
    ) + hessian_mae_rows(
        no_df,
        no_df,
        "ωB97M-V/def2-TZVPD vs B3LYP-D3BJ/dzvp",
        "hessian",
        "reference_hessian",
        PYSCF_PLOT_FACTORS["hessian_rms"],
        THEMOL_PLOT_FACTORS["hessian_rms"],
    )
    if hessian_mae_by_atoms_rows:
        metric_by_atoms_plot(
            hessian_mae_by_atoms_rows,
            "hessian_mae",
            "Hessian MAE (eV/A^2)",
            "Hessian MAE",
            output_dir / "hessian_mae_by_atoms.png",
            args.dpi,
        )
        written.append((output_dir / "hessian_mae_by_atoms.png", len(hessian_mae_by_atoms_rows)))
    else:
        skipped.append("hessian_mae_by_atoms.png")

    eckart_rows = (
        eckart_summary_rows(no_df_modes, df_modes, "ωB97M-V/def2-TZVPD w/wo Dens.Fit", args.negative_eigval_threshold)
        + eckart_summary_rows(no_df_modes, themol_modes, "ωB97M-V/def2-TZVPD vs B3LYP-D3BJ/dzvp", args.negative_eigval_threshold)
    )
    eckart_plot_specs = [
        (
            "eigenvalue_mae",
            "Eckart Hessian eigenvalue MAE (eV/A^2/amu)",
            "Eckart Hessian Eigenvalue MAE",
            output_dir / "eckart_eigenvalue_mae_by_atoms.png",
        ),
        (
            "first_eigenvector_abs_cosine",
            "|cos(first Hessian eigenvector)|",
            "First Eckart Eigenvector Cosine Similarity",
            output_dir / "eckart_first_eigenvector_cosine_by_atoms.png",
        ),
    ]
    for metric, y_label, title, output in eckart_plot_specs:
        if eckart_rows:
            metric_by_atoms_plot(eckart_rows, metric, y_label, title, output, args.dpi)
            written.append((output, len(eckart_rows)))
        else:
            skipped.append(output.name)
    if eckart_rows:
        negative_count_plot(eckart_rows, output_dir / "eckart_negative_eigenvalue_counts.png", args.dpi)
        written.append((output_dir / "eckart_negative_eigenvalue_counts.png", len(eckart_rows)))
    else:
        skipped.append("eckart_negative_eigenvalue_counts.png")

    summary_path = output_dir / "pyscf_comparison_metrics.csv"
    write_summary_csv(summary_path, no_df, df)
    eckart_summary_path = output_dir / "pyscf_eckart_summary_metrics.csv"
    write_eckart_summary_csv(eckart_summary_path, eckart_rows)

    print(f"Read {len(no_df)} no-DF files and {len(df)} DF files.")
    for output, count in written:
        print(f"Wrote {output} ({count} points)")
    for name in skipped:
        print(f"Skipped {name}: no paired data available")
    print(f"Wrote {summary_path}")
    print(f"Wrote {eckart_summary_path}")


if __name__ == "__main__":
    main()
