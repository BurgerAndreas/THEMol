# Data Format

## Units
- Coordinates: Å
- Energy: kcal·mol^-1
- Force: kcal·mol^-1·Å^-1
- Hessian: kcal·mol^-1·Å^-2
- Charge: e
- Dipole: e·Å
- Quadrupole: e·Å^2
- Volume: Å^3
- 1/sigma_i: Å^-1

## Datasets Output Format

 The internal structure for each molecule (keyed by `UUID`) is defined below:

### 1. Hessian Subset

**Output CSV Columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `h5_file`

**HDF5 Structure**:
```text
/<uuid>/
  mapped_nonisomeric_smiles  (utf-8 string object)
  mapped_isomeric_smiles     (utf-8 string object)
  atomic_numbers             (N, 1) int32
  coords                     (N, 3) float64
  hessian                    (3N, 3N) float64
```

### 2. Hessian Relax Subset

**Output CSV Columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `num_steps`, `h5_file`

**HDF5 Structure**:
```text
/<uuid>/
  mapped_nonisomeric_smiles  (utf-8 string object)
  mapped_isomeric_smiles     (utf-8 string object)
  atomic_numbers             (N, 1) int32
  step 0/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
  ...
  step k/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
```

### 3. TorsionScan Subset

**Output CSV Columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `torsion_indices`, `h5_file`, `num_constraints`

**HDF5 Structure**:
```text
/<uuid>/
  mapped_nonisomeric_smiles  (utf-8 string object)
  mapped_isomeric_smiles     (utf-8 string object)
  atomic_numbers             (N, 1) int32
  torsion_atom_indices       (4,) int32        # 0-based [i,j,k,l]
  constraint 0/
    energy                   scalar float64
    coords                   (N, 3) float64
    forces                   (N, 3) float64
  constraint 1/
    ...
```

### 4. TorsionScan Relax Subset

**Output CSV Columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `torsion_indices`, `h5_file`, `num_constraints`, `num_total_steps`

**HDF5 Structure**:
```text
/<uuid>/
  mapped_nonisomeric_smiles  (utf-8 string object)
  mapped_isomeric_smiles     (utf-8 string object)
  atomic_numbers             (N, 1) int32
  torsion_atom_indices       (4,) int32        # 0-based [i,j,k,l]
  constraint 0/
    energy                   (M,) float64      # M is the number of steps
    coords                   (M, N, 3) float64
    forces                   (M, N, 3) float64
  constraint 1/
    ...
```

### 5. MBIS Subset

**Output CSV Columns**: `uuid`, `mapped_nonisomeric_smiles`, `mapped_isomeric_smiles`, `h5_file`

**HDF5 Structure**:
```text
/<uuid>/
  mapped_nonisomeric_smiles  (utf-8 string object)
  mapped_isomeric_smiles     (utf-8 string object)
  atomic_numbers             (N, 1) int32
  coords                     (N, 3) float64
  mbis_info/
    atomic_volumes           (N, 1) float64
    atomic_charge            (N, 1) float64
    atomic_dipole            (N, 3) float64
    atomic_quadrupole        (N, 3, 3) float64
  parameters                 (M, 3) float64    # M MBIS Slater functions. Each row contains:
                                               # 1. atom_idx: 0-based index of the parent atom
                                               # 2. N_i: charge population (amplitude) of the Slater function
                                               # 3. 1/sigma_i: inverse width (decay constant) of the function
```
