# THEMol Hessian HDF5 Summary

Scan completed: 2026-05-20T16:39:25Z

Each `hessian_*.h5` file contains many molecule groups. For every molecule checked, the Hessian shape matched `3N x 3N`, where `N` is the atom count.

## Overall

- Source files: `hessian_0.h5` through `hessian_9.h5`
- Total Hessians: 618,947
- Atom count range: 2 to 70 atoms
- Median atom count: 30 atoms
- Hessian dimension range: `6x6` to `210x210`
- Approximate HDF5 size: 45 GB

## Per-File Summary

| File | Hessians | Atom count min / p50 / max | Hessian dimension range | Unique formulas | Most common formulas |
| --- | ---: | ---: | --- | ---: | --- |
| `hessian_0.h5` | 61,796 | 3 / 30 / 70 | `9x9` to `210x210` | 24,705 | `C12H14N2O` (62), `C13H16N2O` (58), `C13H17NO` (56), `C12H15NO` (53), `C13H14N2O` (52), `C11H14N2O` (50), `C10H11N3O` (49), `C12H15NO2` (49), `C13H16N2O2` (48), `C11H15N3O` (47) |
| `hessian_1.h5` | 61,563 | 4 / 30 / 69 | `12x12` to `207x207` | 24,744 | `C12H15NO2` (60), `C13H16N2O2` (57), `C13H16N2O` (54), `C11H13N3O` (53), `C11H14N2O` (50), `C13H15N3O` (49), `C10H18N2O` (49), `C14H16N2O` (48), `C13H15NO2` (47), `C12H15NO` (47) |
| `hessian_2.h5` | 62,020 | 2 / 30 / 70 | `6x6` to `210x210` | 24,791 | `C13H16N2O` (72), `C11H14N2O` (61), `C12H15NO` (56), `C12H15NO2` (54), `C14H16N2O` (52), `C14H17NO` (50), `C13H18N2O` (50), `C13H14N2O2` (50), `C13H16N2O2` (49), `C10H18N2O2` (49) |
| `hessian_3.h5` | 61,510 | 2 / 30 / 70 | `6x6` to `210x210` | 24,706 | `C11H13N3O` (54), `C12H14N2O` (54), `C11H15N3O` (52), `C11H14N2O` (50), `C12H16N2O` (48), `C13H16N2O` (46), `C13H15N3O` (45), `C13H15NO2` (45), `C13H14N2O` (45), `C11H12N2O2` (44) |
| `hessian_4.h5` | 62,090 | 2 / 30 / 70 | `6x6` to `210x210` | 24,905 | `C13H17NO` (58), `C11H13N3O` (57), `C13H16N2O` (54), `C13H15N3O` (53), `C12H15NO2` (51), `C12H14N2O2` (50), `C13H16N2O2` (50), `C9H16N2O` (49), `C12H13N3O` (48), `C12H16N2O` (48) |
| `hessian_5.h5` | 61,731 | 3 / 30 / 70 | `9x9` to `210x210` | 24,727 | `C13H14N2O` (76), `C11H13N3O` (55), `C12H14N2O` (55), `C13H16N2O` (54), `C11H14N2O` (54), `C12H16N2O` (53), `C11H15NO` (51), `C12H15N3O` (50), `C10H15N3O` (47), `C13H18N2O` (46) |
| `hessian_6.h5` | 61,825 | 3 / 30 / 70 | `9x9` to `210x210` | 24,709 | `C12H14N2O` (59), `C11H13N3O` (58), `C12H16N2O` (53), `C13H16N2O` (52), `C13H15NO` (52), `C12H13NO2` (52), `C13H14N2O2` (51), `C14H18N2O` (51), `C13H18N2O` (50), `C12H15NO` (49) |
| `hessian_7.h5` | 61,858 | 3 / 30 / 70 | `9x9` to `210x210` | 24,675 | `C12H14N2O` (62), `C12H16N2O` (60), `C13H16N2O` (59), `C11H15N3O` (56), `C10H17NO2` (54), `C11H14N2O` (51), `C14H16N2O2` (49), `C12H16N2O2` (49), `C9H16N2O2` (48), `C13H15NO2` (47) |
| `hessian_8.h5` | 62,193 | 4 / 30 / 70 | `12x12` to `210x210` | 24,690 | `C10H11N3O` (61), `C12H16N2O` (58), `C11H15N3O` (56), `C13H16N2O` (53), `C14H18N2O` (53), `C13H14N2O` (51), `C11H14N2O` (51), `C14H16N2O` (48), `C8H15NO` (48), `C13H17NO2` (46) |
| `hessian_9.h5` | 62,361 | 3 / 30 / 70 | `9x9` to `210x210` | 24,891 | `C12H16N2O` (56), `C12H14N2O` (55), `C13H16N2O` (53), `C11H15N3O` (52), `C14H16N2O` (50), `C11H14N2O` (50), `C12H15NO2` (50), `C10H14N2O` (47), `C13H15N3O` (47), `C13H17N3O` (46) |

## Reproduce

```bash
uv run --no-project --python .venv/bin/python python summarize_hessians.py
```

The command above assumes a future `summarize_hessians.py` script with the same metadata scan logic. The current summary was generated from the local scan output, not from a checked-in script.
