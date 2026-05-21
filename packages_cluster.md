# ORCA on This Cluster

ORCA is provided through the cluster module system, but it lives under restricted
Compute Canada CVMFS software. If the load or binary access fails with
`Permission denied`, request ORCA restricted-software access for your account or
project.

Available versions found with `module spider orca`:

- `orca/6.0.0`
- `orca/6.0.1`
- `orca/6.1.0`
- `orca/6.1.1`

Recommended load command:

```bash
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.1
```

Alternative stack for `orca/6.1.1`:

```bash
module load StdEnv/2023 gcc/14.3 openmpi/5.0.8 orca/6.1.1
```

Run ORCA using the full path exposed by the module:

```bash
${EBROOTORCA}/orca orca.inp > orca.out
```

The module root for the recommended stack is:

```text
/cvmfs/restricted.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/gcc12/openmpi4/orca/6.1.1
```

# xTB on This Cluster

xTB is also provided through the cluster module system. It is not in the
restricted CVMFS tree, and both tested versions run normally from this account.

Available versions found with `module spider xtb`:

- `xtb/6.6.1`
- `xtb/6.7.1`

Recommended load command:

```bash
module load StdEnv/2023 gcc/12.3 xtb/6.7.1
```

Run xTB normally after loading the module:

```bash
xtb molecule.xyz --gfn 2 --opt
```

The verified executable path for `xtb/6.7.1` is:

```text
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcc12/xtb/6.7.1/bin/xtb
```

The older tested version can be loaded with:

```bash
module load StdEnv/2023 gcc/12.3 xtb/6.6.1
```

Its verified executable path is:

```text
/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcc12/xtb/6.6.1/bin/xtb
```
