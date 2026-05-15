# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import pytest

from bytemol.core import Molecule, Topology


def test_topology():
    tol_mol = Molecule("c1ccccc1C", fmt="smiles")

    label_name = "atomic_number"
    bonds = tol_mol.get_bonds()
    labels = tol_mol.atomic_numbers

    topo = Topology(bonds=bonds, node_labels={label_name: labels})
    assert tol_mol.natoms == topo.natoms

    # test node
    for atom_idx in range(topo.natoms):
        node = topo.graph.nodes[atom_idx]
        label = node[label_name]
        assert label == labels[atom_idx]

    # test atom
    topo_atoms = topo.atoms
    assert set(topo_atoms) == set(range(topo.natoms))

    # test bonds & angles & propers
    topo_bonds = topo.bonds
    assert set(topo_bonds) == set(bonds)

    topo_angles = topo.angles
    assert len(topo_angles) == 24  # C(3,2) * 6 on the benzene and C(4,2) on the methyl
    assert (0, 1, 2) in topo_angles

    topo_propers = topo.propers
    assert len(topo_propers) == 30  # (2 * 2) * 6 on the benzene and 3 * 2 on the methyl
    assert (0, 1, 2, 3) in topo_propers

    # test rings
    topo_rings = topo.rings
    assert len(topo_rings) == 1
    assert [0, 1, 2, 3, 4, 5] == topo_rings[0]

    # test nonbonded
    nonbonded14_pairs = topo.nonbonded14_pairs
    nonbondedall_pairs = topo.nonbondedall_pairs
    assert len(nonbonded14_pairs) == 27  # 30 propers - 3 para positions
    assert (0, 3) in nonbonded14_pairs
    assert len(nonbondedall_pairs) == 39  # (8 * 3) with H on the methy and 15 on the benzene
    assert (2, 6) in nonbondedall_pairs


def test_topology_ring():
    mol = Molecule.from_smiles('C1CCC1CC2(CCC3)CC3CC2')
    for ring_size, target in [(4, 1), (5, 2), (6, 3), (7, 4), (8, 4), (9, 4)]:
        topo = Topology(bonds=mol.get_bonds(), max_include_ring=ring_size)
        assert len(topo.rings) == target


def test_topology_dimer():
    dimer_mol = Molecule("c1ccccc1.c1ccccc1", fmt="smiles")
    bonds = dimer_mol.get_bonds()
    topo = Topology(bonds=bonds)
    assert topo.natoms == 12 * 2
    assert len(topo.bonds) == 12 * 2
    assert len(topo.angles) == 18 * 2
    assert len(topo.propers) == 24 * 2
    assert len(topo.nonbonded14_pairs) == 21 * 2
    assert len(topo.nonbondedall_pairs) == 15 * 2 + 12 * 12
    assert len(topo.rings) == 2


@pytest.mark.parametrize('smiles', ['CCCCC', 'C1CC1CC', 'C1CC2CC1CC2', 'C1CCCC2C3C1CC2CC3'])
def test_nonobonded(smiles):
    mol = Molecule.from_smiles(smiles)
    topo = Topology(mol.get_bonds())

    nonbonded12 = topo.nonbonded12_pairs
    nonbonded13 = topo.nonbonded13_pairs
    nonbonded14 = topo.nonbonded14_pairs
    nonbonded15 = topo.nonbonded15_pairs
    nonbondedall = topo.nonbondedall_pairs

    for i in range(mol.natoms - 1):
        for j in range(i + 1, mol.natoms):
            length = nx.shortest_path_length(topo.graph, i, j)
            if length == 1:
                assert (i, j) in nonbonded12
            elif length == 2:
                assert (i, j) in nonbonded13
            elif length == 3:
                assert (i, j) in nonbonded14
            elif length == 4:
                assert (i, j) in nonbonded15
                assert (i, j) in nonbondedall
            else:
                assert (i, j) in nonbondedall
