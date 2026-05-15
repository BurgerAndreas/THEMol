# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from operator import itemgetter

import pytest

from bytemol.core import Molecule, MoleculeGraph
from bytemol.core.moleculegraph import Atom, Bond, Hybridization

tol_mol = Molecule("Cc1ccccc1", fmt="smiles")
tol_graph = MoleculeGraph(tol_mol)


class TestAtom:

    def test_idx_init(self):
        atom = Atom(idx=10)
        assert atom.idx == 10
        assert str(atom) == "[]"

    def test_atom_prop(self):
        atom = tol_mol.rkmol.GetAtomWithIdx(1)
        atom = Atom(atom)
        assert atom.idx == 1
        assert atom.aromatic is True
        assert atom.atomic_number == 6
        assert atom.hydrogen_count == 0
        assert atom.hybrdization == Hybridization.SP2

        atom = tol_graph.get_atom(1)
        assert atom.connectivity == 3
        assert atom.ring_connectivity == 2
        assert atom.min_ring_size == 6
        assert str(atom) == "[#6X3]"
        assert atom.hybrdization == Hybridization.SP2

        atom = tol_graph.get_atom(0)
        assert atom.hybrdization == Hybridization.SP3

    def test_get_copy(self):
        atom = tol_graph.get_atom(4)
        catom = atom.get_copy()
        assert atom is not catom
        assert atom.idx == catom.idx
        assert atom.connectivity == catom.connectivity
        assert atom.ring_connectivity == catom.ring_connectivity
        assert atom.min_ring_size == catom.min_ring_size
        assert atom.atomic_number == catom.atomic_number
        assert atom.aromatic == catom.aromatic


class TestBond:

    def test_idx_init(self):
        bond = Bond(idx=3)
        assert bond.idx == 3
        assert str(bond) == "~"

    def test_bond_prop(self):
        bond = tol_mol.rkmol.GetBondWithIdx(2)
        bond = Bond(bond)
        assert bond.idx == 2
        assert bond.order == 1.5
        assert bond.is_conj is True
        assert bond.begin_idx == 2
        assert bond.end_idx == 3
        assert str(bond) == "~"

        bond = tol_graph.get_bond_by_idx(2)
        assert bond.in_ring is True

    def test_get_copy(self):
        bond = tol_graph.get_bond_by_idx(2)
        cbond = bond.get_copy()
        assert bond is not cbond
        assert cbond.idx == cbond.idx
        assert bond.order == cbond.order
        assert bond.is_conj == cbond.is_conj
        assert bond.begin_idx == cbond.begin_idx
        assert bond.end_idx == cbond.end_idx
        assert bond.in_ring == cbond.in_ring


class TestMoleculeGraph:

    def test_get_angles(self):
        ethylene_mol = Molecule("C=C", fmt="smiles")
        ethylene_graph = MoleculeGraph(ethylene_mol)
        angles = ethylene_graph.get_angles()
        assert angles == [(0, 1, 4), (0, 1, 5), (1, 0, 2), (1, 0, 3), (2, 0, 3), (4, 1, 5)]

    def test_get_propers(self):
        ethylene_mol = Molecule("C=C", fmt="smiles")
        ethylene_graph = MoleculeGraph(ethylene_mol)
        propers = ethylene_graph.get_propers()
        assert propers == [(2, 0, 1, 4), (2, 0, 1, 5), (3, 0, 1, 4), (3, 0, 1, 5)]

    def test_get_impropers(self):
        graph = MoleculeGraph(tol_mol)
        impropers = graph.get_impropers()
        assert impropers == [(1, 0, 2, 6), (2, 1, 3, 10), (3, 2, 4, 11), (4, 3, 5, 12), (5, 4, 6, 13), (6, 1, 5, 14)]

    def test_get_nonbonded_pairs(self):
        propylene_mol = Molecule("CC=C", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        nb14, nball = propylene_graph.get_nonbonded_pairs()
        assert nb14 == [(0, 7), (0, 8), (2, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 6), (6, 7), (6, 8)]
        assert nball == [(3, 7), (3, 8), (4, 7), (4, 8), (5, 7), (5, 8)]

        propylene_mol = Molecule("C1C=C1", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        nb14, nball = propylene_graph.get_nonbonded_pairs()
        assert nb14 == [(3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
        assert nball == []

    def test_get_nonring_dihedral_rotate_atoms(self):
        propylene_mol = Molecule("CC=C", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        scanned_list = propylene_graph.get_nonring_dihedral_rotate_atoms(2, 1, 0, 3)
        assert scanned_list == [3, 4, 5]

    def test_num_atoms_bonds(self):
        assert tol_graph.natoms == tol_mol.natoms
        assert tol_graph.nbonds == len(tol_mol.get_bonds())

    def test_get_neighbor(self):
        atoms = tol_graph.get_neighbors(tol_graph.get_atom(1))
        assert tol_graph.get_atom(0) in atoms
        assert tol_graph.get_atom(2) in atoms
        assert tol_graph.get_atom(6) in atoms

    def test_get_intra_topo(self):
        topo = tol_graph.get_intra_topo()
        target_bonds = tol_mol.get_bonds()
        target_bonds.sort(key=itemgetter(0, 1))
        assert topo['Bond'] == target_bonds
        assert topo['Angle'] == tol_graph.get_angles()
        assert topo['ProperTorsion'] == tol_graph.get_propers()
        assert topo['ImproperTorsion'] == tol_graph.get_impropers()

    def test_get_rings(self):
        mol = Molecule("C1CC23CCC14C2.C34", fmt="smiles")
        graph = MoleculeGraph(mol)
        rings = graph.get_rings()
        rings = [set(r) for r in rings]
        assert len(rings) == 6
        assert {0, 1, 2, 5, 7} in rings
        assert {0, 1, 2, 5, 6} in rings
        assert {2, 3, 4, 5, 7} in rings
        assert {2, 5, 6, 7} in rings
        assert {2, 3, 4, 5, 6} in rings
        assert {0, 1, 2, 3, 4, 5} in rings

    def test_get_linear_propers(self):
        propyne_mol = Molecule("C#CC", fmt="smiles")
        propyne_graph = MoleculeGraph(propyne_mol)
        linear_propyne = propyne_graph.get_linear_propers()
        diazo_mol = Molecule("c1ccccc1[N+]#N", fmt="smiles")
        diazo_graph = MoleculeGraph(diazo_mol)
        linear_diazo = diazo_graph.get_linear_propers()
        assert linear_propyne == [(0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6), (2, 1, 0, 3)]
        assert linear_diazo == [(0, 5, 6, 7), (4, 5, 6, 7)]

    def test_get_proper_from_bonds(self):
        propylene_mol = Molecule("CC=C", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        torsions = propylene_graph.get_proper_from_bonds([(0, 1)])
        assert torsions == [(2, 1, 0, 3)]

    def test_get_nonring_rotatable_bonds(self):
        propylene_mol = Molecule("CC=C", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        bonds = propylene_graph.get_nonring_rotatable_bonds()
        assert bonds == [(0, 1), (1, 2)]

    def test_get_nonring_rotatable_propers(self):
        propylene_mol = Molecule("CC=C", fmt="smiles")
        propylene_graph = MoleculeGraph(propylene_mol)
        torsions = propylene_graph.get_nonring_rotatable_propers()
        assert torsions == [(0, 1, 2, 7), (2, 1, 0, 3)]

    def test_get_ring_rotatable_bonds(self):
        mol = Molecule("CC1C=CCC1", fmt="smiles")
        molgraph = MoleculeGraph(mol)
        bonds = molgraph.get_ring_rotatable_bonds()
        assert bonds == [(1, 2), (1, 5), (2, 3), (3, 4), (4, 5)]

    def test_get_ring_rotatable_propers(self):
        mol = Molecule("CC1C=CCC1", fmt="smiles")
        molgraph = MoleculeGraph(mol)
        torsions = molgraph.get_ring_rotatable_propers()
        assert torsions == [(1, 2, 3, 4), (1, 5, 4, 3), (2, 1, 5, 4), (2, 3, 4, 5), (3, 2, 1, 5)]

    def test_get_rotatable_bonds(self):
        mol = Molecule("CC1C=CCC1", fmt="smiles")
        molgraph = MoleculeGraph(mol)
        bonds = molgraph.get_rotatable_bonds()
        assert bonds == [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5)]

    def test_get_rotatable_propers(self):
        mol = Molecule("CC1C=CCC1", fmt="smiles")
        molgraph = MoleculeGraph(mol)
        torsions = molgraph.get_rotatable_propers()
        assert torsions == [(1, 2, 3, 4), (1, 5, 4, 3), (2, 1, 5, 4), (2, 3, 4, 5), (3, 2, 1, 5), (5, 1, 0, 6)]

    def test_multi_mols(self):
        mol = Molecule("C.C", fmt="smiles")
        with pytest.raises(AssertionError):
            MoleculeGraph(mol)

    def test_get_aromatic_rings(self):
        smiles = "C1CCCC1-C1=CC=C1-c1ccccc1-c1c[nH]cc1-c1ccc2occc2c1-c1cc2cccccc2c1"
        mol = Molecule.from_smiles(smiles)
        molgraph = MoleculeGraph(mol)
        rkmol = mol.get_rkmol()

        aromatic_rings = molgraph.get_aromatic_rings()
        assert len(aromatic_rings) == 6
        for ring in aromatic_rings:
            for i in range(len(ring)):
                a0, a1 = ring[i - 1], ring[i]
                if sorted((a0, a1)) != [31, 37]:  # center bond of Azulene
                    bond = rkmol.GetBondBetweenAtoms(a0, a1)
                    assert bond.GetIsAromatic()

        aromatic_rings = molgraph.get_aromatic_rings(max_size=6)
        assert len(aromatic_rings) == 5

    def test_monoatomic_molecule(self):
        mol = Molecule.from_smiles('[Li+]')
        graph = MoleculeGraph(mol)
        assert len(graph.get_neighbors(graph.get_atom(0))) == 0
        assert len(graph.get_bonds()) == 0
