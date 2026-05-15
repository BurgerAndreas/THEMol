# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tempfile

import ase
import ase.io as aio
import numpy as np
import pytest
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from bytemol.core import Conformer, Molecule, assert_good_molecule, read_molecules_from_sdf, read_molecules_from_xyz
from bytemol.units import simple_unit as unit
from bytemol.utils import get_data_file_path, run_command_and_check, temporary_cd

logger = logging.getLogger(__name__)


class TestConformer:
    CO_coords = [[0, 0, 0], [0, 0, 1.1]]
    CO_symbols = ['C', 'O']
    CO_partial_charges = [0.1, -0.1]
    CO_forces = [[0, 0, 0.5], [0, 0, -0.5]]

    def test_conformer_data(self):
        CO = Conformer(self.CO_coords, self.CO_symbols)
        assert CO.natoms == 2
        assert np.array_equal(CO.coords, np.array(self.CO_coords))
        assert CO.symbols == self.CO_symbols

    def test_conformer_to_ase(self):
        CO = Conformer(self.CO_coords, self.CO_symbols)
        CO_atoms = CO.to_ase_atoms()
        assert np.array_equal(CO_atoms.positions, np.array(self.CO_coords))
        assert all(ase_symbol == symbol for ase_symbol, symbol in zip(CO_atoms.symbols, self.CO_symbols))

    def test_conformer_symbols(self):
        mol = Molecule("CC", fmt="smiles", nconfs=1)
        atoms = mol.conformers[0].to_ase_atoms()
        coords = mol.conformers[0].coords
        conf1 = Conformer(coords, mol.atomic_symbols)
        conf2 = Conformer(coords, mol.atomic_numbers)
        conf3 = Conformer(coords, atoms.symbols)
        assert conf1.symbols == conf2.symbols == conf3.symbols == mol.conformers[0].symbols

    def test_from_ase_atoms(self):
        atoms = ase.Atoms(self.CO_symbols,
                          positions=self.CO_coords,
                          info={
                              "one_energy": 1,
                              "two_energy": 2,
                              "energy": 0
                          })
        CO = Conformer.from_ase_atoms(atoms)
        assert np.array_equal(atoms.positions, np.array(self.CO_coords))
        assert np.isclose(CO.confdata['one_energy'], unit.eV_to_kcal_mol(1))
        assert np.isclose(CO.confdata['two_energy'], unit.eV_to_kcal_mol(2))
        assert CO.confdata['energy'] == 0

    def test_to_extxyz(self):
        CO = Conformer(self.CO_coords, self.CO_symbols, confdata={'energy': 1})
        with tempfile.TemporaryDirectory() as tmpdir:
            extxyz_file = os.path.join(tmpdir, "co.extxyz")
            CO.to_extxyz(extxyz_file, append=False)
            CO.to_extxyz(extxyz_file, append=True)
            CO.to_extxyz(extxyz_file, append=True)
            atom_list = aio.read(extxyz_file, index=':')
        assert len(atom_list) == 3
        assert np.array_equal(atom_list[-1].positions, np.array(self.CO_coords))

    def test_confdata_to_extxyz(self):
        confdata = {
            'energy': 1,  # kcal/mol
            'test_energy': 0,
            'partial_charge': self.CO_partial_charges,
            'forces': self.CO_forces
        }
        CO = Conformer(self.CO_coords, self.CO_symbols, confdata=confdata)
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, 'test1.extxyz')
            file2 = os.path.join(tmpdir, 'test2.extxyz')
            file3 = os.path.join(tmpdir, 'test3.extxyz')

            CO.to_xyz(file1)
            atoms1 = aio.read(file1, format='extxyz')
            logger.info(atoms1.calc.results)

            # reserved keywords here https://gitlab.com/ase/ase/-/blob/master/ase/outputs.py?ref_type=heads#L129
            # are  moved from atoms.info to atoms.calc.results
            assert 'energy' in atoms1.calc.results, atoms1.calc.results
            assert np.isclose(atoms1.calc.results['energy'], unit.kcal_mol_to_eV(1))
            assert 'test_energy' in atoms1.info
            assert atoms1.info['test_energy'] == 0
            assert "partial_charge" in atoms1.arrays
            assert np.array_equal(atoms1.arrays['partial_charge'], np.array(self.CO_partial_charges))
            assert "forces" in atoms1.calc.results
            assert np.allclose(atoms1.calc.results['forces'], unit.kcal_mol_A_to_eV_A(np.array(self.CO_forces)))

            CO.to_xyz(file2, confkeys=['energy', 'partial_charge'])
            atoms2 = aio.read(file2)
            assert 'energy' in atoms2.calc.results, atoms2.calc.results
            assert np.isclose(atoms2.calc.results['energy'], unit.kcal_mol_to_eV(1))
            assert "partial_charge" in atoms2.arrays
            assert np.array_equal(atoms2.arrays['partial_charge'], np.array(self.CO_partial_charges))

            CO.to_xyz(file3, confkeys=[])
            atoms3 = aio.read(file3)
            assert not atoms3.info
            assert 'partial_charge' not in atoms3.arrays


def test_mapped_smiles():
    sdf_files = [
        get_data_file_path('9mv_neutral.sdf', 'bytemol.testdata'),
        get_data_file_path('C10H11N4O8P-2_1.sdf', 'bytemol.testdata'),
        get_data_file_path('C12H7Cl3O2_1.sdf', 'bytemol.testdata')
    ]
    for sf in sdf_files:
        mol = Molecule.from_sdf(sf)
        mapped_smi = mol.get_mapped_smiles(isomeric=False)
        mol2 = Molecule.from_mapped_smiles(mapped_smi)
        assert mapped_smi == mol2.get_mapped_smiles(isomeric=False)

        atoms1 = mol.get_rkmol().GetAtoms()
        atoms2 = mol2.get_rkmol().GetAtoms()
        for at1, at2 in zip(atoms1, atoms2):
            assert at1.GetAtomicNum() == at2.GetAtomicNum()

        bonds1 = mol.get_bonds()
        bonds2 = mol2.get_bonds()
        assert bonds1 == bonds2  # same connectivity

    mol = Molecule.from_sdf(get_data_file_path('ethanol.sdf', 'bytemol.testdata'))
    smi = mol.get_mapped_smiles({0: 5, 1: 6, 2: 8})
    assert smi == "[H][C:5]([H])([O:6][H])[C:8]([H])([H])[H]"
    smi = mol.get_mapped_smiles((0, 1, 2))
    assert smi == "[H][C:1]([H])([O:2][H])[C:3]([H])([H])[H]"


def test_init_from_rkmol():
    sdf_file = get_data_file_path('ethanol.sdf', 'bytemol.testdata')
    rkmol = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)[0]
    name = 'ethanol_rkmol'
    mol = Molecule(rkmol, name=name)
    assert mol.natoms == 9
    assert mol.name == name
    assert len(mol.conformers) == mol.nconfs == 1
    conf = mol.get_conformer()
    assert isinstance(conf, Conformer)


def test_init_from_rkmol_noname():
    sdf_file = get_data_file_path('ethanol.sdf', 'bytemol.testdata')
    rkmol = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)[0]
    mol = Molecule(rkmol)
    assert mol.natoms == 9
    assert mol.name.startswith('C2H6O_')
    assert len(mol.conformers) == mol.nconfs == 1
    conf = mol.get_conformer()
    assert isinstance(conf, Conformer)


def test_init_from_rkmol_multiconf_with_props():
    sdf_file = get_data_file_path('ethanol.sdf', 'bytemol.testdata')
    name = 'ethanol_rkmol'
    rkmol = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=False)[0]
    AllChem.EmbedMultipleConfs(rkmol, numConfs=10)
    rkmol.SetProp('name', name)
    confs = rkmol.GetConformers()
    for i, c in enumerate(confs):
        c.SetProp("conf_id", str(i))
    mol = Molecule(rkmol, name=name, keep_conformers=True, keep_mol_prop=True)
    assert len(mol.conformers) == mol.nconfs == 10
    for i, conf in enumerate(mol.conformers):
        print(i, conf.confdata)
        assert conf.confdata['prop_name'] == name
        assert str(conf.confdata['prop_conf_id']) == str(i)


def test_mol_keep_conformers():
    sdf_file = get_data_file_path('ethanol.sdf', 'bytemol.testdata')
    rkmol = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)[0]
    mol = Molecule(rkmol, keep_conformers=False)
    assert mol.nconfs == 0


def test_init_from_sdf():
    sdf_file = get_data_file_path('ethanol.sdf', 'bytemol.testdata')
    mol = Molecule(sdf_file, name='ethanol')
    bonds = mol.get_bonds()
    assert mol.natoms == 9
    assert len(mol.conformers) == 1
    assert mol.name == 'ethanol'
    assert len(bonds) == 8
    assert (2, 8) in bonds

    mol2 = Molecule.from_sdf(sdf_file)
    assert mol2.natoms == 9
    assert len(mol2.conformers) == 1
    assert np.array_equal(mol.get_conformer().coords, mol2.get_conformer().coords)


def test_init_from_multiple_sdf_with_props():
    sdf_file = get_data_file_path('multi_ethanol.sdf', 'bytemol.testdata')
    read_result = read_molecules_from_sdf(sdf_file, check_chiral=False, keep_mol_prop=True)
    assert len(read_result) == 1
    mol = list(read_result.values())[0]
    assert mol.nconfs == 3
    assert mol.get_smiles(isomeric=False) == 'CCO'
    assert np.allclose(mol.conformers[0].coords[3, 0], 0.6634)
    assert np.allclose(mol.conformers[1].coords[3, 0], 0.5254)
    assert np.allclose(mol.conformers[2].coords[3, 0], -0.5631)
    assert np.allclose(mol.conformers[0].confdata['prop_energy'], -100.0)
    assert np.allclose(mol.conformers[1].confdata['prop_energy'], -100.1)
    assert np.allclose(mol.conformers[2].confdata['prop_energy'], -100.2)


def test_update_confdata():
    read_result = read_molecules_from_sdf(get_data_file_path('multi_ethanol.sdf', 'bytemol.testdata'),
                                          check_chiral=False)
    assert len(read_result) == 1
    mol = list(read_result.values())[0]
    confdata = {'energy': [0, 1, 2]}
    mol.update_confdata(confdata)
    for i in range(mol.nconfs):
        conf = mol.get_conformer(i)
        assert conf.confdata['energy'] == confdata['energy'][i]


def test_to_sdf():
    read_result = read_molecules_from_sdf(get_data_file_path('multi_ethanol.sdf', 'bytemol.testdata'),
                                          check_chiral=False)
    assert len(read_result) == 1
    mol = list(read_result.values())[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_0 = os.path.join(tmpdir, '0.sdf')
        sdf_m1 = os.path.join(tmpdir, 'm1.sdf')
        sdf_none = os.path.join(tmpdir, 'none.sdf')

        mol.to_sdf(sdf_0, conf_id=0)
        mol.to_sdf(sdf_m1, conf_id=-1)
        mol.to_sdf(sdf_none)

        mol_0 = Molecule(sdf_0)
        mol_m1 = Molecule(sdf_m1)
        mol_none = list(read_molecules_from_sdf(sdf_none, check_chiral=False).values())[0]

    assert mol_0.nconfs == 1
    assert np.array_equal(mol_0.get_conformer().coords, mol.conformers[0].coords)
    assert mol_m1.nconfs == 1
    assert np.array_equal(mol_m1.get_conformer().coords, mol.conformers[-1].coords)
    assert mol_none.nconfs == mol.nconfs
    assert np.array_equal(mol_none.get_confdata("coords"), mol.get_confdata("coords"))


def test_to_sdf_with_props():
    smiles_list = [
        'Brc1ccc(-c2c[nH]c(C3Cc4ccccc4CN3)n2)cc1',
        'Brc1ccc(C[NH+]2CCCC2)cc1',
        'Brc1cccc(-c2cc(NCc3ccncc3)n3nccc3n2)c1',
        'Brc1cccc(C2CC(c3cccc(Br)c3)n3nnnc3N2)c1',
        'Brc1cnn2c(NCc3cccnc3)cc(-c3ccccc3)nc12',
    ]
    rkmol_list = []
    for idx, smi in enumerate(smiles_list):
        rkmol = Chem.MolFromSmiles(smi)
        rkmol = Chem.AddHs(rkmol, addCoords=True)
        AllChem.EmbedMultipleConfs(rkmol, numConfs=idx + 1)
        rkmol_list.append(rkmol)

    mol_list = []
    for rkmol in rkmol_list:
        mol = Molecule.from_rdkit(rkmol)
        mol.set_mol_prop('smiles', mol.get_smiles(isomeric=False))
        mol.set_mol_prop('nconfs', mol.nconfs)
        mol.set_mol_prop('natoms', mol.natoms)
        mol.set_conf_prop('mark_conf0', 0, conf_id=0)
        mol_list.append(mol)

    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_file = os.path.join(tmpdir, 'mock.sdf')
        mol_list[0].to_sdf(filename=sdf_file, append=False)
        for mol in mol_list[1:]:
            mol.to_sdf(filename=sdf_file, append=True)
        result = read_molecules_from_sdf(sdf_file, keep_mol_prop=True)
        for _, mol in result.items():
            for conf in mol.conformers:
                assert mol.get_smiles(isomeric=False) == conf.confdata['prop_smiles']
                assert mol.nconfs == conf.confdata['prop_nconfs']
                assert mol.natoms == conf.confdata['prop_natoms']
            assert str(mol.conformers[0].confdata['prop_mark_conf0']) == str(0)


def test_to_xyz():
    mol = Molecule.from_xyz(get_data_file_path('multi_ethanol.xyz', 'bytemol.testdata'))
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_0 = os.path.join(tmpdir, '0.xyz')
        xyz_m1 = os.path.join(tmpdir, 'm1.xyz')
        xyz_none = os.path.join(tmpdir, 'none.xyz')

        mol.to_xyz(xyz_0, conf_id=0)
        mol.to_xyz(xyz_m1, conf_id=-1)
        mol.to_xyz(xyz_none)
        mol.to_xyz(xyz_none, conf_id=0, append=True)

        mol_0 = Molecule.from_xyz(xyz_0)
        mol_m1 = Molecule.from_xyz(xyz_m1)
        mol_none = Molecule.from_xyz(xyz_none)

    assert mol_0.nconfs == 1
    assert np.array_equal(mol_0.get_conformer().coords, mol.conformers[0].coords)
    assert mol_m1.nconfs == 1
    assert np.array_equal(mol_m1.get_conformer().coords, mol.conformers[-1].coords)
    assert mol_none.nconfs == mol.nconfs + 1
    assert np.array_equal(mol_none.get_confdata("coords")[:mol.nconfs], mol.get_confdata("coords"))
    assert np.array_equal(mol_none.get_conformer(-1).coords, mol.conformers[0].coords)


def test_init_from_xyz_with_mapped_smiles():
    mol = Molecule.from_xyz(get_data_file_path('multi_ethanol.xyz', 'bytemol.testdata'))
    assert mol.nconfs == 3
    for conf in mol.conformers:
        assert "mapped_smiles" not in conf.confdata


def test_init_from_xyz_with_old_dipole():
    mol = Molecule.from_xyz(get_data_file_path('old_dipole.xyz', 'bytemol.testdata'))
    assert np.allclose(mol.conformers[1].confdata["dft_dipole"][0], 0.0398512)
    assert np.allclose(mol.conformers[1].confdata["dft_quadrupole"][0], -10.956737)
    assert np.allclose(mol.conformers[1].confdata["dft_octopole"][0], 9.8438726)
    assert np.allclose(mol.conformers[1].confdata["dft_hexadecapole"][0], -245.3431184)


def test_to_pdb():
    sdf_file = get_data_file_path('N2H2.sdf', 'bytemol.testdata')
    mol = Molecule.from_sdf(sdf_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "test.pdb")
        mol.to_pdb(pdb_path)
        with open(pdb_path, "r") as file:
            lines = file.readlines()
        assert lines[0] == "COMPND    N2H2\n"
        assert lines[1] == "HETATM    1  N1  UNL     1       0.130   0.310  -0.353  1.00  0.00           N  \n"
        assert lines[-2] == "CONECT    2    4\n"
        new_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        coords = mol.conformers[0].coords
        assert np.allclose(new_mol.GetConformer(id=0).GetPositions(), coords, atol=0.01)
        assert new_mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2.0


@pytest.mark.xfail(reason="change default aromaticity model from mdl to rdkit")
def test_aromaticity():
    rkmol_mdl = Molecule.from_smiles('c1ccsc1').get_rkmol()
    rkmol_rdkit = Molecule.from_smiles('c1ccsc1', aromaticity='rdkit').get_rkmol()

    def get_atom_bond_aromaticity(rkmol: Chem.Mol):
        return [at.GetIsAromatic() for at in rkmol.GetAtoms()], [str(b.GetBondType()) for b in rkmol.GetBonds()]

    atoms, bonds = get_atom_bond_aromaticity(rkmol_mdl)
    assert not np.any(atoms)
    assert 'AROMATIC' not in bonds

    atoms, bonds = get_atom_bond_aromaticity(rkmol_rdkit)
    assert atoms == [True, True, True, True, True, False, False, False, False]
    assert bonds == ['AROMATIC', 'AROMATIC', 'AROMATIC', 'AROMATIC', 'AROMATIC', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE']

    mol = Molecule.from_smiles('c1ccsc1', aromaticity="mdl")  # read using mdl model
    atoms, bonds = get_atom_bond_aromaticity(rkmol_rdkit)
    assert atoms == [False] * mol.natoms

    mol._aromaticity = 'rdkit'
    mol._sanitize()  # change to rdkit model
    rkmol_rdkit = mol.get_rkmol()
    atoms, bonds = get_atom_bond_aromaticity(rkmol_rdkit)
    assert atoms == [True, True, True, True, True, False, False, False, False]
    assert bonds == ['AROMATIC', 'AROMATIC', 'AROMATIC', 'AROMATIC', 'AROMATIC', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE']


def test_from_smiles_and_nconfs():
    # this does not test stereo chemistry
    # all stereo chemistry test are in test_molecule_stereo.py
    # no prune by default
    smiles = 'Brc1ccc(-c2c[nH]c(C3Cc4ccccc4CN3)n2)cc1'
    mol = Molecule.from_smiles(smiles, nconfs=10)
    assert mol.nconfs == 10

    mapped_smiles = mol.get_mapped_smiles(isomeric=False)
    mol2 = Molecule.from_mapped_smiles(mapped_smiles, nconfs=10)
    assert mol2.get_mapped_smiles(isomeric=False) == mapped_smiles
    assert mol2.nconfs == 10

    # pruneRmsThresh
    mol2 = Molecule.from_mapped_smiles(mapped_smiles, nconfs=10, pruneRmsThresh=1.0)
    assert mol2.get_mapped_smiles(isomeric=False) == mapped_smiles
    print(mol2.nconfs)
    assert mol2.nconfs <= 10


def test_inherent_moledata():
    mol = Molecule("c1ccccc1", fmt="smiles")
    assert mol.atomic_numbers == [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]
    assert mol.atomic_masses is not None
    assert mol.atomic_symbols == ['C'] * 6 + ['H'] * 6
    assert np.allclose(mol.formal_charges, np.zeros(12))


def test_SO():
    mol = Molecule.from_smiles('CCCCS(=O)C')
    assert mol.get_smiles() == 'CCCC[S+](C)[O-]'

    mol = Molecule.from_smiles('CCCCS(=O)')
    assert mol.get_smiles() == 'CCCC[SH+][O-]'

    mol = Molecule.from_smiles('O=S-C')
    assert mol.get_smiles() == 'C[SH+][O-]'


def test_N4():
    mol = Molecule.from_smiles('CCCC[N+](C)(C)C')
    assert mol.get_smiles() == 'CCCC[N+](C)(C)C'


def test_order(tmp_path):
    origin_sdf = get_data_file_path('dmso.sdf', 'bytemol.testdata')
    mol = Molecule.from_sdf(origin_sdf)
    # this [C:1]([S+:2][C:3][O-:4] order matchess the order in the sdf file
    assert mol.get_mapped_smiles(isomeric=True) == '[C:1]([S+:2]([C:3]([H:8])([H:9])[H:10])[O-:4])([H:5])([H:6])[H:7]'
    assert mol.get_smiles(isomeric=True) == 'C[S+](C)[O-]'
    logger.info('%s', mol.get_mapped_smiles())
    logger.info('%s', mol.get_smiles())

    write_sdf = os.path.join(tmp_path, 'dmso.sdf')
    mol.to_sdf(write_sdf)

    mol2 = Molecule.from_sdf(write_sdf)
    logger.info('%s', mol2.get_mapped_smiles())
    logger.info('%s', mol2.get_smiles())
    assert np.allclose(mol.conformers[0].coords, mol2.conformers[0].coords)


def test_reorder():
    mol = Molecule.from_mapped_smiles('[C:1]([S:2](=[O:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]')
    assert mol.get_mapped_smiles() == '[C:1]([S+:2]([O-:3])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]'

    mol = Molecule.from_mapped_smiles('[C:1]([S:3](=[O:2])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]')
    assert mol.get_mapped_smiles() == '[C:1]([S+:3]([O-:2])[C:4]([H:8])([H:9])[H:10])([H:5])([H:6])[H:7]'


def test_multiple_sulfoxide():
    mol = Molecule.from_smiles('S(=O)CS(=O)C')
    assert mol.get_smiles() == 'C[S+]([O-])C[SH+][O-]'
    assert mol.get_mapped_smiles(
    ) == '[S+:1]([O-:2])([C:3]([S+:4]([O-:5])[C:6]([H:10])([H:11])[H:12])([H:8])[H:9])[H:7]'

    mol = Molecule.from_mapped_smiles('[S:1](=[O:3])([C:2]([S:7](=[O:5])[C:6]([H:10])([H:11])[H:12])([H:8])[H:9])[H:4]')
    assert mol.get_mapped_smiles(
    ) == '[S+:1]([C:2]([S+:7]([O-:5])[C:6]([H:10])([H:11])[H:12])([H:8])[H:9])([O-:3])[H:4]'


def test_nitro():
    mol = Molecule.from_mapped_smiles(
        '[c:1]1([H:10])[c:2]([H:11])[c:7]([H:12])[c:4]([H:13])[c:5]([H:14])[c:6]1[N:3](=[O:8])=[O:9]')
    assert mol.get_mapped_smiles(
    ) == '[c:1]1([H:10])[c:2]([H:11])[c:7]([H:12])[c:4]([H:13])[c:5]([H:14])[c:6]1[N+:3]([O-:8])=[O:9]'


def test_diazonium():
    # this should remain unchanged and no reaction happened
    mol = Molecule.from_smiles('CC(=O)O[C@H]1CCCC2=C1C([N+]#N)=C1C(=O)[C@H]3C(=O)C=CC=C3C([O-])=C21')
    assert mol.get_smiles() == 'CC(=O)O[C@H]1CCCC2=C1C([N+]#N)=C1C(=O)[C@H]3C(=O)C=CC=C3C([O-])=C21'


def test_trivalent_s():
    mol = Molecule.from_mapped_smiles('[O:3][S:1]([C:2]([H:5])([H:6])[H:7])[H:4]')
    assert mol.get_mapped_smiles() == '[S:1]([C:2]([H:5])([H:6])[H:7])([O:3])[H:4]'


def test_halogen():
    # nothing happens
    mol = Molecule.from_smiles('F')
    assert mol.get_smiles() == 'F'


def test_odd_pyri_oxide():
    mol = Molecule.from_smiles('C=[C-]-[N+2](C)-[O-]')
    assert mol.get_mapped_smiles() == '[C:1](=[C:2]=[N+:3]([C:4]([H:8])([H:9])[H:10])[O-:5])([H:6])[H:7]'

    mol = Molecule.from_smiles('C-[N-]-[N+2](C)-[O-]')
    assert mol.get_mapped_smiles() == '[C:1]([N:2]=[N+:3]([C:4]([H:9])([H:10])[H:11])[O-:5])([H:6])([H:7])[H:8]'


def test_odd_azide():
    mol = Molecule.from_mapped_smiles('[C:1][N-:2][N+:3]#[N:4]')
    assert mol.get_mapped_smiles() == '[C:1][N:2]=[N+:3]=[N-:4]'


def test_sanitize():
    mol = Molecule.from_smiles('BrC1=CC2=C(C=C1)NC=C2')
    smiles = mol.get_smiles()
    logger.info(f'{smiles}')
    mol2 = Molecule.from_smiles(smiles)
    assert mol2.get_smiles() == smiles


@pytest.mark.parametrize(
    "smi",
    [
        r"C[n+]1[cH-]c(=N)on1",  # from chembl frag
        r'c1cc[nH]c1',  # from rdkit issue https://github.com/rdkit/rdkit/issues/2081
        r'Cc1nc2cnc3[nH+]c[cH-]c3c2n1[C@@H]1CCC[C@@H](O)C1',  # from sperrylite set
        r'C[C@@H]1NCC(=N\O)/C(=N/O)C\1=N\O',
        r'O=C1[C@H]2CCN(CC2)[C@@]12CC2',
        r'C1[C@H]2[C@H]3N[C@@H]4[C@@H]5C[C@H]3[C@H]2N[C@@H]5[C@H]14',
        r'CC1(C)NCC(=N\O)/C(=N/O)C\1=N\O',
        r'C[C@H]1C[C@@H]2CC[C@@H](CC2)C1',
        r'C1C[C@H]2[C@@H]3[C@H]4CC[C@H]5[C@@H]3[C@@H]1[C@@H]2[C@@H]45',
        r'C1O[C@]12[C@@H]1C[C@H]3C[C@@H](C1)C[C@@H]2C3',
        r'CN/C(S)=N/[C@H]1C[C@H]2C[C@H](C2)C1',
        r'CC1=C(C)/C=C\C2=C(\C=C/1)CCC2',
        r'C1N[C@]12[C@@H]1C[C@H]3C[C@@H](C1)C[C@@H]2C3',
        r'O=C1[C@H]2CCN(CC2)[C@@]12COCOC2',
        r'C1OO[C@]12[C@@H]1C[C@H]3C[C@@H](C1)C[C@@H]2C3',
        r'F[C@H]1[C@@H]2C[C@@H]3C[C@H]1C[C@H](C2)[C@]31OCOO1',
        r'Fc1ccc(/C=N\[C@H]2C[C@@H]3C[C@@H](C3)C2)cc1',
        r'C(=N/[C@H]1C[C@H]2C[C@H](C2)C1)\c1ccccc1',
        r'C(c1nnn[nH]1)[C@H]1[C@@H]2C[C@H]3C[C@H](C[C@@H]1C3)C2',
        r'Cc1ccccc1/C=N/[C@H]1C[C@@H]2C[C@@H](C2)C1',
        r'Cc1cccc(/C=N/[C@H]2C[C@H]3C[C@H](C3)C2)c1',
        r'Cc1ccc(/C=N/[C@H]2C[C@H]3C[C@H](C3)C2)cc1',
        r'CC1=C/CC[C@H]2[C@@H](\C=C/1)[C@@H]1CCCC[C@H]21',
        r'CC1=C/C[C@@H]2OC(=O)/C(=C3/C=C/C(=C\[C@H](O)C\C=C/1)CO3)C2=O',
        r'CC1=C/CC[C@H]2[C@@H](\C=C/1)[C@]1(C)CCC[C@@]3(CO3)[C@H]21',
        r'CC1=C/[C@@H]2OC(=O)C[C@]2(O)[C@H]2[C@H](\C=C/1)[C@@H]1CCCC[C@@H]12',
        r'F[C@H]1[C@H]2C[C@H]3C[C@@H]1C[C@@H](C2)[C@]31OO[C@]2(CCCCC2)O1',
        r'COC(=O)C1=C/c2cc3c(cc2-c2c(cc(C)c(OC)c2OC)\C=C/1C(=O)OC)OCO3',
        r'COC(=O)C1=C/c2cc3c(cc2-c2c(cc(OC)c(OC)c2C)\C=C/1C(=O)OC)OCO3',
    ],
)
def test_smiles_nonstereo_consistency(smi: str):
    mol1 = Molecule.from_smiles(smi)
    smi1 = mol1.get_smiles(isomeric=False)
    logger.info('%s', smi1)
    mol2 = Molecule.from_smiles(smi1)
    smi2 = mol2.get_smiles(isomeric=False)
    logger.info('%s', smi2)
    assert smi1 == smi2


def test_from_xyz_with_multiple_stereoisomers(tmp_path):
    smi1 = "[C:1]([C@:2]([N:3]([H:9])[H:10])([Cl:4])[H:8])([H:5])([H:6])[H:7]"
    smi2 = "[C:1]([C@@:2]([N:3]([H:9])[H:10])([Cl:4])[H:8])([H:5])([H:6])[H:7]"
    mol1 = Molecule.from_mapped_smiles(smi1, nconfs=1)
    mol2 = Molecule.from_mapped_smiles(smi2, nconfs=2)
    mol3 = Molecule.from_mapped_smiles(smi1, nconfs=2)
    with temporary_cd(tmp_path):
        mol1.to_xyz("1.xyz")
        mol2.to_xyz("2.xyz")
        mol3.to_xyz("3.xyz")
        multi_xyz = "multi.xyz"
        run_command_and_check(f"cat 1.xyz 2.xyz 3.xyz > {multi_xyz}")
        no_chiral_dict = read_molecules_from_xyz(multi_xyz, check_chiral=False)
        chiral_dict = read_molecules_from_xyz(multi_xyz)  # check_chiral=True as default

    assert len(no_chiral_dict) == 1
    no_chiral_smi = mol1.get_mapped_smiles(isomeric=False)
    no_chiral_mol = no_chiral_dict[no_chiral_smi]
    assert no_chiral_mol.nconfs == 5
    assert no_chiral_mol.get_mapped_smiles(isomeric=False) == no_chiral_smi

    assert len(chiral_dict) == 2
    chiral_mol1 = chiral_dict[smi1]
    chiral_mol2 = chiral_dict[smi2]
    assert chiral_mol1.nconfs == 3
    assert chiral_mol2.nconfs == 2
    assert chiral_mol1.get_mapped_smiles(isomeric=True) == smi1
    assert chiral_mol2.get_mapped_smiles(isomeric=True) == smi2


def test_to_xyz_with_hessian(tmp_path):
    mol = Molecule.from_smiles("CCCC", nconfs=1)
    mol.conformers[0].confdata["hessian"] = np.random.random((3 * mol.natoms, 3 * mol.natoms))
    with temporary_cd(tmp_path):
        mol.to_xyz(xyz_path := "test.xyz")
        assert mol.conformers[0].confdata["hessian"] is not None
        mol2 = Molecule.from_xyz(xyz_path)
        assert mol2.get_mapped_smiles() == mol.get_mapped_smiles()


def test_break_in_moledata_and_confdata(tmp_path):
    mol = Molecule.from_smiles("CCCC", nconfs=1)
    mol.moledata["test1"] = "a\nb"
    mol.conformers[0].confdata["test2"] = "c\nd"
    mol.conformers[0].confdata["test3"] = "e\\nf"
    mol.conformers[0].confdata["test4"] = r"g\[h"

    with temporary_cd(tmp_path):
        mol.to_xyz(xyz_path := "test.xyz")
        mol2 = Molecule.from_xyz(xyz_path)

        def get_prop(key):
            return mol2.moledata[key] if key in mol2.moledata else mol2.conformers[0].confdata[key]

        assert get_prop("test1") == "a\nb"
        assert get_prop("test2") == "c\nd"
        assert get_prop("test3") == "e\\nf"
        assert get_prop("test4") == r"g\[h"
        assert get_prop("test4") != "g[h"


def test_mol_name(tmp_path):
    smiles = "CCCC"

    with pytest.raises(AssertionError):
        Molecule.from_smiles(smiles, nconfs=1, name=True)

    with pytest.raises(AssertionError):
        Molecule.from_smiles(smiles, nconfs=1, name='@123')

    with pytest.raises(AssertionError):
        Molecule.from_smiles(smiles, nconfs=1, name='$^123')

    mol = Molecule.from_smiles(smiles, nconfs=1)
    assert mol.name == 'C4H10_f4d7ab73ff083957c0bd665a67c1e2e1'
    mol.name = smiles
    with temporary_cd(tmp_path):
        mol.to_xyz(path := "test.xyz")
        mol = Molecule.from_xyz(path)
        assert mol.name == smiles

        mol = Molecule.from_xyz(path, name=(test_name := "test"))
        assert mol.name == test_name

        name = '1'
        mol.name = name
        mol.to_xyz(path)
        mol = Molecule.from_xyz(path)
        assert mol.name == name

        name = '1.11'
        mol.name = name
        mol.to_xyz(path)
        mol = Molecule.from_xyz(path)
        assert mol.name == name


def test_keep_mol_prop():
    sdf_file = get_data_file_path("1a.sdf", "bytemol.testdata")

    mol_with_prop = Molecule(sdf_file, name='1a', keep_conformers=False, keep_mol_prop=True)
    assert 'prop_r_user_dG.exp' in mol_with_prop.moledata.keys()

    mol_without_prop = Molecule(sdf_file, name='1a', keep_conformers=False)
    assert len(mol_without_prop.moledata) == 0

    mol_with_conf_prop = Molecule(sdf_file, name='1a', keep_mol_prop=True)
    assert 'prop_r_user_dG.exp' in mol_with_conf_prop.conformers[0].confdata.keys()

    mol_with_conf_without_prop = Molecule(sdf_file, name='1a')
    assert len(mol_with_conf_without_prop.moledata) == 0
    assert 'prop_r_user_dG.exp' not in mol_with_conf_without_prop.conformers[0].confdata.keys()


@pytest.mark.parametrize(
    "smi",
    [
        'CC(C)(C1=NC=C(C=N1)C2=CC3=C(C=C2F)N=C4N3C(C5C4C5)C6=CC=CC=C6OC(F)F)O',
        'CC(C)(C1=NC=C(C=N1)C2=CC3=C(C=C2F)N=C4N3[C@H]([C@@H]5[C@H]4C5)C6=CC=CC=C6OC(F)F)O',
        'CC(C)(C1=NC=C(C=N1)C2=CC3=C(C=C2F)N=C4N3[C@@H]([C@H]5[C@@H]4C5)C6=CC=CC=C6OC(F)F)O',
        'CN1CCN(CC1)C2=NC=C(C=N2)C3=CC4=C(C=C3F)N=C5N4[C@H]([C@@H]6[C@H]5C6)C7=CC=CC=C7OC(F)F',
        'C1[C@H]2[C@@H]1C3=NC4=C(N3[C@H]2C5=CC=CC=C5OC(F)F)C=C(C(=C4)F)C6=CC=C(C=C6)S(=O)(=O)N',
        'C1COCCN1C2=NC=C(C=N2)C3=CC4=C(C=C3F)N=C5N4[C@H]([C@@H]6[C@H]5C6)C7=CC=CC=C7OC(F)F',
        'C1[C@H]2[C@@H]1C3=NC4=C(N3[C@H]2C5=CC=CC=C5OC(F)F)C=C(C(=C4)F)C6=CN=C(C=C6)C(=O)N',
        'CNC(=O)C1=NC=C(C=C1)C2=CC3=C(C=C2F)N=C4N3[C@H]([C@@H]5[C@H]4C5)C6=CC=CC=C6OC(F)F',
        'C1CN(CC(=O)N1)C2=NC=C(C=N2)C3=CC4=C(C=C3F)N=C5N4[C@H]([C@@H]6[C@H]5C6)C7=CC=CC=C7OC(F)F',
        'C1[C@H]2[C@@H]1C3=NC4=C(N3[C@H]2C5=CC=CC=C5OC(F)F)C=C(C(=C4)F)C6=CN=C(C=C6)C#N',
        '[O-]P([O-])(=O)OP([O-])([O-])=O',
        'ClO',
        'CC=N[N-]C(=O)c1[nH+]o[nH+]c1C和CC=[N+]=NC(=O)C1=C(C)NON1',
        'CC=N[N-]C(=O)c1no[nH+]c1C和CC=[N+]=NC(=O)C1=C(C)NO[N-]1',
    ],
)
def test_good_molecules(smi):
    mol = Molecule.from_smiles(smi)
    assert_good_molecule(mol)


@pytest.mark.parametrize(
    "smi",
    [
        '[Cs+].[O-]C(=O)[O-].[Cs+]',
        '[Li+].[Al+3].[H-].[H-].[H-].[H-]',
        'Cl[Pt@SP1](Cl)([NH3])[NH3]',
        'Cl*.Cl*.c1ccccc1-c1ccccc1',
        # 'BC',
        # '[Si]C',
        # 'I(=O)(=O)O',
        # 'OCl(=O)(=O)=O',
        'Cl.ClO',
        'S(C)(C)(C)C',
        '[O+]',
        'C1=C[N]C=C1',
        'C1=CC=CC=C[C+]1',
        'C[CH2]',
    ],
)
def test_bad_molecules(smi):
    with pytest.raises(Exception):
        mol = Molecule.from_smiles(smi)
        assert_good_molecule(mol)


def test_deprotonated_sulfonamide():
    mol = Molecule.from_mapped_smiles(
        '[C:1]([C:2]([H:9])([H:10])[H:11])([N:4]=[S@@:8]([C:3]([H:12])([H:13])[H:14])(=[O:6])[O-:7])=[O:5]')
    assert mol.get_mapped_smiles(
    ) == '[C:1]([C:2]([H:9])([H:10])[H:11])([N-:4][S:8]([C:3]([H:12])([H:13])[H:14])(=[O:6])=[O:7])=[O:5]'

    mol = Molecule.from_smiles('CC(=O)N=[S@@](C)(=O)[O-]')
    assert mol.get_smiles() == 'CC(=O)[N-]S(C)(=O)=O'


def test_valid_molecule_passes():
    """verify molecule can pass check"""
    mol = Molecule.from_smiles("CCO")
    assert_good_molecule(mol)


def test_halogen_check_default_off():
    """verify halogen connectivity is not checked by default"""
    mol = Molecule.from_smiles("CCl")
    mol2 = Molecule.from_smiles("[O-]Cl(=O)(=O)")
    logger.info(mol2.get_mapped_smiles())

    assert_good_molecule(mol)
    assert_good_molecule(mol2)


def test_halogen_check_when_enabled():
    mol_good = Molecule.from_smiles("CCl")
    mol_bad = Molecule.from_smiles("[O-]Cl(=O)(=O)")
    logger.info(mol_bad.get_mapped_smiles())

    assert_good_molecule(mol_good, check_halogen_connectivity=True)

    with pytest.raises(AssertionError):
        assert_good_molecule(mol_bad, check_halogen_connectivity=True)
