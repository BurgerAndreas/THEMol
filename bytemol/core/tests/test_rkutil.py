# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
import rdkit.Chem as Chem

from bytemol.core import Molecule, rkutil
from bytemol.utils import get_data_file_path

logger = logging.getLogger(__name__)


def test_find_mapped_smarts_matches():
    # case 1, symmetric
    mol = Chem.MolFromSmiles('CN1C=NC=CN1C')
    mol = rkutil.sanitize_rkmol(mol)
    matches = rkutil.find_mapped_smarts_matches(mol, '[*:1]-[#7X3$(*-[#6X3,#6X2]):2]-[#7X3$(*-[#6X3,#6X2]):3]-[*:4]')
    # matches are not unique
    assert len(matches) == 8
    assert (0, 1, 6, 5) in matches
    assert (0, 1, 6, 7) in matches
    assert (2, 1, 6, 5) in matches
    assert (2, 1, 6, 7) in matches
    # no map number
    matches = rkutil.find_mapped_smarts_matches(mol, '[*]-[#7X3$(*-[#6X3,#6X2])]-[#7X3$(*-[#6X3,#6X2])]-[*]')
    assert len(matches) == 4
    assert (0, 1, 5, 6) in matches
    assert (0, 1, 6, 7) in matches
    assert (1, 2, 5, 6) in matches
    assert (1, 2, 6, 7) in matches

    # case 2, asymmetric
    mol = Chem.MolFromSmiles('CP1C=NC=CN1C')
    mol = rkutil.sanitize_rkmol(mol)
    matches = rkutil.find_mapped_smarts_matches(mol, '[*:1]-[#7X3$(*-[#6X3,#6X2]):2]-[#15X3$(*-[#6X3,#6X2]):3]-[*:4]')
    assert len(matches) == 4
    assert (5, 6, 1, 0) in matches
    assert (5, 6, 1, 2) in matches
    assert (7, 6, 1, 0) in matches
    assert (7, 6, 1, 2) in matches

    # case 3, angle in ring
    mol = Chem.MolFromSmiles('CP1C=NC=CN1C')
    mol = rkutil.sanitize_rkmol(mol)
    matches = rkutil.find_mapped_smarts_matches(mol, '[*;r3:1]1~;@[*;r3:2]~;@[*;r3:3]1')
    assert len(matches) == 0

    mol = Chem.MolFromSmiles('C1CC1')
    mol = rkutil.sanitize_rkmol(mol)
    matches = rkutil.find_mapped_smarts_matches(mol, '[*;r3:1]1~;@[*;r3:2]~;@[*;r3:3]1')
    assert len(matches) == 6
    matches = rkutil.find_mapped_smarts_matches(mol, '[*;r3]1~;@[*;r3]~;@[*;r3]1')
    assert len(matches) == 1


def test_sanitize():
    rkmol = Chem.MolFromSmiles('c1ccsc1')
    rkmol = rkutil.sanitize_rkmol(rkmol)
    assert rkmol.GetRingInfo().NumRings() == 1
    rkmol_copy = rkutil.sanitize_rkmol(Chem.Mol(rkmol))
    assert rkmol_copy.GetRingInfo().NumRings() == 1


def test_sanitize_twice():
    rkmol = Chem.MolFromSmiles('Brc1ccc2[nH]ccc2c1', sanitize=False)
    rkmol = rkutil.sanitize_rkmol(rkmol)
    with pytest.raises(ValueError):
        rkmol = rkutil.sanitize_rkmol(rkmol)


def test_match_improper():
    sdf_path = get_data_file_path('3a_torsion_4_8_12_15.sdf', 'bytemol.testdata')
    mol = Molecule.from_sdf(sdf_path)
    rkmol = mol.get_rkmol()
    matches = rkutil.find_mapped_smarts_matches(rkmol, '[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]')
    assert len(matches) == 6
    for m in matches:
        assert m[1] == 3  # center atom is N


def test_gen_conf_rms():
    # bace, CAT-13a
    rkmol = Molecule.from_smiles('CN1C(=O)[C@@](C)(c2cccc(-c3cccc(Cl)c3)c2)[NH+]=C1N').get_rkmol()
    # generate without prune
    nconfs = 50
    rkmol, success, energies = rkutil.generate_confs(rkmol, nconfs=nconfs, ffopt=True, verbose=True, pruneRmsThresh=-1)
    assert success is not None
    logger.info(energies)
    assert len(energies) == nconfs
    conf_rms = rkutil.get_conf_rms(rkmol, 0, 1)
    logger.info(conf_rms)

    # generate with prune
    min_rms = 1.0
    rkmol, _, _ = rkutil.generate_confs(rkmol, nconfs=nconfs, ffopt=False, verbose=True, pruneRmsThresh=min_rms)
    conf_rms = rkutil.get_conf_rms(rkmol, 0, 1)
    assert conf_rms > min_rms  # default value for pruneRmsThresh
    rms_matrix = rkutil.get_rms_matrix(rkmol)
    logger.info(rms_matrix)
    assert np.min(np.asarray(rms_matrix)) > min_rms


def test_get_indices_mapping_between_isomorphic_mols():
    # CH3CH2OH
    mol1 = Molecule("[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]", fmt="mapped_smiles")
    mol2 = Molecule("[C:2]([C:3]([O:1][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]", fmt="mapped_smiles")
    mapping = rkutil.find_indices_mapping_between_isomorphic_mols(mol1.rkmol, mol2.rkmol)
    assert mapping[:3] == [2, 0, 1]


def test_resonance_matches():
    mol = Molecule.from_smiles('NC(=[NH2+])c1ccc(cc1)C(=O)[O-]')
    rkmol = mol.get_rkmol()
    matches = rkutil.find_mapped_smarts_matches(rkmol, '[CX3:1]-[OX1-:2]', match_resonance=True)
    assert len(matches) == 2
    logger.info(matches)
    matches = rkutil.find_mapped_smarts_matches(rkmol, '[NX3H2+:1]', match_resonance=True)
    assert len(matches) == 2
    logger.info(matches)

    matches = rkutil.find_mapped_smarts_matches(rkmol, '[CX3:1]-[OX1-:2]', match_resonance=False)
    assert len(matches) == 1
    logger.info(matches)
    matches = rkutil.find_mapped_smarts_matches(rkmol, '[NX3H2+:1]', match_resonance=False)
    assert len(matches) == 1
    logger.info(matches)


def test_resonance_match_aromatic():
    mol = Molecule.from_smiles('C#CCCCn1c(Cc2cc(OC)ccc2OC)nc2c(N)[nH+]c(F)nc21')
    rkmol = mol.get_rkmol()
    matches = rkutil.find_mapped_smarts_matches(rkmol, '[cX3:1]:[nX3:2]', match_resonance=True)
    logger.info('%s', matches)
    with pytest.raises(AssertionError):
        assert len(matches) == 4

    matches = rkutil.find_mapped_smarts_matches(rkmol, '[cX3:1]~[nX3:2]', match_resonance=True)
    logger.info('%s', matches)
    assert len(matches) == 4

    matches = rkutil.find_mapped_smarts_matches(rkmol, '[cX3:1]:[nX3:2]', match_resonance=False)
    logger.info('%s', matches)
    assert len(matches) == 4


def test_fmcs_keto_enol():
    keto = Molecule.from_smiles('CC(=O)C')
    enol = Molecule.from_smiles('CC(-O)=C')
    from rdkit.Chem import rdFMCS

    mapping_order = rkutil.find_indices_mapping_between_mols(rkmol0=keto.get_rkmol(),
                                                             rkmol1=enol.get_rkmol(),
                                                             verbose=True,
                                                             bondCompare=rdFMCS.BondCompare.CompareOrder)

    mapping_any = rkutil.find_indices_mapping_between_mols(rkmol0=keto.get_rkmol(),
                                                           rkmol1=enol.get_rkmol(),
                                                           verbose=True,
                                                           bondCompare=rdFMCS.BondCompare.CompareAny)
    assert len(mapping_order) == 5  # only C-C is mapped
    assert len(mapping_any) == 9  # all except 1 H are mapped


def test_resonance():
    rkmol = Molecule.from_smiles('CC=N[N-]C(=O)c1no[nH+]c1C').get_rkmol()
    resoners = rkutil.get_resonance_structures(rkmol)
    print(len(resoners))
    for r in resoners:
        print(Chem.MolToSmiles(r))


def test_canonical_resonance():
    smi = "[H]c1ssc2c([N-]C(=O)C([H])([H])[H])c(=O)n(C([H])([H])[H])c1-2"
    mol = Molecule.from_smiles(smi)
    mol1 = Molecule(rkutil.get_canonical_resoner(mol.get_rkmol()))
    logger.info('%s', mol1.get_smiles(isomeric=False))
    assert mol1.get_smiles(isomeric=False) == mol.get_smiles(isomeric=False)


def test_hbond():
    sdf_file = get_data_file_path('Edoxaban.sdf', 'bytemol.testdata')
    mol = Molecule(sdf_file)
    rkmol = mol.to_rkmol()

    indices_HBD = rkutil.get_hbond_donors(rkmol)
    indices_HBA = rkutil.get_hbond_acceptors(rkmol)
    assert indices_HBD == [9, 13, 23, 30]
    assert indices_HBA == [0, 11, 18, 19, 25, 27, 35, 67]

    intral_hb_info = rkutil.find_intramolecular_hbonds(rkmol)
    assert intral_hb_info == []

    angle = 100
    dist = 3.0
    intral_hb_info = rkutil.find_intramolecular_hbonds(rkmol, dist_upper_bound=dist, angle_lower_bound=angle)
    for i, j, k, d, a in intral_hb_info:
        assert i in indices_HBD
        assert k in indices_HBA
        assert mol.atomic_numbers[j] == 1
        assert d <= dist
        assert a >= angle


@pytest.mark.parametrize('smiles', [
    'CC(=O)[N-][C@H](C)CN(C)N=[PH]=S',
    'C[C@@H](C(=O)[NH+](C)C)N(C)N=[PH]=S',
    'C[C@@H](C(=O)[NH+]1CCCC1)N(C)N=[PH]=S',
])
def test_resoner_formal_charges(smiles):
    mol = Molecule.from_smiles(smiles)
    resoner_rkmol = rkutil.get_canonical_resoner(mol.get_rkmol())
    resoner_mol = Molecule(resoner_rkmol)
    logger.info('%s', mol.get_mapped_smiles())
    logger.info('%s', resoner_mol.get_mapped_smiles())
    assert mol.get_mapped_smiles(isomeric=True) == resoner_mol.get_mapped_smiles(isomeric=True)


@pytest.mark.parametrize('smiles', [
    'O=[11C]([O-])c1ccccc1-c1c2cc(Br)c(=O)c(Br)c-2oc2c(Br)c([O-])c(Br)cc12',
    'O=[13C]([O-])c1ccccc1-c1c2ccc(=O)cc-2oc2cc([O-])ccc12',
    'O=[14C]1C(I)=CC(=C(c2cc(I)c([O-])c(I)c2)c2ccccc2C(=O)[O-])C=C1I',
    'O=C1C=CC(=Nc2ccc([O-])cc2)C=C1',
    'O=c1ccc2[n+]([O-])c3ccc([O-])cc3oc-2c1',
    '[2H]C(CCOS(N)(=O)=O)c1ccc(C(C)(C)C)cc1',
    '[2H]C(CCOS(N)(=O)=O)c1ccccc1',
    '[2H]C(Nc1cc(C(F)(F)F)cc2ncc(N3CCN(C(C)=O)CC3)cc12)c1cccc([N+](=O)[O-])c1',
    '[2H]C(Nc1cc(C(F)(F)F)cc2ncc(N3CCN(C)CC3)cc12)c1cccc([N+](=O)[O-])c1',
])
def test_cleanup_isotope(smiles):
    mol = Molecule.from_smiles(smiles)
    rkmol = rkutil.cleanup_rkmol_isotope(mol.get_rkmol())
    smi = Molecule.from_rdkit(rkmol).get_smiles()
    logger.info('%s %s', smiles, smi)
    assert '[2H]' not in smi
    assert '[11C]' not in smi
    assert '[13C]' not in smi
    assert '[14C]' not in smi
