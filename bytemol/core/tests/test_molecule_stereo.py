# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import rdkit

from bytemol.core import Molecule, read_molecules_from_sdf, rkutil
from bytemol.utils import get_data_file_path, temporary_cd

logger = logging.getLogger(__name__)

# test the following codepath:
# 1. from sdf (with/without H) -> stereo in sdf is correctly marked
# 2. from (mapped) smiles with partial/full stereo -> is correctly preserved
# 3. from (mapped) smiles with genconf -> conf and stereo match

stereo_inring_trans = [
    'C[C@H]1C/C=C/CCCCC1',
    'C[C@H]1C/C=C/CCCCCC1',
    'C[C@H]1C/C=C/CCCCCCC1',
    'C[C@H]1C/C=C/CCCCCCCC1',
    'C[C@@H]1C/C=C/CCCCC1',
    'C[C@@H]1C/C=C/CCCCCC1',
    'C[C@@H]1C/C=C/CCCCCCC1',
    'C[C@@H]1C/C=C/CCCCCCCC1',
]

stereo_inring_cis = [
    r'C[C@H]1C/C=C\CCCCC1',
    r'C[C@H]1C/C=C\CCCCCC1',
    r'C[C@H]1C/C=C\CCCCCCC1',
    r'C[C@H]1C/C=C\CCCCCCCC1',
    r'C[C@@H]1C/C=C\CCCCC1',
    r'C[C@@H]1C/C=C\CCCCCC1',
    r'C[C@@H]1C/C=C\CCCCCCC1',
    r'C[C@@H]1C/C=C\CCCCCCCC1',
    r'Cl/C1=C/CCCCCCCCCC1',
]

stereo_double_bond = [
    r'C/C=C/C',
    r'C/C=C\C',
    r'CC/C=C\C=C\CC',
    r'ClC=C=CCl',
    r'O=C[C@H](O)CO',
    r'C=C[C@@H](O)C=O',
    r'Cl/C(Br)=C(\Br)I',
]

stereo_general = [
    r'C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@@H]4[C@H]3CC[C@@]21CC',
    r'CC[C@@H]1C(=O)N(C)c2cnc(Nc3ccc(C(=O)N[C@@H]4CC[N@H+](C)CC4)cc3OC)nc2N1C1CCCC1',
    r'CC[C@@H]1C(=O)N(C)c2cnc(Nc3ccc(C(=O)N[C@@H]4CC[N@@H+](C)CC4)cc3OC)nc2N1C1CCCC1',
    r'CN1C(=O)[C@@](c2ccccc2)(c2cccc(-c3ccccc3)c2)[NH+]=C1N',
    r'CN1C(=O)[C@@](c2cncnc2)(c2cccc(-c3cccc(Cl)c3)c2)[NH+]=C1N',
    r'COC1=CC2=NC(c3cc(C)c(O)c(C)c3)=NC(=O)[C@@H]2C(OC)=C1',
    r'C[C@@H](CS(C)(=O)=O)Nc1ncc2cc(Oc3ccc(F)cc3F)c(=O)n(C)c2n1',
    r'C[C@H]1C[C@@H](O)c2c(S(=O)(=O)C(F)F)ccc(Oc3cc(F)cc(Cl)c3)c21',
    r'Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2',
    r'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)OC(C)(C)C)c1[nH]nc(C)[n+]1-2',
    r'Cn1cc(-c2ccc(Cc3n[nH]c4ccc(C(=O)N5CC[C@@H](O)C5)cc34)cc2)cn1',
    r'N#C[C@@H]1C[C@@H]1C(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1',
    r'[NH3+][C@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)NCc1cccc(F)c1',
    r'[NH3+][C@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)NCc1ccccc1',
    "CCCCCCCC",
    "NC=O",
]

stereo_xfail = [
    r'COC1=C/C2=NC(c3cc(C)c(O)c(C)c3)=NC(=O)[C@@H]2/C(OC)=C\1',  # cis/trans notation is redundant
    r'Cc1nnc2n1-c1ccc(Cl)cc1/C(c1ccccc1)=N\C2',  # redundant cis/trans for 7-member ring
    r'Cc1sc2c(c1C)/C(c1ccc(Cl)cc1)=N\[C@@H](CC(=O)OC(C)(C)C)c1[nH]nc(C)[n+]1-2',  # redundant cis/trans for 7-member ring
]


@pytest.mark.parametrize("smi", stereo_inring_trans + stereo_inring_cis + stereo_double_bond + stereo_general)
def test_canon_smiles(smi):
    mol = Molecule.from_smiles(smi, verbose=True)
    logger.info('%s', mol.get_smiles(isomeric=True))
    # in these cases, the E/Z notation is redundant and not necessary
    if smi == 'C#C[C@]1(O)CC[C@H]2[C@@H]3CC/C4=C/C(=O)CC[C@@H]4[C@H]3CC[C@@]21CC':
        assert mol.get_smiles(isomeric=True) == 'C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@@H]4[C@H]3CC[C@@]21CC'
    elif smi == r'Cc1sc2c(c1C)/C(c1ccc(Cl)cc1)=N\[C@@H](CC(=O)OC(C)(C)C)c1[nH]nc(C)[n+]1-2':
        # less than 8-member ring
        assert mol.get_smiles(isomeric=True) == 'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)OC(C)(C)C)c1[nH]nc(C)[n+]1-2'
    elif smi == r'COC1=C/C2=NC(c3cc(C)c(O)c(C)c3)=NC(=O)[C@@H]2/C(OC)=C\1':
        # less than 8-member ring
        assert mol.get_smiles(isomeric=True) == 'COC1=CC2=NC(c3cc(C)c(O)c(C)c3)=NC(=O)[C@@H]2C(OC)=C1'
    elif smi == r'Cc1nnc2n1-c1ccc(Cl)cc1/C(c1ccccc1)=N\C2':
        # less than 8-member ring
        assert mol.get_smiles(isomeric=True) == 'Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2'
    elif smi == 'C(/C)(C)=C/C(C)':
        assert mol.get_smiles(isomeric=True) == 'CCC=C(C)C'
    elif smi == 'C[C@H]1C/C=C/CCC1' or smi == r'C[C@H]1C/C=C\CCC1':
        # E/Z notation in small rings below this size is ignored
        assert mol.get_smiles(isomeric=True) == 'C[C@H]1CC=CCCC1'
    else:
        assert mol.get_smiles(isomeric=True) == smi

    mol2 = mol.from_mapped_smiles(mol.get_mapped_smiles(isomeric=True), verbose=True)
    logger.info('%s', mol2.get_mapped_smiles(isomeric=True))
    assert mol2.get_mapped_smiles(isomeric=True) == mol.get_mapped_smiles(isomeric=True)


@pytest.mark.parametrize("smi", stereo_inring_trans + stereo_inring_cis + stereo_double_bond + stereo_general)
def test_genconf_stereo(smi, tmp_path):

    # one step generation
    mol1 = Molecule.from_smiles(smi, nconfs=1, ffopt=True, verbose=True)
    one_step_sdf = f'{tmp_path}/test_onestep_{mol1.name}.sdf'
    mol1.to_sdf(one_step_sdf)

    # two step generation
    mol2 = Molecule.from_smiles(smi)
    rkmol = mol2.get_rkmol()
    rkmol, _, _ = rkutil.generate_confs(rkmol, nconfs=1, ffopt=True, verbose=True)
    mol2 = Molecule.from_rdkit(rkmol)
    two_step_sdf = f'{tmp_path}/test_twostep_{mol1.name}.sdf'
    two_step_xyz = f'{tmp_path}/test_twostep_{mol1.name}.xyz'
    mol2.to_sdf(two_step_sdf)
    mol2.to_xyz(two_step_xyz)

    # verify sdf and xyz
    sdf_mol1 = Molecule.from_sdf(one_step_sdf)
    sdf_mol2 = Molecule.from_sdf(two_step_sdf)
    xyz_mol2 = Molecule.from_xyz(two_step_xyz)

    if smi in stereo_inring_trans:
        logger.warning('conf generation for in-ring cis/trans is not reliable in rdkit, be careful with the result.')
        # genconf is not reliable in this case so mol2 may be different from mol1
        assert mol1.get_smiles(isomeric=True) == mol2.get_smiles(isomeric=True)
        assert mol2.get_smiles(isomeric=True) == sdf_mol2.get_smiles(isomeric=True)
        assert sdf_mol2.get_smiles(isomeric=True) == xyz_mol2.get_smiles(isomeric=True)
        assert mol1.get_mapped_smiles(isomeric=True) == sdf_mol1.get_mapped_smiles(isomeric=True)
        assert mol2.get_mapped_smiles(isomeric=True) == sdf_mol2.get_mapped_smiles(isomeric=True)
        assert sdf_mol2.get_mapped_smiles(isomeric=True) == xyz_mol2.get_mapped_smiles(isomeric=True)
    else:
        # mol1, mol2, sdf_mol1, sdf_mol2 must be all identical
        logger.info('%s', smi)
        logger.info('%s', mol1.get_smiles(isomeric=True))
        logger.info('%s', mol2.get_smiles(isomeric=True))
        logger.info('%s', sdf_mol1.get_smiles(isomeric=True))
        logger.info('%s', sdf_mol2.get_smiles(isomeric=True))
        logger.info('%s', xyz_mol2.get_smiles(isomeric=True))
        assert mol1.get_smiles(isomeric=True) == mol2.get_smiles(isomeric=True)\
          == sdf_mol1.get_smiles(isomeric=True) == sdf_mol2.get_smiles(isomeric=True)\
          == xyz_mol2.get_smiles(isomeric=True) == smi
        assert mol1.get_mapped_smiles(isomeric=True) == mol2.get_mapped_smiles(isomeric=True)\
          == sdf_mol1.get_mapped_smiles(isomeric=True) == sdf_mol2.get_mapped_smiles(isomeric=True)\
          == xyz_mol2.get_mapped_smiles(isomeric=True)


def test_fix_chiral():
    mol = Molecule.from_smiles('C[S@@](=O)C')
    assert mol.get_smiles(isomeric=True) == 'C[S+](C)[O-]'
    mol = Molecule.from_sdf(get_data_file_path('dmso.sdf', 'bytemol.testdata'))
    assert mol.get_smiles(isomeric=True) == 'C[S+](C)[O-]'


@pytest.mark.parametrize(
    "smi",
    [
        '[O:1]=[S:2](=[O:3])([C:4]([C:5]([O:6][H:23])([H:21])[H:22])([H:19])[H:20])[C:7]1=[N:8][O:9][N+:10]([O-:11])=[C:12]1[c:13]1[c:14]([H:24])[c:15]([H:25])[c:16]([H:26])[c:17]([H:27])[c:18]1[H:28]',
        '[C:1]([O:2][C:3](=[O:4])[C:5]([C@@:6]1([H:44])[C:7]([C:8]([H:45])([H:46])[H:47])([C:9]([H:48])([H:49])[H:50])[C@@:10]([O:11][C:12]([C:13]([H:52])([H:53])[H:54])=[O:14])([H:51])[C@@:15]2([H:55])[C:16](=[O:17])[C@:18]1([C:19]([H:56])([H:57])[H:58])[C@@:20]1([H:59])[C:21]([H:60])([H:61])[C:22]([H:62])([H:63])[C@@:23]3([C:24]([H:64])([H:65])[H:66])[C@@:25]([C:26]4=[C:30]([H:70])[O:29][C:28]([H:69])=[C:27]4[H:68])([H:67])[O:31][C:32](=[O:33])[C:34]([H:71])([H:72])[C@@:35]3([H:73])[C@:36]13[O:37][C@:38]23[H:74])([H:42])[H:43])([H:39])([H:40])[H:41]',
        '[C:1]([C:2]([C:3]([C:4]([C:5]([C:6]([C:7]([C:8]([C:9]([C:10]([C:11]([C:12]([C:13]([C:14](=[O:15])[N:16]([C:17]([c:18]1[c:19]([H:65])[c:20]([H:66])[c:21]([C:22](=[O:23])[N:24]([C@:25]([C:26](=[O:27])[O:28][H:69])([C@@:29]([C:30]([H:71])([H:72])[H:73])([C:31]([C:32]([H:76])([H:77])[H:78])([H:74])[H:75])[H:70])[H:68])[H:67])[c:33]([H:79])[c:34]1[H:80])([H:63])[H:64])[H:62])([H:60])[H:61])([H:58])[H:59])([H:56])[H:57])([H:54])[H:55])([H:52])[H:53])([H:50])[H:51])([H:48])[H:49])([H:46])[H:47])([H:44])[H:45])([H:42])[H:43])([H:40])[H:41])([H:38])[H:39])([H:35])([H:36])[H:37]',
        '[C:1](=[C:2]([C:3]([c:4]1[c:5]([H:30])[c:6]([H:31])[c:7]([O:8][C:9](=[O:10])[C:11]23[C:12]([H:32])([H:33])[C:13]4([H:34])[C:14]([H:35])([H:36])[C:15]([H:37])([C:16]([H:38])([H:39])[C:17]([H:40])([C:18]4([H:41])[H:42])[C:19]2([H:43])[H:44])[C:20]3([H:45])[H:46])[c:21]([O:22][C:23]([H:47])([H:48])[H:49])[c:24]1[H:50])([H:28])[H:29])[H:27])([H:25])[H:26]',
        '[C:1]([C@@:2]1([H:36])[C:3]([H:37])([H:38])[N:4]([C:5]([C:6]([C:7](=[O:8])[N:9]([C@@:10]([C:11]([C:12]([c:13]2[c:14]([H:49])[c:15]([H:50])[c:16]([H:51])[c:17]([H:52])[c:18]2[H:53])([H:47])[H:48])([H:45])[H:46])([C:19](=[O:20])[O:21][H:54])[H:44])[H:43])([H:41])[H:42])([H:39])[H:40])[C:22]([H:55])([H:56])[C:23]([H:57])([H:58])[C@@:24]1([C:25]([H:59])([H:60])[H:61])[c:26]1[c:27]([H:62])[c:28]([H:63])[c:29]([H:64])[c:30]([O:31][H:65])[c:32]1[H:66])([H:33])([H:34])[H:35]',
        '[C:1]([c:2]1[c:3]([H:32])[c:4]([H:33])[c:5]([H:34])[c:6]([N:7]2[C:8]([H:35])([H:36])[C:9]([H:37])([H:38])[N:10]([C:11]([C:12]([C:13]([O:14][N:15]3[C:16](=[O:17])[c:18]4[c:19]([H:45])[c:20]([H:46])[c:21]([H:47])[c:22]([H:48])[c:23]4[C:24]3=[O:25])([H:43])[H:44])([H:41])[H:42])([H:39])[H:40])[C:26]([H:49])([H:50])[C:27]2([H:51])[H:52])[c:28]1[H:53])([H:29])([H:30])[H:31]',
        '[N:1](=[C:2]1[S:3][c:4]2[c:5]([H:27])[c:6]([H:28])[c:7]([H:29])[c:8]([H:30])[c:9]2[N:10]1[C:11]([C:12]([N:13]1[C:14]([H:35])([H:36])[C:15]([H:37])([H:38])[C:16]([c:17]2[c:18]([H:40])[c:19]([H:41])[c:20]([F:21])[c:22]([H:42])[c:23]2[H:43])([H:39])[C:24]([H:44])([H:45])[C:25]1([H:46])[H:47])([H:33])[H:34])([H:31])[H:32])[H:26]',
    ],
)
def test_mapped_smiles_stereo_consistency(smi: str):
    mol1 = Molecule.from_mapped_smiles(smi)
    smi1 = mol1.get_mapped_smiles(isomeric=True)
    logger.info('%s', smi1)
    mol2 = Molecule.from_mapped_smiles(smi1)
    smi2 = mol2.get_mapped_smiles(isomeric=True)
    logger.info('%s', smi2)
    assert smi1 == smi2


def test_xyz_stereo():
    mol = Molecule.from_xyz(get_data_file_path('c255_Z_pent_2_ene.xyz', 'bytemol.testdata'))
    logger.info('%s %s', mol.get_smiles(isomeric=True), mol.get_mapped_smiles(isomeric=True))
    assert mol.get_mapped_smiles(
        isomeric=True
    ) == r'[C:1](=[C:2](\[C:5]([C:4]([H:11])([H:12])[H:13])([H:14])[H:15])[H:7])(\[C:3]([H:8])([H:9])[H:10])[H:6]'

    mol = Molecule.from_xyz(get_data_file_path('c289_E_12_dichloroethene.xyz', 'bytemol.testdata'))
    logger.info('%s %s', mol.get_smiles(isomeric=True), mol.get_mapped_smiles(isomeric=True))
    assert mol.get_mapped_smiles(isomeric=True) == r'[C:1](=[C:2](/[Cl:4])[H:6])(\[Cl:3])[H:5]'

    mol = Molecule.from_xyz(get_data_file_path('c290_Z_12_dichloroethene.xyz', 'bytemol.testdata'))
    logger.info('%s %s', mol.get_smiles(isomeric=True), mol.get_mapped_smiles(isomeric=True))
    assert mol.get_mapped_smiles(isomeric=True) == r'[C:1](=[C:2](\[Cl:4])[H:6])(\[Cl:3])[H:5]'


def test_validate_sdf():
    # https://cipvalidationsuite.github.io/ValidationSuite/

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS013.sdf', 'bytemol.testdata'))
    origin_smiles = r'CC\C(\C(\C)=N\O)=N\O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'CCC(=N/O)/C(C)=N/O'

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS018.sdf', 'bytemol.testdata'))
    origin_smiles = r'C1[C@@H]2C=C[C@H](CCCCCC\C=C\C1)CC2'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)

    if rdkit.rdBase.rdkitVersion <= '2022.3.5':
        with pytest.raises(AssertionError):
            # bug of 2022.3.5 legacy mode
            assert mol.get_smiles(isomeric=True) == r'C1=C[C@H]2CC/C=C/CCCCCC[C@@H]1CC2'
    else:
        assert mol.get_smiles(isomeric=True) == r'C1=C[C@H]2CC/C=C/CCCCCC[C@@H]1CC2'

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS019.sdf', 'bytemol.testdata'))
    origin_smiles = r'C1CCCCC\C=C\C1'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'C1=C/CCCCCCC/1'

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS020.sdf', 'bytemol.testdata'))
    origin_smiles = r'CCCCCCCCCC/C(=C(\C#N)/Br)/I'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'CCCCCCCCCC/C(I)=C(/Br)C#N'

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS034.sdf', 'bytemol.testdata'))
    origin_smiles = r'Br[C@H]1[C@H](C[C@H](CC1)Br)Br'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'Br[C@H]1CC[C@@H](Br)[C@@H](Br)C1'

    mol = Molecule.from_sdf(get_data_file_path('stereo/VS035.sdf', 'bytemol.testdata'))
    origin_smiles = r'O[C@@H]1CCCC[C@H]1[C@@H](F)Cl'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O[C@@H]1CCCC[C@H]1[C@@H](F)Cl'

    # this is non-stereo in rdkit, but marked P in original dataset
    mol = Molecule.from_sdf(get_data_file_path('stereo/VS055.sdf', 'bytemol.testdata'))
    origin_smiles = r'BrC=1C(=C(C=CC1)C(=O)O)C=2C(=CC=CC2Br)C(=O)O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O=C(O)c1cccc(Br)c1-c1c(Br)cccc1C(=O)O'

    # this is non-stereo in rdkit, but marked M in original dataset
    mol = Molecule.from_sdf(get_data_file_path('stereo/VS057.sdf', 'bytemol.testdata'))
    origin_smiles = r'[O-][N+](C=1C(=C(C=CC1)C(=O)O)C2=C(C=CC=C2[N+]([O-])=O)C(=O)O)=O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O=C(O)c1cccc([N+](=O)[O-])c1-c1c(C(=O)O)cccc1[N+](=O)[O-]'


def test_read_molecules_from_sdf():
    # https://cipvalidationsuite.github.io/ValidationSuite/

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS013.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'CC\C(\C(\C)=N\O)=N\O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'CCC(=N/O)/C(C)=N/O'

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS018.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'C1[C@@H]2C=C[C@H](CCCCCC\C=C\C1)CC2'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)

    if rdkit.rdBase.rdkitVersion <= '2022.3.5':
        with pytest.raises(AssertionError):
            # bug of 2022.3.5 legacy mode
            assert mol.get_smiles(isomeric=True) == r'C1=C[C@H]2CC/C=C/CCCCCC[C@@H]1CC2'
    else:
        assert mol.get_smiles(isomeric=True) == r'C1=C[C@H]2CC/C=C/CCCCCC[C@@H]1CC2'

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS019.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'C1CCCCC\C=C\C1'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'C1=C/CCCCCCC/1'

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS020.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'CCCCCCCCCC/C(=C(\C#N)/Br)/I'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'CCCCCCCCCC/C(I)=C(/Br)C#N'

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS034.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'Br[C@H]1[C@H](C[C@H](CC1)Br)Br'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'Br[C@H]1CC[C@@H](Br)[C@@H](Br)C1'

    data = read_molecules_from_sdf(get_data_file_path('stereo/VS035.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'O[C@@H]1CCCC[C@H]1[C@@H](F)Cl'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O[C@@H]1CCCC[C@H]1[C@@H](F)Cl'

    # this is non-stereo in rdkit, but marked P in original dataset
    data = read_molecules_from_sdf(get_data_file_path('stereo/VS055.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'BrC=1C(=C(C=CC1)C(=O)O)C=2C(=CC=CC2Br)C(=O)O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O=C(O)c1cccc(Br)c1-c1c(Br)cccc1C(=O)O'

    # this is non-stereo in rdkit, but marked M in original dataset
    data = read_molecules_from_sdf(get_data_file_path('stereo/VS057.sdf', 'bytemol.testdata'))
    mol = list(data.values())[0]
    origin_smiles = r'[O-][N+](C=1C(=C(C=CC1)C(=O)O)C2=C(C=CC=C2[N+]([O-])=O)C(=O)O)=O'
    assert mol.get_smiles(isomeric=True) == Molecule.from_smiles(origin_smiles).get_smiles(isomeric=True)
    assert mol.get_smiles(isomeric=True) == r'O=C(O)c1cccc([N+](=O)[O-])c1-c1c(C(=O)O)cccc1[N+](=O)[O-]'


def test_sf6():
    # initialize from smiles, S is correctly recognized
    # for rdkit >=2022.9, this test requires global switch Chem.SetAllowNontetrahedralChirality(False)
    mol = Molecule.from_smiles('[F]-[S](-[F])(-[F])(-[F])(-[F])-[F]')
    assert mol.get_smiles(isomeric=True) == 'FS(F)(F)(F)(F)F'
    assert mol.get_smiles(isomeric=False) == 'FS(F)(F)(F)(F)F'

    # initialize from xyz, S is incorrectly recognized in rdkit 2022.9.5
    mol = Molecule.from_sdf(get_data_file_path('sf6.sdf', 'bytemol.testdata'))
    assert mol.get_smiles(isomeric=True) == 'FS(F)(F)(F)(F)F'
    assert mol.get_smiles(isomeric=False) == 'FS(F)(F)(F)(F)F'
    assert mol.get_mapped_smiles(isomeric=True) == '[F:1][S:2]([F:3])([F:4])([F:5])([F:6])[F:7]'
    assert mol.get_mapped_smiles(isomeric=False) == '[F:1][S:2]([F:3])([F:4])([F:5])([F:6])[F:7]'


@pytest.mark.xfail()
def test_read_t1696sdf():
    mol = Molecule.from_sdf(get_data_file_path('T000001696.sdf', 'bytemol.testdata'))
    logger.info('%s', mol.get_smiles(isomeric=True))
    # chirality tag is not properly assigned
    assert mol.get_smiles(isomeric=True) == 'O=C(NC1CCCC1Cc1ccccc1)N[C@H]1C[C@@]2(C(=O)O)C[C@@H]12'


@pytest.mark.parametrize(
    "smi",
    [
        'CC',
        'CCC',
        'CCCC',
        'CCCCC',
        'CCCCCC',
        'CCCCCCC',
        'CCCCCCCC',
        'CCCCCCCCC',
    ],
)
def test_non_stereo(smi):
    nconfs = 1
    mol = Molecule.from_smiles(smi, nconfs=nconfs)
    logger.info('%s', mol.get_smiles())
    logger.info('%s', mol.get_mapped_smiles())

    mol2 = Molecule.from_mapped_smiles(mol.get_mapped_smiles(), nconfs=nconfs)
    logger.info('%s', mol2.get_smiles())
    logger.info('%s', mol2.get_mapped_smiles())

    assert '@' not in mol.get_smiles()
    assert '@' not in mol.get_mapped_smiles()
    assert '@' not in mol2.get_smiles()
    assert '@' not in mol2.get_mapped_smiles()


def test_redundant_cis_trans():
    mapped_smiles = "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]"
    mol = Molecule.from_mapped_smiles(mapped_smiles, nconfs=1)
    logger.info('%s', mol.get_smiles(isomeric=True))
    logger.info('%s', mol.get_mapped_smiles(isomeric=True))
    assert '/' not in mol.get_mapped_smiles(isomeric=True)
    assert '\\' not in mol.get_mapped_smiles(isomeric=True)
    assert '/' not in mol.get_smiles(isomeric=True)
    assert '\\' not in mol.get_smiles(isomeric=True)

    mol = Molecule.from_mapped_smiles(mapped_smiles, nconfs=0)
    logger.info('%s', mol.get_smiles(isomeric=True))
    logger.info('%s', mol.get_mapped_smiles(isomeric=True))
    assert '/' not in mol.get_mapped_smiles(isomeric=True)
    assert '\\' not in mol.get_mapped_smiles(isomeric=True)
    assert '/' not in mol.get_smiles(isomeric=True)
    assert '\\' not in mol.get_smiles(isomeric=True)

    mol = Molecule.from_smiles("[C:1](=[C:2]([H:5])[H:6])([H])[H]", nconfs=1)
    logger.info('%s', mol.get_smiles(isomeric=True))
    logger.info('%s', mol.get_mapped_smiles(isomeric=True))
    assert '/' not in mol.get_mapped_smiles(isomeric=True)
    assert '\\' not in mol.get_mapped_smiles(isomeric=True)
    assert '/' not in mol.get_smiles(isomeric=True)
    assert '\\' not in mol.get_smiles(isomeric=True)

    mol = Molecule.from_smiles("[C:1](=[C:2]([H:5])[H:6])([H])[H]", nconfs=0)
    logger.info('%s', mol.get_smiles(isomeric=True))
    logger.info('%s', mol.get_mapped_smiles(isomeric=True))
    assert '/' not in mol.get_mapped_smiles(isomeric=True)
    assert '\\' not in mol.get_mapped_smiles(isomeric=True)
    assert '/' not in mol.get_smiles(isomeric=True)
    assert '\\' not in mol.get_smiles(isomeric=True)


def test_stereo_resonance(tmp_path):

    smi = "[C:1]([N:2]1[C:3](=[O:4])[C@@:5]([C:6]([H:26])([H:27])[H:28])([c:7]2[c:8]([H:29])[c:9]([H:30])[c:10]([H:31])[c:11](-[c:12]3[c:13]([H:32])[c:14]([H:33])[c:15]([H:34])[c:16]([Cl:17])[c:18]3[H:35])[c:19]2[H:36])[N:20]([H:37])[C:21]1=[N+:22]([H:38])[H:39])([H:23])([H:24])[H:25]"
    mol = Molecule.from_mapped_smiles(smi, nconfs=1)
    rkmol = mol.get_rkmol()
    resonance_mols = rkutil.get_resonance_structures(rkmol)
    res_0_rkmol = resonance_mols[0]
    _ = Molecule(res_0_rkmol)

    with temporary_cd(tmp_path):
        sdf_file = "resform3.sdf"
        mol.to_sdf(sdf_file)
        mol = Molecule(sdf_file)


def test_cleanup_isotope_stereo():
    smiles = '[19F][13C@H]([16OH])[35Cl]'
    mol = Molecule.from_smiles(smiles)
    rkmol = rkutil.cleanup_rkmol_isotope(mol.get_rkmol())
    smi = Molecule.from_rdkit(rkmol).get_smiles()
    logger.info('%s %s', smiles, smi)
    assert smi == 'O[C@@H](F)Cl'

    smiles = '[3H][13C@H]([16OH])[35Cl]'
    mol = Molecule.from_smiles(smiles)
    rkmol = rkutil.cleanup_rkmol_isotope(mol.get_rkmol())
    smi = Molecule.from_rdkit(rkmol).get_smiles()
    logger.info('%s %s', smiles, smi)
    assert smi == 'OCCl'

    smiles = 'C[C@H]([13CH3])CI'
    mol = Molecule.from_smiles(smiles)
    rkmol = rkutil.cleanup_rkmol_isotope(mol.get_rkmol())
    smi = Molecule.from_rdkit(rkmol).get_smiles()
    logger.info('%s %s', smiles, smi)
    assert '@' not in smi
    assert smi == 'CC(C)CI'

    smiles = 'C/C(=C/CO)/[11CH3]'
    mol = Molecule.from_smiles(smiles)
    rkmol = rkutil.cleanup_rkmol_isotope(mol.get_rkmol())
    smi = Molecule.from_rdkit(rkmol).get_smiles()
    logger.info('%s %s', smiles, smi)
    assert '\\' not in smi
    assert '/' not in smi
    assert smi == 'CC(C)=CCO'
