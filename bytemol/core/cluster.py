# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Optional

import numpy as np

from bytemol.core.conformer import Conformer
from bytemol.core.molecule import Molecule

logger = logging.getLogger(__name__)


class Cluster:
    """ Handling cluster data, composed of a list of Molecule and cluster informations. """

    def __init__(self, mols: list[Molecule], data: Optional[dict] = None, name: Optional[str] = None):

        for mol in mols:
            assert isinstance(mol, Molecule)
        assert len(mols) >= 1
        if len(mols) > 1:
            confkeys = set(mols[0].conformers[0].confdata)
            for mol in mols[1:]:
                assert set(mol.conformers[0].confdata) == confkeys
                for k in confkeys:
                    if len(mols[0].conformers[0].confdata[k].shape) > 1:
                        assert mols[0].conformers[0].confdata[k].shape[1:] == mol.conformers[0].confdata[k].shape[1:]

        self.mols: list[Molecule] = mols.copy()
        self.name = name if name is not None else '_'.join(mol.name for mol in mols)

        if data is not None:
            assert isinstance(data, dict)
            self.data = data.copy()
        else:
            self.data = {}

    @property
    def nmols(self):
        return len(self.mols)

    @property
    def confdata_keys(self):
        return list(self.mols[0].conformers[0].confdata)

    def get_confdata(self, confdata_key: str, conf_id: int = 0, as_list=False):
        data = [np.array(mol.conformers[conf_id].confdata[confdata_key]) for mol in self.mols]
        data = np.concatenate(data, axis=0)
        if as_list:
            data = data.tolist()
        return data

    def get_coords(self, conf_id: int = 0, as_list=False):
        data = [np.array(mol.conformers[conf_id].coords) for mol in self.mols]
        data = np.concatenate(data, axis=0)
        if as_list:
            data = data.tolist()
        return data

    def set_coords(self, coords, conf_id: int = 0):
        coords = np.array(coords)
        assert coords.shape == (sum([mol.natoms for mol in self.mols]), 3)
        shift = 0
        for mol in self.mols:
            mol.conformers[conf_id].coords = coords[shift:shift + mol.natoms]
            shift += mol.natoms

    def to_xyz_json(self, save_folder: str):
        """ xyz+json files contain multiple conformations.
        json format {
            'cluster_name': self.name,
            'files': list[str],
        }
        """
        if os.path.exists(save_folder):
            assert os.path.isdir(save_folder)
            logger.warning(f'Folder {save_folder} already exists!')
        else:
            os.makedirs(save_folder)

        data = {'cluster_name': self.name, 'files': [f'{self.mols[i].name}_{i}.xyz' for i in range(self.nmols)]}
        data.update(self.data)

        with open(os.path.join(save_folder, 'info.json'), 'w') as file:
            json.dump(data, file, indent=2)

        for mol, name in zip(self.mols, data['files']):
            mol.to_xyz(os.path.join(save_folder, name))

    def to_json(self, json_fp: str, name_as_key=False, conf_id=0):
        """ json file contains only one conformation for now.
        json format {
            'cluster_name': self.name,
            'molecule_names': list[str],
            'mapped_smiles': list[str],
            'structure': list[list[float]],
            'element': list[str],
            'num_atoms': int,
        }
        """

        data = {'molecule_names': [], 'mapped_smiles': [], 'structure': [], 'element': [], 'num_atoms': 0}
        for mol in self.mols:
            data['molecule_names'].append(mol.name)
            data['mapped_smiles'].append(mol.get_mapped_smiles())
            data['element'] += mol.atomic_symbols
            data['num_atoms'] += mol.natoms

        for k in self.confdata_keys:
            confdata = self.get_confdata(k, conf_id, as_list=True)
            if k == 'coords':
                data['structure'] = confdata
            else:
                data[k] = confdata

        data.update(self.data)

        if name_as_key:
            cluster_data = {self.name: data}
        else:
            cluster_data = data
            cluster_data['cluster_name'] = self.name

        with open(json_fp, 'w') as file:
            json.dump(cluster_data, file)

    @classmethod
    def from_json(cls,
                  json_fp: str,
                  mapped_smiles: Optional[list[str]] = None,
                  confdata_keys: Optional[list[str]] = None,
                  name_as_key=False) -> "Cluster":
        """ json format or xyz+json format """

        with open(json_fp) as file:
            data: dict = json.load(file)
        if name_as_key:
            assert len(data) == 1
            cluster_name, data = data.popitem()
        else:
            cluster_name = data.pop('cluster_name')

        mols: list[Molecule] = []

        # xyz + json
        if 'files' in data:
            files = data.pop('files')
            folder = os.path.dirname(os.path.abspath(json_fp))
            for fname in files:
                mols.append(Molecule.from_xyz(os.path.join(folder, fname)))

        # json
        else:
            mol_names = data.pop('molecule_names')
            if mapped_smiles is None:
                if 'mapped_smiles' in data:
                    mapped_smiles = data.pop('mapped_smiles')
                else:
                    raise RuntimeError('No mapped smiles found')

            element = data.pop('element')
            num_atoms = data.pop('num_atoms')
            begin = 0
            for mol_name, mps in zip(mol_names, mapped_smiles):
                mol = Molecule.from_mapped_smiles(mps, name=mol_name)
                assert element[
                    begin:begin +
                    mol.natoms] == mol.atomic_symbols, f"{element[begin:begin + mol.natoms]} != {mol.atomic_symbols}"
                begin += mol.natoms
                mols.append(mol)

            natom_list = [mol.natoms for mol in mols]
            assert sum(natom_list) == num_atoms

            conf_keys = {'structure'}
            if confdata_keys is not None:
                conf_keys |= set(confdata_keys)

            confdata = dict()
            for k in conf_keys:
                confdata[k] = np.array(data.pop(k))
                assert confdata[k].shape[0] == num_atoms

            begin = 0
            for mol in mols:
                confdata_m = {k: v[begin:begin + mol.natoms] for k, v in confdata.items()}
                coords = confdata_m.pop('structure')
                conf = Conformer(coords=coords, symbols=mol.atomic_symbols, confdata=confdata_m)
                mol.append_conformers(conf)
                begin += mol.natoms

        cluster = cls(mols=mols, data=data, name=cluster_name)
        return cluster
