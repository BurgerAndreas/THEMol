# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from bytemol.core import Cluster
from bytemol.utils import get_data_file_path, temporary_cd

cluster_name = '3EC_4LI_4FSI_2PF6_6ce0ab5a'
EC_mps = '[O:1]1[C:2]([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][C:9]1=[O:10]'
LI_mps = '[Li+:1]'
FSI_mps = '[N-:1]([S+:2]([F:3])([O-:4])=[O:5])[S+:6]([O-:7])(=[O:8])[F:9]'
PF6_mps = '[P-:1]([F:2])([F:3])([F:4])([F:5])([F:6])[F:7]'
mapped_smiles = [EC_mps] * 3 + [LI_mps] * 4 + [FSI_mps] * 4 + [PF6_mps] * 2
json_fp = get_data_file_path(f'cluster/{cluster_name}.json', 'bytemol.testdata')


def test_cluster_json():

    with open(json_fp) as file:
        raw_data = json.load(file)[cluster_name]

    cluster = Cluster.from_json(json_fp, confdata_keys=['gradient'], mapped_smiles=mapped_smiles, name_as_key=True)
    assert 'clean' in cluster.data

    coords = cluster.get_coords()
    gradient = cluster.get_confdata('gradient')
    assert np.allclose(coords, np.array(raw_data['structure']))
    assert np.allclose(gradient, np.array(raw_data['gradient']))

    with temporary_cd():
        tmp_path = './test.json'
        cluster.to_json(tmp_path, name_as_key=True)
        with open(tmp_path) as file:
            converted_data = json.load(file)[cluster_name]

    for k, v in raw_data.items():
        assert k in converted_data
        if k in ['structure', 'gradient']:
            assert np.allclose(np.array(v), np.array(converted_data[k]))
        else:
            assert v == converted_data[k], k


def test_cluster_xyz_json():

    with open(json_fp) as file:
        raw_data = json.load(file)[cluster_name]

    cluster = Cluster.from_json(json_fp, confdata_keys=['gradient'], mapped_smiles=mapped_smiles, name_as_key=True)
    with temporary_cd():
        cluster.to_xyz_json('./test')
        new_cluster = Cluster.from_json('./test/info.json')
        new_json = 'test.json'
        new_cluster.to_json(new_json)
        with open(new_json) as file:
            converted_data = json.load(file)

    for k, v in raw_data.items():
        assert k in converted_data
        if k in ['structure', 'gradient']:
            assert np.allclose(np.array(v), np.array(converted_data[k]))
        else:
            assert v == converted_data[k], k
