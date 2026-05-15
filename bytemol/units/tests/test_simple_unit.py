# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from bytemol.units import simple_unit as unit

logger = logging.getLogger(__name__)


class TestUnit:

    def test_energy_unit(self):
        # 0.037 Hartree = 1 eV = 23 kca/mol = 96 kJ/mol
        Hartree = 0.03674932247
        eV = 1
        kcal_mol = 23.0605
        kJ_mol = 96.48533
        assert np.isclose(unit.eV_to_kJ_mol(1), kJ_mol / eV)
        assert np.isclose(unit.eV_to_kcal_mol(1), kcal_mol / eV)
        assert np.isclose(unit.eV_to_Hartree(1), Hartree / eV)
        assert np.isclose(unit.kJ_mol_to_kcal_mol(1), kcal_mol / kJ_mol)
        assert np.isclose(unit.kcal_mol_to_kJ_mol(1), kJ_mol / kcal_mol)
        assert np.isclose(unit.kcal_mol_to_eV(1), eV / kcal_mol)
        assert np.isclose(unit.Hartree_to_eV(1), eV / Hartree)
        assert np.isclose(unit.Hartree_to_kcal_mol(1), kcal_mol / Hartree)

    def test_force_unit(self):
        # 1 eV/A = 23 kcal/mol/A = 965 kJ/mol/nm
        eV_A = 1
        kcal_mol_A = 23.0605
        kJ_mol_nm = 964.8533
        assert np.isclose(unit.eV_A_to_kJ_mol_nm(1), kJ_mol_nm / eV_A)
        assert np.isclose(unit.eV_A_to_kcal_mol_A(1), kcal_mol_A / eV_A)
        assert np.isclose(unit.kJ_mol_nm_to_kcal_mol_A(1), kcal_mol_A / kJ_mol_nm)
        assert np.isclose(unit.kcal_mol_A_to_eV_A(1), eV_A / kcal_mol_A)

    def test_distance_unit(self):
        # 1 Angstrom = 0.1 nm = 1.9 Bohr
        Angstrom = 1
        nm = 0.1
        Bohr = 1.889726
        assert np.isclose(unit.A_to_nm(1), nm / Angstrom)
        assert np.isclose(unit.nm_to_A(1), Angstrom / nm)
        assert np.isclose(unit.Bohr_to_A(1), Angstrom / Bohr)
        assert np.isclose(unit.A_to_Bohr(1), Bohr / Angstrom)

    def test_bond_k_unit(self):
        # 1 kcal/mol/A^2 = 418 kJ/mol/nm^2
        kcal_mol_A2 = 1
        kJ_mol_nm2 = 418.4
        assert np.isclose(unit.kJ_mol_nm2_to_kcal_mol_A2(1), kcal_mol_A2 / kJ_mol_nm2)
        assert np.isclose(unit.kcal_mol_A2_to_kJ_mol_nm2(1), kJ_mol_nm2 / kcal_mol_A2)
