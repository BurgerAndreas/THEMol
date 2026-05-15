# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# DO NOT import .pint_unit or .openmm here. otherwise openmm is imported everywhere in bytemol and causes conflicts.
# do this in user code:
# from bytemol.units.pint_unit import Quantity, unit
