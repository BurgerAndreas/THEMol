# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


class MissingOpenMMUnitError(Exception):
    """Raised when a unit cannot be converted to an equivalent OpenMM unit"""


class NoneQuantityError(Exception):
    """Raised when attempting to convert `None` between unit packages as a quantity object"""


class NoneUnitError(Exception):
    """Raised when attempting to convert `None` between unit packages as a unit object"""
