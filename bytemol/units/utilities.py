# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from bytemol.utils import get_data_file_path


def get_defaults_path() -> str:
    """Get the full path to the ``defaults.txt`` file"""
    return get_data_file_path("defaults.txt", "bytemol.units")
