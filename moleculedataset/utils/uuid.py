# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import uuid


def deterministic_uuid(mapped_isomeric_smiles: str) -> str:
    namespace = uuid.UUID(int=0)
    return uuid.uuid5(namespace, mapped_isomeric_smiles).hex
