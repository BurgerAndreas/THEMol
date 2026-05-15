# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


def _cast_string(types, word: str):
    if isinstance(types, tuple):
        words = word.split()
        assert len(words) == len(types)
        return tuple(_cast_string(t, l) for (t, l) in zip(types, words))
    elif types == str:
        return word
    else:
        return types(word)


def _read_stream(types, arg: Any, fill: Any, prompt: str, invalid: Callable, istream=sys.stdin):
    assert isinstance(types, tuple) or types in (int, float, str, str.upper, str.lower)
    if arg is None:
        input_fail = 1
    elif arg == "\n":
        if fill is None:
            input_fail = 1
        else:
            arg = fill
            input_fail, arg = invalid(arg)
    else:
        input_fail, arg = invalid(arg)
    ostream = sys.stdout
    while input_fail:
        ostream.write(prompt)
        ostream.flush()

        line = istream.readline()
        line = line.strip()
        words = line.split()
        if len(words) == 0:
            arg = fill
        else:
            arg = _cast_string(types, line)
        if arg is None:
            input_fail = 1
        else:
            input_fail, arg = invalid(arg)
    return arg


def read_argv(types, fill: Any, prompt: str, invalid: Callable, argv: List[str]):
    exist = len(argv)
    if exist:
        word = argv.pop(0)
        arg = _cast_string(types=types, word=word)
    else:
        arg = None

    return _read_stream(types=types, arg=arg, fill=fill, prompt=prompt, invalid=invalid)
