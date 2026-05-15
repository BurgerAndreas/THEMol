# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from typing import Union

logger = logging.getLogger(__name__)


def _get_stem_version(src: str, ext: str):
    if ext == "" or "." not in src:
        # case 1
        if os.path.isfile(src):
            stem, ext = os.path.splitext(src)
        else:
            stem = src
    elif len(ext) >= 2 and ext[0] == "." and src.endswith(ext):
        # case 2
        stem = src[:-len(ext)]
    elif src[-1] != "." and "." in src:
        filename = f"{src}{ext}"
        if os.path.isfile(filename):
            # case 4
            stem, ext = os.path.splitext(filename)
        else:
            # case 3
            stem, ext = os.path.splitext(src)
    else:
        msg = f"Cannot process src=\"{src}\" and ext=\"{ext}\""
        raise ValueError(msg)

    match = re.search(r"\._(\d+)$", stem)
    if match:
        vernum = match.group(1)
        stem = os.path.splitext(stem)[0]
    else:
        vernum = "0"
    vernum = int(vernum)

    return stem, vernum


def _suggest_filename(stem: str, ext: str, vernum: int, wd: Union[int, str]):
    assert vernum >= 0
    stem1 = stem
    if isinstance(wd, int) and wd == 0:
        stem = stem1
    elif isinstance(wd, str):
        basename = os.path.basename(stem1)
        if wd in ("", "./"):
            stem = basename
        else:
            stem = f"{wd}/{basename}"
    else:
        assert False
    if vernum == 0:
        return f"{stem}{ext}"
    else:
        return f"{stem}._{vernum}{ext}"


def suggest_new_filename(src: str, oldext: str, newext: str, wd: Union[int, str] = "", overwrite: bool = True):
    stem, vernum = _get_stem_version(src, oldext)

    while True:
        ver0 = vernum
        ver1 = ver0 + 1
        vernum += 1

        cdd0 = _suggest_filename(stem, newext, ver0, wd)
        cdd1 = _suggest_filename(stem, newext, ver1, wd)
        if not os.path.isfile(cdd0):
            return cdd0
        elif not os.path.isfile(cdd1):
            return cdd1
        else:
            if overwrite:
                logger.warning(msg=f"{cdd1} already exists and will be overwritten.")
                return cdd1
            continue


def suggest_old_filename(src: str, oldext: str, newext, wd: Union[int, str] = ""):
    if isinstance(newext, tuple):
        filename = None
        for ne in newext:
            filename = suggest_old_filename(src, oldext, ne, wd)
            if filename is not None:
                return filename
        return filename

    stem, vernum = _get_stem_version(src, oldext)

    cdd1 = _suggest_filename(stem, newext, vernum, wd)
    if os.path.isfile(cdd1):
        if vernum > 0:
            # file._2.xyz
            # case 11
            return cdd1
        else:
            # from file.xyz, find the latest file._N.xyz
            while True:
                cdd2 = cdd1
                vernum += 1
                cdd1 = _suggest_filename(stem, newext, vernum, wd)
                if not os.path.isfile(cdd1):
                    # case 12
                    return cdd2
    elif vernum == 0:
        # cannot find file.xyz
        _msg = f"Cannot find file {cdd1}"
        return None
    else:
        # cannot find file._2.xyz, try to use file.xyz as a fallback
        cdd0 = _suggest_filename(stem, newext, 0, wd)
        if os.path.isfile(cdd0):
            # case 13
            return cdd0
        else:
            _msg = f"Cannot find {cdd0} or {cdd1}"
            return None
