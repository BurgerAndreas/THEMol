# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from bytemol.utils import run_command_and_check, suggest_new_filename, suggest_old_filename, temporary_cd
from bytemol.utils.ixinput import _cast_string
from bytemol.utils.suggest_filename import _get_stem_version


def test_cast_string():
    assert 42 == _cast_string(int, "42")
    assert -42.25 == _cast_string(float, "-42.25")
    assert " -42.25 " == _cast_string(str, " -42.25 ")
    assert " all lower " == _cast_string(str.lower, " ALL LOWER ")
    assert " ALL UPPER " == _cast_string(str.upper, " all upper ")
    assert (42, -42.25, "-42.25", "all_lower", "ALL_UPPER") == _cast_string((int, float, str, str.lower, str.upper),
                                                                            ("42 -42.25 -42.25 ALL_LOWER all_upper"))


def test_get_stem_version(tmp_path):
    with temporary_cd(tmp_path):
        run_command_and_check(r"mkdir -p test_files/")
        run_command_and_check(r"touch test_files/zz._2.xyz")

        # case 1
        src, ext = "yy", ""
        stem, ver = "yy", 0
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 1
        src, ext = "yy", ".xyz"
        stem, ver = "yy", 0
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 1
        src, ext = "zz._2.mbis.npz", ""
        stem, ver = "zz._2.mbis.npz", 0
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 2
        src, ext = "zz._2.mbis.npz", ".npz"
        stem, ver = "zz._2.mbis", 0
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 2
        src, ext = "zz._2.mbis.npz", ".mbis.npz"
        stem, ver = "zz", 2
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 2
        src, ext = "zz._2.any.npz", ".any.npz"
        stem, ver = "zz", 2
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 3
        src, ext = "yy.xyz", ".*"
        stem, ver = "yy", 0
        assert _get_stem_version(src, ext) == (stem, ver)

        # case 4
        src, ext = "test_files/zz._2", ".xyz"
        stem, ver = "test_files/zz", 2
        assert _get_stem_version(src, ext) == (stem, ver)


def test_suggest_new_filename(tmp_path):
    with temporary_cd(tmp_path):
        run_command_and_check(r"mkdir -p test_files")
        run_command_and_check(r"touch test_files/zz.xyz")
        run_command_and_check(r"touch test_files/zz._1.xyz")
        run_command_and_check(r"touch test_files/zz._2.xyz")
        run_command_and_check(r"touch test_files/zz.itp")
        run_command_and_check(r"touch test_files/zz._1.itp")
        run_command_and_check(r"touch test_files/zz._2.itp")

        wd = 0

        src, oldext, newext = "test_files/zz", ".xyz", ".itp"
        ref = "test_files/zz._3.itp"
        assert suggest_new_filename(src, oldext, newext, wd, overwrite=False) == ref

        src, oldext, newext = "test_files/zz._1", ".xyz", ".itp"
        ref = "test_files/zz._3.itp"
        assert suggest_new_filename(src, oldext, newext, wd, overwrite=False) == ref

        src, oldext, newext = "test_files/zz._2.xyz", ".*", ".itp"
        ref = "test_files/zz._3.itp"
        assert suggest_new_filename(src, oldext, newext, wd, overwrite=False) == ref

        src, oldext, newext = "test_files/zz._2.xyz", "", ".itp"
        ref = "test_files/zz._3.itp"
        assert suggest_new_filename(src, oldext, newext, wd, overwrite=False) == ref


def test_suggest_old_filename(tmp_path):
    with temporary_cd(tmp_path):
        run_command_and_check(r"mkdir -p test_files")
        run_command_and_check(r"touch test_files/zz.xyz")
        run_command_and_check(r"touch test_files/zz.itp")
        run_command_and_check(r"touch test_files/zz._1.itp")
        run_command_and_check(r"touch test_files/zz.mbis.npz")

        wd = 0

        # case 11
        src, oldext, newext = "test_files/zz._1", ".xyz", ".itp"
        ref = "test_files/zz._1.itp"
        assert suggest_old_filename(src, oldext, newext, wd) == ref

        # case 12
        src, oldext, newext = "test_files/zz", ".xyz", ".mbis.npz"
        ref = "test_files/zz.mbis.npz"
        assert suggest_old_filename(src, oldext, newext, wd) == ref

        # case 13
        src, oldext, newext = "test_files/zz._2", ".xyz", ".mbis.npz"
        ref = "test_files/zz.mbis.npz"
        assert suggest_old_filename(src, oldext, newext, wd) == ref
