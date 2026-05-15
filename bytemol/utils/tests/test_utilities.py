# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from bytemol.utils.utilities import get_data_file_path, run_command_and_check, split_array_evenly, temporary_cd


def compare_paths(path_1: str, path_2: str) -> bool:
    """Checks whether two paths are the same.

    Parameters
    ----------
    path_1
        The first path.
    path_2
        The second path.

    Returns
    -------
    True if the paths are equivalent.
    """
    return os.path.normpath(path_1) == os.path.normpath(path_2)


def test_get_data_file_path():
    """Tests that the `get_data_file_path` can correctly find
    data files.
    """

    # Test a path which should exist.
    data_file_path = get_data_file_path("data.dat", package_name="bytemol.utils")
    assert os.path.isfile(data_file_path)

    # Ensure a double-checking through data/ takes place
    data_file_path = get_data_file_path("data/data.dat", package_name="bytemol.utils")
    assert os.path.isfile(data_file_path)

    # Test a path which should not exist.
    with pytest.raises(FileNotFoundError):
        get_data_file_path("missing.file", package_name="bytemol.utils")


def test_temporary_cd():
    """Tests that temporary cd works as expected"""

    original_directory = os.getcwd()

    # Move to the parent directory
    with temporary_cd(os.pardir):

        current_directory = os.getcwd()
        expected_directory = os.path.abspath(os.path.join(original_directory, os.pardir))

        assert compare_paths(current_directory, expected_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to a temporary directory
    with temporary_cd():
        assert not compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to the same directory
    with temporary_cd(""):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    with temporary_cd(os.curdir):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)


def test_run_command_and_check(tmp_path):
    with temporary_cd(tmp_path):
        retcode, stdout, stderr = run_command_and_check('gcc', allow_error=True, separate_stderr=True)
        assert 'gcc: fatal error: no input files' in stderr
        assert 'compilation terminated.' in stderr
        assert len(stdout) == 0
        assert retcode != 0

    with temporary_cd(tmp_path):
        retcode, stdout, stderr = run_command_and_check('gcc --version', allow_error=False, separate_stderr=False)
        assert retcode == 0
        assert stderr is None

    retcode, stdout, stderr = run_command_and_check('env | grep MYENV', env={'MYENV': 'TEST'})
    assert stdout == 'MYENV=TEST\n'


class TestSplitJobs:

    @pytest.fixture
    def job_list(self):
        return [1, 2, 3, 4, 5, 6, 7]

    def test_even_split(self, job_list):
        num_workers = 2
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[1, 2, 3, 4], [5, 6, 7]]
        assert result == expected_result

    def test_uneven_split(self, job_list):
        num_workers = 3
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[1, 2, 3], [4, 5], [6, 7]]
        assert result == expected_result

    def test_one_job_per_worker(self, job_list):
        num_workers = len(job_list)
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[1], [2], [3], [4], [5], [6], [7]]
        assert result == expected_result

    def test_more_workers_than_jobs(self, job_list):
        num_workers = 10
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[1], [2], [3], [4], [5], [6], [7], [], [], []]
        assert result == expected_result

    def test_empty_job_list(self):
        job_list = []
        num_workers = 5
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[], [], [], [], []]
        assert result == expected_result

    def test_one_worker(self):
        job_list = [1, 2, 3, 4, 5]
        num_workers = 1
        result = split_array_evenly(job_list, num_workers)
        expected_result = [[1, 2, 3, 4, 5]]
        assert result == expected_result
