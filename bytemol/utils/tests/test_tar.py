# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import random
from collections import OrderedDict
from itertools import chain
from uuid import uuid4

import numpy as np

from bytemol.utils.tar import MultiTarFileReader, TarFile


class TestTarFile:
    FILES = [f'test{i}.tar' for i in range(5)]
    files = []

    @staticmethod
    def gen_random_tar(tarfn):
        d = OrderedDict()
        with TarFile(tarfn, 'w') as wfp:
            for _ in range(10):
                fn = uuid4().hex
                content = np.random.bytes(1024)
                d[fn] = content
                wfp.write(content, fn)
        return d

    @classmethod
    def setup_class(cls):
        cls.files = [(tarfn, cls.gen_random_tar(tarfn)) for tarfn in cls.FILES]

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to
        setup_class.
        """
        for f in cls.FILES:
            os.remove(f)

    def test_single_tar_random_read(self):
        fn, d = self.files[0]
        fp = TarFile(fn)
        assert len(fp) == len(d)
        for m in fp.members:
            assert fp[m.name] == d[m.name]

    def test_single_tar_sequential_read(self):
        fn, d = self.files[0]
        fp = TarFile(fn)
        assert len(fp) == len(d)
        for c, v in zip(fp, d.values()):
            assert c == v

    def test_multiple_tar_random_read(self):
        fp = MultiTarFileReader([f[0] for f in self.files])
        assert len(fp) == sum(len(d) for _, d in self.files)
        for name, d in self.files:
            for k, v in d.items():
                assert fp[name, k] == v
                # we can expect uuids cannot be the same
                assert fp[k] == v

    def test_multiple_tar_sequential_read(self):
        fp = MultiTarFileReader([f[0] for f in self.files])
        assert len(fp) == sum(len(d) for _, d in self.files)
        for c, v in zip(fp, chain.from_iterable(d.values() for _, d in self.files)):
            assert c == v

    def test_single_tar_filter(self):
        fn, d = self.files[0]
        file_filter = [v for _, v in sorted(random.sample(list(enumerate(d.keys())), k=5))]
        fp = TarFile(fn, file_filter=file_filter)
        assert len(fp) == len(file_filter)
        for c, v in zip(fp, (d[k] for k in file_filter)):
            assert c == v

    def test_multiple_tar_filter(self):
        filters = {}
        for fn, d in self.files:
            file_filter = [v for _, v in sorted(random.sample(list(enumerate(d.keys())), k=5))]
            filters[fn] = file_filter

        fp = MultiTarFileReader([f[0] for f in self.files], filters=filters)
        assert len(fp) == 5 * len(self.files)
        for c, v in zip(fp, [d[i] for f, d in self.files for i in filters[f]]):
            assert c == v

    def test_write_tar_no_key(self):
        with TarFile('test_no_key.tar', 'w') as wfp:
            for _ in range(10):
                content = np.random.bytes(1024)
                wfp.write(content)
        with TarFile('test_no_key.tar') as fp:
            assert fp['8'] == fp[8]
        os.remove('test_no_key.tar')
