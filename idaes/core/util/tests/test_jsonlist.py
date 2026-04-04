#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Tests for jlist.JsonList module
"""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import idaes.core.util.jsonlist as jlist

# debug
jlist._log.setLevel(logging.DEBUG)


@pytest.fixture
def sample_jlist_empty():
    with TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        data_file = tmpdir / "sample.json"
        index_file = tmpdir / "sample-index.csv"
        yield jlist.JsonList(data_file, index_file)


# global, for test comparisons
line1 = '{"fake": 123}'
line2 = '{"fake": 456}'


@pytest.fixture
def sample_jlist_with_data():
    with TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        data_file = tmpdir / "sample.json"
        with open(data_file, "w") as f:
            f.write(f"{line1}\n{line2}\n")
        index_file = tmpdir / "sample-index.csv"
        with open(index_file, "w") as f:
            f.write("offset,hash,desc,tags\n")
            f.write("0,001,fake1,tag_1;tag_2\n")
            n = len(line1)
            f.write(f"{n},002,fake2,tag_1;tag_2\n")
        yield jlist.JsonList(data_file, index_file)


@pytest.mark.unit
def test_init_bad_arguments():
    with pytest.raises(TypeError):
        jlist.JsonList()


@pytest.mark.unit
def test_init_bad_files():
    with pytest.raises(jlist.BadIndexFile):
        jlist.JsonList("ignore.json", "/this/path/should/not/exist.csv")


@pytest.mark.unit
def test_init_ok(tmpdir):

    # 1..2 arguments
    for arg1 in (tmpdir / "foo.json", str(tmpdir / "foo.json")):
        for arg2 in (None, tmpdir / "bar.csv", str(tmpdir / "bar.csv")):
            jl = jlist.JsonList(arg1, arg2) if arg2 else jlist.JsonList(arg1)
            assert str(jl.data_file) == str(arg1)
            if arg2 is None:
                assert (
                    jl.index_file.name
                    == Path(arg1).stem + jlist.JsonList.INDEX_FILE_SUFFIX
                )
            else:
                assert str(jl.index_file) == str(arg2)


@pytest.mark.unit
def test_init_nonempty(sample_jlist_with_data):
    j1 = sample_jlist_with_data
    j2 = jlist.JsonList(j1.data_file, j1.index_file)
    assert len(j2) == len(j1)


@pytest.mark.unit
def test_append(sample_jlist_empty):
    jl = sample_jlist_empty
    data = {"test": 123}
    # populate
    count = 0
    for meta in (
        {},
        {"hash": "123"},
        {"hash": "123", "desc": "hello"},
        {"hash": "123", "desc": "hello", "tags": ["a"]},
        {"hash": "123", "desc": "hello", "tags": ["a", "to be or not to b$$"]},
        {"hash": "123", "desc": "hello,\nworld", "tags": ["a", "to be or not to b$$"]},
    ):
        for doc in (data, json.dumps(data)):
            jl.append(doc, **meta)
            count += 1
    # check contents
    assert len(jl) == count, f"{len(jl)} != {count}"


@pytest.mark.unit
def test_get_index(sample_jlist_with_data):
    jl = sample_jlist_with_data
    for i, row in enumerate(jl.metadata):
        print(f"row: {row}")
        if i == 0:
            assert row["hash"] == "001"
            assert row["desc"] == "fake1"
        else:
            assert row["hash"] == "002"
            assert row["desc"] == "fake2"
        assert row["tags"] == ["tag_1", "tag_2"]


@pytest.mark.unit
def test_get_doc_bad_index(sample_jlist_with_data):
    jl = sample_jlist_with_data
    for i in -2, 2, 100:
        with pytest.raises(IndexError):
            jl[i]


@pytest.mark.unit
def test_get_doc(sample_jlist_with_data):
    jl = sample_jlist_with_data
    doc = jl[0]
    assert doc == json.loads(line1)
    doc = jl[1]
    assert doc == json.loads(line2)
    doc = jl[-1]
    assert doc == json.loads(line2)
