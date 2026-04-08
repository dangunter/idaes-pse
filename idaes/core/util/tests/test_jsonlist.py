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

from collections import namedtuple
import json
import logging
from pathlib import Path
from random import random, randint
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import pytest

import idaes.core.util.jsonlist as jlist

# debug
jlist._log.setLevel(logging.DEBUG)

# Fixtures
# --------


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
        ts = time.time()
        with open(index_file, "w") as f:
            f.write("offset,timestamp,hash,desc,tags,ext\n")
            f.write(f"0,{ts},001,fake1,tag_1;tag_2,{{}}\n")
            n = len(line1)
            f.write(f'{n},{ts + 1},002,fake2,tag_1;tag_2,"{{}}"\n')
        yield jlist.JsonList(data_file, index_file)


@pytest.fixture
def big_obj():
    """For the performance test"""
    return _create_obj(20)


@pytest.fixture
def medium_obj():
    return _create_obj(5)


def _create_obj(n):
    def random_key():
        return str(uuid4())

    def random_value():
        return random() * 1e6

    obj = {}
    for i in range(n):
        d = []
        for j in range(n):
            e = {}
            for k in range(n):
                e[random_key()] = random_value()
            d.append(e)
        obj[random_key()] = d

    return {"object": obj}


# Unit tests
# ----------


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
        {"file_hash": "123"},
        {"file_hash": "123", "desc": "hello"},
        {"file_hash": "123", "desc": "hello", "tags": ["a"]},
        {"file_hash": "123", "desc": "hello", "tags": ["a", "to be or not to b$$"]},
        {
            "file_hash": "123",
            "desc": "hello,\nworld",
            "tags": ["a", "to be or not to b$$"],
        },
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


@pytest.mark.unit
def test_user_metadata(tmpdir):
    data_file = tmpdir / "extra-columns.json"
    long_text = """Early one morning the sun was shining
I was laying in bed
Wondering if she'd changed at all
If her hair was still red"""
    user_data = {
        "values": [{"x": 1.0, "y": 100.0, "text": long_text}, {"x": 2.0, "y": 200.0}]
    }
    jl = jlist.JsonList(data_file)
    jl.append({"hello": "world"}, ext=user_data)
    # close and reopen
    jl = jlist.JsonList(data_file)
    # now check from saved
    for item in jl.metadata:
        assert item["ext"] == user_data


@pytest.mark.unit
def test_delete(tmpdir, medium_obj):
    data_file = tmpdir / "delete.json"
    jl = jlist.JsonList(data_file)

    # add some objects, adding a number 'i'
    for i in range(15):
        medium_obj["i"] = i
        jl.append(medium_obj, ext={"index": i})

    # delete first 5
    jl.delete(start=0, num=5)
    # check that there are 10 now
    assert len(jl) == 10
    # check that they have i == 5..14
    # since 0..4 were deleted
    meta = list(jl.metadata)
    for i in range(10):
        print(f"check1: metadata[{i}] = {meta[i]}")
        print(f"check1: jl[{i}]['i'] == {jl[i]['i']}")
        assert jl[i]["i"] == i + 5
        assert meta[i]["ext"]["index"] == i + 5

    # delete last 5
    jl.delete(start=5, num=5)
    # check that there are 5 now
    assert len(jl) == 5
    # check that they have i == 5..9
    meta = list(jl.metadata)
    for i in range(5):
        print(f"check2: metadata[{i}] = {meta[i]}")
        print(f"check2: jl[{i}]['i'] == {jl[i]['i']}")
        assert jl[i]["i"] == i + 5
        assert meta[i]["ext"]["index"] == i + 5

    # delete third
    jl.delete(start=2, num=1)
    # check that there are 4 now
    assert len(jl) == 4
    # check that they have i == 5, 6, 8, 9
    meta = list(jl.metadata)
    assert jl[0]["i"] == 5
    assert meta[0]["ext"]["index"] == 5
    assert jl[1]["i"] == 6
    assert meta[1]["ext"]["index"] == 6
    assert jl[2]["i"] == 8
    assert meta[2]["ext"]["index"] == 8
    assert jl[3]["i"] == 9
    assert meta[3]["ext"]["index"] == 9


@pytest.mark.unit
def test_delete_bad_args(sample_jlist_empty):
    jl = sample_jlist_empty

    for bad_start in (-1, len(jl)):
        with pytest.raises(ValueError):
            jl.delete(start=bad_start)

    with pytest.raises(ValueError):
        jl.delete(start=0)

    for bad_start, bad_num in ((0, len(jl) + 1), (1, len(jl))):
        with pytest.raises(ValueError):
            jl.delete(start=bad_start, num=bad_num)


# Integration tests
# -----------------


def timing_summary(title, times, ms=False):
    num = len(times)
    t_tot = sum(times)
    t_mean = t_tot / num
    t_min, t_min_idx, t_max, t_max_idx = 1e6, -1, -1, -1
    for idx, t in enumerate(times):
        if t < t_min:
            t_min, t_min_idx = t, idx
        if t > t_max:
            t_max, t_max_idx = t, idx
    if ms:
        t_tot *= 1000
        t_mean *= 1000
        t_min *= 1000
        t_max *= 1000
        units = "ms"
    else:
        units = "s"
    print(
        f"{title} {num}: total={t_tot:.3f}{units} average={t_mean:.3f}{units} "
        f"min({t_min_idx})={t_min:.3f}{units} max({t_max_idx})={t_max:.3f}{units}"
    )


@pytest.mark.integration
def test_perf(big_obj, sample_jlist_empty):
    big_obj_str = json.dumps(big_obj)
    size = len(big_obj_str)
    num = 20

    print(f"Creating JsonList with {num} objects of size {size} bytes")

    jl = sample_jlist_empty

    times = []
    for i in range(num):
        t0 = time.time()
        jl.append(
            big_obj_str, file_hash="hello", desc="hello, world", tags=["perf", str(i)]
        )
        t1 = time.time()
        times.append(t1 - t0)
    timing_summary("Append", times, ms=True)

    times = []
    tries = 10
    print(f"Deserialize {tries} objects")
    for i in range(tries):
        idx = randint(0, num - 1)
        t0 = time.time()
        obj = json.loads(big_obj_str)
        t1 = time.time()
        times.append(t1 - t0)
    timing_summary("Deserialize", times, ms=True)
    print("Will subtract min deserialization time from retrieve time")
    d_time = min(times)

    times = []
    tries = 100
    print(f"Retrieve {tries} objects")
    for i in range(tries):
        idx = randint(0, num - 1)
        t0 = time.time()
        obj = jl[idx]
        t1 = time.time()
        times.append(t1 - t0 - d_time)
    timing_summary("Retrieve (minus min(deserialize))", times, ms=True)


@pytest.mark.integration
def test_with_structfs_report(sample_jlist_empty):
    from idaes.core.util.structfs.tests import flash_flowsheet

    # run twice
    for i in range(2):
        flash_flowsheet.FS.run_steps()
        rpt = flash_flowsheet.FS.report()
        jl = sample_jlist_empty
        jl.append(rpt, desc="Flash flowsheet for test")
