"""
This module has common classes, definitions, and functions for
IDAES model I/O (serialization).
"""

# stdlib
from __future__ import annotations
from collections import namedtuple
from io import IOBase
import logging
import re
from statistics import median
from typing import Any, Iterable, Optional
import time

# third-party
from google.protobuf import json_format
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.component import ComponentData
from pyomo.core.base.param import ParamData, IndexedParam
from pyomo.core.base.var import IndexedVar, VarData, ScalarVar
from pyomo.core.base.boolean_var import IndexedBooleanVar
from pyomo.environ import Var, Param, Block, Suffix
from pyomo.network.arc import ScalarArc, IndexedArc, Arc
import msgpack

try:
    import ujson as json
except ImportError:
    import json

# pkg
from idaes.logger import getLogger

# from idaes.core.io import pyomo_model_state_pb2 as pb

__author__ = "Dan Gunter (LBNL)"

_log = getLogger(__name__)

FORMAT_VERSION = "0.0.1"

SECT_VAR, SECT_VAR_B = "variable", b"variable"
SECT_SUFFIX, SECT_SUFFIX_B = "suffix", b"suffix"
BT_OTHER, BT_VAR, BT_PARAM = 0, 1, 2


class ModelWriter:
    # Flags for things to `include``
    SUFFIXES, ARCS, CONFIGS, ALL_BLOCKS = (1 << n for n in range(4))
    ALL = SUFFIXES | ARCS | CONFIGS
    # Types
    # ARC_TYPES = {ScalarArc, IndexedArc}
    # BLOCK_TYPES = {
    #     Var: 1,
    #     Param: 2,
    #     Suffix: 3,
    #     ConfigDict: 4,
    # }

    def __init__(self, model: Block, include: int = 0):
        self._m = model
        self._opt_suffix = bool(include & self.SUFFIXES)
        self._opt_arc = bool(include & self.ARCS)
        self._opt_config = bool(include & self.CONFIGS)
        self._opt_all_blocks = bool(include & self.ALL_BLOCKS)
        self._model = model

    def write(self, stream: IOBase, text=False):
        m = self._model
        dump = self._dump_json_line if text else msgpack.pack
        # metadata
        meta = {"version": FORMAT_VERSION, "ts": time.time()}
        dump(meta, stream)
        # variables
        dump({"section": SECT_VAR}, stream)
        for c in m.component_objects():
            if c.is_variable_type():
                subtype = BT_VAR
            elif c.is_parameter_type():
                subtype = BT_PARAM
            else:
                subtype = BT_OTHER
            # subtype, id, name, parent_id, [is_indexed, num]
            b = [subtype, id(c), c.name, id(c.parent_block())]
            if subtype == BT_OTHER:
                if self._opt_arc or self._opt_all_blocks:
                    dump(b, stream)
            else:
                items = []
                for index in c:
                    v = c[index]
                    if index is None:
                        b.append(False)  # not indexed
                    else:
                        b.append(True)  # indexed
                    # item: index, value, [fixed, stale, lb, ub]
                    if subtype == BT_VAR:
                        item = (index, v.value, v.fixed, v.stale, v.lb, v.ub)
                    elif subtype == BT_PARAM:
                        item = (index, v.value)
                    items.append(item)
                b.append(len(items))
                dump(b, stream)
                for item in items:
                    dump(item, stream)
        if self._opt_suffix:
            dump({"section": SECT_SUFFIX}, stream)
            for c in m.component_objects(Suffix):
                print("@@ SUFFIX")
        if self._opt_arc:
            for c in m.component_objects(Arc):
                src_id, dst = c.source.parent_block().name, c.dest.parent_block()

    @staticmethod
    def _dump_json_line(obj, stream):
        json.dump(obj, stream)
        stream.write("\n")


def dump(obj, stream):
    ModelWriter(obj).write(stream)


class ModelReader:
    def __init__(self, handler=None):
        self._stream = None
        if handler is None:
            self._h = BaseHandler()
        else:
            self._h = handler

    def read(self, stream: IOBase, text=False):
        var_state = {BT_VAR: 3, BT_PARAM: 4, BT_OTHER: 2}
        self._stream = stream
        if text:
            obj_stream = self._read_json_lines
            section_key = "section"
            section_map = {SECT_VAR: 2, SECT_SUFFIX: 5}
        else:
            obj_stream = self._read_msgpack
            section_key = b"section"
            section_map = {SECT_VAR_B: 2, SECT_SUFFIX_B: 5}

        ext_count = 0
        n, i = 0, 0
        block_id = None
        obj_iter = obj_stream()

        obj = next(obj_iter)
        self._h.metadata(obj)
        state, count = 1, 1

        for obj in obj_iter:
            # generic check for new section
            if isinstance(obj, dict):
                if 3 <= state <= 4:
                    raise ValueError("Section header in middle of block data")
                key = obj[section_key]
                try:
                    state = section_map[key]
                except KeyError:
                    state, ext_name, ext_count = -1, key, 0
            # other known states
            elif state == 2:
                data_type, block_id = obj[0], obj[1]
                nm = obj[2] if text else obj[2].decode()
                if data_type == BT_OTHER:
                    self._h.block(block_id, nm, obj[3])
                else:
                    self._h.block(block_id, nm, *obj[3:])
                state = var_state[data_type]
                n, i = obj[-1], 0
            elif 3 <= state <= 4:
                if state == 3:
                    self._h.block_data_var(block_id, *obj)
                else:
                    self._h.block_data_param(block_id, *obj)
                i += 1
                if i == n:
                    state = 2  # next var block
            elif state == 5:  # suffix
                self._h.suffix(obj)
            # other state => extension
            else:
                self._h.ext(ext_name, obj, ext_count)
                ext_count += 1
            count += 1

        return count

    def _read_json_lines(self):
        for line in self._stream:
            yield json.loads(line)

    def _read_msgpack(self):
        unpacker = msgpack.Unpacker(self._stream)
        for msg in unpacker:
            yield msg


class BaseHandler:
    def metadata(self, d):
        pass

    def block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        pass

    def block_data_var(
        self,
        block_id: int,
        index: float | str,
        value: float | bool,
        fixed: bool,
        stale: bool,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        pass

    def block_data_param(self, block_id: int, index: int, value: float):
        pass

    def suffix(self, d):
        pass

    def ext(self, name, obj, count):
        pass


class BaseModelHandler(BaseHandler):
    def __init__(self, model: Block):
        self._m = model
        self._cur_block = None

    def block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        b = self._m
        nm = str(name)
        parts = nm.split(".")
        n, i = len(parts), 0
        last = n - 1
        while i < n:
            if i != last and parts[i + 1][-1] == "]":
                # foo.bar[0.0].baz -> i='bar[0', i+1='0]'
                lbr = parts[i].rfind("[")
                c_name = parts[i][:lbr]
                b = getattr(b, c_name)
                c_idx = float(parts[i][lbr + 1 :] + "." + parts[i + 1][:-1])
                b = b[c_idx]
                i += 2
            else:
                b = getattr(b, parts[i])
                i += 1
        self.model_block(b, block_id, is_indexed, num)

    def model_block(self, b, block_id, is_indexed, num):
        self._cur_block = b
        self._cur_block_id = block_id
        self._cur_block_ix = is_indexed
        self._cur_block_num = num
        self._cur_data_block_count = 0

    def block_data_var(
        self,
        block_id: int,
        index: float | str,
        value: float | bool,
        fixed: bool,
        stale: bool,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        if self._cur_block is None:
            raise ValueError("Unexpected variable")
        if self._cur_data_block_count >= self._cur_block_num:
            raise ValueError(
                f"Too many data blocks: {self._cur_data_block_count} >= {self._cur_block_num}"
            )
        if self._cur_block_ix:
            d = self._cur_block[index]
        else:
            d = self._cur_block[None]
        self.model_block_data(d, value, fixed, stale, lb, ub)
        self._cur_data_block_count += 1

    def model_block_data(
        self,
        data_block: Block,
        value: float | bool,
        fixed: bool,
        stale: bool,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        # print(f"set value {value} on data block {data_block}")
        pass


if __name__ == "__main__":
    import sys
    from idaes.core.io.tests.flowsheets import demo_model

    m = demo_model()
    w = ModelWriter(m)
    if len(sys.argv) > 1 and sys.argv[1] == "json":
        text = True
        ext = "json"
        mode = "w"
    else:
        text = False
        ext = "msgp"
        mode = "wb"
    fname = f"model.{ext}"
    with open(fname, mode) as f:
        t0 = time.time()
        w.write(f, text=text)
        t1 = time.time()
    print(f"write {(t1 - t0):.6f}s")
    print(fname)
    #
    mode = "r" if text else "rb"
    r = ModelReader(handler=BaseModelHandler(m))
    with open(fname, mode) as f:
        t0 = time.time()
        n = r.read(f, text=text)
        t1 = time.time()
    print(f"read {(t1 - t0):.6f}s")
    print(n)
