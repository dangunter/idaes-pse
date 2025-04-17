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
# from google.protobuf import json_format
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.component import ComponentData
from pyomo.core.base.param import ParamData, IndexedParam
from pyomo.core.base.var import IndexedVar, VarData, ScalarVar
from pyomo.core.base.boolean_var import IndexedBooleanVar
from pyomo.environ import Var, Param, Block, value
from pyomo.core.base.suffix import Suffix, SuffixDataType
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

SECT_BLOCK, SECT_BLOCK_B = "block", b"block"
SECT_SUFFIX, SECT_SUFFIX_B = "suffix", b"suffix"
SECT_ARC, SECT_ARC_B = "arc", b"arc"
SECT_CONFIG, SECT_CONFIG_B = "config", b"config"
SECT_BLOCK2, SECT_BLOCK2_B = "extra_blocks", b"extra_blocks"

BT_OTHER, BT_VAR, BT_PARAM = 0, 1, 2


def try_float(x):
    try:
        return float(x)
    except ValueError:
        print(f"@@ not float: {x}")
        return x


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
        if not self._opt_all_blocks and (self._opt_arc or self._opt_suffix):
            var_block_ids = set()
            if self._opt_arc:
                arc_blocks = set()
            if self._opt_suffix:
                sfx_block_names = set()
        else:
            var_block_ids, arc_blocks, sfx_block_names = None, None, None
        # metadata
        meta = {"version": FORMAT_VERSION, "ts": time.time()}
        dump(meta, stream)
        # variables
        dump({"section": SECT_BLOCK}, stream)
        for c in m.component_objects():
            id_c = id(c)
            if var_block_ids is not None:
                var_block_ids.add(id_c)
            if c.is_variable_type():
                subtype = BT_VAR
            elif c.is_parameter_type():
                subtype = BT_PARAM
            else:
                subtype = BT_OTHER
            # subtype, id, name, parent_id, [is_indexed, num]
            b = [subtype, id_c, c.name, id(c.parent_block())]
            if subtype == BT_OTHER:
                if self._opt_all_blocks:
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
                parent_id = id(c.parent_block())
                n = len(c.keys())
                dtype = c.datatype
                item = (parent_id, c.local_name, int(c.direction), dtype, n)
                dump(item, stream)
                for pyo_k, pyo_v in c.items():
                    k = pyo_k.name
                    v = int(pyo_v) if dtype == SuffixDataType.INT else float(pyo_v)
                    dump((k, v), stream)
                    if sfx_block_names is not None:
                        sfx_block_names.add(k)
        if self._opt_arc:
            dump({"section": SECT_ARC}, stream)
            for c in m.component_objects(Arc):
                src_id, dst_id = id(c.source.parent_block()), id(c.dest.parent_block())
                dump((c.local_name, src_id, dst_id), stream)
                if arc_blocks is not None:
                    arc_blocks.add(c)
        # extra blocks for arc/suffix references
        if var_block_ids:
            dump({"section": SECT_BLOCK2}, stream)
            for nm in sfx_block_names:
                b = m.find_component(nm)
                block_id = id(b)
                if block_id not in var_block_ids:
                    item = (block_id, nm)
                    dump(item, stream)
            for b in arc_blocks:
                for block_ref in b.src, b.dest:
                    block_id = id(block_ref)
                    if block_id not in var_block_ids:
                        item = (block_id, nm)
                        dump(item, stream)

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

    ST_START, ST_BLOCK, ST_BLOCK2 = 1, 2, 8
    ST_DATA_VAR, ST_DATA_PARAM = 3, 4
    ST_SUFFIX, ST_SUFFIX_ITEM = 5, 6
    ST_ARC = 7

    def read(self, stream: IOBase, text=False):
        var_state = {
            BT_VAR: self.ST_DATA_VAR,
            BT_PARAM: self.ST_DATA_PARAM,
            BT_OTHER: self.ST_BLOCK,
        }
        self._stream = stream
        if text:
            obj_stream = self._read_json_lines
            section_key = "section"
            section_map = {
                SECT_BLOCK: self.ST_BLOCK,
                SECT_BLOCK2: self.ST_BLOCK2,
                SECT_SUFFIX: self.ST_SUFFIX,
                SECT_ARC: self.ST_ARC,
            }
        else:
            obj_stream = self._read_msgpack
            section_key = b"section"
            section_map = {
                SECT_BLOCK_B: self.ST_BLOCK,
                SECT_BLOCK2_B: self.ST_BLOCK2,
                SECT_SUFFIX_B: self.ST_SUFFIX,
                SECT_ARC_B: self.ST_ARC,
            }

        ext_count = 0
        n, i = 0, 0
        block_id = None
        obj_iter = obj_stream()

        obj = next(obj_iter)
        self._h.metadata(obj)
        state, count = self.ST_START, 1
        in_block_data = False

        for obj in obj_iter:
            # generic check for new section
            if isinstance(obj, dict):
                if in_block_data:
                    raise ValueError("Section header in middle of block data")
                key = obj[section_key]
                try:
                    state = section_map[key]
                except KeyError:
                    state, ext_name, ext_count = -1, key, 0
            # other known states
            elif state == self.ST_BLOCK:
                data_type, block_id = obj[0], obj[1]
                nm = obj[2] if text else obj[2].decode()
                if data_type == BT_OTHER:
                    self._h.var_block(block_id, nm, obj[3])
                else:
                    self._h.var_block(block_id, nm, *obj[3:])
                    in_block_data = True
                state = var_state[data_type]
                n, i = obj[-1], 0
            elif in_block_data:
                if state == self.ST_DATA_VAR:
                    self._h.block_data_var(block_id, *obj)
                else:
                    self._h.block_data_param(block_id, *obj)
                i += 1
                if i == n:
                    state = self.ST_BLOCK  # next var block
                    in_block_data = False
            elif state == self.ST_SUFFIX:
                # parent_id, c.local_name, int(c.direction), dtype, n
                suffix_num, suffix_count = obj[-1], 0
                suffix_is_int = obj[3] == SuffixDataType.INT
                suffix_obj = obj[:-1]
                suffix_values = []
                state = self.ST_SUFFIX_ITEM
            elif state == self.ST_SUFFIX_ITEM:
                # key, value
                k, sfx_v = obj
                v = int(sfx_v) if suffix_is_int else float(sfx_v)
                suffix_values.append((k, v))
                suffix_count += 1
                if suffix_count == suffix_num:
                    suffix_obj.append(suffix_values)
                    self._h.suffix(*suffix_obj)
                    state = self.ST_SUFFIX  # next suffix
            elif state == self.ST_ARC:
                name = obj[0] if text else obj[0].decode()
                self._h.arc(name, obj[1], obj[2])
            elif state == self.ST_BLOCK2:
                self._h.extra_block(*obj)
            # other state => extension
            else:
                self._h.ext(ext_name, obj, ext_count)
                ext_count += 1
            count += 1

        self._h.finalize()

        return count

    def _read_json_lines(self):
        for line in self._stream:
            yield json.loads(line)

    def _read_msgpack(self):
        unpacker = msgpack.Unpacker(self._stream)
        for msg in unpacker:
            yield msg


class BaseHandler:
    def finalize(self):
        pass

    def metadata(self, d):
        pass

    def var_block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        pass

    def extra_block(self, block_id: int, name: str):
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

    def suffix(
        self,
        parent_id: int,
        name: str,
        direction: int,
        dtype: int,
        values: list[int | float],
    ):
        pass

    def arc(self, name: str, src_block_id: int, dst_block_id: int):
        pass

    def ext(self, name, obj, count):
        pass


class BaseModelHandler(BaseHandler):
    def __init__(self, model: Block):
        self._m = model
        self._cur_block = None
        self._block_id_map = {}

    def var_block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        b = self._m.find_component(name)
        self.model_var_block(b, block_id, is_indexed, num)

    def model_var_block(self, b, block_id, is_indexed, num):
        self._cur_block = b
        self._cur_block_id = block_id
        self._cur_block_ix = is_indexed
        self._cur_block_num = num
        self._cur_data_block_count = 0

    def extra_block(self, block_id: int, name: str):
        b = self._m.find_component(name)
        self._block_id_map[block_id] = b

    ## TODO: Arc and Suffix, store them up until the end
    ## TODO: add some sort of close() that will then
    ## TODO  run back through stored arcs and suffixes and resolve them
    ## TODO: to blocks, then call model_{arc,suffix} which the user can
    ## TODO: impmlement to eg set the values

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
    import argparse, sys
    from idaes.core.io.tests.flowsheets import demo_model
    from prommis.uky import uky_flowsheet as uky_flowsheet

    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    p.add_argument("--model", choices=["demo", "uky"])
    args = p.parse_args()

    if args.model == "demo":
        m = demo_model()
    else:
        m = uky_flowsheet.build()
    w = ModelWriter(m, include=ModelWriter.SUFFIXES | ModelWriter.ARCS)

    if args.json:
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

    # nm = str(name)
    # parts = nm.split(".")
    # n, i = len(parts), 0
    # last = n - 1

    # while i < n:
    #     if i != last and parts[i + 1][-1] == "]":
    #         # foo.bar[0.0].baz -> i='bar[0', i+1='0]'
    #         lbr = parts[i].rfind("[")
    #         c_name = parts[i][:lbr]
    #         b = getattr(b, c_name)
    #         idx_expr = parts[i][lbr + 1 :] + "." + parts[i + 1][:-1]
    #         if "," in idx_expr:
    #             idx_list = map(try_float, idx_expr.split(","))
    #             b = b[*idx_list]
    #         else:
    #             b = b[try_float(idx_expr)]
    #         i += 2
    #     else:
    #         b = getattr(b, parts[i])
    #         i += 1
