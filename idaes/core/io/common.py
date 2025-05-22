"""
This module has common classes, definitions, and functions for
IDAES model I/O (serialization).
"""

from __future__ import annotations

# stdlib
from base64 import b64encode, b64decode
from collections import namedtuple
import gzip
from io import IOBase
import logging
import pickle
from statistics import median
from typing import Any, Iterable, Optional
import time

# third-party
# pyomo
from pyomo.environ import Block
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.component import ComponentData
from pyomo.core.base.param import ParamData, IndexedParam
from pyomo.core.base.var import IndexedVar, VarData, ScalarVar
from pyomo.core.base.boolean_var import IndexedBooleanVar
from pyomo.core.base.suffix import Suffix, SuffixDataType, SuffixDirection
from pyomo.network.arc import ScalarArc, IndexedArc, Arc

# other
import msgpack

try:
    import ujson as json
except ImportError:
    import json

__author__ = "Dan Gunter (LBNL)"

_log = logging.getLogger(__name__)

FORMAT_VERSION = (1, 0, 0)

DEFAULT_ENCODING = "utf-8"

SECT_BLOCK, SECT_BLOCK_B = "block", b"block"
SECT_SUFFIX, SECT_SUFFIX_B = "suffix", b"suffix"
SECT_ARC, SECT_ARC_B = "arc", b"arc"
SECT_CONFIG, SECT_CONFIG_B = "config", b"config"
SECT_BLOCK2, SECT_BLOCK2_B = "extra_blocks", b"extra_blocks"

BT_OTHER, BT_VAR, BT_PARAM = 0, 1, 2

KEY_SECTION, KEY_SECTION_B = "section", b"section"
KEY_VERSION = "version"


class Writer:
    """Serialize a model to an output stream."""

    # Flags for things to `include``
    SUFFIXES, ARCS, CONFIGS, ALL_BLOCKS = (1 << n for n in range(4))
    ALL = SUFFIXES | ARCS | CONFIGS

    def __init__(self, model: Block, include: int = 0):
        self._m = model
        self._opt_suffix = bool(include & self.SUFFIXES)
        self._opt_arc = bool(include & self.ARCS)
        self._opt_config = bool(include & self.CONFIGS)
        self._opt_all_blocks = bool(include & self.ALL_BLOCKS)
        self._model = model
        self._dbg = _log.isEnabledFor(logging.DEBUG)

    def write(self, stream: IOBase, text: bool = False, gz: bool = False):
        m = self._model
        if gz:
            if text:
                kwargs: dict[str, str] = {"mode": "wt", "encoding": DEFAULT_ENCODING}
            else:
                kwargs = {"mode": "wb"}
            stream = gzip.open(stream, **kwargs)
        dump = self._dump_json_line if text else msgpack.pack
        if not self._opt_all_blocks and (self._opt_arc or self._opt_suffix):
            extra_block_ids = set()
            if self._opt_arc:
                arc_blocks = set()
            if self._opt_suffix:
                sfx_block_names = set()
        else:
            extra_block_ids, arc_blocks, sfx_block_names = None, None, None
        configs = {} if self._opt_config else None
        # metadata
        meta: dict[str, float | tuple[int, int, int]] = {
            KEY_VERSION: FORMAT_VERSION,
            "ts": time.time(),
        }
        dump(meta, stream)
        # variables
        dump({KEY_SECTION: SECT_BLOCK}, stream)
        for c in m.component_objects():
            id_c = id(c)
            if c.is_variable_type():
                subtype = BT_VAR
            elif c.is_parameter_type():
                subtype = BT_PARAM
            else:
                subtype = BT_OTHER
            # subtype, id, name, parent_id, [is_indexed, num]
            b = [subtype, id_c, c.name, id(c.parent_block())]
            if subtype == BT_OTHER:
                do_config = self._opt_config and hasattr(c, "CONFIG")
                if self._opt_all_blocks or do_config:
                    dump(b, stream)
                    if do_config:
                        config_data = self._serialize_config(c.config, text)
                        if config_data:
                            configs[id_c] = config_data
                    if extra_block_ids is not None:
                        extra_block_ids.add(id_c)
            else:
                if extra_block_ids is not None:
                    extra_block_ids.add(id_c)
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
            dump({KEY_SECTION: SECT_SUFFIX}, stream)
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
            dump({KEY_SECTION: SECT_ARC}, stream)
            for c in m.component_objects(Arc):
                # src_id, dst_id = id(c.source.parent_block()), id(c.dest.parent_block())
                src_id, dst_id = id(c.source), id(c.dest)
                dump((c.local_name, src_id, dst_id), stream)
                if arc_blocks is not None:
                    arc_blocks.add(c)
        if self._opt_config:
            dump({KEY_SECTION: SECT_CONFIG}, stream)
            for k, v in configs.items():
                dump((k, v), stream)
        # extra blocks for arc/suffix references
        if extra_block_ids:
            dump({KEY_SECTION: SECT_BLOCK2}, stream)
            for nm in sfx_block_names:
                b = m.find_component(nm)
                block_id = id(b.parent_block())
                if block_id not in extra_block_ids:
                    item = (block_id, nm)
                    dump(item, stream)
            for b in arc_blocks:
                missing = False
                for endpt in b.src, b.dest:
                    endpt_id = id(endpt)
                    if endpt_id not in extra_block_ids:
                        item = (endpt_id, endpt.name)
                        dump(item, stream)

    def _serialize_config(self, cfg, text: bool):
        data = None
        if isinstance(cfg, ConfigDict):
            data = []
            for k, v in cfg.items():
                s = self._cpickle(v, text, self._dbg)
                if s:
                    data.append((k, s))
        elif isinstance(cfg, ConfigList):
            data = []
            for v in cfg:
                s = self._cpickle(v, text, self._dbg)
                if s:
                    data.append((None, s))
        else:  # ConfigValue
            s = self._cpickle(v, text, self._dbg)
            if s:
                data = [(None, s)]
        return data

    @staticmethod
    def _cpickle(obj, text, dbg: bool) -> str:
        try:
            s = pickle.dumps(obj)
            if text:
                s = b64encode(s).decode()
        except pickle.PicklingError as err:
            s = None
            if dbg:
                _log.debug(f"PickleError: {err}")
        return s

    @staticmethod
    def _dump_json_line(obj, stream):
        json.dump(obj, stream)
        stream.write("\n")


def dump(obj, stream):
    Writer(obj).write(stream)


class Reader:
    """Deserialize a model from an input stream."""

    def __init__(self, handler=None):
        self._stream = None
        if handler is None:
            self._h = DataHandler()
        else:
            self._h = handler

    ST_START, ST_BLOCK, ST_BLOCK2 = 1, 2, 9
    ST_DATA_VAR, ST_DATA_PARAM = 3, 4
    ST_SUFFIX, ST_SUFFIX_ITEM = 5, 6
    ST_ARC = 7
    ST_CONFIG = 8

    def read(self, stream: IOBase, text=False):
        var_state = {
            BT_VAR: self.ST_DATA_VAR,
            BT_PARAM: self.ST_DATA_PARAM,
            BT_OTHER: self.ST_BLOCK,
        }
        self._stream = stream
        if text:
            obj_stream = self._read_json_lines
            section_key = KEY_SECTION
            section_map = {
                SECT_BLOCK: self.ST_BLOCK,
                SECT_BLOCK2: self.ST_BLOCK2,
                SECT_SUFFIX: self.ST_SUFFIX,
                SECT_ARC: self.ST_ARC,
                SECT_CONFIG: self.ST_CONFIG,
            }
        else:
            obj_stream = self._read_msgpack
            section_key = KEY_SECTION_B
            section_map = {
                SECT_BLOCK_B: self.ST_BLOCK,
                SECT_BLOCK2_B: self.ST_BLOCK2,
                SECT_SUFFIX_B: self.ST_SUFFIX,
                SECT_ARC_B: self.ST_ARC,
                SECT_CONFIG_B: self.ST_CONFIG,
            }

        ext_count = 0
        n, i = 0, 0
        block_id = None
        obj_iter = obj_stream()

        obj = next(obj_iter)
        if text:
            meta_obj = obj
        else:
            meta_obj = obj  # {key.decode(): value for key, value in obj.items()}
        self._h.metadata(meta_obj)
        state, count = self.ST_START, 1
        in_block_data = False

        for obj in obj_iter:
            # generic check for new section
            if isinstance(obj, dict):
                if in_block_data:
                    raise ValueError("Section header in middle of block data")
                skey = section_key if text else section_key.decode()
                key = obj[skey]
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
                    self._h.var_block_data(block_id, *obj)
                else:
                    self._h.param_block_data(block_id, *obj)
                i += 1
                if i == n:
                    state = self.ST_BLOCK  # next var block
                    in_block_data = False
            elif state == self.ST_SUFFIX:
                # parent_id, c.local_name, int(c.direction), dtype, n
                suffix_num, suffix_count = obj[-1], 0
                suffix_is_int = obj[3] == SuffixDataType.INT
                suffix_obj = obj[:-1]
                if not text:
                    suffix_obj[1] = suffix_obj[1].decode()
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
            elif state == self.ST_CONFIG:
                if text:
                    values = [(k, b64decode(v)) for k, v in obj[1]]
                else:
                    values = [(k.decode(), v) for k, v in obj[1]]
                self._h.config(obj[0], values)
            elif state == self.ST_BLOCK2:
                nm = obj[1] if text else obj[1].decode()
                self._h.extra_block(obj[0], nm)
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


############
# Handlers #
############


class DataHandler:
    def metadata(self, d: dict, skip_version_check=False):
        if not skip_version_check:
            try:
                version = d[KEY_VERSION]
            except KeyError:
                raise ValueError("Missing version")
            if version[0] != FORMAT_VERSION[0]:
                raise ValueError(
                    f"Major version mismatch, expected {FORMAT_VERSION[0]} != {version[0]}"
                )
            if version[1] != FORMAT_VERSION[1]:
                raise ValueError(
                    f"Minor version mismatch, expected {FORMAT_VERSION[1]} != {version[1]}"
                )

    def var_block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        pass

    def var_block_data(
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

    def param_block_data(self, block_id: int, index: int, value: float):
        pass

    def extra_block(self, block_id: int, name: str):
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

    def config(self, block_id: int, data: str):
        pass

    def ext(self, name, obj, count):
        pass

    def finalize(self):
        pass


class ModelHandler:
    def model_var(
        self,
        data_block: Block,
        value: float | bool,
        fixed: bool,
        stale: bool,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        pass

    def model_param(
        self,
        data_block: Block,
        value: float | bool,
    ):
        pass

    def model_arc(self, name: str, src: Block, dst: Block):
        pass

    def model_config(self, target: Block, values: list[ConfigValue]):
        pass

    def model_suffix(
        self,
        ref: Block,
        name: str,
        direction: SuffixDirection,
        values_type: SuffixDataType,
        values: list[int | float],
    ):
        pass


class NoopModelHandler(ModelHandler):
    pass


class PrintHandler(ModelHandler):
    def __init__(self):
        self.section("Model", "=")
        self._var = False
        self._param = False
        self._config = False

    def model_var(
        self,
        data_block: Block,
        value: float | bool,
        fixed: bool,
        stale: bool,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        if not self._var:
            self.section("Variables", "-")
            self._var = True
        print_value, print_lb, print_ub = value, lb, ub
        if value is None:
            print_value = "\u2205"
        if lb is None:
            print_lb = "-\u221e"
        if ub is None:
            print_ub = "\u221e"
        print(f"{data_block.name} = {print_value} [{print_lb} .. {print_ub}]")

    def model_param(
        self,
        data_block: Block,
        value: float | bool,
    ):
        if not self._param:
            self.section("Parameters", "-")
            self._param = True
        if value is None:
            value = "-"
        print(f"{data_block.name} = {value}")

    def section(self, s, delim):
        n = len(s) + 2
        print("+" + delim * n + "+")
        print("| " + s + " |")
        print("+" + delim * n + "+")

    def model_config(self, target, values):
        if not self._config:
            self.section("Config", "-")
            self._config = True
        if values:
            print(f">>> {target.name}")
            for k, v in values:
                print(f"  {k} = {v}")


class DataToModel(DataHandler):
    def __init__(self, model: Block, model_handler: ModelHandler = None):
        self._m = model
        self._mh = model_handler or NoopModelHandler()
        self._cur_block = None
        self._block_id_map = {}
        self._stored_suffixes, self._stored_arcs = [], []
        self._stored_configs = []

    def var_block(
        self,
        block_id: int,
        name: str,
        parent_id: int,
        is_indexed: bool = False,
        num: int = 0,
    ):
        b = self._m.find_component(name)
        self._block_id_map[block_id] = b
        self._cur_block = b
        self._cur_block_id = block_id
        self._cur_block_ix = is_indexed
        self._cur_block_num = num
        self._cur_data_block_count = 0

    def var_block_data(
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
                f"Too many variable data blocks: {self._cur_data_block_count} >= {self._cur_block_num}"
            )
        if self._cur_block_ix:
            d = self._cur_block[index]
        else:
            d = self._cur_block[None]
        self._mh.model_var(d, value, fixed, stale, lb, ub)
        self._cur_data_block_count += 1

    def param_block_data(self, block_id, index, value):
        if self._cur_block is None:
            raise ValueError("Unexpected parameter")
        if self._cur_data_block_count >= self._cur_block_num:
            raise ValueError(
                f"Too many parameter data blocks: {self._cur_data_block_count} >= {self._cur_block_num}"
            )
        if self._cur_block_ix:
            d = self._cur_block[index]
        else:
            d = self._cur_block[None]
        self._mh.model_param(d, value)
        self._cur_data_block_count += 1

    def extra_block(self, block_id: int, name: str):
        b = self._m.find_component(name)
        self._block_id_map[block_id] = b

    def suffix(
        self,
        parent_id: int,
        name: str,
        direction: int,
        dtype: int,
        values: list[int | float],
    ):
        self._stored_suffixes.append((parent_id, name, direction, dtype, values))

    def arc(self, name: str, src_block_id: int, dst_block_id: int):
        self._stored_arcs.append((name, src_block_id, dst_block_id))

    def config(self, block_id: int, data: list[tuple[str, str]]):
        values = [(k, pickle.loads(item)) for k, item in data]
        self._stored_configs.append((block_id, values))

    def finalize(self):
        for parent_id, name, direction, dtype, values in self._stored_suffixes:
            try:
                ref_block = self._block_id_map[parent_id]
            except KeyError:
                _log.error(f"Block for suffix '{name}' not found")
                continue
            self._mh.model_suffix(ref_block, name, direction, dtype, values)

        for name, src_block_id, dst_block_id in self._stored_arcs:
            try:
                src_block = self._block_id_map[src_block_id]
                try:
                    dst_block = self._block_id_map[src_block_id]
                except KeyError:
                    _log.error(f"arc destination block ({dst_block_id}) not found")
                    continue
            except KeyError:
                _log.error(f"arc source block ({src_block_id}) not found")
                continue
            self._mh.model_arc(name, src_block, dst_block)

        for cfg_block_id, obj in self._stored_configs:
            try:
                cfg_block = self._block_id_map[cfg_block_id]
            except KeyError:
                _log.error(f"Block for config ({cfg_block_id}) not found")
                continue
            self._mh.model_config(cfg_block, obj)
