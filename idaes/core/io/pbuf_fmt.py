"""
Protobuf model state I/O
"""

# stdlib
from collections import namedtuple
from enum import Enum
import struct
import time
from typing import Iterable, Any

# third-party
from pyomo.environ import Suffix, Block

try:
    import ojson as json
except ImportError:
    import json

# package
from . import pyomo_model_state_pb2 as pb
from .model_state import ModelState
from .common import ModelSerializerInterface, ModelDeserializerInterface
from .common import DSBlock, DSBool, DSMeta, DSParam, DSVar, OutOfOrder

__author__ = "Dan Gunter (LBNL)"

# map from Pyomo Suffix constant to matching PB constant
_sfx_dir_map = {
    getattr(Suffix, s): getattr(pb.SuffixDirection, s)
    for s in ("LOCAL", "IMPORT", "EXPORT", "IMPORT_EXPORT")
}
_sfx_dt_map = {
    Suffix.FLOAT: pb.SuffixDatatype.FLOAT,
    Suffix.INT: pb.SuffixDatatype.INT,
    None: pb.SuffixDatatype.NONE,
}


class ProtobufDeserializer:

    class State(Enum):
        Metadata = (pb.Metadata, "metadata")
        Blocks = (pb.Block, "blocks")
        OptionalData = (pb.OptionalData, "optional data")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.State.Metadata

    def metadata(self) -> DSMeta:
        if self._state != self.State.Metadata:
            raise OutOfOrder("metadata", self._state.value[1])
        m = self._get_message()
        return DSMeta(m.format_version, m.created, m.author, m.coauthors)

    def block(self) -> DSBlock | DSParam | DSBool | DSVar | None:
        if self._state != self.State.Block:
            raise OutOfOrder("metadata", self._state.value[1])
        m = self._get_message()
        if m.idx == -1:
            return None
        if m.WhichOneOf("variable_index") == "var_index_f":
            key = m.var_index_f
        else:
            key = m.var_index_s
        if m.HasField("lb"):
            result = DSVar(
                m.name,
                m.idx,
                m.type_index,
                m.parent_index,
                m.has_variable_index,
                key,
                m.value,
                m.fixed,
                m.stale,
                m.lb,
                m.ub,
            )
        elif m.HasField("fixed"):
            result = DSBool(
                m.name,
                m.idx,
                m.type_index,
                m.parent_index,
                m.has_variable_index,
                key,
                m.value,
                m.fixed,
                m.stale,
            )
        elif m.HasField("value"):
            result = DSParam(
                m.name,
                m.idx,
                m.type_index,
                m.parent_index,
                m.has_variable_index,
                key,
                m.value,
            )
        else:
            result = DSBlock(
                m.name, m.idx, m.type_index, m.parent_index, m.has_variable_index, key
            )
        return result


class ProtobufSerializer(ModelSerializerInterface):
    """Serialize using protocol buffers (protobuf).

    The output format will be a stream with 3 types of messages:
    Message 1    : Metadata
    Messages 2..N: Block (last block will be a marker)
    Message N+1  : OptionalData

    Since streaming isn't really natively supported, we DIY it.
    Each message will be preceded by a 4-byte unsigned length
    so, e.g. at any position `p` in a buffer `buf`
    ```
    n = struct.unpack("!I", buf[p:p + 4])
    <Message>.ParseFromString(buf[p + 4: 4 + n + p])
    p += n + 4 # move to next message
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = pb.Metadata()
        meta.format_version = self.meta["version"]
        meta.created = self.meta.get("created", time.time())
        meta.author = self.meta.get("author", "unknown")
        n = self._write(meta)
        print(f"@@ meta was {n} bytes")
        self._opt_data = None

    def _write(self, message) -> int:
        # streaming writes:
        # - message-length
        # - message
        data = message.SerializeToString()
        data_sz = len(data)
        sz_msg = struct.pack("!I", data_sz)
        self.strm.write(sz_msg)
        self.strm.write(data)
        return 4 + data_sz

    def begin_blocks(self):
        self._block_num = 0

    def end_blocks(self):
        # write a zero-size as a marker
        size = struct.pack("!I", 0)
        self.strm.write(size)

    def close(self):
        # write the optional data sections before closing
        self._write(self._get_opt_data())
        super().close()

    def _block(self, name, type_i, parent, indexed, key):
        b = pb.Block()
        b.idx = self._block_num
        self._block_num += 1
        b.name = name
        b.type_index = type_i
        b.parent_index = parent
        if indexed:
            b.has_variable_index = True
            if isinstance(key, float):
                b.var_index_f = key
            else:
                b.var_index_s = str(key)
        else:
            b.has_variable_index = False
        return b

    def block(self, e, name, type_i, parent, key, indexed):
        b = self._block(name, type_i, parent, indexed, key)
        b.subtype = pb.BlockType.BASE
        self._write(b)

    def indexed_param(self, e, name, type_i, parent, key):
        b = self._block(name, type_i, parent, True, key)
        b.subtype = pb.BlockType.PARAM
        b.value = e
        self._write(b)

    def indexed_bool(self, e, name, type_i, parent, key):
        b = self._block(name, type_i, parent, True, key)
        b.subtype = pb.BlockType.BOOL
        b.value = e.value
        b.fixed, b.stale = e.fixed, e.stale
        self._write(b)

    def indexed_var(self, e, name, type_i, parent, key):
        b = self._block(name, type_i, parent, True, key)
        b.subtype = pb.BlockType.VAR
        b.value = e.value
        b.fixed, b.stale = e.fixed, e.stale
        b.lb, b.ub = e.lb, e.ub
        self._write(b)

    def _get_opt_data(self):
        if self._opt_data is None:
            self._opt_data = pb.OptionalData()
        return self._opt_data

    def type_names(self, type_names: Iterable[str]):
        types = pb.Types()
        types.type_names[:] = type_names
        self._write(types)

    def suffixes(self, suffixes: Iterable[tuple[str, int, int, int, int, dict]]):
        od = self._get_opt_data()
        for name, tidx, pidx, dr, dt, vals in suffixes:
            b = od.suffixes.add()
            b.name = name
            b.type_index = tidx
            b.parent_index = pidx
            b.direction = _sfx_dir_map[dr]
            b.datatype = _sfx_dt_map[dt]
            for key, val in vals.items():
                skey = str(key)
                if b.datatype == Suffix.FLOAT:
                    b.float_data[skey] = val
                elif b.datatype == Suffix.INT:
                    b.int_data[skey] = int(val)
                else:
                    b.any_data[skey] = str(val)

    def arcs(self, arcs: Iterable[tuple[str, int, int]]):
        od = self._get_opt_data()
        for name, src_i, dst_i in arcs:
            a = od.arcs.add()
            a.name, a.src_index, a.dst_index = name, src_i, dst_i

    def configs(self, configs: Iterable[tuple[int, str, dict[str, Any]]]):
        od = self._get_opt_data()
        for idx, key, val in configs:
            sval = json.dumps(val)
            cfg = od.configs.add()
            cfg.block_index = idx
            cfg.key = key
            cfg.val = sval


# def deserialize_into(buf: bytes, model: Block):
#     pb = pb_model.Model()
#     pb.ParseFromString(buf)
#     block_iter = ((b.name, b.parent_idx) for b in pb.core.blocks)
#     block_obj = get_block_obj(model, block_iter)
#     # data (float vars, params, bools)
#     for d in pb.core.blocks:
#         block_type = d.type_index
#         if not Subtypes.is_data(block_type, False):
#             continue
#         # index is either string or float (defined with protobuf 'oneof')
#         vidx = d.var_index_s if d.HasField("var_index_s") else d.var_index_f
#         # retrieve corrsponding Pyomo block object
#         item = block_obj[d.parent_index][vidx]
#         # set values into block object
#         item.value = d.value
#         if block_type == Subtypes.st_var:
#             item.fixed, item.stale = d.fixed, d.stale
#             lb = d.lb if d.has_lb else None
#             ub = d.ub if d.has_ub else None
#             item.bounds = (lb, ub)
#         elif block_type == Subtypes.st_bool:
#             item.fixed, item.stale = d.fixed, d.stale
#         # nothing else to do for param
#     # suffixes
#     for sfx in pb.core.suffixes:
#         item = block_obj[sfx.idx]
#         item.direction, item.datatype = sfx.direction, sfx.datatype
#         item.clear_values()
#         for data_values in (sfx.int_data, sfx.float_data):
#             for k, v in data_values.items():
#                 component = block_obj[k]
#                 item.set_value(component, v)
#         for k, v in sfx.any_data.items():
#             component = block_obj[k]
#             item.set_value(component, v)
