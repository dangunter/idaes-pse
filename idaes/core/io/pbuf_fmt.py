"""
Protobuf model state I/O
"""

# stdlib
import time

# third-party
from pyomo.environ import (
    Suffix,
    Block
)

try:
    import ojson as json
except ImportError:
    import json
# package
from . import model_state_pb2 as pb_model
from .model_state import ModelState
from .common import FORMAT_VERSION, Subtypes, get_block_obj

__author__ = "Dan Gunter (LBNL)"

# map from Pyomo Suffix constant to matching PB constant
_sfx_dir_map = {
    getattr(Suffix, s): getattr(pb_model.SuffixDirection, s)
    for s in ("LOCAL", "IMPORT", "EXPORT", "IMPORT_EXPORT")
}
_sfx_dt_map = {
    Suffix.FLOAT: pb_model.SuffixDatatype.FLOAT,
    Suffix.INT: pb_model.SuffixDatatype.INT,
    None: pb_model.SuffixDatatype.NONE,
}


def serialize(model_state: ModelState) -> bytes:
    m = pb_model.Model()
    # metadata
    m.meta.format_version = FORMAT_VERSION
    m.meta.created = time.time()
    m.meta.author = "OpenPSE Team"
    # data: blocks
    type_enum = pb_model.BlockType
    for i, b in enumerate(model_state.core.blocks):
        type_idx = b[1]
        if type_idx == Subtypes.st_suffix:
            o = m.core.suffixes.add()
        else:
            o = m.core.blocks.add()
        o.name, o.type_index, o.parent_index = b[:3]
        # fill in values according to sub-type
        if type_idx <= Subtypes.st_param:
            # set var_index
            if isinstance(b[3], float):
                o.var_index_f = b[3]
                o.var_index_s = ""
            else:
                o.var_index_s = str(b[3])
            # set value
            o.value = b[4]
            # add fixed/stale for var & bool
            if type_idx < Subtypes.st_param:
                o.fixed, o.stale = b[5], b[6]
                # add lb, ub for var
                if type_idx == Subtypes.st_var:
                    o.subtype = type_enum.VAR
                    if b[7] is None:
                        o.has_lb = False
                    else:
                        o.has_lb, o.lb = True, b[7]
                    if b[8] is None:
                        o.has_ub = False
                    else:
                        o.has_ub, o.ub = True, b[8]
                else:
                    o.subtype = type_enum.BOOL
            else:
                o.subtype = type_enum.PARAM
        elif type_idx == Subtypes.st_suffix:
            o.idx = i
            o.subtype = type_enum.SUFFIX
            o.direction = _sfx_dir_map[b[3]]
            datatype = b[4]
            o.datatype = _sfx_dt_map[datatype]
            for key, val in b[5].items():
                skey = str(key)
                if datatype == Suffix.FLOAT:
                    o.float_data[skey] = val
                elif datatype == Suffix.INT:
                    o.int_data[skey] = val
                else:
                    o.any_data[skey] = str(val)
        else:
            o.subtype = type_enum.BASE
    # data: connectivity
    for item in model_state.core.conn:
        o = m.core.conn.add()
        o.arc_index, o.src_index, o.dst_index = item
    # data: config values
    for item in model_state.core.configs:
        o = m.core.config.add()
        o.block_index = item[0]
        o.key = item[1]
        o.val = json.dumps(item[2])
    # done
    return m.SerializeToString()


def deserialize_into(buf: bytes, model: Block):
    pb = pb_model.Model()
    pb.ParseFromString(buf)
    block_iter = ((b.name, b.parent_idx) for b in pb.core.blocks))
    block_obj = get_block_obj(model, block_iter)
    # core
    # pre-allocate target array since blocks and suffixes interleaved
    b = [None] * (len(pb.core.blocks) + len(pb.core.suffixes))
    # all block types except suffixes
    for block in pb.core.blocks:
        if block.subtype == pb_model.BASE:
            item = (block.name, block.type_index, block.parent_index)
        elif block.subtype in (pb_model.VAR, pb_model.BOOL, pb_model.PARAM):
            # use string index if non-empty else float
            if block.var_index_s:
                var_index = block.var_index_s
            else:
                var_index = block.var_index_f
            if block.subtype == pb_model.VAR:
                # lower and upper bound values, or None
                lb = block.lb if block.has_lb else None
                ub = block.ub if block.has_ub else None
                item = (
                    block.name,
                    block.type_index,
                    block.parent_index,
                    var_index,
                    block.value,
                    block.fixed,
                    block.stale,
                    lb,
                    ub,
                )
            elif block.subtype == pb_model.BOOL:
                item = (
                    block.name,
                    block.type_index,
                    block.parent_index,
                    var_index,
                    block.value,
                    block.fixed,
                    block.stale,
                )
            else:
                item = (
                    block.name,
                    block.type_index,
                    block.parent_index,
                    var_index,
                    block.value,
                )
        b[block.idx] = item
    # suffix blocks
    for sfx in pb.core.suffixes:
        values = sfx.int_data
        values.update(sfx.float_data)
        values.update(sfx.any_data)
        b[sfx.idx] = (
            sfx.name,
            sfx.type_index,
            sfx.parent_index,
            sfx.direction,
            sfx.datatype,
            values,
        )
    # done with core blocks
    m.core.blocks = b
    # connection info
    m.core.conn = [x.arc_index, x.src_index, x.dst_index] for x in pb.core.conn
    # config dicts
    c = []
    for cfg in pb.core.config:
        val = json.loads(cfg.val)
        c.append((cfg.block_index, cfg.key, val))
    m.core.configs = c
    # done
    return m
