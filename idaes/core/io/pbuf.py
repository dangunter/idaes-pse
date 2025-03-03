"""
Protobuf model state I/O
"""

# stdlib
import time

# third-party
from pyomo.environ import Suffix

try:
    import ojson as json
except ImportError:
    import json
# package
from . import model_state_pb2 as pb_model
from .model_state import ModelState
from .common import FORMAT_VERSION, Subtypes


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
    for b in model_state.core.blocks:
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
                if type_idx < Subtypes.st_param:
                    if b[7] is None:
                        o.has_lb = False
                    else:
                        o.has_lb, o.lb = True, b[7]
                    if b[8] is None:
                        o.has_ub = False
                    else:
                        o.has_ub, o.ub = True, b[8]
        elif type_idx == Subtypes.st_suffix:
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
                    o.any_data[skey] = val
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
