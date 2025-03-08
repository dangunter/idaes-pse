"""
JSON model state I/O
"""

from pyomo.environ import Block

try:
    import ojson as json
except ImportError:
    import json

from .model_state import ModelState
from .common import ENCODING, get_block_obj, Subtypes


def serialize(model_state: ModelState) -> bytes:
    d = model_state.as_dict()
    return json.dumps(d, check_circular=False, encoding=ENCODING)


def deserialize_into(buf: bytes, model: Block):
    txt = bytes.decode(encoding=ENCODING)
    mstate = ModelState.model_validate_json(txt)
    block_obj = get_block_obj(model)
    for block_i, block in enumerate(mstate.core.blocks):
        block_type = block[1]
        if Subtypes.is_data(block_type, False):
            pidx, vidx = block[2], block[3]
            item = block_obj[pidx][vidx]
            item.value = block[4]
            if block_type == Subtypes.st_var:
                item.fixed = block[5]
                item.stale = block[6]
                item.bounds = (block[7], block[8])
            elif block_type == Subtypes.st_bool:
                item.fixed = block[5]
                item.stale = block[6]
            # nothing else to do for st_param
        elif block_type == Subtypes.st_suffix:
            item = block_obj[block_i]
            item.direction, item.datatype = block[3], block[4]
            item.clear_values()
            for k, v in block[5].items():
                component = block_obj[k]
                item.set_value(component, v)
        # do nothing for other blocks
