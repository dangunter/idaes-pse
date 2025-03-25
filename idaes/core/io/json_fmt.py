"""
JSON model state I/O
"""

# stdlib
from typing import Any, Iterable

# third-party
from pyomo.environ import Block

try:
    import ojson as json
except ImportError:
    import json
# pkg
from .model_state import ModelState
from .common import ModelSerializerInterface

ENCODING = "utf-8"


# def serialize(model_state: ModelState) -> bytes:
#     d = model_state.as_dict()
#     return json.dumps(d, check_circular=False, encoding=ENCODING)


# def deserialize_into(buf: bytes, model: Block):
#     txt = bytes.decode(encoding=ENCODING)
#     mstate = ModelState.model_validate_json(txt)
#     block_obj = get_block_obj(model)
#     for block_i, block in enumerate(mstate.core.blocks):
#         block_type = block[1]
#         if Subtypes.is_data(block_type, False):
#             pidx, vidx = block[2], block[3]
#             item = block_obj[pidx][vidx]
#             item.value = block[4]
#             if block_type == Subtypes.st_var:
#                 item.fixed = block[5]
#                 item.stale = block[6]
#                 item.bounds = (block[7], block[8])
#             elif block_type == Subtypes.st_bool:
#                 item.fixed = block[5]
#                 item.stale = block[6]
#             # nothing else to do for st_param
#         elif block_type == Subtypes.st_suffix:
#             item = block_obj[block_i]
#             item.direction, item.datatype = block[3], block[4]
#             item.clear_values()
#             for k, v in block[5].items():
#                 component = block_obj[k]
#                 item.set_value(component, v)
#         # do nothing for other blocks


class BlockTuple:
    # Meaning of positions in tuples.
    # var, bool, param
    NAME = 0
    TYPE_IDX = 1
    PARENT = 2
    IDX = 3
    IS_INDEXED = 4
    VALUE = 5
    FIXED = 6
    STALE = 7
    LB = 8
    UB = 9
    # & suffix
    DIRECTION = 3
    DATATYPE = 4
    VALUES = 5


class JsonModelSerializer(ModelSerializerInterface):
    def __init__(self):
        self._m = ModelState()

    def to_str(self) -> str:
        return json.dumps(self._m.as_dict(), check_circular=False)

    def to_bytes(self) -> bytes:
        return self.to_str().encode(encoding=ENCODING)

    def block(self, e, name, type_i, parent, key, indexed):
        self._m.core.blocks.append(
            (
                name,
                type_i,
                parent,
                key,
                indexed,
            )
        )

    def indexed_param(self, e, name, type_i, parent, key):
        self._m.core.blocks.append(
            (
                name,
                type_i,
                parent,
                key,
                True,
                e,
            )
        )

    def indexed_bool(self, e, name, type_i, parent, key):
        v = e[key]
        self._m.core.blocks.append(
            (
                name,
                type_i,
                parent,
                key,
                True,
                e.value,
                e.fixed,
                e.stale,
            )
        )

    def indexed_var(self, e, name, type_i, parent, key):
        self._m.core.blocks.append(
            (
                name,
                type_i,
                parent,
                key,
                True,
                e.value,
                e.fixed,
                e.stale,
                e.lb,
                e.ub,
            )
        )

    def scalar_var(self, e, name, type_i, parent):
        self._m.core.blocks.append(
            (
                name,
                type_i,
                parent,
                None,
                False,
                e.value,
                e.fixed,
                e.stale,
                e.lb,
                e.ub,
            )
        )

    def suffix(
        self,
        name: str,
        type_i: int,
        pidx: int,
        direction: int,
        datatype: int,
        values: dict,
    ):
        self._m.core.blocks.append((name, type_i, pidx, direction, datatype, values))

    def type_names(self, type_names: Iterable[str]):
        self._m.core.block_types = list(type_names)

    def arcs(self, arcs: Iterable[tuple[str, int, int]]):
        self._m.core.conn = list(arcs)

    def config(self, index: int, key: str, value: dict[str, Any]):
        self._m.core.configs.append((index, key, value))
