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


class JsonModelSerializer(ModelSerializerInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strm.write('{"meta":')
        json.dump(self.meta, self.strm)

    def begin_blocks(self):
        self._num_blocks = 0
        self.strm.write(',"core": {"blocks":[')

    def end_blocks(self):
        self.strm.write("]")

    def close(self):
        self.strm.write("}")
        super().close()

    def block(self, e, name, type_i, parent, key, indexed):
        idx_i = "1" if indexed else "0"
        key_s = "null" if key is None else str(key)
        self._add_block(f'"{name}",{type_i},{parent},"{key_s}",{idx_i}')

    def indexed_param(self, e, name, type_i, parent, key):
        key_s = "null" if key is None else str(key)
        self._add_block(f'"{name}",{type_i},{parent},"{key_s}",1,{e}')

    def indexed_bool(self, e, name, type_i, parent, key):
        key_s = "null" if key is None else str(key)
        fixed_i, stale_i = "1" if e.fixed else "0", "1" if e.stale else "0"
        self._add_block(
            f'"{name}",{type_i},{parent},"{key_s}",1,{e.value},{fixed_i},{stale_i}'
        )

    def indexed_var(self, e, name, type_i, parent, key):
        key_s = "null" if key is None else str(key)
        fixed_i, stale_i = "1" if e.fixed else "0", "1" if e.stale else "0"
        self._add_block(
            f'"{name}",{type_i},{parent},"{key_s}",1,{e.value},{fixed_i},{stale_i},{e.lb},{e.ub}'
        )

    def suffix(self, name: str, tidx: int, pidx: int, dr: int, dt: int, vals: dict):
        values_d = json.dumps(vals)
        self._add_block(f'"{name}",{tidx},{pidx},{dr},{dt},{values_d}')

    def _add_block(self, data: str):
        comma = "" if self._num_blocks == 0 else ","
        self.strm.write(f"{comma}[{data}]")
        self._num_blocks += 1

    def type_names(self, type_names: Iterable[str]):
        self.strm.write(',"types":[')
        first = True
        for nm in type_names:
            if first:
                self.strm.write('"' + nm + '"')
                first = False
            else:
                self.strm.write(',"' + nm + '"')
        self.strm.write("]")

    def configs(self, configs: Iterable[tuple[int, str, dict[str, Any]]]):
        self.strm.write(',"configs":[')
        first = True
        for idx, key, value in configs:
            v = json.dumps(value)
            if first:
                self.strm.write(f'[{idx},"{key}",{v}]')
                first = False
            else:
                self.strm.write(f',[{idx},"{key}",{v}]')
        self.strm.write("]")

    def arcs(self, arcs: Iterable[tuple[str, int, int]]):
        # self._m.core.conn = list(arcs)
        self.strm.write(',"arcs":[')
        first = True
        for name, src_idx, dst_idx in arcs:
            if first:
                self.strm.write(f'["{name}",{src_idx},{dst_idx}]')
                first = False
            else:
                self.strm.write(f',["{name}",{src_idx},{dst_idx}]')
        self.strm.write("]")
