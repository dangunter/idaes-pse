"""
Pydantic definition of model state
"""

# stdlib
import time
from typing import Any
from typing_extensions import TypeAliasType

# third-party
import pydantic
from pyomo.environ import Block
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.network import Arc

# package
from .common import FORMAT_VERSION
from idaes.logger import getLogger

_log = getLogger(__name__)

# For data, the index can be a tuple or a scalar
IndexType = TypeAliasType("Index", tuple | int | float | str)


class ModelMetadata(pydantic.BaseModel):
    format_version: str = pydantic.Field(
        default=FORMAT_VERSION,
        title="Format version",
        description="Version number for the format, fixed by the module",
        frozen=True,
    )
    created: float = pydantic.Field(
        default_factory=time.time,
        description="When the model serialization was created",
    )
    other: dict[str, Any] = pydantic.Field(
        default={},
        title="other metadata",
        description="Other metadata sections",
        examples=[{"section-name": {"data-key": "data-value"}}],
    )


class ModelCore(pydantic.BaseModel):
    """Core information about a model.

    The data is organized like a set of tables, where the
    index of each row is implicitly its position in the list.
    As with tables, the data is held in tuples with implicit field names.
    """

    #: - common (header): name, type_index, parent_index, ...
    #: Then for ..., first few type_index values are special,
    #: corresponding to Subtypes indexed from 0
    #: - st_var: ... = var_index, value, fixed, stale, lb, ub
    #: - st_bool: ... = var_index, value, fixed, stale
    #: - st_param: ... = var_index, value
    #: - st_suffix: ... = direction, datatype, values
    #:    - 'values' is a dict key=block (by index), value=a value
    blocks: list[tuple] = []
    #: type names
    block_types: list[str] = []

    # connectivity: stream name, source block index, dest block index
    conn: list[tuple[str, int, int]] = []

    # config: block index, key, value
    configs: list[tuple[int, str, Any]] = []

    model_config = pydantic.ConfigDict()  # this is internal to Pydantic

    # def build(
    #     self,
    #     block: Block,
    #     include_suffixes: bool = False,
    #     include_conn: bool = False,
    #     include_configs: bool = False,
    # ):
    #     """Build contents from a block representing a model or portion of a model
    #        that is being serialized.

    #     Args:
    #         block: Root block of the model
    #         include_suffixes: Inlude Suffix objects
    #         include_conn: Include connectivity information in serialization
    #         include_configs: Include ConfigDict information in serialization
    #     """
    #     comp = [(block, -1, None)]
    #     # add data subtypes as initial types
    #     self.block_types = [str(type(t)) for t in Subtypes.types]
    #     next_type_idx = len(self.block_types)
    #     # for looking up existing block types
    #     type_lookup = {}

    #     # index of previous block added
    #     block_idx = -1

    #     # extra data structures for optional parts
    #     if include_suffixes:
    #         block_idmap, suffixes = {}, []
    #     if include_conn:
    #         arcs = []
    #     if include_conn or include_configs:
    #         block_rmap = {}

    #     # main loop: traverse model breadth-first
    #     cp, cs = 0, len(comp)  # cp=current position, cs=current size
    #     while cp < cs:
    #         block, parent_idx, data_idx = comp[cp]
    #         cp += 1
    #         block_type = type(block)
    #         subtype = Subtypes.get(block_type)
    #         # data
    #         if data_idx or Subtypes.is_data(subtype):
    #             self._add_data_block(block, parent_idx, data_idx, subtype)
    #             # add to map for suffixes too
    #             if include_suffixes:
    #                 block_idmap[id(block)] = block_idx
    #             block_idx += 1
    #         # suffix
    #         elif subtype == Subtypes.st_suffix:
    #             if include_suffixes:
    #                 self._add_suffix(block, suffixes, subtype, parent_idx)
    #         # every other kind of block
    #         else:
    #             name = block.getname(fully_qualified=False)
    #             if "[" in name:
    #                 full_name = block.getname(fully_qualified=True)
    #                 raise ValueError()
    #             # get index for block type
    #             type_idx, next_type_idx = self._get_type(
    #                 block_type, type_lookup, next_type_idx
    #             )
    #             # add this block
    #             self.blocks.append((name, type_idx, parent_idx))
    #             if include_suffixes:
    #                 block_idmap[id(block)] = block_idx
    #             block_idx += 1
    #             # add all sub-blocks
    #             cs = self._add_subblocks_to_components(comp, block, block_idx, cs)
    #             # add config entries
    #             if include_configs:
    #                 if isinstance(getattr(block, "config", None), ConfigDict):
    #                     for key, config_val in block.config.items():
    #                         cf_val = self._config_val(config_val)
    #                         entry = (block_idx, key, cf_val)
    #                         self.configs.append(entry)
    #             # add connectivity; store idx/block map, with Arcs being special
    #             if include_conn:
    #                 if isinstance(block, Arc):
    #                     arcs.append((block_idx, block))
    #                 else:
    #                     block_rmap[name] = block_idx
    #     # end of main loop

    #     # build connectivity from the stored Arcs
    #     if include_conn:
    #         for arc_idx, arc in arcs:
    #             self.conn.append(
    #                 (
    #                     arc_idx,
    #                     block_rmap[arc.source.parent_block().getname()],
    #                     block_rmap[arc.dest.parent_block().getname()],
    #                 )
    #             )

    #     # map suffix block ids to block indexes then add to blocks
    #     if include_suffixes:
    #         self._suffix_blockid_to_index(suffixes, block_idmap)
    #         self.blocks.extend(suffixes)

    # def _get_type(self, block_type, lookup, next_idx) -> tuple[int, int]:
    #     """Get index for block's type.

    #     If the type is new, a new entry is added to `lookup` and `self.block_types`.

    #     Return:
    #         Index of type and next index in array (incremented if new type added)
    #     """
    #     try:
    #         idx = lookup[block_type]
    #     except KeyError:
    #         # add type to 'block_types' list and lookup dict
    #         idx, lookup[block_type] = next_idx, next_idx
    #         # add the new type
    #         self.block_types.append(str(block_type))
    #         next_idx += 1
    #     return idx, next_idx

    # @staticmethod
    # def _add_subblocks_to_components(
    #     comp: list[tuple],
    #     block: Block,
    #     block_idx: int,
    #     cs: int,
    # ) -> int:
    #     """Add all child (sub) blocks to the list of components to process in `comp`.

    #     Returns:
    #         New current size of component list
    #     """
    #     # add child components
    #     if _can_have_subcomponents(block):
    #         for child in block.component_objects(descend_into=False):
    #             comp.append((child, block_idx, None))
    #             cs += 1
    #     # add indexed process blocks
    #     if getattr(block, "__process_block__", None) == "indexed":
    #         for i in range(len(block)):
    #             comp.append((block[i], block_idx, None))
    #             cs += 1
    #     # add (the parents of) data blocks
    #     if hasattr(block, "keys"):
    #         first = True
    #         for key in iter(block._index_set):
    #             if first:
    #                 first = False
    #                 item = block[key]
    #                 data_subtype = Subtypes.get(type(item), None)
    #                 if Subtypes.is_data(data_subtype):
    #                     comp.append((item, block_idx, key))
    #                     cs += 1
    #                 else:
    #                     break  # assume all types are the same
    #             else:
    #                 # already passed first, assume rest too
    #                 comp.append((block[key], block_idx, key))
    #                 cs += 1
    #     return cs

    # def _add_data_block(
    #     self, b: Block, parent_idx: int, data_idx: float | str, subtype: int
    # ):
    #     item = [b.local_name, subtype, parent_idx, data_idx]
    #     if subtype == Subtypes.st_var:
    #         item.extend((b.value, b.fixed, b.stale, b.lb, b.ub))
    #     elif subtype == Subtypes.st_bool:
    #         item.extend((b.value, b.fixed, b.stale))
    #     elif subtype == Subtypes.st_param:
    #         item.append(b.value)
    #     else:
    #         raise ValueError(f"Unexpected subtype={subtype}")
    #     self.blocks.append(item)

    # def _add_suffix(self, block, suffixes, subtype, parent_idx):
    #     suffix_data = {}
    #     for k in block.keys():
    #         data_subtype = Subtypes.get(type(k))
    #         suffix_data[id(k)] = block[k]
    #     item = [
    #         block.local_name,
    #         subtype,
    #         parent_idx,
    #         block.direction.value,
    #         block.datatype.value,
    #         suffix_data,
    #     ]  # list instead of tuple because last element will be modified
    #     suffixes.append(item)

    # @staticmethod
    # def _suffix_blockid_to_index(suffixes: list[tuple], mapping: dict[int, int]):
    #     """Replacing suffix block id's with indexes.
    #     If the block id is not found, the suffix is dropped.
    #     """
    #     for sfx in suffixes:
    #         d, x = {}, 0
    #         for k, v in sfx[-1].items():
    #             try:
    #                 d[mapping[k]] = v
    #             except KeyError as err:
    #                 x += 1  # TODO: not sure why???
    #         # if _log.isEnabledFor(logging.DEBUG):
    #         #    _log.debug(f"found {len(d)} suffixes, couldn't find {x}")
    #         sfx[-1] = d

    # @classmethod
    # def _config_val(cls, config_val) -> dict | str:
    #     """Create a new configuration value."""
    #     if type(config_val) in {int, str, float, bool}:
    #         cf_val = config_val
    #     elif isinstance(config_val, ConfigDict):
    #         cf_val = {k: cls._config_val(v) for k, v in config_val.items()}
    #     elif isinstance(config_val, ConfigList):
    #         cf_val = {}
    #         for i, v in enumerate(config_val):
    #             cf_val[i] = cls._config_val(v)
    #     elif isinstance(config_val, ConfigValue):
    #         return {
    #             "value": config_val.value(),
    #             "default_value": config_val._default,
    #             "description": config_val._description,
    #         }
    #     else:
    #         cf_val = str(config_val)  # shrug
    #     return cf_val


class ModelState(pydantic.BaseModel):
    """Represents the state of a model."""

    #: Metadata like author and date
    meta: ModelMetadata = ModelMetadata()
    #: Core info: blocks, variables, etc.
    core: ModelCore = ModelCore()
    #: Extensions, arbitrary info in form {"type of info": {...info...}}
    ext: dict[str, dict] = {}

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.core.build(model)

    def as_dict(self):
        return {"meta": dict(self.meta), "core": dict(self.core), "ext": dict(self.ext)}


#     def set_values(self, model: Block):
#         """Set values in Pyomo `model` from those found in this instance.

#         Args:
#             model: Pyomo model, to modify

#         Returns:
#             None
#         """
#         block_obj = get_block_obj(
#             model, ((b[BH.NAME], b[BH.PARENT_IDX]) for b in self.core.blocks)
#         )
#         for block_i, block in enumerate(self.core.blocks):
#             block_type = block[BH.TYPE_IDX]
#             if Subtypes.is_data(block_type, False):
#                 pidx, vidx = block[BH.PARENT_IDX], block[BH.VAR_IDX]
#                 item = block_obj[pidx][vidx]
#                 if block_type == Subtypes.st_var:
#                     item.fixed = block[BH.FIXED]
#                     item.stale = block[BH.STALE]
#                     item.bounds = (block[BH.LB], block[BH.UB])
#                 elif block_type == Subtypes.st_bool:
#                     item.fixed = block[BH.FIXED]
#                     item.stale = block[BH.STALE]
#                 # nothing else to do for st_param
#             elif block_type == Subtypes.st_suffix:
#                 item = block_obj[block_i]
#                 item.direction, item.datatype = block[BH.DIRECTION], block[BH.DATATYPE]
#                 item.clear_values()
#                 for k, v in block[BH.VALUES].items():
#                     component = block_obj[k]
#                     item.set_value(component, v)
#             # do nothing for other blocks


# def _can_have_subcomponents(o):
#     return hasattr(o, "component_objects") and hasattr(o.component_objects, "__call__")
