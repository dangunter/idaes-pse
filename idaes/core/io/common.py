"""
This module has common classes, definitions, and functions for
IDAES model I/O (serialization).
"""

# stdlib
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
import gzip
from io import IOBase
import logging
from statistics import median
from typing import Any, Iterable

# third-party
import gzip
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.component import ComponentData
from pyomo.core.base.param import ParamData, IndexedParam
from pyomo.core.base.var import IndexedVar, VarData, ScalarVar
from pyomo.core.base.boolean_var import IndexedBooleanVar
from pyomo.environ import Block, Suffix
from pyomo.network import Arc
from pyomo.network.arc import ScalarArc, IndexedArc

try:
    import orjson as json
except ImportError:
    import json

# pkg
from idaes.logger import getLogger

__author__ = "Dan Gunter (LBNL)"

_log = getLogger(__name__)

FORMAT_VERSION = "0.0.1"


class ModelSerializerInterface(ABC):

    @abstractmethod
    def to_str(self) -> str:
        pass

    @abstractmethod
    def to_bytes(self) -> bytes:
        pass

    @abstractmethod
    def block(self, e, name, type_i, parent, key, indexed):
        pass

    @abstractmethod
    def indexed_bool(self, e, name, type_i, parent, key):
        pass

    @abstractmethod
    def indexed_param(self, e, name, type_i, parent, key):
        pass

    @abstractmethod
    def indexed_var(self, e, name, type_i, parent, key):
        pass

    @abstractmethod
    def scalar_var(self, e, name, type_i, parent, key):
        pass

    @abstractmethod
    def type_names(self, type_names: Iterable[str]):
        pass

    @abstractmethod
    def arcs(self, arcs: Iterable[tuple[str, int, int]]):
        pass

    @abstractmethod
    def suffix(
        self,
        name: str,
        type_idx: int,
        parent: int,
        direction: int,
        datatype: int,
        values: dict,
    ):
        pass

    @abstractmethod
    def config(self, index: int, key: str, value: dict[str, Any]):
        pass


# Types for data in blocks
DataTypes = namedtuple(
    "DataTypes",
    ("IndexedParam", "IndexedVar", "IndexedBooleanVar", "VarData", "ScalarVar"),
)
DT = DataTypes(IndexedParam, IndexedVar, IndexedBooleanVar, VarData, ScalarVar)


class Builder:
    # Flags for things to `include``
    SUFFIXES, CONN, CONFIGS = (1 << n for n in range(3))
    ALL = SUFFIXES | CONN | CONFIGS

    def __init__(self, serializer: ModelSerializerInterface, include: int = 0):
        self._ser = serializer
        self._callbacks = {
            DT.IndexedParam: self._ser.indexed_param,
            DT.IndexedVar: self._ser.indexed_var,
            DT.IndexedBooleanVar: self._ser.indexed_bool,
            # DT.VarData: self._ser.scalar_var,
            # DT.ScalarVar: self._ser.scalar_var,
        }
        self._suffixes = bool(include & self.SUFFIXES)
        self._conn = bool(include & self.CONN)
        self._arc_types = {ScalarArc, IndexedArc}
        self._configs = bool(include & self.CONN)

    def build(self, model: Block) -> None:
        comp_arr = [(model, -1)]

        # initialize type arr/map with known types
        type_arr = list(DT)
        type_map = {t: i for i, t in enumerate(type_arr)}

        # tracking data structures
        obj_id_map = {}
        obj_name_map = {}
        obj_indexed_name_map = {}

        # data structures for optional parts
        if self._suffixes:
            suffixes = []
        if self._conn:
            arcs = {}  # unique streams

        # main loop: until no more objects to process
        # cp = current position => block index
        # cs = current size => next block index
        cp, cs = 0, len(comp_arr)
        log_dbg = _log.isEnabledFor(logging.DEBUG)  # check once
        while cp < cs:
            obj, parent = comp_arr[cp]
            # if parent > 0:
            #    print(f"{obj.name} parent={parent}")

            if log_dbg:
                _log.debug(f"object '{obj.name}' cp={cp} cs={cs} parent={parent}")

            obj_id = id(obj)

            # if duplicate, move to next immediately
            if obj_id in obj_id_map:
                cp += 1
                continue

            obj_id_map[obj_id] = cs  # where block will be in list

            obj_name = obj.getname()
            obj_fullname = obj.name

            # get an index for the component type
            obj_type = type(obj)
            type_idx = type_map.get(obj_type, None)
            if type_idx is None:
                type_idx = len(type_arr)
                type_arr.append(obj_type.__qualname__)
                type_map[obj_type] = type_idx

            if obj_type is Suffix:
                if self._suffixes:
                    suffixes.append(
                        (
                            obj_name,
                            type_idx,
                            obj.direction.value,
                            obj.datatype.value,
                            {str(sk): sv for sk, sv in obj.items()},
                        )
                    )
            elif obj_type in self._arc_types:
                if self._conn:
                    # arc -> names of parent blocks of src/dst ports
                    arcs[obj_name] = (
                        obj.source.parent_block().name,
                        obj.dest.parent_block().name,
                    )
                    # don't put the arc in the block list
            else:
                # not a suffix or arc
                # separate code based on whether indexed
                if hasattr(obj, "keys"):
                    cb = None
                    # is_component_data = isinstance(obj, ComponentData)
                    indexed, start_cs, added = None, cs, []
                    # get each indexed item
                    for key, element in obj.items():
                        # add element
                        if cb is None:
                            cb = self._callbacks.get(type(element), self._ser.block)
                            # subsequent iterations, use same callback -> assumes
                            # all items are of the same type
                        cb(element, obj_name, type_idx, parent, key, True)
                        added.append(cs)
                        # put name in map if serializing connectivity or suffixes
                        if (self._conn or self._suffixes) and hasattr(
                            element, "getname"  # may be scalar
                        ):
                            elt_name = element.getname(fully_qualified=True)
                            obj_name_map[elt_name] = cs
                        # add subcomponents (if any)
                        cs = self._build_add_subcomponents(element, comp_arr, cp, cs)
                        # block config
                        if (
                            self._configs
                            and hasattr(element, "config")
                            and isinstance(element.config, ConfigDict)
                        ):
                            self._build_add_configs(element, cs)
                    # suffixes referring to name w/o index refer to all of the
                    # ones we just added in the loop above
                    if self._suffixes:
                        obj_indexed_name_map[obj_fullname] = added
                else:
                    # add the object
                    cb = self._callbacks.get(obj_type, self._ser.block)
                    cb(obj, obj_name, type_idx, parent, key, False)
                    # put name in map if serializing connectivity or suffixes
                    if self._conn or self._suffixes:
                        obj_name_map[obj_name] = cp
                    # add subcomponents (if any)
                    cs = self._build_add_subcomponents(obj, comp_arr, cp, cs)
                    # optionally add block configs
                    if (
                        self._configs
                        and hasattr(obj, "config")
                        and isinstance(obj.config, ConfigDict)
                    ):
                        self._build_add_configs(obj, cp)
            # move to next object
            cp += 1
        # end of main loop

        # +type names
        self._ser.type_names((str(t) for t in type_arr))

        # +connectivity
        if self._conn:
            self._ser.arcs(
                (
                    (name, obj_name_map[src], obj_name_map[dst])
                    for name, (src, dst) in arcs.items()
                )
            )

        # +suffixes
        if self._suffixes:
            cur_idx = len(comp_arr)  # adding at end
            for name, type_idx, direction, datatype, d in suffixes:
                # map names to blocks
                d2 = {}
                for k, v in d.items():
                    try:
                        # try in regular name map
                        datum_obj_idx = obj_name_map[k]
                        d2[datum_obj_idx] = v
                    except KeyError:
                        try:
                            # try in name map that refers to a list of indexes
                            idx_list = obj_indexed_name_map[k]
                            for idx in idx_list:
                                d2[idx] = v
                        except KeyError:
                            _log.warning(f"Cannot find object for suffix name={k}")
                            continue
                self._ser.suffix(name, type_idx, cur_idx, direction, datatype, d2)
                cur_idx += 1

    @staticmethod
    def _build_add_subcomponents(element, comp_arr, parent_idx, cs: int) -> int:
        """If can have subcomponents, add them."""
        try:
            subcomponents = element.component_objects(descend_into=False)
            for subobj in subcomponents:
                comp_arr.append((subobj, parent_idx))
                cs += 1
        # no subcomponents; that's ok
        except (AttributeError, TypeError):
            pass
        return cs

    def _build_add_configs(self, element, element_idx):
        for key, config_val in element.config.items():
            cf_val = self._config_val(config_val)
            self._ser.config(element_idx, key, cf_val)

    @classmethod
    def _config_val(cls, config_val) -> dict[str, Any]:
        """Create a new configuration value, somewhat standardized."""
        # standardized structure being returned
        result = {
            "value": None,  # scalar or dict
            # "default_value": None,  # may be absent
            # "description": "",  # may be absent
        }
        if type(config_val) in {int, str, float, bool}:
            result["value"] = config_val
        elif isinstance(config_val, ConfigDict):
            result["value"] = {k: cls._config_val(v) for k, v in config_val.items()}
        elif isinstance(config_val, ConfigList):
            list_map = {}
            for i, v in enumerate(config_val):
                list_map[i] = cls._config_val(v)
            result["value"] = list_map
        elif isinstance(config_val, ConfigValue):
            result.update(
                {
                    "value": config_val.value(),
                    "default_value": config_val._default,
                    "description": config_val._description,
                }
            )
        else:
            result["value"] = str(config_val)  # shrug
        return result
