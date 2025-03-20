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
from typing import Iterable

# third-party
import gzip
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.component import ComponentData
from pyomo.core.base.param import ParamData, IndexedParam
from pyomo.core.base.var import IndexedVar
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
    def type_names(self, type_name_iter):
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


DataTypes = namedtuple("DataTypes", ("IndexedParam", "IndexedVar", "IndexedBooleanVar"))
DT = DataTypes(IndexedParam, IndexedVar, IndexedBooleanVar)


class Builder:
    def __init__(
        self,
        serializer: ModelSerializerInterface,
        include_suffixes: bool = False,
        include_conn: bool = False,
        include_configs: bool = False,
    ):
        self._ser = serializer
        self._callbacks = {
            DT.IndexedParam: self._ser.indexed_param,
            DT.IndexedVar: self._ser.indexed_var,
            DT.IndexedBooleanVar: self._ser.indexed_bool,
        }
        self._suffixes = include_suffixes
        self._conn = include_conn
        if self._conn:
            self._arc_types = {ScalarArc, IndexedArc}
        self._configs = include_configs

    def build(self, model: Block) -> None:
        comp_arr = [(model, -1)]

        # initialize type arr/map with known types
        type_arr = list(DT)
        type_map = {t: i for i, t in enumerate(type_arr)}

        obj_id_map = {}
        obj_name_map = {}
        obj_indexed_name_map = {}

        last_comp = -1

        # extra data structures for optional parts
        if self._suffixes:
            suffixes = []
        if self._conn:
            arcs = {}  # unique streams

        # add blocks (get list of unique types)
        cp, cs, block_count = 0, len(comp_arr), 0
        while cp < cs:
            obj, parent = comp_arr[cp]
            cp += 1

            obj_id = id(obj)
            if obj_id in obj_id_map:
                continue
            obj_id_map[obj_id] = block_count

            obj_name = obj.getname()
            obj_fullname = obj.name

            # get an index for the component type
            obj_type = type(obj)
            type_idx = type_map.get(obj_type, None)
            if type_idx is None:
                type_idx = len(type_arr)
                type_arr.append(type_idx)
                type_map[obj_type] = type_idx

            if obj_type is Suffix:
                suffixes.append(
                    (
                        obj_name,
                        type_idx,
                        obj.direction.value,
                        obj.datatype.value,
                        {str(sk): sv for sk, sv in obj.items()},
                    )
                )
            elif self._conn and obj_type in self._arc_types:
                # arc -> names of parent blocks of src/dst ports
                arcs[obj_name] = (
                    obj.source.parent_block().name,
                    obj.dest.parent_block().name,
                )
                # don't put the arc in the block list
            else:
                try:
                    obj_keys = obj.keys()
                except AttributeError:
                    obj_keys = [None]
                cb = None
                is_component_data = isinstance(obj, ComponentData)
                indexed, start_block_count = None, block_count
                # get each indexed item
                for key in obj_keys:
                    indexed = not (key is None and is_component_data)
                    if indexed:
                        try:
                            element = obj[key]
                        except TypeError:  # not subscriptable
                            continue  # skip
                    else:
                        element = obj
                    if cb is None:
                        cb = self._callbacks.get(type(element), self._ser.block)
                        # subsequent iterations, use same callback -> assumes
                        # all items are of the same type
                    cb(element, obj_name, type_idx, parent, key, indexed)
                    # put name in map if serializing connectivity or suffixes
                    if self._conn or self._suffixes:
                        element_name = element.getname(fully_qualified=True)
                        obj_name_map[element_name] = block_count
                    # if can have subcomponents, add them
                    try:
                        subcomponents = obj.component_objects()
                        for subobj in subcomponents:
                            comp_arr.append((subobj, cp))
                            cs += 1
                    # no subcomponents; that's ok
                    except (AttributeError, TypeError):
                        pass
                    # block config
                    if self._configs and isinstance(
                        getattr(obj, "config", None), ConfigDict
                    ):
                        for key, config_val in obj.config.items():
                            cf_val = self._config_val(config_val)
                            self._ser.config(block_count, key, cf_val)
                    # done with this block
                    block_count += 1
                # suffixes referring to name w/o index refer to all of the
                # ones we just added in the loop above
                if self._suffixes and indexed:
                    obj_indexed_name_map[obj_fullname] = range(
                        start_block_count, block_count
                    )

        # type names
        self._ser.type_names((str(t) for t in type_arr))

        # connectivity
        if self._conn:
            self._ser.arcs(
                (
                    (name, obj_name_map[src], obj_name_map[dst])
                    for name, (src, dst) in arcs.items()
                )
            )

        # suffixes
        if self._suffixes:
            cur_idx = block_count
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

        # config params

    @classmethod
    def _config_val(cls, config_val) -> dict | str:
        """Create a new configuration value."""
        result = None
        if type(config_val) in {int, str, float, bool}:
            result = config_val
        elif isinstance(config_val, ConfigDict):
            result = {k: cls._config_val(v) for k, v in config_val.items()}
        elif isinstance(config_val, ConfigList):
            result = {}
            for i, v in enumerate(config_val):
                result[i] = cls._config_val(v)
        elif isinstance(config_val, ConfigValue):
            return {
                "value": config_val.value(),
                "default_value": config_val._default,
                "description": config_val._description,
            }
        else:
            result = str(config_val)  # shrug
        return result
