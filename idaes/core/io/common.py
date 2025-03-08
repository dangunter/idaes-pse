"""
This module has common classes, definitions, and functions for
IDAES model I/O (serialization).
"""

from __future__ import annotations
from enum import Enum
import gzip
from io import IOBase
import logging
from statistics import median

from idaes.logger import getLogger
import gzip
from pyomo.environ import (
    Var,
    BooleanVar,
    Block,
    Suffix,
)

from pyomo.core.base.param import ParamData

try:
    import orjson as json
except ImportError:
    import json

__author__ = "Dan Gunter (LBNL)"

_log = getLogger(__name__)

FORMAT_VERSION = "0.0.1"
ENCODING = "utf-8"


class DataFormat(Enum):
    """Supported data I/O formats."""

    JSON = "JSON"
    PROTOBUF = "protobuf"


class StateOption:
    """Constants to specify selected parts of the state."""

    VAR = 1
    BOOL = 2
    PARAM = 4
    VALUE = VAR | BOOL | PARAM
    SUFFIX = 8
    CONN = 16
    CONFIG = 32
    ALL = VALUE | SUFFIX | CONN | CONFIG


class Subtypes:
    types = (
        Var._ComponentDataClass,  # var
        BooleanVar._ComponentDataClass,  # boolean
        ParamData,  # param
        Suffix,
    )
    st_var, st_bool, st_param, st_suffix = range(len(types))
    _lookup = {}
    for i in range(len(types)):
        _lookup[types[i]] = i

    @classmethod
    def get(cls, key, default_value=None):
        return cls._lookup.get(key, default_value)

    @classmethod
    def is_data(cls, t) -> bool:
        return False if t is None else t < cls.st_suffix


def get_block_obj(pyomo_model, block_iter: iter[str, int]) -> dict[int, Block]:
    """Get Pyomo block objects for each block in this model.

    Args:
        pyomo_model: Corresponding Pyomo model
        block_list: Iterable of (name, parent_index) with the
                    position in list being the block index.

    Returns:
        Mapping from index in Model.core.block_info to the
        corresponding Pyomo block object.
    """
    block_obj = {0: pyomo_model}
    i = 0
    for name, parent_idx in block_iter:
        if i > 0:
            parent_obj = block_obj[parent_idx]
            if name[-1] == "]":  # indexed block, e.g. 'fs[1]'
                span_start = name.rfind("[")
                idx_str = name[span_start + 1 : -1]
                idx = _index_from_str(idx_str)
                block_obj[i] = parent_obj[idx]
            else:
                block_obj[i] = getattr(parent_obj, name)
        i += 1
    return block_obj


def _index_from_str(s) -> int | float | str:
    # quoted string, strip quotes and we're done
    if s[0] == "'" or s[0] == '"':
        idx = s[1:-1]
    else:
        # try to parse as int, then float
        try:
            idx = int(s)
        except ValueError:
            try:
                idx = float(s)
            except ValueError:
                idx = s  # default is string
    return idx
