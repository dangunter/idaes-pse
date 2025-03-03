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
from pathlib import Path
import sys
import time
from typing import Annotated, Any, ByteString, Dict, List, Tuple, Union, Optional
from typing_extensions import TypeAliasType

from idaes.logger import getLogger
import gzip
from pyomo.environ import (
    Var,
    BooleanVar,
    Block,
    Suffix,
)
from pyomo.common.config import ConfigDict, ConfigList, ConfigValue
from pyomo.core.base.param import ParamData
from pyomo.network import Arc

try:
    import ojson as json
except ImportError:
    import json

__author__ = "Dan Gunter (LBNL)"

_log = getLogger(__name__)

FORMAT_VERSION = "0.0.1"


class DataFormat(Enum):
    """Supported data I/O formats."""

    JSON = "json"
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


type DataFile = str | Path | IOBase


class FileTools:
    @classmethod
    def open(cls, f: DataFile, is_output: bool, gz=False) -> IOBase:
        if isinstance(f, IOBase):
            return f
        if isinstance(f, str):
            path = Path(f)
        if gz:
            path = cls._add_file_ext(path, ".gz")
            mode = "wb" if is_output else "rb"
            fp = gzip.open(path, mode)
        else:
            mode = "wb" if is_output else "rb"
            fp = path.open(path, mode)
        return fp

    @staticmethod
    def _add_file_ext(p: Path, ext: str) -> Path:
        parts = list(p.parts)
        if parts[-1].endswith(ext):
            return p  # do nothing, already has suffix
        parts[-1] += ext
        return Path(*parts)


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


# Public API


def as_dict(model: Block, **build_options) -> Dict:
    """Serialize the model as a JSON-ready dictionary.

    Args:
        model: The model to serialize to a dict
        build_options: Keyword options for `ModelCore.build()`

    Returns:
        Python dictionary representation of the model serializer object
    """
    m = ModelState()
    m.core.build(model, **build_options)
    return m.as_dict()


def save(
    model: Union[Block, ModelState],
    fp: Union[str, Path, IOBase],
    data_format: DataFormat = DataFormat.JSON,
    encoding: str = "utf8",
    gz=False,
    build_kw: Optional[Dict] = None,
    **dump_kw,
):
    """Save a model to a file or other IO stream.

    This is a convenience function to build a ModelSerializer object
    from the model and dump it to a file with one call.

    Args:
        model: The model object, or a serialized model object
        fp: A filename as a Path or string, or file object
        data_format: Format to save in, default is JSON
        encoding: If `fp` is a filename, the encoding to use when creating the file obj
        gz: If True, use gzip to compress the file as it is written. The output file, if a str
            or Path, will have the '.gz' extension appended unless already ends in '.gz'
        build_kw: Keywords to pass to `ModelSerializer.build`
        dump_kw: Other keywords passed to `ojson.dump` or,
                 if *ojson* was not available, `json.dump`.

    Returns:
        None
    """
    # get model
    if isinstance(model, ModelState):
        mstate = model
    else:
        build_kw = build_kw or {}
        mstate = ModelState()
        mstate.core.build(model, **build_kw)
    # format model contents to buffer
    if data_format == DataFormat.PROTOBUF:
        t0 = time.time()
        buf = _protobuf_bytes(mstate)
        t1 = time.time()
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"protobuf dump time={(t1 - t0):.3g}s")
        mode, encoding = "wb", None
    else:
        t0 = time.time()
        buf = json.dumps(mstate.as_dict(), check_circular=False, **dump_kw)
        t1 = time.time()
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"json dump time={(t1 - t0):.3g}s")
        mode = "wt"
    # open output
    if isinstance(fp, str):
        fp = Path(fp)
    if isinstance(fp, Path):
        if gz:
            fp = _add_suffix(fp, ".gz")
            f = gzip.open(fp, mode)
        else:
            f = fp.open(mode, encoding=encoding)
    else:
        f = fp
    # write buffer to output
    f.write(buf)
    f.close()


# map from Pyomo Suffix constant to matching PB constant
_sfx_dir_map = {
    getattr(Suffix, s): getattr(idaes_pb2.SuffixDirection, s)
    for s in ("LOCAL", "IMPORT", "EXPORT", "IMPORT_EXPORT")
}
_sfx_dt_map = {
    Suffix.FLOAT: idaes_pb2.SuffixDatatype.FLOAT,
    Suffix.INT: idaes_pb2.SuffixDatatype.INT,
    None: idaes_pb2.SuffixDatatype.NONE,
}


def _protobuf_bytes(mstate: ModelState) -> ByteString:
    m = idaes_pb2.Model()
    # metadata
    m.meta.format_version = FORMAT_VERSION
    m.meta.created = time.time()
    m.meta.author = "OpenPSE Team"
    # data: blocks
    type_enum = idaes_pb2.BlockType
    for b in mstate.core.blocks:
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
    for item in mstate.core.conn:
        o = m.core.conn.add()
        o.arc_index, o.src_index, o.dst_index = item
    # data: config values
    for item in mstate.core.configs:
        o = m.core.config.add()
        o.block_index = item[0]
        o.key = item[1]
        o.val = json.dumps(item[2])
    # done
    return m.SerializeToString()


def load(
    model: Block,
    fp: Union[str, Path, IOBase],
    data_format: DataFormat = DataFormat.JSON,
    encoding: str = "utf8",
    set_values=True,
    gz=False,
) -> ModelState:
    """Load into an existing model from a file.

    This is a convenience function to build a ModelSerializer object from JSON
    data in a file (with Pydantic's `.model_validate_json()` method) and
    set values from the loaded data, with a single call.

    Args:
        model: The model object
        fp: A filename as a Path or string, or file object
        data_format: Format of input file, default is JSON
        encoding: If `fp` is a filename, the expected text encoding
        set_values: If True, set values in model from loaded data
        gz: If True, use gzip to decompress the file

    Raises:
        ValueError: If model validation fails, will contain Pydantic error message

    Returns:
        The serialized model object
    """
    if isinstance(fp, str):
        fp = Path(fp)
    if gz:
        read_mode = "rt" if data_format == DataFormat.JSON else "rb"
        f = gzip.open(fp, mode=read_mode)
    else:
        read_mode = "r" if data_format == DataFormat.JSON else "rb"
        encoding = encoding if data_format == DataFormat.JSON else None
        f = fp.open(read_mode, encoding=encoding) if isinstance(fp, Path) else fp
    data = f.read()
    if data_format == DataFormat.JSON:
        try:
            mstate = ModelState.model_validate_json(data)
        except pydantic.ValidationError as err:
            raise ValueError(f"JSON validation failed: {err}")
        if set_values:
            mstate.set_values(model)
    elif data_format == DataFormat.PROTOBUF:
        mstate = idaes_pb2.Model()
        mstate.ParseFromString(data)
        if set_values:
            set_values_protobuf(mstate, model)
    return mstate


def set_values_protobuf(m: idaes_pb2.Model, model: Block):
    pass  # TODO: Set the values!!


def test_create_model(days):
    from idaes_models.models.stepload_case.flowsheets.heating_flowsheet_stepload_additional_constraints import (
        build_model,
        setup_model,
    )

    ts = 24 * days  # hours * days
    print(f"create model with {ts} timesteps")
    pbd = build_model(timesteps=ts, load_kw=1000)
    setup_model(pbd, timesteps=ts, load_kw=1000)
    return pbd


def test_json_write(pbd, fname, gz=False, **kwargs):
    model = ModelState()
    model.core.build(pbd, **kwargs)
    save(model, fname + ".json", data_format=DataFormat.JSON, gz=gz)


def test_json_read(fname, gz=False, encoding="utf-8"):
    filename = fname + ".json.gz" if gz else fname + ".json"
    return load(None, filename, gz=gz, data_format=DataFormat.JSON, set_values=False)


def test_protobuf_write(pbd, fname, gz=False, **kwargs):
    model = ModelState()
    model.core.build(pbd, **kwargs)
    save(model, fname + ".pbuf", data_format=DataFormat.PROTOBUF, gz=gz)


def test_protobuf_read(fname, gz=False):
    filename = fname + ".pbuf.gz" if gz else fname + ".pbuf"
    return load(
        None, filename, data_format=DataFormat.PROTOBUF, gz=gz, set_values=False
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--verbose", "-v", action="count")
    args = p.parse_args()

    if args.verbose:
        _log.setLevel(logging.DEBUG)

    pbd = test_create_model(args.days)

    for kwargs in (
        {},
        {"include_suffixes": True, "include_configs": True, "include_conn": True},
    ):
        print("================================")
        if not kwargs:
            print("flags = default")
        else:
            flags_str = ", ".join(kwargs.keys())
            print(f"flags = {flags_str}")
        print("--------------------------------")
        times = {
            "protobuf": {"read": {"gz": [], "raw": []}, "write": {"gz": [], "raw": []}},
            "json": {"read": {"gz": [], "raw": []}, "write": {"gz": [], "raw": []}},
        }
        for i in range(3):
            for gz in (True, False):
                gz_key = "gz" if gz else "raw"
                print(f"run {i + 1} gzip={gz}")
                flags = "_".join((k[8:] for k in kwargs.keys()))
                if not flags:
                    filename = f"test-base-{i}"
                else:
                    filename = f"test-{flags}-{i}"
                print("  protobuf")
                t0 = time.time()
                test_protobuf_write(pbd, filename, gz=gz, **kwargs)
                times["protobuf"]["write"][gz_key].append(time.time() - t0)
                t0 = time.time()
                test_protobuf_read(filename, gz=gz)
                times["protobuf"]["read"][gz_key].append(time.time() - t0)
                print("  json")
                t0 = time.time()
                test_json_write(pbd, filename, gz=gz, **kwargs)
                times["json"]["write"][gz_key].append(time.time() - t0)
                t0 = time.time()
                test_json_read(filename, gz=gz)
                times["json"]["read"][gz_key].append(time.time() - t0)
        for fmt in times:
            for direction in ("read", "write"):
                for gz_key in ("raw", "gz"):
                    t = times[fmt][direction][gz_key]
                    median_time = median(t)
                    time_info = " ".join((f"{v:.3g}" for v in t))
                    print(
                        f"{fmt} + {direction} + {gz_key}: {median_time:.3g} [ {time_info} ]"
                    )
