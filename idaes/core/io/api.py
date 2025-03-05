"""
Main API.

Usage:

    # `my_model` contains IDAES/Pyomo model

    # SAVE
    sf = StateFile("my-model.binpb")
    sf.save(my_model)

    # LOAD
    sf = StateFile("my-model.binpb")
    sf.load_into(my_model)
"""

from io import FileIO
import gzip
from pathlib import Path
from warnings import warn

# third=party
from pyomo.environ import Block
from .common import DataFormat, ModelState, StateOption
from idaes.logger import getLogger

# package
from . import pbuf, jsondata


_log = getLogger(__name__)


class StateFile:

    SER_FN = {
        DataFormat.PROTOBUF: pbuf.serialize,
        DataFormat.JSON: jsondata.serialize,
    }
    DESER_FN = {
        DataFormat.PROTOBUF: pbuf.deserialize_into,
        DataFormat.JSON: jsondata.deserialize_into,
    }

    def __init__(
        self,
        f: str | Path | FileIO,
        fmt: DataFormat = None,
        options: int = StateOption.VALUE,
        gz: bool = False,
    ):
        self._fp = None
        if isinstance(f, Path):
            self._path = f
        elif isinstance(f, str):
            self._path = Path(f)
        else:
            self._path, self._fp = Path(getattr(f, "name", "")), f
        self._opt, self._gz, self._fmt = options, gz, fmt
        # infer format, if not given
        if self._fmt is None:
            suffix = self._path.suffix
            if suffix == ".binpb":
                self._fmt = DataFormat.PROTOBUF
            elif suffix == ".json":
                self._fmt = DataFormat.JSON
            else:
                self._fmt = DataFormat.JSON
                warn(
                    f"Save file extension '{suffix}' not '.binpb' or '.json'. "
                    f"Defaulting to {self._fmt.value} output."
                )

    def save(self, model):
        if self._fp is None:
            self._open_file("wb")
        model_state = ModelState(model, self._opt)
        serialize = self.SER_FN[self._fmt]
        buf = serialize(model_state)
        try:
            self._fp.write(buf)
        finally:
            self._fp = None

    def load_into(self, model):
        if self._fp is None:
            self._open_file("rb")
        try:
            buf = self._fp.read()
        finally:
            self._fp = None
        self.DESER_FN[self._fmt](buf, model)

    def _open_file(self, mode):
        if self._gz:
            if self._path.suffix != ".gz":
                self._add_suffix(self._path, ".gz")
            self._fp = gzip.open(self._path, mode=mode)
        else:
            self._fp = open(self._path, mode=mode)


def to_json(f, model, gz: bool = False):
    StateFile(f, fmt=DataFormat.JSON, gz=gz).save(model)


def as_dict(model: Block, **build_options) -> dict:
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
