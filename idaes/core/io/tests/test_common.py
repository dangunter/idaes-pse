"""
Tests for idaes.core.io.common module
"""

# third-party
import pytest

# pkg
from idaes.core.io import common
from idaes.core.io.tests.flowsheets import demo_model

__author__ = "Dan Gunter (LBNL)"


@pytest.fixture
def dummy_serializer():
    from typing import Any, Iterable

    class DummySerializer(common.ModelSerializerInterface):
        def __init__(self):
            self.reset()

        def reset(self):
            self.blocks = []

        def to_str(self) -> str:
            return ""

        def to_bytes(self) -> bytes:
            return b""

        def block(self, e, name, type_i, parent, key, indexed):
            self.blocks.append({"name": name})

        def indexed_bool(self, e, name, type_i, parent, key):
            return

        def indexed_param(self, e, name, type_i, parent, key):
            return

        def indexed_var(self, e, name, type_i, parent, key):
            return

        def type_names(self, type_name_iter):
            return

        def type_names(self, type_names: Iterable[str]):
            return

        def arcs(self, arcs: Iterable[tuple[str, int, int]]):
            return

        def suffix(
            self,
            name: str,
            type_idx: int,
            parent: int,
            direction: int,
            datatype: int,
            values: dict,
        ):
            return

        def config(self, index: int, key: str, value: dict[str, Any]):
            return

    return DummySerializer()


@pytest.fixture(scope="module")
def model():
    return demo_model()


@pytest.mark.unit
def test_builder_init(dummy_serializer):
    builder = common.Builder(dummy_serializer)


@pytest.mark.unit
@pytest.mark.parametrize("flags", list(range(1 << 3)))
def test_builder_build1(dummy_serializer, model, flags):
    bld = common.Builder(dummy_serializer, include=flags)
    bld.build(model)
    # m.fs = FlowsheetBlock
    # m.fs.M01 = Mixer(...)
    # m.fs.H02 = Heater(...)
    # m.fs.F03 = Flash(...)
    units = {"M01", "H02", "F03"}
    found = 0
    for b in dummy_serializer.blocks:
        if b["name"] in units:
            found += 1
    assert found == len(units)


#        dummy_serializer.reset()
