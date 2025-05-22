"""
Tests for idaes.core.io.common module
"""

# stdlib
import gzip
from pathlib import Path
import time

# third-party
import pytest

# idaes
from idaes.models.properties.swco2 import SWCO2ParameterBlock
from idaes.models.unit_models import Heater, PressureChanger
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core import FlowsheetBlock
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from pyomo.environ import (
    Var,
    Param,
    Block,
    value,
    TransformationFactory,
    ConcreteModel,
    Block,
)
from idaes.models.unit_models import Flash, Mixer

# pkg
from idaes.core.io.common import *

__author__ = "Dan Gunter (LBNL)"


def _timed_write(w, fname, gz: bool = False, text: bool = False, mode: str = ""):
    with open(fname, mode) as f:
        t0 = time.time()
        w.write(f, text=text, gz=gz)
        t1 = time.time()
    return t1 - t0


def _timed_read(
    m, fname: str, gz: bool = False, text: bool = False, do_print: bool = False
):
    if gz:
        if text:
            f = gzip.open(fname, mode="rt")
        else:
            f = gzip.open(fname, mode="r")
    else:
        mode = "r" if text else "rb"
        f = open(fname, mode)
    model_handler = PrintHandler() if do_print else None
    r = Reader(handler=DataToModel(m, model_handler=model_handler))
    t0 = time.time()
    n = r.read(f, text=text)
    t1 = time.time()
    return t1 - t0, n


def _demo_model() -> Block:
    """Semi-complicated demonstration flowsheet."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.BT_props = BTXParameterBlock()
    m.fs.M01 = Mixer(property_package=m.fs.BT_props)
    m.fs.H02 = Heater(property_package=m.fs.BT_props)
    m.fs.F03 = Flash(property_package=m.fs.BT_props)
    m.fs.s01 = Arc(source=m.fs.M01.outlet, destination=m.fs.H02.inlet)
    m.fs.s02 = Arc(source=m.fs.H02.outlet, destination=m.fs.F03.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m.fs)

    m.fs.properties = SWCO2ParameterBlock()
    m.fs.main_compressor = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.bypass_compressor = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.turbine = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=False,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )
    m.fs.boiler = Heater(
        dynamic=False, property_package=m.fs.properties, has_pressure_change=True
    )
    m.fs.FG_cooler = Heater(
        dynamic=False, property_package=m.fs.properties, has_pressure_change=True
    )
    m.fs.pre_boiler = Heater(
        dynamic=False, property_package=m.fs.properties, has_pressure_change=False
    )
    m.fs.HTR_pseudo_tube = Heater(
        dynamic=False, property_package=m.fs.properties, has_pressure_change=True
    )
    m.fs.LTR_pseudo_tube = Heater(
        dynamic=False, property_package=m.fs.properties, has_pressure_change=True
    )
    return m


@pytest.mark.integration
def test_write_and_read():
    files = []
    m: Block = _demo_model()
    do_print = True
    for use_json in False, True:
        for use_gzip in False, True:
            if use_json:
                text = True
                ext = "json"
                if use_gzip:
                    mode = "wb"
                else:
                    mode = "w"
            else:
                text = False
                ext = "msgp"
                mode = "wb"
            if use_gzip:
                ext += ".gz"

            fname = f"model.{ext}"
            w = Writer(m, include=0)

            print(f"Parameters: ext={ext} text={text} mode={mode}")
            duration = _timed_write(w, fname, text=text, mode=mode, gz=use_gzip)
            print(f"write '{fname}' in {duration:.6f}s")

            duration, n = _timed_read(
                m, fname, gz=use_gzip, text=text, do_print=do_print
            )
            print(f"read {n} records from '{fname}' in {duration:.6f}s")
            files.append(fname)

    print("cleaning files")
    for f in files:
        Path(f).unlink()
