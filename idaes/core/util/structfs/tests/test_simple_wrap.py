#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Tests for simplified wrapper API.
"""

import pytest
from idaes.core.util.structfs.simple_wrap import _Wrapper, RESULT_FLOWSHEET_KEY
from idaes.core.util.doctesting import Docstring

### Basic Flash flowsheet code

from pyomo.environ import ConcreteModel, SolverFactory
from idaes.core import FlowsheetBlock

# Import idaes logger to set output levels
import idaes.logger as idaeslog
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from idaes.models.unit_models import Flash

# imitate statement in subpackage's __init__.py
fi_main = _Wrapper.main


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = BTXParameterBlock(
        valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal", state_vars="FTPz"
    )
    m.fs.flash = Flash(property_package=m.fs.properties)
    # assert degrees_of_freedom(m) == 7
    m.fs.flash.inlet.flow_mol.fix(1)
    m.fs.flash.inlet.temperature.fix(368)
    m.fs.flash.inlet.pressure.fix(101325)
    m.fs.flash.inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.flash.inlet.mole_frac_comp[0, "toluene"].fix(0.5)
    m.fs.flash.heat_duty.fix(0)
    m.fs.flash.deltaP.fix(0)

    return m


def initialize(m):
    m.fs.flash.initialize(outlvl=idaeslog.INFO)


### end basic Flash flowsheet code


@fi_main
def my_main():
    """Create and solve Flash flowsheet"""

    m = build()
    initialize(m)

    solver = SolverFactory("ipopt")
    result = solver.solve(m, tee=True)

    return m, result


@pytest.mark.integration
def test_main():
    m, result = my_main()
    assert m
    assert result
    # see if flowsheet steps were run
    assert result[RESULT_FLOWSHEET_KEY]


#####
# Test docstring in _Wrapper class
#####


# pacify linter by defining functions
# (will be overwritten by exec() below)
def fi_wrap_my_main_function(x, y, z=None):
    return None


def fi_wrap_build_flowsheet():
    return None


def fi_wrap_solve_flowsheet():
    return None


# redefine 2 functions that will be needed by original name
def build_flowsheet():
    return fi_wrap_build_flowsheet()


def solve_flowsheet():
    return fi_wrap_solve_flowsheet()


#  load the functions from the docstring
_ds1 = Docstring(_Wrapper.__doc__)
exec(_ds1.code("simplewrap", func_prefix="fi_wrap_"))


@pytest.mark.unit
def test_wrap_docstr():
    model, solve_result = fi_wrap_my_main_function("arg1", "arg2", keyword="foo")
    # model should be whatever the fake build returns
    assert type(model) == type(fi_wrap_build_flowsheet())
    # empty solve dict should only have the 'extra' key with the added object
    assert len(solve_result) == 1
    assert solve_result[RESULT_FLOWSHEET_KEY]
