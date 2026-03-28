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
import pytest
from types import SimpleNamespace
from pyomo.environ import ConcreteModel, SolverStatus, TerminationCondition
from idaes.core import FlowsheetBlock
from ..fsrunner import FlowsheetRunner, BaseFlowsheetRunner
from .flash_flowsheet import FS as flash_fs
from idaes.core.util import structfs
from idaes.core.util.doctesting import Docstring
from pyomo.environ import assert_optimal_termination


@pytest.mark.unit
def test_run_all():
    flash_fs.run_steps()
    assert_optimal_termination(flash_fs.results)


@pytest.mark.unit
def test_rerun():

    flash_fs.run_steps()
    first_model = flash_fs.model

    print("-- rerun --")

    # model not changed
    flash_fs.run_steps(first="solve_initial", last="solve_initial")
    assert flash_fs.model == first_model


@pytest.mark.unit
def test_rerun_reset():
    flash_fs.run_steps()
    first_model = flash_fs.model

    print("-- rerun --")

    # reset forces new model
    flash_fs.reset()
    flash_fs.run_steps(last="solve_initial")
    assert flash_fs.model != first_model


@pytest.mark.unit
def test_annotation():
    runner = flash_fs
    runner.run_steps("build")
    print(runner.timings.history)

    ann = runner.annotate_var  # alias
    flash = runner.model.fs.flash  # alias
    category = "flash"
    kw = {"input_category": category, "output_category": category}

    ann(
        flash.inlet.flow_mol,
        key="fs.flash.inlet.flow_mol",
        title="Inlet molar flow",
        desc="Flash inlet molar flow rate",
        **kw,
    ).fix(1)
    ann(flash.inlet.temperature, units="Centipedes", **kw).fix(368)
    ann(flash.inlet.pressure, **kw).fix(101325)
    ann(flash.inlet.mole_frac_comp[0, "benzene"], **kw).fix(0.5)
    ann(flash.inlet.mole_frac_comp[0, "toluene"], **kw).fix(0.5)
    ann(flash.heat_duty, **kw).fix(0)
    ann(flash.deltaP, is_input=False, **kw).fix(0)

    ann = runner.annotated_vars
    print("-" * 40)
    print(ann)
    print("-" * 40)
    assert ann["fs.flash.inlet.flow_mol"]["title"] == "Inlet molar flow"
    assert (
        ann["fs.flash.inlet.flow_mol"]["description"] == "Flash inlet molar flow rate"
    )
    assert ann["fs.flash.inlet.flow_mol"]["input_category"] == category
    assert ann["fs.flash.inlet.flow_mol"]["output_category"] == category
    assert runner.model.fs.flash.inlet.flow_mol[0].value == 1
    assert ann["fs.flash._temperature_inlet_ref"]["units"] == "Centipedes"
    assert ann["fs.flash.deltaP"]["is_input"] == False


#####
# Test the code blocks in the structfs/__init__.py
#####

# pacify linters:
sfi_before_build_model = sfi_before_set_operating_conditions = sfi_before_init_model = (
    sfi_before_solve
) = lambda x: None
SolverStatus, FS = None, None

#  load the functions from the docstring
_ds1 = Docstring(structfs.__doc__)
exec(_ds1.code("before", func_prefix="sfi_before_"))
exec(_ds1.code("after", func_prefix="sfi_after_"))


@pytest.mark.unit
def test_sfi_before():
    m = sfi_before_build_model()
    sfi_before_set_operating_conditions(m)
    sfi_before_init_model(m)
    result = sfi_before_solve(m)
    assert result.solver.status == SolverStatus.ok


@pytest.mark.unit
def test_sfi_after():
    FS.run_steps()
    assert FS.results.solver.status == SolverStatus.ok


# pacify linters
annotate_vars_example = lambda x: None
# load example function from docstring
_ds2 = Docstring(BaseFlowsheetRunner.annotate_var.__doc__)
exec(_ds2.code("annotate_vars"))


@pytest.mark.unit
def test_ann_docs():
    annotate_vars_example(fr := FlowsheetRunner())
    ex = fr.annotated_vars["example"]
    assert ex["fullname"] == "ScalarVar"
    assert ex["title"] == "Example variable"
