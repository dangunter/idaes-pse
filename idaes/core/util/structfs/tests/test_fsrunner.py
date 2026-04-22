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
from pyomo.environ import ConcreteModel, SolverStatus, TerminationCondition, Var
from idaes.core import FlowsheetBlock
from ..fsrunner import (
    FlowsheetRunner,
    BaseFlowsheetRunner,
    global_flowsheet,
    wrapped_main,
    run_wrapped_main,
)

from .flash_flowsheet import FS as flash_fs
from idaes.core.util import structfs
from idaes.core.util.doctesting import Docstring
from pyomo.environ import assert_optimal_termination


@pytest.mark.unit
def test_run_all():
    flash_fs.run_steps()
    assert_optimal_termination(flash_fs.result)


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
    assert FS.result.solver.status == SolverStatus.ok


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


#####
# Test utilities to find wrapped functions
#####


@pytest.mark.unit
def test_find_wrapped():
    from . import test_simple_wrap

    assert global_flowsheet(test_simple_wrap) is None
    assert wrapped_main(test_simple_wrap)


@pytest.mark.integration
def test_run_wrapped():
    from . import test_simple_wrap
    from pprint import pprint

    wmain = test_simple_wrap.my_main
    report = run_wrapped_main(wmain)
    assert report
    assert "actions" in report
    actions = report["actions"]
    for k in (
        "degrees_of_freedom",
        "model_variables",
        "capture_solver_output",
        "mermaid_diagram",
        "timings",
    ):
        assert k in actions
        print(f"action={k}")
        pprint(actions[k])


@pytest.mark.unit
def test_wrapped_main():
    from . import test_simple_wrap

    # with the doctest in that module, there are
    # actually 2 wrapped mains. Either will do.
    wmain = wrapped_main(test_simple_wrap)


@pytest.mark.unit
def test_context_solve_and_properties(monkeypatch):
    seen = {}

    class FakeSolver:
        def solve(self, model, tee):
            seen["call"] = (model, tee)
            return "solved"

    monkeypatch.setattr(fsrunner, "SolverFactory", lambda name: FakeSolver())

    ctx = Context(model="model", solver=None, tee=False)
    ctx.solve()
    assert seen["call"] == ("model", False)
    assert ctx.result == "solved"
    assert ctx.result == "solved"

    ctx.result = "updated"
    ctx.model = "new-model"
    ctx.solver = "solver"
    assert ctx.result == "updated"
    assert ctx.model == "new-model"
    assert ctx.solver == "solver"
    assert ctx.tee is False


@pytest.mark.unit
def test_base_flowsheet_runner_set_solver_branches(monkeypatch):
    default_solver = object()
    monkeypatch.setattr(fsrunner, "get_solver", lambda: default_solver)

    class SolverWithOptions:
        def __init__(self):
            self.options = None

    monkeypatch.setattr(fsrunner, "SolverFactory", lambda name: f"factory:{name}")

    r1 = BaseFlowsheetRunner()
    r1._set_solver()
    assert r1._context.solver is default_solver

    r2 = BaseFlowsheetRunner(solver="ipopt_v2")
    r2._set_solver()
    assert r2._context.solver == "factory:ipopt_v2"

    supplied = SolverWithOptions()
    r3 = BaseFlowsheetRunner(solver=supplied, solver_options={"tol": 1e-6})
    r3._set_solver()
    assert r3._context.solver is supplied
    assert r3._context.solver.options == {"tol": 1e-6}


@pytest.mark.unit
def test_base_flowsheet_runner_before_after_and_annotate_error():
    class TinyRunner(BaseFlowsheetRunner):
        STEPS = ("build", "middle", "end")

        def _create_model(self):
            return SimpleNamespace(created=True)

    rn = TinyRunner()

    @rn.step("build")
    def build(ctx):
        ctx["calls"] = ["build"]

    @rn.step("middle")
    def middle(ctx):
        if "calls" not in ctx:
            ctx["calls"] = []
        ctx["calls"].append("middle")

    @rn.step("end")
    def end(ctx):
        if "calls" not in ctx:
            ctx["calls"] = []
        ctx["calls"].append("end")

    rn.run_steps(before="end")
    assert rn["calls"] == ["build", "middle"]

    rn.reset()
    rn.run_steps(after="build")
    assert rn["calls"] == ["middle", "end"]

    v = Var()
    v.construct()
    with pytest.raises(ValueError, match="One of 'is_input', 'is_output' must be True"):
        rn.annotate_var(v, is_input=False, is_output=False)

    rn.annotate_var(v, key="v")
    ann = rn.annotated_vars
    print(f"@@ before: _ann = {rn._ann}")
    ann["v"]["title"] = "changed"
    print(f"@@ after: _ann = {rn._ann}")
    assert rn.annotated_vars["v"]["title"] != "changed"


@pytest.mark.unit
def test_flowsheetrunner_set_solve_step(monkeypatch):
    rn = FlowsheetRunner(solve_step="special_solve")
    assert rn.get_action("solver_output")._user_solve_step == "special_solve"

    monkeypatch.setattr(rn, "run_step", lambda name: setattr(rn, "_build_called", name))
    monkeypatch.setattr(
        rn, "run_steps", lambda **kwargs: setattr(rn, "_solve_called", kwargs)
    )
    rn.build()
    rn.solve_initial()
    assert rn._build_called == "build"
    assert rn._solve_called == {"last": "solve_initial"}
