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
#################################################################################import pprint
import sys
import time
from types import SimpleNamespace

import pytest
from pytest import approx
import pyomo.environ as pyo
from pyomo.network import Port
from .. import runner
from ..fsrunner import FlowsheetRunner
from ..runner_actions import (
    ComponentList,
    CaptureSolverOutput,
    Diagnostics,
    MermaidDiagram,
    ModelVariables,
    NumericalIssuesData,
    StructuralIssuesData,
    Timer,
    UnitDofChecker,
)
from . import flash_flowsheet


@pytest.mark.unit
def test_class_timer(monkeypatch):
    def step_name(i):
        return f"step{i}"

    # n runs with m steps per run
    n, m = 2, 3

    # set up Timer action attached to Runner instance
    steps = [step_name(i) for i in range(m)]
    rnr = runner.Runner(steps)
    for step_i in steps:
        rnr.add_step(step_i, lambda ctx: None)
    timer = Timer(rnr)

    # control results of next batch of calls to time.time()
    run_time = 10
    # generate list of start/end timings for n runs with m steps
    times = []
    for i in range(n):
        times.append(i * run_time)
        for j in range(m):
            times.append(i * run_time + j)
            times.append(i * run_time + j + 1)
        times.append((i + 1) * run_time)
    # print(f"times: {times}")
    times = iter(times)
    monkeypatch.setattr(time, "time", lambda: next(times))

    # emulate a 'n' runs with 'm' steps each
    for i in range(n):
        timer.before_run()
        for j in range(m):
            name = step_name(j)
            # print(f"run {i}, step {j}")
            timer.before_step(name)
            timer.after_step(name)
        timer.after_run()

    # check that times match expected
    time_history = timer.get_history()
    # print(time_history)
    assert len(time_history) == n
    for run_num, run in enumerate(time_history):
        run_t = run["run"]
        # each run should take 'run_time' sec
        assert run_t == run_time
        steps = run["steps"]
        assert len(steps) == m
        for i in range(m):
            step_t = steps[step_name(i)]
            # each step should take 1 sec
            assert step_t == 1
        # inclusive time is sum of step times
        assert run["inclusive"] == m
        # exclusive time is run time minus step times
        assert run["exclusive"] == run_time - m

    # check report, which contains last run only
    rpt = timer.report()
    for i in range(m):
        step_t = rpt.timings[step_name(i)]
        assert step_t == 1


@pytest.mark.unit
def test_unit_dof_action_base():
    rn = flash_flowsheet.FS
    rn.reset()
    turn_off_mermaid_server(rn)

    def check_step(name, data):
        print(f"check_step {name} data: {data}")
        assert "fs.flash" in data
        if name == "solve_initial":
            assert data["fs.flash"] == 0

    def check_run(step_dof, model_dof):
        assert model_dof == 0

    rn.add_action(
        "check_dof",
        UnitDofChecker,
        "fs",
        ["build", "solve_initial"],
        check_step,
        check_run,
    )

    rn.run_steps("build", "solve_initial")

    pprint.pprint(rn.get_action("check_dof").get_dof())


@pytest.mark.unit
def test_unit_dof_action_getters():
    rn = flash_flowsheet.FS
    rn.reset()
    turn_off_mermaid_server(rn)

    aname = "check_dof"
    rn.add_action(
        aname,
        UnitDofChecker,
        "fs",
        ["build", "solve_initial"],
    )
    rn.run_steps()

    act = rn.get_action(aname)

    steps = act.steps()
    dofs = []
    for s in steps:
        step_dof = act.get_dof()[s]
        assert step_dof
        dofs.append(step_dof)
    assert dofs[0] != dofs[1]

    assert act.steps() == act.steps(only_with_data=True)


@pytest.mark.unit
def test_dof_report():
    rn = flash_flowsheet.FS
    rn.reset()
    turn_off_mermaid_server(rn)
    check_steps = (
        "build",
        "set_operating_conditions",
        "initialize",
        "solve_initial",
    )
    rn.add_action("dof", UnitDofChecker, "fs", check_steps)
    rn.run_steps()
    report = rn.get_action("dof").report()
    assert report
    report_data = report.model_dump()
    assert report_data["model"] == 0  # model has DOF=0
    for step_name in check_steps:
        assert step_name in report_data["steps"]
        for unit, value in report_data["steps"][step_name].items():
            assert value >= 0  # DOF > 0 in all (step, unit)


@pytest.mark.unit
def test_mermaid_report():
    rn = flash_flowsheet.FS
    rn.reset()
    rn.add_action("diagram", MermaidDiagram)
    rn.run_steps()
    action = rn.get_action("diagram")
    action.set_model_root("fs")
    report = action.report()
    if action.diagram is None:
        print("Connectivity not installed")
        assert report == {}
    else:
        print("Connectivity IS installed")
        assert report.diagram != {}


@pytest.mark.unit
def test_unit_dof_checker_empty_steps_error():
    with pytest.raises(ValueError, match="At least one step name must be provided"):
        UnitDofChecker(runner.Runner(["build"]), "fs", [])


@pytest.mark.unit
def test_unit_dof_checker_after_step_after_run_and_summary(monkeypatch):
    class FakeUnit:
        def __init__(self, name):
            self.name = name

    class FakeFlowsheet:
        def __init__(self):
            self.flash = FakeUnit("fs.flash")
            self.heater = FakeUnit("fs.heater")

        def component_objects(self, descend_into=True):
            return [self.flash, self.heater, object()]

    class FakeRunner:
        def __init__(self):
            self.model = SimpleNamespace(fs=FakeFlowsheet())

        def normalize_name(self, name):
            return runner.Runner.normalize_name(name)

        def list_steps(self):
            return ["build", "solve"]

    seen = {}

    def step_func(name, data):
        seen["step"] = (name, data.copy())

    def run_func(step_dof, model_dof):
        seen["run"] = (step_dof.copy(), model_dof)

    monkeypatch.setattr(
        UnitDofChecker,
        "_is_unit_model",
        staticmethod(lambda block: isinstance(block, FakeUnit)),
    )
    monkeypatch.setattr(
        UnitDofChecker,
        "_get_dof",
        staticmethod(
            lambda block, fix_inlets=True: {"fs.flash": 0, "fs.heater": 1}[block.name]
        ),
    )
    monkeypatch.setattr(
        "idaes.core.util.structfs.runner_actions.degrees_of_freedom",
        lambda block: 2 if isinstance(block, FakeFlowsheet) else -1,
    )

    action = UnitDofChecker(
        FakeRunner(),
        "fs",
        ["build"],
        step_func=step_func,
        run_func=run_func,
    )

    action.after_step("solve")
    assert action.get_dof() == {}

    action.after_step("build")
    action.after_run()

    assert action.get_dof() == {"build": {"fs": 2, "fs.flash": 0, "fs.heater": 1}}
    assert action.get_dof_model() == 2
    assert action.steps(only_with_data=True) == ["build"]
    assert set(action.steps()) == {"build"}
    assert seen["step"][0] == "build"
    assert seen["step"][1]["fs.heater"] == 1
    assert seen["run"][1] == 2

    text = action.summary(stream=None)
    assert "Degrees of freedom: 2" in text
    assert "build:" in text
    assert "fs.flash" in text

    step_text = action.summary(stream=None, step="build")
    assert "fs.heater" in step_text
    assert "build:" not in step_text


@pytest.mark.unit
def test_unit_dof_checker_get_dof_fixes_and_frees_inlets(monkeypatch):
    class FakeScalarPort:
        def __init__(self, name, fixed=False):
            self.name = name
            self._fixed = fixed
            self.fix_calls = 0
            self.free_calls = 0

        def is_fixed(self):
            return self._fixed

        def fix(self):
            self._fixed = True
            self.fix_calls += 1

        def free(self):
            self._fixed = False
            self.free_calls += 1

    class FakeBlock:
        def __init__(self, components):
            self._components = components

        def component_objects(self, descend_into=False):
            return self._components

    inlet = FakeScalarPort("unit.inlet")
    recycle = FakeScalarPort("unit.recycle", fixed=True)
    outlet = FakeScalarPort("unit.outlet")
    block = FakeBlock([inlet, recycle, outlet])

    monkeypatch.setattr(
        "idaes.core.util.structfs.runner_actions.ScalarPort", FakeScalarPort
    )
    monkeypatch.setattr(
        "idaes.core.util.structfs.runner_actions.degrees_of_freedom", lambda _: 7
    )

    assert UnitDofChecker._get_dof(block) == 7
    assert inlet.fix_calls == 1
    assert inlet.free_calls == 1
    assert recycle.fix_calls == 0
    assert recycle.free_calls == 0
    assert outlet.fix_calls == 0


@pytest.mark.unit
def test_capture_solver_output_default_and_custom_step():
    action = CaptureSolverOutput(runner.Runner(["solve_model", "custom_solve"]))
    stdout = sys.stdout

    action.before_step("build")
    assert sys.stdout is stdout

    action.before_step("solve_model")
    print("solver output line")
    action.after_step("solve_model")
    assert sys.stdout is stdout
    assert action.report().output["solve_model"] == "solver output line\n"

    action.set_solve_step("custom_solve")
    action.before_step("solve_model")
    assert sys.stdout is stdout

    action.before_step("custom_solve")
    print("custom output")
    action.after_step("custom_solve")
    assert action.report().output["custom_solve"] == "custom output\n"


@pytest.mark.unit
def test_model_variables_helpers():
    action = ModelVariables(FlowsheetRunner())
    m = pyo.ConcreteModel()
    m.fs = pyo.Block()
    m.fs.scalar = pyo.Var(initialize=3, units=pyo.units.m)
    m.fs.indexed = pyo.Var([0.0], initialize=4)
    m.fs.param = pyo.Param([1], initialize=5, mutable=True)
    m.fs.inlet = Port()
    m.fs.inlet.add(m.fs.scalar, "flow")

    values, indexed = action._get_values(m.fs.scalar, action.VAR_TYPE)
    assert indexed is False
    assert len(values) == 1
    assert values[0][0] is None
    assert values[0][1] == 3
    assert values[0][2] == "m"
    assert values[0][3] is False
    assert values[0][5] is None
    assert values[0][6] is None
    assert values[0][7] == "Reals"

    values, indexed = action._get_values(m.fs.param, action.PARAM_TYPE)
    assert indexed is True
    assert values == [(1, 5, "")]

    tree = {}
    action._add_block(tree, "fs.indexed[0.0].value", "payload")
    assert tree["fs"]["indexed[0.0]"]["value"] == "payload"

    action._extract_vars(m)
    report = action.report()
    print(f"report: variables={report.variables}\nport_aliases={report.port_aliases}")
    assert report.variables["fs"]["scalar"][0] == "V"
    assert report.variables["fs"]["param"][0] == "P"
    assert report.port_aliases["fs.inlet.flow"] == "fs.scalar"


@pytest.mark.unit
def test_mermaid_diagram_with_mocked_connectivity(monkeypatch):
    calls = {}

    class FakeConnectivity:
        def __init__(self, input_model):
            calls["input_model"] = input_model

    class FakeMermaid:
        def __init__(self, conn, component_images=True):
            calls["conn"] = conn
            calls["component_images"] = component_images

        def write(self, _):
            return "graph TD\nA-->B"

    action = MermaidDiagram(runner.Runner([]))
    action._runner.model = SimpleNamespace(fs=SimpleNamespace(unit="root"))

    monkeypatch.setattr(
        "idaes.core.util.structfs.runner_actions.Connectivity", FakeConnectivity
    )
    monkeypatch.setattr("idaes.core.util.structfs.runner_actions.Mermaid", FakeMermaid)

    action.show_unit_images(False)
    action.set_model_root("fs")
    action.after_run()

    assert calls["input_model"] is action._runner.model.fs
    assert calls["component_images"] is False
    assert action.report().diagram == ["graph TD", "A-->B"]


@pytest.mark.unit
def test_diagnostics_report_with_and_without_model(monkeypatch):
    action = Diagnostics(runner.Runner([]))
    action._runner.model = None
    action.after_run()

    empty = action.report()
    assert empty.valid is False
    assert empty.variables is None

    class FakeDiagnosticsData:
        def __init__(self, model):
            self.model = model

        def all_as_obj(self):
            return {
                "variables": ComponentList(components=[]),
                "constraints": ComponentList(components=[]),
                "structural_issues": StructuralIssuesData(warnings={}, cautions={}),
                "numerical_issues": NumericalIssuesData(warnings={}, cautions={}),
            }

    monkeypatch.setattr(
        "idaes.core.util.structfs.runner_actions.DiagnosticsData", FakeDiagnosticsData
    )

    action._runner.model = object()
    action.after_run()
    report = action.report()

    assert report.valid is True
    assert isinstance(report.variables, ComponentList)
    assert isinstance(report.constraints, ComponentList)


@pytest.mark.unit
def test_model_variables():
    rn = flash_flowsheet.FS
    rn.reset()
    turn_off_mermaid_server(rn)

    # get model vars
    rn.run_steps()
    mv = rn.get_action("model_variables")
    report = mv.report()

    # check the report root
    fs = report.variables["fs"]
    assert fs

    # check flash unit's heat_duty variable
    print(fs["flash"]["heat_duty"])
    assert fs["flash"]["heat_duty"] == [
        "V",
        True,
        [(0.0, 0, True, True, None, None, "Reals")],
    ]


def turn_off_mermaid_server(runner):
    dg = runner.get_action("mermaid_diagram")
    dg.show_unit_images(False)
