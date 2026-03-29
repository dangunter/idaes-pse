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
import json
import logging
import pprint
import sys
import time
from pathlib import Path
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
def test_class_timer():
    timer = Timer(runner.Runner([]))
    n, m = 2, 3
    for i in range(n):
        timer.before_run()
        time.sleep(0.1)
        for j in range(m):
            name = f"step{j}"
            timer.before_step(name)
            time.sleep(0.1 * (j + 1))
            timer.after_step(name)
        time.sleep(0.1)
        timer.after_run()

    s = timer.get_history()
    # [ {'run': 0.8005404472351074,
    #    'steps': {
    #       'step0': 0.10010385513305664,
    #       'step2': 0.3000965118408203,
    #       'step1': 0.20009303092956543
    #      },
    #     'inclusive': 0.6002933979034424,
    #     'exclusive': 0.20024704933166504
    #    },
    #    ...
    # ]
    eps = 0.1  # big variance needed for Windows
    for r in s:
        print(f"Timings: {r}")
        assert r["run"] == approx(0.8, abs=eps)
        assert r["inclusive"] + r["exclusive"] == approx(r["run"])
        for name, t in r["steps"].items():
            assert t == approx(0.1 + 0.1 * i, abs=eps)


@pytest.mark.component
def test_timer_runner():
    rn = runner.Runner(["step1", "step2", "step3"])

    @rn.step("step1")
    def sleepy1(context):
        time.sleep(0.1)

    @rn.step("step2")
    def sleepy2(context):
        time.sleep(0.1)

    @rn.step("step3")
    def sleepy3(context):
        time.sleep(0.1)

    rn.add_action("timer", Timer)

    rn.run_steps()

    s = rn.get_action("timer").get_history()

    eps = 0.1  # big variance needed for Windows
    for r in s:
        print(f"Timings: {r}")
        assert r["run"] == approx(0.3, abs=eps)
        assert r["inclusive"] + r["exclusive"] == approx(r["run"])
        for name, t in r["steps"].items():
            # assert name == f"step{i + 1}"
            assert t == approx(0.1, abs=eps)


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
def test_timer_report():
    rn = flash_flowsheet.FS
    rn.reset()
    turn_off_mermaid_server(rn)
    rn.add_action("timer", Timer)
    rn.run_steps()
    report = rn.get_action("timer").report()
    # {'build': 0.053082942962646484,
    # 'set_operating_conditions': 0.0004742145538330078,
    # 'initialize': 0.22397446632385254,
    # 'set_solver': 7.581710815429688e-05,
    # 'solve_initial': 0.03623509407043457}
    expect_steps = (
        "build",
        "set_operating_conditions",
        "initialize",
        "set_solver",
        "solve_initial",
    )
    assert report
    print(f"Hey! here's the report: {report}")
    for step_name in expect_steps:
        assert step_name in report.timings
        assert report.timings[step_name] < 1


@pytest.mark.unit
def test_timer_missing_begin_and_empty_summary(monkeypatch, caplog):
    rn = runner.Runner(["step1", "step2"])
    rn.add_step("step1", lambda ctx: None)
    rn.add_step("step2", lambda ctx: None)
    timer = Timer(rn)

    assert timer.summary() == ""

    times = iter([10.0, 11.0, 13.5, 20.0])
    monkeypatch.setattr(time, "time", lambda: next(times))

    timer.before_run()
    timer.before_step("step1")
    timer.after_step("step1")
    timer.after_run()

    assert len(timer) == 1
    assert timer.step_times[-1] == {"step1": 2.5, "step2": -1}
    assert timer.report().timings == {"step1": 2.5, "step2": -1}


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
