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

# stdlib
from io import StringIO
import logging
import pprint
import sys
import time
from types import SimpleNamespace

import pytest
from pytest import approx
import pyomo.environ as pyo
from pyomo.network import Port
from .. import runner
from . import flash_flowsheet, hda_flowsheet
from ..fsrunner import FlowsheetRunner
from ..runner_actions import (
    ComponentList,
    SolverActionBase,
    CaptureSolverOutput,
    SolverResult,
    GetSolverResults,
    Diagnostics,
    MermaidDiagram,
    ModelVariables,
    NumericalIssuesData,
    StructuralIssuesData,
    Timer,
    UnitDofChecker,
    StreamTable,
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
def test_stream_table_action_create():
    obj = StreamTable(flash_flowsheet.FS)
    assert obj


@pytest.mark.unit
def test_stream_table_action_report():
    rn = hda_flowsheet.FS  # need a flowsheet with Arcs
    rn.reset()
    rn._actions = {}  # no other actions
    rn.add_action("streamtable", StreamTable)
    rn.build()  # don't need to solve
    report = rn.get_action("streamtable").report()
    print(f"report: {report}")
    # perform some basic checks on the contents
    assert report
    assert report.index[0].startswith("flow_mol")
    assert report.units[0] == "mole / second"
    assert "s03" in report.columns
    assert len(report.columns) == len(report.data[0])
    assert len(report.data) == len(report.index)
    v1 = report.data[0][0]
    assert len(v1) == 2
    assert v1[0] > 0
    assert v1[0] < 1
    assert v1[1] == "unfixed"
