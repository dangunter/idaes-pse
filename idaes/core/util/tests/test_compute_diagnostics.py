# -*- coding: utf-8 -*-
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
Tests for compute_diagnostics module
"""

from io import StringIO
from types import SimpleNamespace

import pytest

import idaes.core.util.compute_diagnostics as cd
from idaes.core.util.structfs.tests import flash_flowsheet


@pytest.fixture(scope="module")
def flash_build_model():
    """Build flash flowsheet (one time), and return its model"""
    flash_flowsheet.FS.build()
    return flash_flowsheet.FS.model.clone()


@pytest.fixture(scope="module")
def flash_solve_model():
    """Solve the flash flowsheet (one time), and return its model"""
    flash_flowsheet.FS.run_steps()
    return flash_flowsheet.FS.model.clone()


class _FakeVar:
    def __init__(
        self, name, value=0.0, lb=None, ub=None, fixed=False, units="dimensionless"
    ):
        self.name = name
        self.value = value
        self.lb = lb
        self.ub = ub
        self.fixed = fixed
        self._units = units

    def get_units(self):
        return self._units


def _make_diagnostics(monkeypatch, model=None, toolbox=None):
    if model is None:
        model = object()
    if toolbox is None:
        toolbox = SimpleNamespace(model=model, config=SimpleNamespace())
    monkeypatch.setattr(cd, "get_jacobian", lambda m: ("jac", "nlp"))
    return cd.DiagnosticsData(toolbox=toolbox)


@pytest.mark.unit
def test_init_requires_toolbox_or_model():
    with pytest.raises(ValueError, match="cannot both be None"):
        cd.DiagnosticsData()


@pytest.mark.unit
def test_block_list_names_and_vcset_from_blocks():
    block = SimpleNamespace(name="b1")
    indexed_block = [SimpleNamespace(name="b2"), SimpleNamespace(name="b3")]

    assert cd._block_list_names([block, indexed_block]) == ["b1", "b2", "b3"]

    vcset = cd.VCSet.from_blocks([block], [indexed_block])
    assert vcset.variables == ["b1"]
    assert vcset.constraints == ["b2", "b3"]


@pytest.mark.unit
def test_all_as_dict_and_json(monkeypatch):
    diag = _make_diagnostics(monkeypatch)

    monkeypatch.setattr(
        diag,
        "variables",
        lambda: cd.ComponentList(
            components=[
                cd.ComponentListData(tag="variables", description="vars", names=["x"])
            ]
        ),
    )
    monkeypatch.setattr(
        diag,
        "constraints",
        lambda: cd.ComponentList(
            components=[
                cd.ComponentListData(tag="constraints", description="cons", names=["c"])
            ]
        ),
    )
    monkeypatch.setattr(
        diag,
        "structural_issues",
        lambda: cd.StructuralIssuesData(
            warnings=cd.StructuralWarningsData(dof=1),
            cautions=cd.StructuralCautionsData(),
        ),
    )
    monkeypatch.setattr(
        diag,
        "numerical_issues",
        lambda: cd.NumericalIssuesData(warnings=cd.NumericalWarningsData()),
    )

    data = diag.all_as_dict()
    assert data["variables"]["components"][0]["names"] == ["x"]
    assert data["structural_issues"]["warnings"]["dof"] == 1

    json_text = diag.all_as_json(sort_keys=True)
    assert '"variables"' in json_text
    assert '"dof": 1' in json_text

    stream = StringIO()
    assert diag.all_as_json(stream=stream, sort_keys=True) is None
    assert stream.getvalue() == json_text


@pytest.mark.unit
def test_get_variables_for_condition_at_or_outside_bounds(monkeypatch):
    config = SimpleNamespace(variable_bounds_violation_tolerance=1e-6)
    diag = _make_diagnostics(
        monkeypatch, toolbox=SimpleNamespace(model=object(), config=config)
    )
    vars_found = [
        _FakeVar("x", value=5.0, lb=0.0, ub=1.0, fixed=False),
        _FakeVar("y", value=-1.0, lb=0.0, ub=2.0, fixed=True),
    ]
    monkeypatch.setattr(
        cd, "vars_violating_bounds", lambda model, tolerance: vars_found
    )

    data = diag._get_variables_for_condition(cd.VariableCondition.at_or_outside_bounds)

    assert data.tag == cd.VariableCondition.at_or_outside_bounds.value
    assert data.names == ["x (free)", "y (fixed)"]
    assert data.values == [5.0, -1.0]
    assert data.ranges == [(0.0, 1.0), (0.0, 2.0)]
    assert data.bounds == {"tol": 1e-6}
    assert data.bounds_desc == "value range"


@pytest.mark.unit
def test_get_constraints_for_condition_mismatched_terms(monkeypatch):
    diag = _make_diagnostics(monkeypatch)
    monkeypatch.setattr(
        diag,
        "_verify_active_variables_initialized",
        lambda toolbox: None,
    )
    monkeypatch.setattr(
        diag,
        "_collect_constraint_mismatches",
        lambda toolbox: (["c1: large mismatch", "c2: small mismatch"], [], []),
    )

    data = diag._get_constraints_for_condition(cd.ConstraintCondition.mismatched_terms)

    assert data.tag == cd.ConstraintCondition.mismatched_terms.value
    assert data.names == ["c1", "c2"]
    assert data.details == ["large mismatch", "small mismatch"]
    assert data.values == [None, None]


@pytest.mark.integration
def test_diagnostics_built_flash(flash_build_model):
    diag = cd.DiagnosticsData(model=flash_build_model)
    # merely built model will not work
    with pytest.raises(cd.DiagnosticsError):
        results_dict = diag.all_as_dict()
    with pytest.raises(cd.DiagnosticsError):
        results_json = diag.all_as_json()
    # solved model will work


@pytest.mark.integration
def test_diagnostics_solved_flash(flash_solve_model):
    diag = cd.DiagnosticsData(model=flash_solve_model)
    results_dict = diag.all_as_dict()
    assert results_dict
    results_json = diag.all_as_json()
    assert results_json
    # check structure and contents of returned data
    struct_cautions = results_dict["structural_issues"]["cautions"]
    assert struct_cautions["zero_vars"] == [
        "fs.flash.control_volume.heat[0.0]",
        "fs.flash.control_volume.deltaP[0.0]",
    ]
