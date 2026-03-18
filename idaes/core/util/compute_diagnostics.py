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
Compute diagnostics values and return as Pydantic data models.

Usage::

    from idaes.core.util.compute_diagnostics import DiagnosticsData, DiagnosticsToolbox

    model = build_and_run_model()  # XXX: replace with real code
    toolbox = DiagnosticsToolbox(model)
    diag = DiagnosticsData(toolbox)
    # get and print all the info as JSON
    print(diag.all_as_json(indent=2))
    # same, but write to a file
    diag.all_as_json(open("diagnostics.json", "w"))
    # also can get as a dict
    d = diag.all_as_dict()
    # example of using the dict
    def count(data, key, subkey):
        x = data[key][subkey]
        return sum(map(lambda y: x[y] is not None, x))
    print(f"{count(d, 'structural', 'warnings')} structural warnings")
    print(f"{count(d, 'structural', 'cautions')} structural cautions")
    print(f"{count(d, 'numerical', 'warnings')} numerical warnings")
"""

__author__ = "Dan Gunter (LBNL)"

# stdlib
from enum import StrEnum
import json
from math import log
from typing import Callable, TypeVar

# third party
from pydantic import BaseModel, Field, computed_field
from pyomo.environ import (
    # Binary,
    # Integers,
    # Block,
    # check_optimal_termination,
    # ComponentMap,
    # ConcreteModel,
    Constraint,
    # Expression,
    # Objective,
    # Param,
    # RangeSet,
    # Set,
    # SolverFactory,
    value,
    # Var,
)
from pyomo.util.check_units import identify_inconsistent_units

# package
from idaes.core.scaling.util import (
    get_jacobian,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    # activated_blocks_set,
    # deactivated_blocks_set,
    # activated_equalities_set,
    # deactivated_equalities_set,
    # activated_inequalities_set,
    # deactivated_inequalities_set,
    # activated_objectives_set,
    # deactivated_objectives_set,
    variables_in_activated_constraints_set,
    variables_not_in_activated_constraints_set,
    variables_with_none_value_in_activated_equalities_set,
    # number_activated_greybox_equalities,
    # number_deactivated_greybox_equalities,
    # activated_greybox_block_set,
    # deactivated_greybox_block_set,
    # greybox_block_set,
    # unfixed_greybox_variables,
    # greybox_variables,
    # large_residuals_set,
    large_residuals_set,
    variables_near_bounds_set,
)
from .model_diagnostics import (
    DiagnosticsToolbox,
    _extreme_jacobian_rows,
    _extreme_jacobian_columns,
    _var_in_block,
    _vars_fixed_to_zero,
    _vars_near_zero,
    _vars_violating_bounds,
    _vars_with_extreme_values,
    _vars_with_none_value,
    check_parallel_jacobian,
)

# -------------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------------


def _block_list_names(blocks) -> list[str]:
    """Retrieve names of blocks, including indexed blocks, into a list."""
    b_items = []
    for b in blocks:
        if hasattr(b, "name"):
            b_items.append(b.name)
        else:  # indexed
            for i in b:
                b_items.append(i.name)
    return b_items


class ComponentListData(BaseModel):
    #: simplify check for whether this has any data
    @computed_field
    @property
    def empty(self) -> bool:
        return len(self.names) == 0

    #: short description of condition these components satisfy
    tag: str
    #: longer description of condition these components satisfy
    description: str
    #: names of components
    names: list[str] = Field(default_factory=list)
    #: optional values of components
    values: list[float | None] = Field(default_factory=list)
    value_format: str = ".6E"
    #: optional bounds
    bounds: dict = Field(default_factory=dict)
    bounds_desc: str | None = None
    #: optional descriptive details for components
    details: list[str] = Field(default_factory=list)


class VariableListData(ComponentListData):
    #: units for each variable
    units: list[str] = Field(default_factory=list)
    #: range of each variable (optional)
    ranges: list[tuple[float | None, float | None]] = Field(default_factory=list)


ConstraintListData = ComponentListData  # identical, at least for now


class ComponentList(BaseModel):
    components: list[ComponentListData]


class VCSet(BaseModel):
    """Combined variables and constraints.

    Not returned directly; used by `StructuralWarningsData`.
    """

    variables: list[str]
    constraints: list[str]

    @classmethod
    def from_blocks(cls, var_blocks, const_blocks) -> "VCSet":
        v_items = _block_list_names(var_blocks)
        c_items = _block_list_names(const_blocks)
        return VCSet(variables=v_items, constraints=c_items)


class EvalErrorData(BaseModel):
    component_name: str
    message: str


class StructuralWarningsData(BaseModel):
    """Structural warnings.

    All possibilities are listed, the value will be None if it is not
    an issue for this model.
    """

    dof: int | None = None
    inconsistent_units: list[str] | None = None
    underconstrained_set: VCSet | None = None
    overconstrained_set: VCSet | None = None
    evaluation_errors: list[EvalErrorData] | None = None


class StructuralCautionsData(BaseModel):
    """Structural cautions.

    All possibilities are listed, the value will be None if it is not
    an issue for this model.
    """

    zero_vars: list[str] | None = None
    unused_vars_free: list[str] | None = None
    unused_vars_fixed: list[str] | None = None


class StructuralIssuesData(BaseModel):
    """Structural issues: warnings and cautions."""

    warnings: StructuralWarningsData
    cautions: StructuralCautionsData


class NumericalWarningsData(BaseModel):
    """Numerical warnings."""

    constraints_with_large_residuals: ComponentListData | None = None
    constraints_with_extreme_jacobians: ComponentListData | None = None
    constraints_parallel: ComponentList | None = None
    variables_parallel: ComponentList | None = None


class NumericalIssuesData(BaseModel):
    """Numerical warnings and.. other things"""

    warnings: NumericalWarningsData


# -------------------------------------------------------------------------------
# Interface
# -------------------------------------------------------------------------------


class VariableCondition(StrEnum):
    external = "are external variables that appear in constraints"
    unused = "do not appear in any activated constraints"
    fixed_to_zero = "are fixed to zero"
    at_or_outside_bounds = "have values that fall at or outside their bounds"
    with_none_value = "have a value of none"
    value_near_zero = "have a value near zero"
    extreme_values = "have extreme values"
    near_bounds = "have values close to their bounds"
    extreme_jacobians = "corresponding to Jacobian columns with extreme norms"


class ConstraintCondition(StrEnum):
    large_residuals = "have residuals greater than specified tolerance"
    no_free_variables = "do not have any free variables"
    canceling_terms = "have additive terms which potentially cancel each other"
    mismatched_terms = "have additive terms of different magnitude"
    extreme_jacobians = "corresponding to Jacobian rows with extreme L2 norms"


class DiagnosticsData:
    """Interface to get diagnostics data"""

    VC = VariableCondition  # alias

    def __init__(self, toolbox: DiagnosticsToolbox = None, model=None):
        if toolbox is None:
            if model is None:
                raise ValueError("Arguments `toolbox` and `model` cannot both be None")
            self._toolbox = DiagnosticsToolbox(model)
            self._model = model
        else:
            self._toolbox = toolbox
            self._model = self._toolbox.model
        # get jacobian only once (since model does not change)
        self._jac, self._nlp = get_jacobian(self._model)

    def all_as_dict(self) -> dict[str, dict]:
        """Return all the diagnostics as a single dictionary.

        Returns:
            dict: Diagnostics data
        """
        return {k: v.model_dump() for k, v in self.all_as_obj()}

    def all_as_json(self, stream=None, **kwargs) -> str | None:
        """Write (or return) all the diagnostics as JSON-formatted text.

        Args:
            stream: Stream to write JSON. If None, return a string instead.
            kwargs: Keyword arguments passed through to `json.dump()` or `dumps()` method

        Returns:
            str | None: String returned when `stream` argument is None, otherwise None
        """
        obj = self.all_as_dict()
        if stream is None:
            result = json.dumps(obj, **kwargs)
        else:
            json.dump(obj, stream, **kwargs)
            result = None
        return result

    def all_as_obj(self) -> dict[str, BaseModel]:
        """Same as `all_as_dict` except each top-level value is an object instead of a dict.

        Note that the keys returned here should match the attributes
        of `idaes.core.util.structfs.runner_actions.Diagnostics.Report`.

        Returns:
            dict: Diagnostics data with string keys and Pydantic model object values
        """
        return {
            "variables": self.variables(),
            "constraints": self.constraints(),
            "structural_issues": self.structural_issues(),
            "numerical_issues": self.numerical_issues(),
        }

    def variables(
        self, conditions: list[VariableCondition] | None = None
    ) -> ComponentList | list[dict]:
        """Compute the list of variables meeting some condition(s).

        Args:
            conditions: Zero or more conditions. If zero, look for all
                  conditions. If one, just return one. If multiple, return one per condition.

        Returns:
            Selected variables and associated metadata
        """
        return self._get_components_for_conditions(
            conditions, VariableCondition, self._get_variables_for_condition
        )

    def constraints(
        self, conditions: list[ConstraintCondition] | None = None
    ) -> ComponentList | list[dict]:
        """Compute the list of constraints meeting some condition(s).

        Args:
            conditions: Zero or more conditions. If zero, look for all
                  conditions. If one, just return one. If multiple, return one per condition.

        Returns:
            Selected constraints and associated metadata
        """
        return self._get_components_for_conditions(
            conditions, ConstraintCondition, self._get_constraints_for_condition
        )

    def _get_components_for_conditions(self, conditions, cond_enum, getter):
        if not conditions:
            conditions = list(cond_enum)
        components = []
        for cond in conditions:
            item = getter(cond)
            components.append(item)
        return ComponentList(components=components)

    def structural_issues(
        self, evaluation_errors=True, unit_consistency=True
    ) -> StructuralIssuesData:
        """Compute structural warnings and cautions.

        Args:
            evaluation_errors: Include potential evaluation errors
            unit_consistency: Include unit consistency checks

        Returns:
            Found structural issues
        """
        tbx, model = self._toolbox, self._toolbox.model
        uc = [] if unit_consistency else identify_inconsistent_units(model)
        uc_var, uc_con, oc_var, oc_con = tbx.get_dulmage_mendelsohn_partition()
        w, c = StructuralWarningsData(), StructuralCautionsData()

        # Warnings

        dof = degrees_of_freedom(model)
        if dof != 0:
            w.dof = dof

        if len(uc) > 0:
            w.inconsistent_units = uc
        if len(uc_var) + len(uc_con) > 0:
            # uc_set = set(uc_var) + set(uc_con)
            w.underconstrained_set = VCSet.from_blocks(uc_var, uc_con)
        if len(oc_var) + len(oc_con) > 0:
            w.overconstrained_set = VCSet.from_blocks(oc_var, oc_con)

        if not evaluation_errors:
            eval_warnings = tbx._collect_potential_eval_errors()
            if len(eval_warnings) > 0:
                w.evaluation_errors = []
                for ew_raw in eval_warnings:
                    ew_comp, ew_msg = ew_raw.split(":")
                    ee = EvalErrorData(
                        component_name=ew_comp.strip(), message=ew_msg.strip()
                    )
                    w.evaluation_errors.append(ee)

        # Cautions

        zero_vars = _vars_fixed_to_zero(model)
        if len(zero_vars) > 0:
            c.zero_vars = _block_list_names(zero_vars)

        unused_vars = variables_not_in_activated_constraints_set(model)
        if len(unused_vars) > 0:
            uv_free, uv_fixed = [], []
            for v in unused_vars:
                if v.fixed:
                    uv_fixed.append(v)
                else:
                    uv_free.append(v)
            if uv_fixed:
                c.unused_vars_fixed = _block_list_names(uv_fixed)
            if uv_free:
                c.unused_vars_free = _block_list_names(uv_free)

        return StructuralIssuesData(warnings=w, cautions=c)

    def numerical_issues(self, parallel_components=True) -> ComponentList:
        # warnings
        tbx, model = self._toolbox, self._toolbox.model
        wrn = NumericalWarningsData()

        data = self._get_constraints_for_condition(ConstraintCondition.large_residuals)
        if not data.empty:
            wrn.constraints_with_large_residuals = data

        data = self._get_constraints_for_condition(
            ConstraintCondition.extreme_jacobians
        )
        if not data.empty:
            wrn.constraints_with_extreme_jacobians = data

        if parallel_components:
            partol = tbx.config.parallel_component_tolerance
            for direction, ctype in ("row", "constraint"), ("column", "variable"):
                pairs = check_parallel_jacobian(
                    model,
                    tolerance=partol,
                    direction=direction,
                    jac=self._jac,
                    nlp=self._nlp,
                )
                if len(pairs) == 0:
                    continue
                items = [
                    ComponentListData(
                        tag=f"parallel {ctype}",
                        description=f"nearly parallel {ctype}",
                        names=[c1.name, c2.name],
                        bounds={"tol": partol},
                    )
                    for c1, c2 in pairs
                ]
                comp_list = ComponentList(components=items)
                if ctype == "constraint":
                    wrn.constraints_parallel = comp_list
                else:
                    wrn.variables_parallel = comp_list

        return NumericalIssuesData(warnings=wrn)

    def _get_variables_for_condition(
        self, cond: VariableCondition
    ) -> ComponentListData:
        tbx = self._toolbox  # alias
        kwargs = {}  # additional keywords for return value constructor
        desc = str(cond)  # default description
        details, values = None, None
        bounds, bounds_desc = {}, None
        names, ranges, values, details, units = [], [], [], [], []

        def _set_variables(
            it,
            nm=names,
            rg=ranges,
            vl=values,
            dt=details,
            un=units,
            nm_func=None,
            vl_func=None,
            dt_func=None,
            rg_func=None,
            un_func=None,
        ):
            for v in it:
                nm.append(v.name if nm_func is None else nm_func(v))
                vl.append(v.value if vl_func is None else vl_func(v))
                dt.append("" if dt_func is None else dt_func(v))
                rg.append((v.lb, v.ub) if rg_func is None else rg_func(v))
                if un_func is None:
                    u = str(v.get_units())
                    # replace 'None' with an empty string
                    # -- to match `runner_actions.ModelVariables`
                    if u == "None":
                        u == ""
                else:
                    u = un_func(v)
                un.append(u)

        if cond == VariableCondition.external:
            _set_variables(variables_in_activated_constraints_set(tbx._model))
        elif cond == VariableCondition.unused:
            _set_variables(variables_not_in_activated_constraints_set(tbx._model))
        elif cond == VariableCondition.fixed_to_zero:
            _set_variables(_vars_fixed_to_zero(tbx._model))
        elif cond == VariableCondition.at_or_outside_bounds:
            t_zero = tbx.config.variable_bounds_violation_tolerance
            _set_variables(
                _vars_violating_bounds(tbx._model, tolerance=t_zero),
                nm_func=lambda v: f"{v.name} ({'fixed' if v.fixed else 'free'})",
            )
            bounds = {"tol": t_zero}
            bounds_desc = "value range"
        elif cond == VariableCondition.with_none_value:
            _set_variables(_vars_with_none_value(tbx._model))
        elif cond == VariableCondition.value_near_zero:
            t_zero = tbx.config.variable_zero_value_tolerance
            _set_variables(_vars_near_zero(tbx._model, t_zero))
            bounds = {"tol": t_zero}
            bounds_desc = "zero value"
        elif cond == VariableCondition.extreme_values:
            t_small, t_large, t_zero = (
                tbx.config.variable_small_value_tolerance,
                tbx.config.variable_large_value_tolerance,
                tbx.config.variable_zero_value_tolerance,
            )
            _set_variables(
                _vars_with_extreme_values(
                    model=tbx._model, large=t_large, small=t_small, zero=t_zero
                )
            )
            bounds = {"small": t_small, "large": t_large, "zero": t_zero}
            bonds_desc = "extreme values"
        elif cond == VariableCondition.near_bounds:
            t_abs = tbx.config.variable_bounds_absolute_tolerance
            t_rel = tbx.config.variable_bounds_relative_tolerance
            _set_variables(
                variables_near_bounds_set(tbx._model, abs_tol=t_abs, rel_tol=t_rel)
            )
            bounds = {"abs": t_abs, "rel": t_rel}
            bounds_desc = "near bounds"
        elif cond == VariableCondition.extreme_jacobians:
            tbx._verify_active_variables_initialized()
            t_small, t_large = (
                tbx.config.jacobian_small_value_caution,
                tbx.config.jacobian_large_value_caution,
            )
            # compute the extreme jacobians
            xjc = _extreme_jacobian_columns(
                jac=self._jac, nlp=self._nlp, large=t_large, small=t_small
            )
            xjc.sort(key=lambda i: abs(log(i[0])), reverse=True)
            # each value is (variable, extreme-value)
            _set_variables(
                xjc,
                nm_func=lambda v: str(v[1]),
                vl_func=lambda v: v[0],
                rg_func=lambda v: (None, None),
                un_func=lambda v: "",
            )
            kwargs["value_format"] = ".3E"
            bounds = {"small": t_small, "large": t_large}
            bounds_desc = "extreme jacobians"

        # return as Pydantic data object
        return VariableListData(
            tag=cond.value,
            description=desc,
            names=names,
            details=details,
            values=values,
            bounds=bounds,
            bounds_desc=bounds_desc,
            ranges=ranges,
            units=units,
            **kwargs,
        )

    def _get_constraints_for_condition(
        self, cond: ConstraintCondition
    ) -> ComponentListData:
        tbx = self._toolbox  # alias
        kwargs = {}  # additional keywords for return value constructor
        desc = str(cond)  # default description
        details, values = None, None
        bounds, bounds_desc = {}, None
        names, values, details = [], [], []

        if cond == ConstraintCondition.large_residuals:
            residuals = large_residuals_set(
                tbx._model,
                tol=tbx.config.constraint_residual_tolerance,
                return_residual_values=True,
            )
            for c, residual in residuals.items():
                names.append(c.name)
                values.append(residual)
            kwargs["value_format"] = ".3E"
            bounds = {"tol": tbx.config.constraint_residual_tolerance}
            bounds_desc = "residual tolerance"
        elif cond == ConstraintCondition.no_free_variables:
            tbx._verify_active_variables_initialized()
            _, _, names = tbx._collect_constraint_mismatches()
        elif cond == ConstraintCondition.canceling_terms:
            tbx._verify_active_variables_initialized()
            _, cancellation, _ = tbx._collect_constraint_mismatches()
            for item in cancellation:
                name, _, detail = item.partition(": ")
                names.append(name)
                details.append(detail)
        elif cond == ConstraintCondition.mismatched_terms:
            tbx._verify_active_variables_initialized()
            mismatch, _, _ = tbx._collect_constraint_mismatches()
            for item in mismatch:
                name, _, detail = item.partition(": ")
                names.append(name)
                details.append(detail)
        elif cond == ConstraintCondition.extreme_jacobians:
            tbx._verify_active_variables_initialized()
            t_small, t_large = (
                tbx.config.jacobian_small_value_caution,
                tbx.config.jacobian_large_value_caution,
            )
            xjr = _extreme_jacobian_rows(
                jac=self._jac, nlp=self._nlp, large=t_large, small=t_small
            )
            xjr.sort(key=lambda i: abs(log(i[0])), reverse=True)
            for c in xjr:
                names.append(c[1].name)
                values.append(c[0])
            kwargs["value_format"] = ".3E"
            bounds = {"small": t_small, "large": t_large}
            bounds_desc = "extreme jacobians"
        else:
            raise ValueError(f"Unhandled constraint condition: {cond}")

        if not details:
            details = [""] * len(names)
        if not values:
            values = [None] * len(names)

        return ComponentListData(
            tag=cond.value,
            description=desc,
            names=names,
            details=details,
            values=values,
            bounds=bounds,
            bounds_desc=bounds_desc,
            **kwargs,
        )
