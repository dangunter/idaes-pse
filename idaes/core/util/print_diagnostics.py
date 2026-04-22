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
Print diagnostics values in Pydantic data models.
"""

import sys

from idaes.core.util.compute_diagnostics import (
    StructuralCautionsData,
    StructuralWarningsData,
    VariableListData,
)


class Report:
    """Interface class to pretty-print the contents of the associated data model
    to the console.

    Subclasses should implement `get_lines()` to return a title
    and list of lines for the content.
    """

    def print(self, stream=None, indent=4, width=80, border=True):
        """Write the formatted report to a stream.

        Args:
            stream: Destination stream. Defaults to `sys.stdout` when None.
            indent: Number of spaces to prefix each content line.
            width: Width used for divider lines.
            border: If True, render top and bottom border lines.
        """
        # use stdout if no stream is given
        if stream is None:
            stream = sys.stdout
        # setup
        tab = " " * indent

        # write top border
        if border:
            stream.write("=" * width)
            stream.write("\n")

        # get title and content
        title, lines = self.get_lines()

        if lines is None:
            stream.write(title)
            stream.write("\n")
        else:
            # write title and divider
            if title:
                stream.write(title)
                stream.write("\n")
                stream.write("-" * width)
                stream.write("\n")

            # write content
            for line in lines:
                stream.write(f"{tab}{line}\n")

        # write bottom divider
        if border:
            stream.write("=" * width)
            stream.write("\n")

    def get_lines(self) -> tuple[str, list[str]]:
        """Return the title and content lines for this report.

        Returns:
            A `(title, lines)` tuple for rendering.
        """
        return "", []  # override in subclasses


def _plural(n, word):
    suffix = "s" if abs(n) > 1 else ""
    return f"{n} {word}{suffix}"


class VariableListReport(Report):
    """Pretty-printer for variable diagnostic lists."""

    def __init__(self, data: VariableListData):
        self._data = data

    def get_lines(self) -> tuple[str, list[str] | None]:
        data = self._data
        num_variables = len(data.variables)
        if num_variables == 0:
            title = f"No model variables {data.description}"
            return title, None
        title = f"Model variables that {data.description} ({num_variables})"
        lines = []
        for i in range(num_variables):
            items = [data.variables[i]]
            if data.details[i]:
                items.append(data.details[i])
            if data.values[i] is not None:
                items.append(f"value={format(data.values[i], data.value_format)}")
            line = " ".join(items)
            lines.append(line)
        return title, lines


class StructuralWarningsReport(Report):
    """Pretty-printer for structural warning diagnostics."""

    def __init__(self, data: StructuralWarningsData):
        self._data = data

    def get_lines(self):
        data = self._data
        lines = []
        if data.dof is not None:
            lines.append(f"WARNING: {_plural(data.dof, 'Degree')} of Freedom")
        if data.inconsistent_units is not None:
            lines.append(
                f"WARNING: {_plural(len(data.inconsistent_units), 'Component')} with inconsistent units"
            )
        if (
            data.underconstrained_set is not None
            or data.overconstrained_set is not None
        ):
            indent = " " * 4
            ucv, ucc = len(data.underconstrained_set.variables), len(
                data.underconstrained_set.constraints
            )
            ocv, occ = len(data.overconstrained_set.variables), len(
                data.overconstrained_set.constraints
            )
            lines.extend(
                [
                    "WARNING: Structural singularity found",
                    f"{indent}Under-Constrained Set: {ucv} variables, {ucc} constraints",
                    f"{indent}Over-Constrained Set: {ocv} variables, {occ} constraints",
                ]
            )
        if data.evaluation_errors is not None:
            lines.append(
                f"WARNING: Found {len(data.evaluation_errors)} potential evaluation errors."
            )

        return "Structural warnings", lines


class StructuralCautionsReport(Report):
    """Pretty-printer for structural caution diagnostics."""

    def __init__(self, data: StructuralCautionsData):
        self._data = data

    def get_lines(self):
        data = self._data
        lines = []
        if data.zero_vars is not None:
            lines.append(
                f"Caution: {_plural(len(data.zero_vars), 'variable')} fixed to 0"
            )
        if data.unused_vars_free is not None or data.unused_vars_fixed is not None:
            num_free = (
                0 if data.unused_vars_free is None else len(data.unused_vars_free)
            )
            num_fixed = (
                0 if data.unused_vars_fixed is None else len(data.unused_vars_fixed)
            )
            lines.append(
                f"Caution: {_plural(num_free + num_fixed, 'unused variable')} ({num_fixed} fixed)"
            )
        return "Structural cautions", lines


class StructuralIssuesReport(Report):
    """Pretty-printer for combined structural issue diagnostics."""

    def __init__(self, data: StructuralCautionsData):
        self._data = data

    def get_lines(self):
        data = self._data
        warnings_title, warning_lines = data.warnings.get_lines()
        cautions_title, caution_lines = data.cautions.get_lines()
        return (
            "Structural issues",
            [warnings_title, "-" * len(warnings_title)]
            + warning_lines
            + ["", cautions_title, "-" * len(cautions_title)]
            + caution_lines,
        )
