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

from .fsrunner import BaseFlowsheetRunner, RESULT_FLOWSHEET_KEY


class FlowsheetRunnerWithMain(BaseFlowsheetRunner):
    """Rewrite FlowsheetRunner constructor to:
    (a) skip timings,
    (b) consider the build step a solve step, and
    (c) have an attribute for the main() function
    """

    def __init__(self, *args, **kwargs):
        """Constructor."""
        from .runner_actions import (  # pylint: disable=C0415
            UnitDofChecker,
            CaptureSolverOutput,
            ModelVariables,
            MermaidDiagram,
        )

        super().__init__(*args, **kwargs)
        self.main_func = None
        self.add_action("degrees_of_freedom", UnitDofChecker, "fs", ["build"])
        self.add_action("capture_solver_output", CaptureSolverOutput, solve_re=r"build")
        self.add_action("model_variables", ModelVariables)
        self.add_action("mermaid_diagram", MermaidDiagram)


"""
Create an instance of FlowsheetRunnerWithMain and add a build
step that simply calls the provided main function to build & solve the model.
"""
_FS = FlowsheetRunnerWithMain()


@_FS.step("build")
def _build(ctx):
    model, solve_result = _FS.main_func()
    ctx.model = model
    ctx["results"] = solve_result


def fi_main(main_fn):
    """Decorator for function that returns the tuple (model, results)
    after building and solving a model, so that it provides
    information through the FlowsheetRunner API.
    """

    # note: don't change 'fi_wrapper' name
    def fi_wrapper(*args, **kwargs):
        _FS.main_func = main_fn
        _FS.run_steps()
        _FS.results[RESULT_FLOWSHEET_KEY] = _FS
        return _FS.model, _FS.results

    return fi_wrapper
