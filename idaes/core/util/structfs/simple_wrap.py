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
**this** is the documentation for the simple wrapper
"""
# stdlib
import inspect
import os

# package
from .fsrunner import BaseFlowsheetRunner, RESULT_FLOWSHEET_KEY


class SimpleFlowsheetRunner(BaseFlowsheetRunner):
    """Rewrite FlowsheetRunner constructor to:
    (a) consider the build step (also) a solve step, and
    (b) have an attribute `main_func` for the main function
    """

    def __init__(self, *args, **kwargs):
        """Constructor."""
        from .runner_actions import (  # pylint: disable=C0415
            UnitDofChecker,
            CaptureSolverOutput,
            ModelVariables,
            MermaidDiagram,
            Timer,
        )

        super().__init__(*args, **kwargs)
        self.main_func = None
        self.main_func_args = []
        self.main_func_kwargs = {}
        self.add_action("timings", Timer)
        self.add_action("degrees_of_freedom", UnitDofChecker, "fs", ["build"])
        self.add_action("capture_solver_output", CaptureSolverOutput, solve_re=r"build")
        self.add_action("model_variables", ModelVariables)
        self.add_action("mermaid_diagram", MermaidDiagram)


# create an instance of FlowsheetRunnerWithMain
_FS = SimpleFlowsheetRunner()


class _Wrapper:
    """
    ### Usage

    The functionality of the API is imported with the name `fi_main`
    in the `idaes.core.util.structfs` package, so normal usage requires only a
    single function, listed as `my_main_function()` in the example below
    (some extra classes and functions are added so this can be a self-contained and
    working example):
    ```{code} python
    :caption: Simple Wrapper Usage

    from idaes.core.util.structfs import fi_main

    @fi_main
    def my_main_function(some, args, keyword=None): # can take any arguments
        # build the flowsheet -> model
        model = build_flowsheet()
        # initialize the flowsheet
        # solve the flowsheet -> solve_result
        solve_result = solve_flowsheet()

        # **Important!**: return the model and solve result as a tuple
        return model, solve_result


    #------------------------------------------------------------------

    # Some classes so the build/solve can nominally succeed

    class FakeFlowsheet:
        is_indexed = lambda x: False
        def component_data_objects(self, *arg, **kw):
            return []
        def component_objects(self, *arg, **kw):
            return []

    class FakeModel:
        fs = FakeFlowsheet()
        def component_objects(self, *arg, **kw):
            return []

    # Fake build and solve functions

    def build_flowsheet():
        # Fake build of flowsheet
        return FakeModel()

    def solve_flowsheet():
        # Fake solve of flowsheet
        return {}

    ```
    So, in summary, the steps to enable your flowsheets are:

    1. Create a function that returns the tuple `(model, solve_result)` after
       building and solving the model.

    2. Add the import statement `from idaes.core.util.structfs import fi_main` and
       then decorate the function in (1) with  `@fi_main`

    That's it! Now the Flowsheet Inspector can run your flowsheet and show the diagram,
    model variables, degrees of freedom, diagnostics, etc.
    """

    # add a build step that simply calls the provided main function
    # to build & solve the model.
    @_FS.step("build")
    def _build(ctx):
        model, solve_result = _FS.main_func(*_FS.main_func_args, **_FS.main_func_kwargs)
        ctx.model = model
        ctx["results"] = solve_result  # pylint: disable=E1137

    @classmethod
    def main(cls, **main_kw):
        """Decorator *factory* for function that returns the tuple (model, results)
        after building and solving a model, so that it provides
        information through the FlowsheetRunner API.
        """

        def fi_wrapper_factory(main_fn):
            # note: don't change 'fi_wrapper' name, since this
            # is used for auto-detection of the method in user's code
            def fi_wrapper(*args, **kwargs):
                if "module" not in main_kw:
                    main_kw["module"] = inspect.getmodule(main_fn).__name__
                if "filename" not in main_kw:
                    main_kw["filename"] = os.path.abspath(inspect.getfile(main_fn))
                _FS.main_func = main_fn
                _FS.main_func_args = args
                _FS.main_func_kwargs = kwargs
                _FS.set_report_target(**main_kw)
                # run the flowsheet
                _FS.run_steps()
                # stash object in result dict under 'special' key
                _FS.results[RESULT_FLOWSHEET_KEY] = _FS
                return _FS.model, _FS.results

            return fi_wrapper

        return fi_wrapper_factory
