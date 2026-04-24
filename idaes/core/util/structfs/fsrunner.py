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
Specialize the generic `Runner` class to running a flowsheet,
in `FlowsheetRunner`.
"""

# stdlib
from copy import deepcopy
from enum import Enum
from types import FunctionType
from typing import Sequence

# third-party
from pyomo.environ import ConcreteModel, SolverFactory
from pyomo.environ import units as pyunits
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver

try:
    from idaes_connectivity import Connectivity
    from idaes_connectivity.jupyter import display_connectivity
except ImportError:
    Connectivity = None

# package
from .runner import Runner

DEFAULT_SOLVER_NAME = "ipopt"

# Constants
#: Special key used to embed a flowsheet runner instance in a result dict
RESULT_FLOWSHEET_KEY = "__fi"


class Context(dict):
    """Syntactic sugar for the dictionary for the 'context' passed into each
    step of the `FlowsheetRunner` class.
    """

    @property
    def model(self):
        """The model being run."""
        return self["model"]

    @model.setter
    def model(self, value):
        """The model being run."""
        self["model"] = value

    @property
    def solver(self):
        """The solver used to solve the model."""
        return self["solver"]

    @solver.setter
    def solver(self, value):
        """The solver used to solve the model."""
        self["solver"] = value

    def solve(self):
        """Perform solve, store result"""
        if self.solver is None:
            self.solver = SolverFactory(DEFAULT_SOLVER_NAME)
        self.results = self.solver.solve(self.model, tee=self["tee"])

    @property
    def tee(self):
        """Return whether solver output should be streamed.

        Returns:
            True if solver output should be echoed, otherwise False.
        """
        return self["tee"]

    @property
    def results(self):
        """Return the stored solver results, if any.

        Returns:
            The stored solver results object, or None if no solve has run.
        """
        return self.get("results", None)

    @results.setter
    def results(self, value):
        """Store solver results in the context.

        Args:
            value: Solver results object to store.
        """
        self["results"] = value


class BaseFlowsheetRunner(Runner):
    """Specialize the base `Runner` to handle IDAES flowsheets.

    This class pre-determine the name and order of steps to run

    Attributes:
        STEPS: List of defined step names.
    """

    _SET_SOLVER_STEP = "set_solver"

    STEPS = (
        "build",
        "set_solver",
        "initialize",
        "set_operating_conditions",
        "set_scaling",
        "solve_initial",
        "add_costing",
        "initialize_costing",
        "setup_optimization",
        "solve_optimization",
    )

    def __init__(
        self,
        solver=None,
        tee=True,
        solver_options: dict | None = None,
        steps: Sequence[str] = None,
    ):
        if steps is None:
            steps = self.STEPS
        self.build_step = steps[0]
        self._solver, self._tee = solver, tee
        self._solver_options = solver_options or {}
        self._ann = {}
        super().__init__(steps)  # needs to be last

    def set_solve_steps(self, solve_steps: Sequence[str]):
        """Set `solve_steps` for all contained actions which have this attribute."""
        for _, action in self._actions.items():
            if hasattr(action, "solve_steps"):
                action.solve_steps = solve_steps

    def run_steps(
        self,
        first: str = Runner.STEP_ANY,
        last: str = Runner.STEP_ANY,
        before=None,
        after=None,
        **kwargs,
    ):
        """Run the steps.

        Before it calls the superclass to run the steps, checks
        if the step name matches the `build_step` attribute and,
        if so, creates an empty Pyomo ConcreteModel to use as
        the base model for the flowsheet.
        """
        self._set_solver()
        from_step_name = self.normalize_name(first)
        if (
            from_step_name == "-"
            or from_step_name == self.build_step
            or self._context.model is None
        ):
            self._context.model = self._create_model()

        # replace first/last with before/after, if present
        if before is not None:
            kwargs["before"] = before
            last = ""
        if after is not None:
            kwargs["after"] = after
            first = ""

        super().run_steps(first, last, **kwargs)

    def _set_solver(self):
        """Set the solver, if no 'set_solver' step exists."""
        if self._SET_SOLVER_STEP not in self._steps:
            if self._solver is None:
                # use default solver, if none given
                self._context.solver = get_solver()
            elif isinstance(self._solver, str):
                # create solver from string
                self._context.solver = SolverFactory(self._solver)
            else:
                self._context.solver = self._solver
            if self._solver_options:
                self._context.solver.options = self._solver_options

    def reset(self):
        """Reset the runner context to its initial state."""
        self._context = Context(solver=self._solver, tee=self._tee, model=None)

    def _create_model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        return m

    @property
    def model(self):
        """Syntactic sugar to return the model."""
        return self._context.model

    @property
    def results(self):
        """Syntactic sugar to return the `result` in the context.
        Returns:
            results from Pyomo, or None if not set
        """
        return self._context.results

    def annotate_var(
        self,
        variable: object,
        key: str = None,
        title: str = None,
        desc: str = None,
        units: str = None,
        rounding: int = 3,
        is_input: bool = True,
        is_output: bool = True,
        input_category: str = "main",
        output_category: str = "main",
    ) -> object:
        """Annotate a Pyomo variable.

        Args:

        - variable: Pyomo variable being annotated
        - key: Key for this block in dict. Defaults to object name.
        - title: Name / title of the block. Defaults to object name.
        - desc: Description of the block. Defaults to object name.
        - units: Units. Defaults to string value of native units.
        - rounding: Significant digits
        - is_input: Is this variable an input
        - is_output: Is this variable an output
        - input_category: Name of input grouping to display under
        - output_category: Name of output grouping to display under

        Returns:

        - Input block (for chaining)

        Raises:

        - ValueError: if `is_input` and `is_output` are both False

        # Example

        Here is an example of annotating a single Pyomo variable.
        You can apply this same technique to any Pyomo, and thus IDAES,
        object.

        ```{code}
        :name: annotate_vars
        from idaes.core.util.structfs.fsrunner import FlowsheetRunner
        from pyomo.environ import *

        def example(f: FlowsheetRunner):
            v = Var()
            v.construct()
            f.annotate_var(v, key="example", title="Example variable").fix(1)
        ```

        To retrieve the annotated variables, use the `annotated_vars`
        property:

        ```{code}
        example(fr := FlowsheetRunner())
        print(fr.annotated_vars)
        # prints something like this:
        # {'example': {'var': <pyomo.core.base.var.ScalarVar object at 0x762ffb124b40>,
        # 'fullname': 'ScalarVar', 'title': 'Example variable',
        # 'description': 'ScalarVar', 'units': 'dimensionless',
        # 'rounding': 3, 'is_input': True, 'is_output': True, 'input_category': 'main',
        # 'output_category': 'main'}}
        ```

        """
        if not is_input and not is_output:
            raise ValueError("One of 'is_input', 'is_output' must be True")

        qual_name = variable.name
        key = key or variable.name

        self._ann[key] = {
            "var": variable,
            "fullname": qual_name,
            "title": title or qual_name,
            "description": desc or qual_name,
            "units": units or str(pyunits.get_units(variable)),
            "rounding": rounding,
            "is_input": is_input,
            "is_output": is_output,
            "input_category": input_category,
            "output_category": output_category,
        }

        return variable

    @property
    def annotated_vars(self) -> dict[str,]:
        """Get (a copy of) the annotated variables."""
        return deepcopy(self._ann)


class DiagnosticReportType(Enum):
    """Different types of diagnostic reports"""

    STRUCTURAL = "structural"
    NUMERICAL = "numerical"


class FlowsheetRunner(BaseFlowsheetRunner):
    """Interface for running and inspecting IDAES flowsheets."""

    class DegreesOfFreedom:
        """Wrapper for the UnitDofChecker action"""

        def __init__(self, runner):
            """Create the degrees-of-freedom helper.

            Args:
                runner: Flowsheet runner that owns the underlying action.
            """
            from .runner_actions import UnitDofChecker  # pylint: disable=C0415

            # check DoF after build, initial solve, and optimization solve
            self._a = runner.add_action(
                "degrees_of_freedom",
                UnitDofChecker,
                "fs",
                ["build", "solve_initial", "solve_optimization"],
            )
            self._rnr = runner

        def model(self):
            """Get the model."""
            return self._a.get_dof_model()

        def __getattr__(self, name):
            """Naming the step prints a summary of that step."""
            if name not in set(self._rnr.list_steps()):
                raise AttributeError(f"No step named '{name}'")
            self._a.summary(step=name)

        def __str__(self):
            return self._a.summary(stream=None)

        def _ipython_display_(self):
            self._a.summary()

    class Timings:
        """Wrapper for the Timer action"""

        def __init__(self, runner):
            """Create the timings helper.

            Args:
                runner: Flowsheet runner that owns the underlying action.
            """
            from .runner_actions import Timer  # pylint: disable=C0415

            self._a: Timer = runner.add_action("timings", Timer)

        @property
        def values(self) -> list[dict]:
            """Get timing values."""
            return self._a.get_history()

        @property
        def history(self) -> str:
            """Get a text report of the timing history"""
            h = []
            for i in range(len(self._a)):
                h.append(f"== Run {i + 1} ==")
                h.append("")
                h.append(self._a.summary(run_idx=i))
            return "\n".join(h)

        def __str__(self):
            return self._a.summary()

        def _ipython_display_(self):
            self._a._ipython_display_()  # pylint: disable=protected-access

    def __init__(self, solve_steps: list[str] = None, **kwargs):
        """Initialize a flowsheet runner with default inspection actions.

        Args:
            solve_step: Optional list of steps considered solves.
            **kwargs: Additional keyword arguments passed to
                `BaseFlowsheetRunner`.
        """
        from .runner_actions import (  # pylint: disable=C0415
            CaptureSolverOutput,
            GetSolverResults,
            ModelVariables,
            MermaidDiagram,
            Diagnostics,
            StreamTable,
        )

        super().__init__(**kwargs)
        self.dof = self.DegreesOfFreedom(self)
        self.timings = self.Timings(self)
        self.add_action("solver_output", CaptureSolverOutput)
        self.add_action("solver_results", GetSolverResults)
        self.add_action("diagnostics", Diagnostics)
        self.add_action("model_variables", ModelVariables)
        self.add_action("mermaid_diagram", MermaidDiagram)
        self.add_action("stream_table", StreamTable)

    def build(self):
        """Run just the build step"""
        self.run_step("build")

    def solve_initial(self):
        """Perform all steps up to 'solve_initial'"""
        self.run_steps(last="solve_initial")

    def show_diagram(self):
        """Return the diagram."""
        if Connectivity is not None:
            return display_connectivity(input_model=self.model)
        else:
            return ""

    @property
    def solver_status(self) -> str:
        """Solver status, from Pyomo

        Returns:
            str: Pyomo value (e.g., "ok") or "unknown" if it cannot be found
        """
        rpt = self.get_action("solver_results").report()
        if len(rpt.results) > 0:
            value = rpt.results[0].solver["Status"]
        else:
            value = "unknown"
        return value


## Utility code to find modules


def global_flowsheet(a_module):
    """Find the global flowhseet object

    Return:
        The flowsheet object, or None if not found.
    """
    return getattr(a_module, "FS", None)


def wrapped_main(a_module) -> FunctionType | None:
    """Find a wrapped flowsheet main function.

    Returns:
       The wrapped function, or None if not found
    """
    if not hasattr(a_module, "fi_main"):
        return None
    found = None
    for k in dir(a_module):
        fn = getattr(a_module, k)
        if isinstance(fn, FunctionType):
            name = fn.__name__
            if name == "fi_wrapper":
                found = k
                break
    if found:
        return getattr(a_module, found)
    return None


def run_wrapped_main(func) -> dict[str, dict]:
    """Run a wrapped flowsheet returned by `wrapped_flowsheet()`
    and return the report from the embedded runner object.
    """
    m, result = func()  # pylint: disable=W0612
    return result[RESULT_FLOWSHEET_KEY].report()
