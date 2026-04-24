from enum import Enum

DEFAULT_SOLVER_NAME = "ipopt"

#: Special key used to embed a flowsheet runner instance in a result dict
RESULT_FLOWSHEET_KEY = "__fi"


class ActionNames(Enum):
    SOLVER_OUTPUT = "solver_output"
    SOLVER_RESULTS = "solver_results"
    DIAGNOSTICS = "diagnostics"
    MODEL_VARIABLES = "model_variables"
    MERMAID_DIAGRAM = "mermaid_diagram"
    STREAM_TABLE = "stream_table"
    DOF = "degrees_of_freedom"
    TIMINGS = "timings"
