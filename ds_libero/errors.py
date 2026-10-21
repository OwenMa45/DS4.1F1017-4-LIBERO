class ExperimentError(Exception):
    status = "configuration_error"


class InvalidAction(ExperimentError):
    status = "invalid_output"


class APIError(ExperimentError):
    status = "api_error"


class BudgetExhausted(ExperimentError):
    status = "budget_exhausted"


class SimulationError(ExperimentError):
    status = "simulation_error"
