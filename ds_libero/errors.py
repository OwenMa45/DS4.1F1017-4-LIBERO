class ExperimentError(Exception):
    status = "configuration_error"


class InvalidAction(ExperimentError):
    status = "invalid_output"


class ActionFormatError(InvalidAction):
    """A rejected reply whose formatting can be reported back to the policy."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


class ActionDimensionError(ActionFormatError):
    def __init__(self, kind, message, dimensions):
        super().__init__(kind, message)
        self.dimensions = dimensions


class APIError(ExperimentError):
    status = "api_error"


class BudgetExhausted(ExperimentError):
    status = "budget_exhausted"


class SimulationError(ExperimentError):
    status = "simulation_error"
