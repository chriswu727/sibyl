"""Safe, actionable errors at the research request boundary."""


class ResearchRequestError(ValueError):
    """An application-authored message safe to return to the host."""
