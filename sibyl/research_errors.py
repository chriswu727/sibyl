"""Safe, actionable errors at the research request boundary."""


class ResearchRequestError(ValueError):
    """An application-authored message safe to return to the host."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        field: str | None = None,
        claim_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.claim_id = claim_id
