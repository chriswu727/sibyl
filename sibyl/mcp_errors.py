"""Structured, sanitized recovery at the agent transport boundary."""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from .report_exports import ReportExportError
from .research_errors import ResearchRequestError
from .runtime import SibylRuntime


def json_text(payload: dict[str, Any]) -> TextContent:
    """Serialized structuredContent for hosts that only pass text to the model."""
    return TextContent(
        type="text", text=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


class SafeToolError(ToolError):
    def __init__(self, error: ResearchRequestError) -> None:
        super().__init__(str(error))
        self.error = error


def error_result(
    runtime: SibylRuntime, operation: str, arguments: dict[str, Any], exc: ToolError
) -> CallToolResult:
    error = ResearchRequestError(
        "Operation failed. Inspect the current job before retrying.",
        code="operation_failed",
    )
    cause: BaseException | None = exc
    for _ in range(8):
        if isinstance(cause, SafeToolError):
            error = cause.error
            if isinstance(error, ReportExportError):
                payload = error.manifest.model_dump(mode="json")
                return CallToolResult(
                    isError=True,
                    content=[json_text(payload)],
                    structuredContent=payload,
                )
            break
        if isinstance(cause, ValidationError):
            error = ResearchRequestError(
                "Tool arguments do not match the input schema. Correct the indicated field.",
                code="invalid_parameters",
            )
            locations = cause.errors(include_input=False, include_context=False)
            if locations:
                location = locations[0]["loc"]
                if locations[0]["type"] == "extra_forbidden":
                    location = location[:-1]
                error.field = ".".join(str(part) for part in location)
            break
        if cause is None:
            break
        cause = cause.__cause__
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "view": "error",
        "status": "error",
        "operation": operation,
        "error": {
            "code": error.code,
            "message": str(error),
            "field": error.field,
            "claim_id": error.claim_id,
        },
        "job_id": None,
        "current_revision": None,
        "job_status": None,
        "recovery": {
            "action": "correct_request",
            "instructions": "Correct the request. Do not infer success or guess a new revision.",
            "read": None,
        },
    }
    job_id = arguments.get("job_id")
    if (
        isinstance(job_id, str)
        and re.fullmatch(r"[a-f0-9]{32}", job_id)
        and runtime._research_jobs is not None
    ):
        try:
            job = runtime.research_jobs.store.get(job_id)
        except Exception:
            pass
        else:
            result.update(
                job_id=job_id,
                current_revision=job["revision"],
                job_status=job["status"],
            )
            result["recovery"]["read"] = {
                "tool": "get_research",
                "arguments": {"job_id": job_id, "view": "brief"},
            }
    actions = {
        "unsupported_mode": (
            "start_new_research",
            "Legacy autonomous execution is not supported. Read/export saved state when available, or explicitly start a new host investigation.",
        ),
        "stale_revision": (
            "reload_research",
            "Read the current brief and reassess against its revision. A rejected assessment does not increment it.",
        ),
        "invalid_claim": (
            "repair_assessment",
            "Correct the indicated claim and resubmit the complete question claim set at the current revision. Do not invent evidence to satisfy validation.",
        ),
        "report_not_finalized": (
            "review_and_finalize",
            "Read the current brief, repair rejected reviews, then finalize existing claims before exporting. No export path is confirmed by this error.",
        ),
        "immutable_report": (
            "start_new_research",
            "Completed reports cannot be changed. Read/export this job, or explicitly start a new investigation.",
        ),
        "research_gaps": (
            "inspect_gaps",
            "Inspect the current research_plan and investigate/review gaps. Use allow_incomplete only to disclose unresolved gaps, not to bypass invalid claims.",
        ),
    }
    if error.code in actions:
        action, instructions = actions[error.code]
        result["recovery"].update(action=action, instructions=instructions)
    return CallToolResult(
        isError=True, content=[json_text(result)], structuredContent=result
    )


class AgentMCP(FastMCP):
    def __init__(self, runtime: SibylRuntime, **kwargs: Any) -> None:
        self.runtime = runtime
        super().__init__("sibyl", **kwargs)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as exc:
            return error_result(self.runtime, name, arguments, exc)
