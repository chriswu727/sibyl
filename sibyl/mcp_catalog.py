"""MCP registration and request boundary; application tools remain transport-light."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from functools import wraps
from typing import get_type_hints

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult
from pydantic import BaseModel

from .async_cache import CapacityError
from .evidence import SourceBundle
from .mcp_errors import AgentMCP, SafeToolError, json_text
from .report_exports import ExportManifest, ReportExportError
from .research_errors import ResearchRequestError
from .research_state import ResearchBudgetExceeded
from .research_views import research_brief
from .runtime import SibylRuntime, current_runtime, get_runtime


def _validate_payload(values):
    size = 0

    def visit(value, depth=0):
        nonlocal size
        if isinstance(value, BaseModel):
            value = value.model_dump()
        if depth > 8:
            raise ToolError("Tool input is too deeply nested.")
        if isinstance(value, str):
            if len(value) > 10000:
                raise ToolError("Text parameters must be at most 10000 characters.")
            size += len(value)
        elif isinstance(value, (list, tuple, dict)):
            if len(value) > 128:
                raise ToolError("Tool input contains too many items.")
            for item in value.values() if isinstance(value, dict) else value:
                visit(item, depth + 1)
        if size > 64000:
            raise ToolError("Combined tool text must be at most 64000 characters.")

    visit(values)


class ToolCatalog:
    def __init__(self):
        self.tools = []

    def tool(self, **options):
        def register(function):
            @wraps(function)
            async def bounded(*args, **kwargs):
                async def execute():
                    result = await function(*args, **kwargs)
                    if isinstance(result, ExportManifest) and result.status == "failed":
                        raise ReportExportError(result)
                    return result

                try:
                    _validate_payload([*args, *kwargs.values()])
                    return await get_runtime().execute(function.__name__, execute)
                except (
                    CapacityError,
                    ResearchBudgetExceeded,
                    ResearchRequestError,
                ) as exc:
                    error = (
                        exc
                        if isinstance(exc, ResearchRequestError)
                        else ResearchRequestError(
                            str(exc),
                            code="capacity_exceeded"
                            if isinstance(exc, CapacityError)
                            else "budget_exhausted",
                        )
                    )
                    raise SafeToolError(error) from None
                except ValueError:
                    raise SafeToolError(
                        ResearchRequestError(
                            "Invalid request or configuration. Check the tool parameters."
                        )
                    ) from None
                except asyncio.TimeoutError:
                    raise SafeToolError(
                        ResearchRequestError(
                            "Operation deadline exceeded. Inspect state before retrying with a narrower request.",
                            code="deadline_exceeded",
                        )
                    ) from None
                except ToolError as exc:
                    raise SafeToolError(ResearchRequestError(str(exc))) from None
                except Exception:
                    # Provider exceptions can contain credentials or full request URLs.
                    raise SafeToolError(
                        ResearchRequestError(
                            "Operation failed. Check configuration and retry.",
                            code="operation_failed",
                        )
                    ) from None

            bounded.__annotations__ = get_type_hints(function)
            self.tools.append((bounded, options))
            return bounded

        return register

    def create_server(
        self, runtime: SibylRuntime, instructions: str, *, agent_mode: bool = False
    ) -> FastMCP:
        @asynccontextmanager
        async def lifespan(server):
            try:
                await runtime.start()
                yield runtime
            finally:
                await runtime.aclose()

        server_options = dict(
            log_level="WARNING", instructions=instructions, lifespan=lifespan
        )
        server = (
            AgentMCP(runtime, **server_options)
            if agent_mode
            else FastMCP("sibyl", **server_options)
        )
        for function, options in self.tools:

            def bind(function):
                @wraps(function)
                async def invoke(*args, **kwargs):
                    token = current_runtime.set(runtime)
                    try:
                        result = await function(*args, **kwargs)
                        if agent_mode and isinstance(result, SourceBundle):
                            # Same SourceBundle payload; the large schema stays in docs.
                            payload = result.to_dict()
                            return CallToolResult(
                                content=[json_text(payload)],
                                structuredContent=payload,
                            )
                        if isinstance(result, ExportManifest):
                            if not agent_mode:
                                return result.text()
                            payload = result.model_dump(mode="json")
                            return CallToolResult(
                                content=[json_text(payload)],
                                structuredContent=payload,
                            )
                        if agent_mode and function.__name__ in {
                            "start_research",
                            "continue_research",
                            "get_research",
                            "assess_research",
                            "complete_research",
                            "cancel_research",
                        }:
                            if (
                                "research_plan" in result
                                and kwargs.get("view", "auto") == "auto"
                            ):
                                result = research_brief(result)
                            return CallToolResult(
                                content=[json_text(result)],
                                structuredContent=result,
                            )
                        return result
                    finally:
                        current_runtime.reset(token)

                if function.__annotations__.get("return") is ExportManifest:
                    signature = inspect.signature(function)
                    hints = get_type_hints(function)
                    invoke.__signature__ = signature.replace(
                        parameters=[
                            parameter.replace(annotation=hints[parameter.name])
                            for parameter in signature.parameters.values()
                        ],
                        return_annotation=ExportManifest if agent_mode else str,
                    )
                return invoke

            if agent_mode and function.__annotations__.get("return") is SourceBundle:
                options = {**options, "structured_output": False}
            server.tool(**options)(bind(function))
        return server


catalog = ToolCatalog()
