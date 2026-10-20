"""MCP registration and request boundary; application tools remain transport-light."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import wraps
from typing import get_type_hints

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .async_cache import CapacityError
from .runtime import SibylRuntime, current_runtime, get_runtime
from .research_state import ResearchBudgetExceeded


def _validate_payload(values):
    size = 0

    def visit(value, depth=0):
        nonlocal size
        if depth > 8:
            raise ToolError("Tool input is too deeply nested.")
        if isinstance(value, str):
            if len(value) > 10000:
                raise ToolError("Text parameters must be at most 10000 characters.")
            size += len(value)
        elif isinstance(value, (list, tuple, dict)):
            if len(value) > 128:
                raise ToolError("Tool input contains too many items.")
            for item in (value.values() if isinstance(value, dict) else value):
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
                _validate_payload([*args, *kwargs.values()])
                try:
                    return await get_runtime().execute(
                        function.__name__, lambda: function(*args, **kwargs)
                    )
                except (CapacityError, ResearchBudgetExceeded) as exc:
                    raise ToolError(str(exc)) from None
                except ValueError:
                    raise ToolError(
                        "Invalid request or configuration. Check the tool parameters."
                    ) from None
                except asyncio.TimeoutError:
                    raise ToolError(
                        "Operation deadline exceeded. Retry with a narrower request."
                    ) from None
                except ToolError:
                    raise
                except Exception:
                    # Provider exceptions can contain credentials or full request URLs.
                    raise ToolError(
                        "Operation failed. Check configuration and retry."
                    ) from None

            bounded.__annotations__ = get_type_hints(function)
            self.tools.append((bounded, options))
            return bounded

        return register

    def create_server(self, runtime: SibylRuntime, instructions: str) -> FastMCP:
        @asynccontextmanager
        async def lifespan(server):
            try:
                yield runtime
            finally:
                await runtime.aclose()

        server = FastMCP(
            "sibyl", log_level="WARNING", instructions=instructions, lifespan=lifespan
        )
        for function, options in self.tools:

            def bind(function):
                @wraps(function)
                async def invoke(*args, **kwargs):
                    token = current_runtime.set(runtime)
                    try:
                        return await function(*args, **kwargs)
                    finally:
                        current_runtime.reset(token)

                return invoke

            server.tool(**options)(bind(function))
        return server
