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


class ToolCatalog:
    def __init__(self):
        self.tools = []

    def tool(self, **options):
        def register(function):
            @wraps(function)
            async def bounded(*args, **kwargs):
                if any(
                    isinstance(value, str) and len(value) > 10000
                    for value in (*args, *kwargs.values())
                ):
                    raise ToolError("Text parameters must be at most 10000 characters.")
                try:
                    return await get_runtime().execute(
                        function.__name__, lambda: function(*args, **kwargs)
                    )
                except CapacityError as exc:
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
