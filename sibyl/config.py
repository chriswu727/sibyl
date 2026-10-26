"""Optional retrieval configuration. Model execution belongs to the MCP host."""

from __future__ import annotations

from dataclasses import dataclass, fields

import yaml


@dataclass
class Config:
    search_engine: str = "all"
    extractor: str = "bs4"
    jina_fallback: bool = False
    js_render: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as stream:
            data = yaml.safe_load(stream)
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(
                "Sibyl configuration must be a retrieval settings mapping."
            )
        allowed = {field.name for field in fields(cls)}
        if set(data) - allowed:
            raise ValueError(
                "Unsupported Sibyl configuration. Only search_engine, extractor, "
                "jina_fallback and js_render are supported. Remove legacy model "
                "settings: reasoning now belongs to your MCP host."
            )
        if data.get("extractor", "bs4") not in {"bs4", "trafilatura"}:
            raise ValueError("extractor must be bs4 or trafilatura")
        if not isinstance(data.get("search_engine", "all"), str):
            raise ValueError("search_engine must be a string")
        for flag in ("jina_fallback", "js_render"):
            if flag in data and not isinstance(data[flag], bool):
                raise ValueError(f"{flag} must be a boolean")
        return cls(**data)

    @classmethod
    def from_env(cls) -> Config:
        """Model credentials in the host environment do not configure Sibyl."""
        return cls()
