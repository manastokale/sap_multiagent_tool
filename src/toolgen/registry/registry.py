"""Tool Registry — the central store of parsed API definitions.

Provides lookup, search, filtering, and statistics over the loaded tools.
"""

from __future__ import annotations

import logging
from pathlib import Path

from toolgen.models import APIEndpoint, RegistryStats, Tool
from toolgen.registry.loader import load_tools_from_directory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Indexed registry of tools and their API endpoints."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: list[Tool] = tools or []
        self._by_endpoint_id: dict[str, APIEndpoint] = {}
        self._by_tool_name: dict[str, Tool] = {}
        self._by_category: dict[str, list[Tool]] = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._by_endpoint_id.clear()
        self._by_tool_name.clear()
        self._by_category.clear()
        for tool in self._tools:
            self._by_tool_name[tool.name] = tool
            cat = tool.category
            if cat not in self._by_category:
                self._by_category[cat] = []
            self._by_category[cat].append(tool)
            for ep in tool.endpoints:
                self._by_endpoint_id[ep.endpoint_id] = ep

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        toolenv_dir: Path,
        max_tools: int | None = None,
        categories: list[str] | None = None,
    ) -> ToolRegistry:
        """Load from a ToolBench toolenv/tools directory."""
        tools = load_tools_from_directory(
            toolenv_dir,
            max_tools=max_tools,
            categories=categories,
        )
        return cls(tools)

    # -- Lookups ----------------------------------------------------------

    def get_endpoint(self, endpoint_id: str) -> APIEndpoint | None:
        return self._by_endpoint_id.get(endpoint_id)

    def get_tool(self, tool_name: str) -> Tool | None:
        return self._by_tool_name.get(tool_name)

    def get_tools_by_category(self, category: str) -> list[Tool]:
        return self._by_category.get(category, [])

    def list_categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    def all_endpoints(self) -> list[APIEndpoint]:
        return list(self._by_endpoint_id.values())

    def all_endpoint_ids(self) -> list[str]:
        return list(self._by_endpoint_id.keys())

    def all_tools(self) -> list[Tool]:
        return list(self._tools)

    # -- Search -----------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[APIEndpoint]:
        """Simple text search over endpoint names and descriptions."""
        query_lower = query.lower()
        results: list[tuple[int, APIEndpoint]] = []
        for ep in self._by_endpoint_id.values():
            score = 0
            if query_lower in ep.endpoint_name.lower():
                score += 2
            if query_lower in ep.description.lower():
                score += 1
            if query_lower in ep.tool_name.lower():
                score += 1
            if score > 0:
                results.append((score, ep))
        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in results[:limit]]

    # -- Stats ------------------------------------------------------------

    def stats(self) -> RegistryStats:
        total_params = sum(
            len(ep.parameters)
            for ep in self._by_endpoint_id.values()
        )
        total_endpoints = len(self._by_endpoint_id)
        with_schema = sum(
            1 for ep in self._by_endpoint_id.values() if ep.response_schema
        )
        return RegistryStats(
            total_tools=len(self._tools),
            total_endpoints=total_endpoints,
            total_categories=len(self._by_category),
            categories={
                cat: sum(len(t.endpoints) for t in tools)
                for cat, tools in self._by_category.items()
            },
            endpoints_with_response_schema=with_schema,
            avg_params_per_endpoint=(
                round(total_params / total_endpoints, 2) if total_endpoints else 0.0
            ),
        )

    def __len__(self) -> int:
        return len(self._by_endpoint_id)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ToolRegistry(tools={s.total_tools}, "
            f"endpoints={s.total_endpoints}, "
            f"categories={s.total_categories})"
        )
