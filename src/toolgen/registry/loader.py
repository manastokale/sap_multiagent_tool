"""ToolBench JSON loader.

Parses the raw ToolBench toolenv/tools directory structure into clean
internal models. Handles the many inconsistencies in the raw data:
  - Missing or malformed parameter types
  - Parameters as dicts vs lists
  - Missing descriptions, methods, or required fields
  - Duplicate endpoint names within a tool
  - Unicode/encoding issues in tool names
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from toolgen.models import (
    APIEndpoint,
    HTTPMethod,
    Parameter,
    ParameterType,
    Tool,
)

logger = logging.getLogger(__name__)


def _parse_parameter(raw: dict[str, Any], required: bool = False) -> Parameter | None:
    """Parse a single parameter from ToolBench's raw format.

    ToolBench parameters look like:
        {"name": "city", "type": "STRING", "description": "...", "default": ""}
    But type, description, and default are often missing or inconsistent.
    """
    name = raw.get("name")
    if not name or not isinstance(name, str):
        return None

    name = name.strip()
    if not name:
        return None

    param_type = ParameterType.from_raw(raw.get("type"))
    description = str(raw.get("description", "")).strip()
    default = raw.get("default")

    # ToolBench sometimes puts "" as default for required params
    if default == "" and required:
        default = None

    enum_values = raw.get("enum")
    if enum_values and not isinstance(enum_values, list):
        enum_values = None

    return Parameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        default=default,
        enum=enum_values,
    )


def _parse_parameters(raw_api: dict[str, Any]) -> list[Parameter]:
    """Parse both required and optional parameters from a raw API entry."""
    params: list[Parameter] = []
    seen_names: set[str] = set()

    def _coerce_param_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            coerced = []
            for name, spec in value.items():
                if isinstance(spec, dict):
                    coerced.append({"name": name, **spec})
                else:
                    coerced.append({"name": name, "type": spec})
            return coerced
        return []

    # Parse required parameters
    for p_raw in _coerce_param_list(raw_api.get("required_parameters", [])):
        param = _parse_parameter(p_raw, required=True)
        if param and param.name not in seen_names:
            params.append(param)
            seen_names.add(param.name)

    # Parse optional parameters
    for p_raw in _coerce_param_list(raw_api.get("optional_parameters", [])):
        param = _parse_parameter(p_raw, required=False)
        if param and param.name not in seen_names:
            params.append(param)
            seen_names.add(param.name)

    return params


def _parse_api_endpoint(
    raw_api: dict[str, Any],
    tool_name: str,
    category: str,
) -> APIEndpoint | None:
    """Parse a single API endpoint from ToolBench's api_list entry."""
    name = raw_api.get("name")
    if not name or not isinstance(name, str):
        return None

    name = name.strip()
    if not name:
        return None

    description = str(raw_api.get("description", "")).strip()
    method = HTTPMethod.from_raw(raw_api.get("method"))
    parameters = _parse_parameters(raw_api)
    response_schema = raw_api.get("response_schema") or raw_api.get("response")
    if not isinstance(response_schema, dict):
        response_schema = None

    return APIEndpoint(
        tool_name=tool_name,
        endpoint_name=name,
        description=description,
        method=method,
        category=category,
        parameters=parameters,
        response_schema=response_schema,
    )


def load_tool_from_json(filepath: Path, category: str = "") -> Tool | None:
    """Load a single tool from a ToolBench JSON file.

    Expected format:
    {
        "tool_name": "...",
        "tool_description": "...",
        "title": "...",
        "api_list": [...]
    }
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to parse %s: %s", filepath, e)
        return None

    if not isinstance(data, dict):
        logger.warning("Expected dict in %s, got %s", filepath, type(data).__name__)
        return None

    tool_name = data.get("tool_name", "")
    if not tool_name:
        tool_name = data.get("title", filepath.stem)
    tool_name = str(tool_name).strip()

    description = str(data.get("tool_description", "")).strip()

    api_list = data.get("api_list", [])
    if not isinstance(api_list, list):
        logger.warning("api_list is not a list in %s", filepath)
        api_list = []

    endpoints: list[APIEndpoint] = []
    seen_names: set[str] = set()

    for raw_api in api_list:
        if not isinstance(raw_api, dict):
            continue
        ep = _parse_api_endpoint(raw_api, tool_name, category)
        if ep is None:
            continue
        # Handle duplicate endpoint names within a tool
        if ep.endpoint_name in seen_names:
            ep.endpoint_name = f"{ep.endpoint_name}_dup{len(seen_names)}"
            ep.endpoint_id = f"{tool_name}/{ep.endpoint_name}"
        seen_names.add(ep.endpoint_name)
        endpoints.append(ep)

    if not endpoints:
        logger.debug("No valid endpoints in %s", filepath)
        return None

    return Tool(
        name=tool_name,
        description=description,
        category=category,
        endpoints=endpoints,
    )


def load_tools_from_directory(
    toolenv_dir: Path,
    max_tools: int | None = None,
    categories: list[str] | None = None,
) -> list[Tool]:
    """Load tools from the ToolBench toolenv/tools directory.

    Directory structure:
        toolenv/tools/
            Category1/
                tool1.json
                tool2.json
            Category2/
                ...

    Args:
        toolenv_dir: Path to toolenv/tools directory.
        max_tools: Optional limit on total tools loaded (for dev).
        categories: Optional list of categories to load (for dev).

    Returns:
        List of parsed Tool objects.
    """
    tools: list[Tool] = []

    if not toolenv_dir.is_dir():
        logger.error("toolenv directory does not exist: %s", toolenv_dir)
        return tools

    count = 0
    for path in sorted(toolenv_dir.iterdir()):
        if path.is_file() and path.suffix == ".json":
            candidate_files = [(path, "Uncategorized")]
        elif path.is_dir():
            category = path.name
            if categories and category not in categories:
                continue
            candidate_files = [
                (tool_file, category)
                for tool_file in sorted(path.rglob("*.json"))
                if not any(part.startswith(".") for part in tool_file.relative_to(path).parts)
            ]
        else:
            continue

        for tool_file, category in candidate_files:
            if max_tools and count >= max_tools:
                return tools

            tool = load_tool_from_json(tool_file, category=category)
            if tool:
                tools.append(tool)
                count += 1

    logger.info("Loaded %d tools from %s", len(tools), toolenv_dir)
    return tools
