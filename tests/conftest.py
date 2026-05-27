"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolgen.models import (
    APIEndpoint,
    HTTPMethod,
    Parameter,
    ParameterType,
    ToolChain,
    ChainPattern,
)
from toolgen.registry.registry import ToolRegistry


@pytest.fixture
def sample_tool_json() -> dict:
    """A realistic ToolBench-style tool JSON."""
    return {
        "tool_name": "hotel_api",
        "tool_description": "Search and book hotels worldwide",
        "title": "Hotel API",
        "api_list": [
            {
                "name": "search_hotels",
                "url": "https://api.example.com/hotels/search",
                "description": "Search for hotels in a city",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "STRING", "description": "City name"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "NUMBER", "description": "Max price per night", "default": ""},
                    {"name": "rating", "type": "NUMBER", "description": "Minimum rating", "default": "3.0"},
                    {"name": "currency", "type": "STRING", "description": "Currency code", "default": "USD"},
                ],
            },
            {
                "name": "get_hotel_details",
                "url": "https://api.example.com/hotels/details",
                "description": "Get detailed info about a specific hotel",
                "method": "GET",
                "required_parameters": [
                    {"name": "hotel_id", "type": "STRING", "description": "Hotel ID from search"},
                ],
                "optional_parameters": [],
            },
            {
                "name": "book_hotel",
                "url": "https://api.example.com/hotels/book",
                "description": "Book a hotel room",
                "method": "POST",
                "required_parameters": [
                    {"name": "hotel_id", "type": "STRING", "description": "Hotel ID"},
                    {"name": "check_in", "type": "STRING", "description": "Check-in date"},
                    {"name": "check_out", "type": "STRING", "description": "Check-out date"},
                ],
                "optional_parameters": [
                    {"name": "guests", "type": "INTEGER", "description": "Number of guests", "default": "1"},
                ],
            },
        ],
    }


@pytest.fixture
def sample_flight_json() -> dict:
    """A second tool for multi-tool testing."""
    return {
        "tool_name": "flight_api",
        "tool_description": "Search and book flights",
        "title": "Flight API",
        "api_list": [
            {
                "name": "search_flights",
                "url": "",
                "description": "Search for flights between cities",
                "method": "GET",
                "required_parameters": [
                    {"name": "origin", "type": "STRING", "description": "Origin city"},
                    {"name": "destination", "type": "STRING", "description": "Destination city"},
                    {"name": "date", "type": "STRING", "description": "Travel date"},
                ],
                "optional_parameters": [
                    {"name": "max_price", "type": "NUMBER", "description": "Max price"},
                ],
            },
            {
                "name": "book_flight",
                "url": "",
                "description": "Book a flight",
                "method": "POST",
                "required_parameters": [
                    {"name": "flight_id", "type": "STRING", "description": "Flight ID"},
                ],
                "optional_parameters": [],
            },
        ],
    }


@pytest.fixture
def malformed_tool_json() -> dict:
    """A ToolBench JSON with common inconsistencies."""
    return {
        "tool_name": "  messy_api  ",
        "tool_description": "",
        "api_list": [
            {
                "name": "get_data",
                "description": "Get some data",
                # Missing method
                "required_parameters": [
                    {"name": "id", "type": "int", "description": ""},  # 'int' not 'INTEGER'
                ],
                "optional_parameters": "not_a_list",  # Wrong type
            },
            {
                # Missing name
                "description": "Bad endpoint",
            },
            {
                "name": "  duplicate  ",
                "description": "First",
                "required_parameters": [],
                "optional_parameters": [],
            },
            {
                "name": "duplicate",
                "description": "Second",
                "required_parameters": [],
                "optional_parameters": [],
            },
        ],
    }


@pytest.fixture
def tool_json_dir(sample_tool_json, sample_flight_json, tmp_path) -> Path:
    """Create a temporary directory with tool JSON files."""
    travel_dir = tmp_path / "Travel"
    travel_dir.mkdir()

    (travel_dir / "hotel_api.json").write_text(json.dumps(sample_tool_json))
    (travel_dir / "flight_api.json").write_text(json.dumps(sample_flight_json))

    return tmp_path


@pytest.fixture
def sample_registry(tool_json_dir) -> ToolRegistry:
    """A populated registry for testing."""
    return ToolRegistry.from_directory(tool_json_dir)


@pytest.fixture
def sample_endpoints() -> list[APIEndpoint]:
    """Pre-built endpoints for testing."""
    return [
        APIEndpoint(
            tool_name="hotel_api",
            endpoint_name="search_hotels",
            description="Search for hotels",
            method=HTTPMethod.GET,
            category="Travel",
            parameters=[
                Parameter(name="city", type=ParameterType.STRING, required=True),
                Parameter(name="max_price", type=ParameterType.NUMBER),
            ],
        ),
        APIEndpoint(
            tool_name="hotel_api",
            endpoint_name="book_hotel",
            description="Book a hotel room",
            method=HTTPMethod.POST,
            category="Travel",
            parameters=[
                Parameter(name="hotel_id", type=ParameterType.STRING, required=True),
                Parameter(name="check_in", type=ParameterType.STRING, required=True),
            ],
        ),
        APIEndpoint(
            tool_name="flight_api",
            endpoint_name="search_flights",
            description="Search for flights",
            method=HTTPMethod.GET,
            category="Travel",
            parameters=[
                Parameter(name="origin", type=ParameterType.STRING, required=True),
                Parameter(name="destination", type=ParameterType.STRING, required=True),
            ],
        ),
    ]


@pytest.fixture
def sample_chain(sample_endpoints) -> ToolChain:
    """A sample tool chain for testing."""
    return ToolChain(
        endpoints=sample_endpoints,
        pattern=ChainPattern.SEARCH_AND_ACT,
    )
