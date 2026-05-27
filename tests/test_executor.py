"""Tests for the mock executor."""

import pytest

from toolgen.executor.mock_executor import MockExecutor
from toolgen.models import APIEndpoint, HTTPMethod, Parameter, ParameterType


@pytest.fixture
def executor():
    return MockExecutor(seed=42)


@pytest.fixture
def search_endpoint():
    return APIEndpoint(
        tool_name="hotel_api",
        endpoint_name="search_hotels",
        description="Search for hotels in a city",
        method=HTTPMethod.GET,
        parameters=[
            Parameter(name="city", type=ParameterType.STRING, required=True),
            Parameter(name="max_price", type=ParameterType.NUMBER),
        ],
    )


@pytest.fixture
def get_endpoint():
    return APIEndpoint(
        tool_name="hotel_api",
        endpoint_name="get_hotel_details",
        description="Get detailed info about a specific hotel",
        method=HTTPMethod.GET,
        parameters=[
            Parameter(name="hotel_id", type=ParameterType.STRING, required=True),
        ],
    )


@pytest.fixture
def book_endpoint():
    return APIEndpoint(
        tool_name="hotel_api",
        endpoint_name="book_hotel",
        description="Book a hotel room",
        method=HTTPMethod.POST,
        parameters=[
            Parameter(name="hotel_id", type=ParameterType.STRING, required=True),
            Parameter(name="check_in", type=ParameterType.STRING, required=True),
        ],
    )


class TestMockExecutor:
    def test_search_returns_results(self, executor, search_endpoint):
        result = executor.execute(search_endpoint, {"city": "Paris"})
        assert "results" in result
        assert len(result["results"]) >= 2
        # Each result should have an ID
        for item in result["results"]:
            assert "id" in item
            assert "name" in item

    def test_search_results_stored_in_session(self, executor, search_endpoint):
        executor.execute(search_endpoint, {"city": "Paris"})
        summary = executor.get_session_summary()
        assert "hotel" in summary["entities"]
        assert len(summary["entities"]["hotel"]) >= 2

    def test_get_uses_session_id(self, executor, search_endpoint, get_endpoint):
        """ID from search should be retrievable via get."""
        search_result = executor.execute(search_endpoint, {"city": "Paris"})
        hotel_id = search_result["results"][0]["id"]

        detail = executor.execute(get_endpoint, {"hotel_id": hotel_id})
        assert detail["id"] == hotel_id

    def test_create_returns_confirmation(self, executor, book_endpoint):
        result = executor.execute(book_endpoint, {
            "hotel_id": "htl_test",
            "check_in": "2026-06-01",
        })
        assert "status" in result
        assert result["status"] == "confirmed"
        assert "confirmation_id" in result

    def test_multi_step_chain_consistency(
        self, executor, search_endpoint, get_endpoint, book_endpoint
    ):
        """Full chain: search → get details → book. IDs must chain correctly."""
        # 1. Search
        search_result = executor.execute(search_endpoint, {"city": "Paris"})
        hotel_id = search_result["results"][0]["id"]

        # 2. Get details using the ID from search
        detail = executor.execute(get_endpoint, {"hotel_id": hotel_id})
        assert detail["id"] == hotel_id

        # 3. Book using the same ID
        booking = executor.execute(book_endpoint, {
            "hotel_id": hotel_id,
            "check_in": "2026-06-01",
        })
        assert booking["status"] == "confirmed"

    def test_reset_clears_state(self, executor, search_endpoint):
        executor.execute(search_endpoint, {"city": "Paris"})
        assert executor.get_session_summary()["entities"]

        executor.reset()
        assert not executor.get_session_summary()["entities"]

    def test_generic_endpoint(self, executor):
        ep = APIEndpoint(
            tool_name="weather_api",
            endpoint_name="get_weather",
            description="Get weather forecast",
        )
        result = executor.execute(ep, {"city": "Tokyo"})
        assert "temperature" in result or "status" in result

    def test_delete_operation(self, executor):
        ep = APIEndpoint(
            tool_name="hotel_api",
            endpoint_name="cancel_booking",
            description="Cancel a hotel booking",
            parameters=[
                Parameter(name="booking_id", type=ParameterType.STRING, required=True),
            ],
        )
        result = executor.execute(ep, {"booking_id": "bk_1234"})
        assert result["status"] == "deleted"
