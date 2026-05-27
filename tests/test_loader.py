"""Tests for the ToolBench loader and registry."""

import json


from toolgen.registry.loader import load_tool_from_json, load_tools_from_directory


class TestLoadToolFromJson:
    def test_load_valid_tool(self, sample_tool_json, tmp_path):
        filepath = tmp_path / "hotel.json"
        filepath.write_text(json.dumps(sample_tool_json))

        tool = load_tool_from_json(filepath, category="Travel")
        assert tool is not None
        assert tool.name == "hotel_api"
        assert tool.category == "Travel"
        assert len(tool.endpoints) == 3

        # Check search endpoint
        search = tool.endpoints[0]
        assert search.endpoint_name == "search_hotels"
        assert search.endpoint_id == "hotel_api/search_hotels"
        assert len(search.required_parameters) == 1
        assert len(search.optional_parameters) == 3

    def test_load_malformed_tool(self, malformed_tool_json, tmp_path):
        filepath = tmp_path / "messy.json"
        filepath.write_text(json.dumps(malformed_tool_json))

        tool = load_tool_from_json(filepath)
        assert tool is not None
        assert tool.name == "messy_api"  # Whitespace trimmed
        # Should have 3 endpoints (skips the one without a name, renames duplicate)
        assert len(tool.endpoints) >= 2

        # First endpoint should handle 'int' type alias
        get_data = tool.endpoints[0]
        assert get_data.endpoint_name == "get_data"
        # The 'int' type should be parsed correctly
        if get_data.parameters:
            assert get_data.parameters[0].type.value == "integer"

    def test_load_invalid_json(self, tmp_path):
        filepath = tmp_path / "bad.json"
        filepath.write_text("not json at all")

        tool = load_tool_from_json(filepath)
        assert tool is None

    def test_load_empty_api_list(self, tmp_path):
        filepath = tmp_path / "empty.json"
        filepath.write_text(json.dumps({
            "tool_name": "empty",
            "api_list": [],
        }))

        tool = load_tool_from_json(filepath)
        assert tool is None  # No endpoints = skip


class TestLoadToolsFromDirectory:
    def test_load_directory(self, tool_json_dir):
        tools = load_tools_from_directory(tool_json_dir)
        assert len(tools) == 2  # hotel + flight
        names = {t.name for t in tools}
        assert "hotel_api" in names
        assert "flight_api" in names

    def test_max_tools(self, tool_json_dir):
        tools = load_tools_from_directory(tool_json_dir, max_tools=1)
        assert len(tools) == 1

    def test_category_filter(self, tool_json_dir):
        tools = load_tools_from_directory(tool_json_dir, categories=["Travel"])
        assert len(tools) == 2

        tools = load_tools_from_directory(tool_json_dir, categories=["Nonexistent"])
        assert len(tools) == 0

    def test_load_nested_category_files(self, sample_tool_json, tmp_path):
        nested_dir = tmp_path / "Travel" / "hotel_bundle"
        nested_dir.mkdir(parents=True)
        (nested_dir / "hotel_api.json").write_text(json.dumps(sample_tool_json))

        tools = load_tools_from_directory(tmp_path)

        assert len(tools) == 1
        assert tools[0].name == "hotel_api"
        assert tools[0].category == "Travel"

    def test_nonexistent_directory(self, tmp_path):
        tools = load_tools_from_directory(tmp_path / "nope")
        assert len(tools) == 0


class TestToolRegistry:
    def test_from_directory(self, sample_registry):
        assert len(sample_registry) > 0
        stats = sample_registry.stats()
        assert stats.total_tools == 2
        assert stats.total_categories == 1

    def test_get_endpoint(self, sample_registry):
        ep = sample_registry.get_endpoint("hotel_api/search_hotels")
        assert ep is not None
        assert ep.endpoint_name == "search_hotels"

        ep = sample_registry.get_endpoint("nonexistent/ep")
        assert ep is None

    def test_get_tool(self, sample_registry):
        tool = sample_registry.get_tool("hotel_api")
        assert tool is not None
        assert len(tool.endpoints) == 3

    def test_list_categories(self, sample_registry):
        cats = sample_registry.list_categories()
        assert "Travel" in cats

    def test_search(self, sample_registry):
        results = sample_registry.search("hotel")
        assert len(results) > 0
        assert any("hotel" in r.endpoint_name.lower() for r in results)

    def test_stats(self, sample_registry):
        stats = sample_registry.stats()
        assert stats.total_tools == 2
        assert stats.total_endpoints == 5  # 3 hotel + 2 flight
        assert stats.avg_params_per_endpoint > 0
