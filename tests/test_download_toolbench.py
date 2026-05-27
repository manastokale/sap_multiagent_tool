"""Tests for ToolBench downloader conversion helpers."""

import json

from scripts.download_toolbench import _api_from_hf_row, _derive_tool_name, _safe_relative_path


def test_hf_row_converter_maps_api_doc_to_toolbench_endpoint():
    row = {
        "row": {
            "text": json.dumps(
                {
                    "category_name": "Logistics",
                    "required_parameters": [{"name": "task_id", "type": "STRING"}],
                    "optional_parameters": [],
                    "method": "GET",
                    "template_response": {"statusCode": "int"},
                    "name": "SQUAKE_Checkhealth",
                    "description": "Health check.",
                }
            )
        }
    }

    api = _api_from_hf_row(row)

    assert api is not None
    assert api["_category"] == "Logistics"
    assert api["_tool_name"] == "SQUAKE"
    assert api["name"] == "SQUAKE_Checkhealth"
    assert api["response_schema"] == {"statusCode": "int"}


def test_derive_tool_name_uses_prefix_for_related_apis():
    assert _derive_tool_name("SQUAKE_Projects") == "SQUAKE"
    assert _derive_tool_name("single") == "single"


def test_safe_relative_path_blocks_parent_traversal():
    assert str(_safe_relative_path("../secret/tool.json")) == "secret/tool.json"
    assert str(_safe_relative_path("./nested/../../tool.json")) == "nested/tool.json"
