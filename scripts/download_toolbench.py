"""Download ToolBench tool definitions from GitHub via the REST API.

The bundled fixture corpus under ``data/toolenv/tools`` is intentionally small.
This script downloads a larger ToolBench slice into ``.toolbench_tmp`` by default
so local verification can use real ToolBench files without polluting the checked-in
fixture data.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = "OpenBMB/ToolBench"
BRANCH = "master"
DEFAULT_TOOLS_PATH = "data/toolenv/tools"
API_BASE = "https://api.github.com"
HF_ROWS_BASE = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "mteb/ToolBench"
HF_CONFIG = "ToolBench-corpus"
HF_SPLIT = "test"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TARGET_DIR = PROJECT_ROOT / ".toolbench_tmp" / "toolenv" / "tools"


def json_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Fetch a JSON document with basic retry handling."""
    req = urllib.request.Request(url, headers=headers or {})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                retry_after = e.headers.get("Retry-After", "60")
                wait = int(retry_after) if retry_after.isdigit() else 60
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            print(f"  Network error (attempt {attempt + 1}): {e.reason}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def github_get(url: str, token: str = "") -> dict[str, Any]:
    """Make a GET request to the GitHub API."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return json_get(url, headers=headers)


def get_tree_sha(path: str, token: str = "") -> str:
    """Get the tree SHA for a specific path in the repo."""
    url = f"{API_BASE}/repos/{REPO}/git/trees/{BRANCH}?recursive=0"
    data = github_get(url, token)

    parts = [part for part in path.split("/") if part]
    current_sha = str(data["sha"])

    for part in parts:
        url = f"{API_BASE}/repos/{REPO}/git/trees/{current_sha}"
        tree_data = github_get(url, token)
        for item in tree_data["tree"]:
            if item["path"] == part and item["type"] == "tree":
                current_sha = str(item["sha"])
                break
        else:
            raise RuntimeError(f"Path component '{part}' not found in {path!r}")

    return current_sha


def download_github_tools(
    token: str = "",
    source_path: str = DEFAULT_TOOLS_PATH,
    target_dir: Path = DEFAULT_TARGET_DIR,
    max_categories: int | None = None,
    max_tools: int | None = None,
    overwrite: bool = False,
) -> None:
    """Download ToolBench-style JSON files from a GitHub tree."""
    print(f"==> Resolving tree for {source_path}...")
    tools_sha = get_tree_sha(source_path, token)

    print(f"==> Fetching category listing (tree SHA: {tools_sha[:8]})...")
    tools_tree = github_get(f"{API_BASE}/repos/{REPO}/git/trees/{tools_sha}", token)

    categories = [
        item
        for item in tools_tree["tree"]
        if item["type"] == "tree" and not item["path"].startswith(".")
    ]
    if max_categories:
        categories = categories[:max_categories]

    print(f"==> Found {len(categories)} categories")
    target_dir.mkdir(parents=True, exist_ok=True)

    total_tools = 0
    for cat_idx, cat in enumerate(categories):
        if max_tools is not None and total_tools >= max_tools:
            break

        cat_name = _safe_name(str(cat["path"]))
        cat_dir = target_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{cat_idx + 1}/{len(categories)}] {cat_name}...", end="", flush=True)
        cat_tree = github_get(
            f"{API_BASE}/repos/{REPO}/git/trees/{cat['sha']}?recursive=1",
            token,
        )
        json_files = [
            item
            for item in cat_tree["tree"]
            if item["type"] == "blob" and item["path"].endswith(".json")
        ]

        cat_count = 0
        for json_file in json_files:
            if max_tools is not None and total_tools >= max_tools:
                break

            target_path = cat_dir / _safe_relative_path(str(json_file["path"]))
            if target_path.exists() and not overwrite:
                continue

            try:
                blob = github_get(json_file["url"], token)
                content = base64.b64decode(blob["content"]).decode("utf-8")
                parsed = json.loads(content)
                if not _looks_like_toolbench_tool(parsed):
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                cat_count += 1
                total_tools += 1
            except Exception as e:
                print(f"\n    Warning: Failed to download {json_file['path']}: {e}")
                continue

            if total_tools % 25 == 0:
                time.sleep(0.5)

        print(f" {cat_count} new tools")

    total_on_disk = sum(1 for _ in target_dir.rglob("*.json"))
    print(f"\n==> Done: {total_tools} new tools downloaded to {target_dir}")
    print(f"==> Total tools on disk: {total_on_disk}")
    print(f"==> Use with: export TOOLGEN_TOOLENV_DIR={target_dir}")


def download_hf_toolbench_corpus(
    target_dir: Path = DEFAULT_TARGET_DIR,
    max_categories: int | None = None,
    max_apis: int | None = None,
    page_size: int = 100,
    overwrite: bool = False,
) -> None:
    """Download API docs from the Hugging Face ToolBench retrieval corpus.

    The HF corpus stores one API per row. This converter groups rows into
    ToolBench-like ``tool_name.json`` files with an ``api_list``.
    """
    print(f"==> Fetching {HF_DATASET}/{HF_CONFIG}/{HF_SPLIT} from Hugging Face...")
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    categories_seen: list[str] = []
    kept_apis = 0
    offset = 0

    while max_apis is None or kept_apis < max_apis:
        length = min(page_size, max_apis - kept_apis) if max_apis else page_size
        params = urllib.parse.urlencode(
            {
                "dataset": HF_DATASET,
                "config": HF_CONFIG,
                "split": HF_SPLIT,
                "offset": offset,
                "length": length,
            }
        )
        data = json_get(f"{HF_ROWS_BASE}?{params}")
        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            api_record = _api_from_hf_row(row)
            if api_record is None:
                continue

            category = api_record.pop("_category")
            tool_name = api_record.pop("_tool_name")
            if category not in categories_seen:
                if max_categories is not None and len(categories_seen) >= max_categories:
                    continue
                categories_seen.append(category)

            grouped.setdefault((category, tool_name), []).append(api_record)
            kept_apis += 1
            if max_apis is not None and kept_apis >= max_apis:
                break

        offset += len(rows)
        total_rows = int(data.get("num_rows_total") or 0)
        if total_rows and offset >= total_rows:
            break

    written_tools = 0
    for (category, tool_name), api_list in grouped.items():
        filename = f"{_safe_name(tool_name)}.json"
        target_path = target_dir / _safe_name(category) / filename
        if target_path.exists() and not overwrite:
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tool_name": tool_name,
            "tool_description": f"ToolBench retrieval corpus APIs for {tool_name}.",
            "title": tool_name,
            "api_list": api_list,
        }
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written_tools += 1

    print(f"==> Done: {kept_apis} APIs grouped into {written_tools} tool files")
    print(f"==> Total tool files on disk: {sum(1 for _ in target_dir.rglob('*.json'))}")
    print(f"==> Data at: {target_dir}")
    print(f"==> Use with: export TOOLGEN_TOOLENV_DIR={target_dir}")


def _looks_like_toolbench_tool(parsed: Any) -> bool:
    return isinstance(parsed, dict) and isinstance(parsed.get("api_list"), list)


def _api_from_hf_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_text = row.get("row", {}).get("text")
    if not isinstance(raw_text, str):
        return None
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    api_name = str(raw.get("name") or row.get("row_idx") or "api").strip()
    if not api_name:
        return None

    category = str(raw.get("category_name") or "Uncategorized").strip() or "Uncategorized"
    tool_name = _derive_tool_name(api_name)
    return {
        "_category": category,
        "_tool_name": tool_name,
        "name": api_name,
        "description": str(raw.get("description") or "").strip(),
        "method": raw.get("method") or "GET",
        "required_parameters": raw.get("required_parameters") or [],
        "optional_parameters": raw.get("optional_parameters") or [],
        "response_schema": raw.get("template_response") or raw.get("response_schema") or {},
    }


def _derive_tool_name(api_name: str) -> str:
    parts = [part for part in api_name.split("_") if part]
    if len(parts) <= 1:
        return api_name
    return parts[0]


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return cleaned.strip("._") or "unnamed"


def _safe_relative_path(value: str) -> Path:
    parts = [
        _safe_name(part)
        for part in Path(value).parts
        if part not in {"", ".", ".."}
    ]
    if not parts:
        return Path("unnamed.json")
    return Path(*parts)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "legacy_max_categories",
        nargs="?",
        type=int,
        help="Backward-compatible positional alias for --max-categories.",
    )
    parser.add_argument(
        "--source",
        choices=["hf-corpus", "github"],
        default="hf-corpus",
        help="Download from the HF ToolBench corpus or a GitHub tree.",
    )
    parser.add_argument("--source-path", default=DEFAULT_TOOLS_PATH)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument(
        "--max-tools",
        type=int,
        default=None,
        help="Limit API rows for hf-corpus, or JSON files for github.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_categories is None:
        args.max_categories = args.legacy_max_categories
    return args


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.source == "hf-corpus":
        download_hf_toolbench_corpus(
            target_dir=args.target_dir,
            max_categories=args.max_categories,
            max_apis=args.max_tools,
            page_size=args.page_size,
            overwrite=args.overwrite,
        )
    else:
        download_github_tools(
            token=os.environ.get("GITHUB_TOKEN", ""),
            source_path=args.source_path,
            target_dir=args.target_dir,
            max_categories=args.max_categories,
            max_tools=args.max_tools,
            overwrite=args.overwrite,
        )
