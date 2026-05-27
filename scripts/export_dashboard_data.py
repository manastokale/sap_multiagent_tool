"""Export generated ToolGen artifacts into a browser-friendly dashboard bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_bundle(
    output_dir: Path,
    artifacts_dir: Path,
    dataset: Path,
    max_conversations: int,
) -> dict[str, Any]:
    conversations = _read_jsonl(dataset, limit=max_conversations)
    return {
        "source": {
            "dataset": str(dataset),
            "artifacts_dir": str(artifacts_dir),
        },
        "conversations": conversations,
        "liveSamples": [
            row
            for path in [
                output_dir / "live_strict_smoke.jsonl",
                output_dir / "live_gemini_smoke.jsonl",
                output_dir / "gemini_smoke.jsonl",
                output_dir / "model_routing_smoke.jsonl",
            ]
            for row in _read_jsonl(path, limit=10)
        ],
        "runA": _read_jsonl(output_dir / "run_a_seed42.jsonl", limit=max_conversations),
        "runB": _read_jsonl(output_dir / "run_b_seed42.jsonl", limit=max_conversations),
        "diversity": _read_json(output_dir / "diversity_analysis.json", {}),
        "artifacts": {
            "registryStats": _read_json(artifacts_dir / "registry_stats.json", {}),
            "graphStats": _read_json(artifacts_dir / "graph_stats.json", {}),
            "endpoints": _read_jsonl(artifacts_dir / "endpoints.jsonl"),
            "edges": _read_jsonl(artifacts_dir / "graph_edges.jsonl"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("output/artifacts"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/toolbench_hf_100.jsonl"),
        help="Conversation JSONL file to preload into the dashboard.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("dashboard/public/toolgen-data/bundle.json"),
    )
    parser.add_argument("--max-conversations", type=int, default=250)
    args = parser.parse_args()

    bundle = build_bundle(
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        dataset=args.dataset,
        max_conversations=args.max_conversations,
    )
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {args.target} with {len(bundle['conversations'])} conversations")


if __name__ == "__main__":
    main()
