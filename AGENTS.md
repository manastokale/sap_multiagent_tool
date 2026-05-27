# AGENTS.md

Repository-local guide for coding agents working on ToolGen.

## Project Intent

ToolGen creates synthetic multi-turn, multi-tool conversations grounded in
ToolBench-style API schemas. Treat the project as a data-generation system, not a
generic chatbot demo. The priority order is:

1. valid, auditable tool traces
2. reproducible runs and tests
3. clear quality metrics and repair metadata
4. useful dashboard inspection
5. live LLM fluency when credentials and quota are available

## Read First

- `README.md` for setup and user-facing workflow
- `DESIGN.md` for architecture and algorithm choices
- `pyproject.toml` for package commands and dependencies
- `.env.example` for configuration names only

Do not print, copy, or document real values from `.env`.

## Source Map

- `src/toolgen/registry/`: ToolBench JSON parsing and endpoint registry
- `src/toolgen/graph/`: endpoint graph construction and constrained sampling
- `src/toolgen/agents/`: planner, user simulator, assistant, LLM client
- `src/toolgen/executor/`: mock tool outputs and session-local grounding
- `src/toolgen/judge/`: LLM-as-judge and deterministic fallback scoring
- `src/toolgen/repair/`: repair and retry loop
- `src/toolgen/pipeline.py`: orchestration, experiments, JSONL writing
- `dashboard/`: React/Vite/Material UI audit interface
- `scripts/`: ToolBench download and dashboard export helpers
- `tests/`: focused unit and integration coverage

## Engineering Rules

- Keep the offline path deterministic and runnable without credentials.
- Keep strict live mode strict: with `TOOLGEN_REQUIRE_LIVE_LLM=true`, do not silently
  fall back to templates or heuristic judging.
- Keep `TOOLGEN_LIVE_PROFILE=full` as the assessment-fidelity path. Use
  `TOOLGEN_LIVE_PROFILE=hybrid` only to save quota while keeping planner, assistant
  tool decisions, and judge live.
- Preserve generated JSONL contracts unless the docs and tests are updated together.
- Add focused tests when changing schema parsing, graph sampling, LLM routing, scoring,
  repair, or output serialization.
- Use provider headers for secrets. Do not put API keys into URLs, frontend bundles,
  logs, dashboard JSON, or docs.
- Generated files under `output/`, `dashboard/dist/`, caches, and local ToolBench dumps
  should stay ignored.

## Verification

Run the relevant subset, and for final checks prefer the full set:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
cd dashboard && npm run build
```

For dashboard changes, export a bundle first:

```bash
.venv/bin/python scripts/export_dashboard_data.py --dataset output/live_strict_smoke.jsonl
```

## Dashboard Style

The dashboard should remain an operational audit tool:

- Material UI components
- light Material UI theme
- sharp edges, no rounded decorative cards
- single-page trace workbench, not a multi-tab reporting app
- left pane for generated chats, center pane for the selected transcript
- right pane for ADK-style Events plus Event, Request, Response, and Graph details
- internal scroll regions instead of page overflow
