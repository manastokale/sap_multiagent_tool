"""CLI interface for the ToolGen pipeline.

Commands:
  toolgen generate          Generate conversations
  toolgen evaluate          Re-score existing conversations
  toolgen analyze           Compute diversity metrics
  toolgen run-experiment    Run diversity experiment (Run A vs Run B)
  toolgen info              Show registry and graph stats
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from toolgen.config import get_settings
from toolgen.pipeline import Pipeline, compute_diversity_metrics
from toolgen.models import Conversation

console = Console()
SECRET_KEY_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """ToolGen: Multi-agent tool-calling conversation generator."""
    _setup_logging(verbose)


@main.command()
@click.option("--max-tools", default=None, type=int, help="Limit tools loaded (for dev)")
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Category to include. Can be passed multiple times.",
)
def build(max_tools: int | None, categories: tuple[str, ...]) -> None:
    """Ingest ToolBench data and build registry/graph artifacts."""
    settings = get_settings()
    pipeline = Pipeline(settings)

    with console.status("Building registry and graph artifacts..."):
        paths = pipeline.build_artifacts(
            max_tools=max_tools,
            categories=list(categories) if categories else None,
        )

    console.print("[bold]Build complete[/bold]")
    for name, path in paths.items():
        console.print(f"  {name}: {path}")


@main.command("config")
def show_config() -> None:
    """Show resolved settings without printing secret values."""
    settings = get_settings()

    table = Table(title="ToolGen Config")
    table.add_column("Setting", style="cyan")
    table.add_column("Resolved Value", style="green")
    table.add_row("Project root", str(settings.project_root))
    table.add_row(
        "Env file",
        f"{settings.env_file} ({'found' if settings.env_file.exists() else 'missing'})",
    )
    table.add_row("Precedence", "constructor/CLI overrides > project .env > shell env > defaults")
    table.add_row("Mode", _format_mode(settings.llm_provider, settings.use_offline_llm))
    table.add_row("LLM provider", settings.llm_provider)
    table.add_row("Live profile", settings.normalized_live_profile)
    table.add_row("Randomize models", str(settings.randomize_models))
    table.add_row("Generation models", _format_models(settings.generation_models))
    table.add_row("Planner models", _format_models(settings.planner_models))
    table.add_row("Assistant models", _format_models(settings.assistant_models))
    table.add_row("User simulator models", _format_models(settings.user_models))
    table.add_row("Summary models", _format_models(settings.summary_models))
    table.add_row("Judge models", _format_models(settings.judge_models))
    table.add_row("Parallel conversations", str(settings.normalized_max_parallel_conversations))
    table.add_row("Max turns/conversation", str(settings.max_turns_per_conversation))
    table.add_row("LLM max output tokens", str(settings.llm_max_output_tokens))
    table.add_row("Require live LLM", str(settings.require_live_llm))
    table.add_row("Requests/minute", str(settings.llm_requests_per_minute))
    table.add_row("Gemini key", _secret_state(settings.gemini_api_key))
    table.add_row("Groq key (unused)", _secret_state(settings.groq_api_key))
    console.print(table)

    shell_keys = _relevant_shell_keys()
    if shell_keys:
        console.print(
            "[yellow]Shell variables also exist, but project .env wins when it sets "
            "the same key:[/yellow]"
        )
        console.print("  " + ", ".join(shell_keys))


@main.command()
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--num-conversations", "-n", default=100, help="Number of conversations")
@click.option(
    "--no-cross-conversation-steering",
    is_flag=True,
    help="Disable cross-conversation diversity steering (for Run A)",
)
@click.option("--no-repair", is_flag=True, help="Disable automatic repair")
@click.option("--max-tools", default=None, type=int, help="Limit tools loaded (for dev)")
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Category to include. Can be passed multiple times.",
)
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSONL path")
@click.option(
    "--progress-log/--no-progress-log",
    default=False,
    help="Print detailed generation events above the compact progress display.",
)
def generate(
    seed: int,
    num_conversations: int,
    no_cross_conversation_steering: bool,
    no_repair: bool,
    max_tools: int | None,
    categories: tuple[str, ...],
    output: str | None,
    progress_log: bool,
) -> None:
    """Generate synthetic tool-calling conversations."""
    settings = get_settings()

    if settings.llm_provider.lower() != "offline" and settings.use_offline_llm:
        console.print("[yellow]No live LLM key available; falling back to offline mode.[/yellow]")

    console.print("[bold]ToolGen Generate[/bold]")
    console.print(f"  Seed: {seed}")
    console.print(f"  Conversations: {num_conversations}")
    console.print(f"  Steering: {'disabled' if no_cross_conversation_steering else 'enabled'}")
    console.print(f"  Repair: {'disabled' if no_repair else 'enabled'}")
    console.print(f"  Mode: {_format_mode(settings.llm_provider, settings.use_offline_llm)}")
    if not settings.use_offline_llm:
        console.print(f"  Live profile: {settings.normalized_live_profile}")
    console.print(f"  Generation models: {_format_models(settings.generation_models)}")
    console.print(f"  Planner models: {_format_models(settings.planner_models)}")
    console.print(f"  Assistant models: {_format_models(settings.assistant_models)}")
    console.print(f"  User models: {_format_models(settings.user_models)}")
    console.print(f"  Summary models: {_format_models(settings.summary_models)}")
    console.print(f"  Judge models: {_format_models(settings.judge_models)}")
    console.print(f"  Parallel conversations: {settings.normalized_max_parallel_conversations}")
    console.print(f"  Max turns/conversation: {settings.max_turns_per_conversation}")
    console.print(f"  LLM max output tokens: {settings.llm_max_output_tokens}")
    if categories:
        console.print(f"  Categories: {', '.join(categories)}")
    console.print()

    pipeline = Pipeline(settings)

    with console.status("Loading tool registry..."):
        registry = pipeline.load_registry(
            max_tools=max_tools,
            categories=list(categories) if categories else None,
        )
    console.print(f"  Registry: {registry}")

    with console.status("Building tool graph..."):
        pipeline.build_graph()

    output_path = Path(output) if output else None

    with _make_progress() as progress:
        task_id = progress.add_task("Generating chats | initializing", total=num_conversations)
        status_task_id = progress.add_task("Run status | waiting", total=None)
        conversation_tasks: dict[str, int] = {}
        live_calls = 0
        latest_event = "starting"
        completed_count = 0
        event_count = 0
        progress_lock = threading.Lock()

        def refresh_progress() -> None:
            progress.update(
                task_id,
                description=(
                    f"Generating chats | done={completed_count}/{num_conversations} "
                    f"| live_calls={live_calls} | {_short_progress_label(latest_event, 90)}"
                ),
            )
            progress.update(
                status_task_id,
                description=f"Run status | {_short_progress_label(latest_event, 120)}",
            )

        def update_progress(current: int, total: int, label: str) -> None:
            nonlocal completed_count, latest_event
            with progress_lock:
                completed_count = current
                conv_id = _extract_conversation_id(label)
                if conv_id and conv_id in conversation_tasks:
                    progress.update(
                        conversation_tasks[conv_id],
                        description=f"{conv_id} | done | {_short_progress_label(label, 80)}",
                        visible=False,
                    )
                    del conversation_tasks[conv_id]
                latest_event = _format_compact_event(event_count, label, live_calls)
                progress.update(task_id, total=total, completed=current)
                refresh_progress()

        def update_live_calls(count: int) -> None:
            nonlocal live_calls
            with progress_lock:
                live_calls = count
                refresh_progress()

        def update_status(label: str) -> None:
            nonlocal event_count, latest_event
            with progress_lock:
                event_count += 1
                conv_id, stage, detail = _parse_status_label(label)
                latest_event = _format_compact_event(event_count, label, live_calls)
                if conv_id and _show_conversation_progress(stage):
                    task_description = _format_conversation_progress(
                        conv_id,
                        event_count,
                        stage,
                        detail,
                    )
                    task_id_for_conv = conversation_tasks.get(conv_id)
                    if task_id_for_conv is None:
                        task_id_for_conv = progress.add_task(task_description, total=None)
                        conversation_tasks[conv_id] = task_id_for_conv
                    else:
                        progress.update(
                            task_id_for_conv,
                            description=task_description,
                            visible=True,
                        )
                refresh_progress()
                if progress_log:
                    progress.console.log(
                        f"[cyan]event {event_count:04d}[/cyan] {label} "
                        f"[dim](live_calls={live_calls})[/dim]"
                    )

        conversations = pipeline.generate(
            num_conversations=num_conversations,
            seed=seed,
            enable_steering=not no_cross_conversation_steering,
            enable_repair=not no_repair,
            output_path=output_path,
            progress_callback=update_progress,
            live_call_callback=update_live_calls,
            status_callback=update_status,
        )

    # Summary table
    _print_summary(conversations)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def evaluate(input_path: str) -> None:
    """Re-score conversations from a JSONL file."""
    settings = get_settings()

    pipeline = Pipeline(settings)
    input_file = Path(input_path)
    total = _count_jsonl_records(input_file)
    with _make_progress() as progress:
        task_id = progress.add_task("Evaluating chats", total=total)

        def update_progress(current: int, progress_total: int, label: str) -> None:
            progress.update(
                task_id,
                total=progress_total,
                completed=current,
                description=f"Evaluating chats | {label}",
            )

        conversations = pipeline.evaluate(input_file, progress_callback=update_progress)

    # Write updated scores
    output_path = input_file.with_suffix(".scored.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")

    console.print(f"Scored {len(conversations)} conversations → {output_path}")
    _print_summary(conversations)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
def analyze(input_path: str) -> None:
    """Compute diversity metrics for a JSONL dataset."""
    input_file = Path(input_path)
    conversations = []
    total = _count_jsonl_records(input_file)
    with _make_progress() as progress:
        task_id = progress.add_task("Loading chats", total=total)
        with input_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                conversation = Conversation(**data)
                conversations.append(conversation)
                progress.update(
                    task_id,
                    completed=len(conversations),
                    description=f"Loading chats | {conversation.conversation_id}",
                )
        progress.update(task_id, completed=total, description="Computing diversity metrics")

    metrics = compute_diversity_metrics(conversations)

    table = Table(title="Diversity Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Tool Combination Entropy", str(metrics.tool_combination_entropy))
    table.add_row("Domain Coverage CV", str(metrics.domain_coverage_cv))
    table.add_row("Unique Tool Pairs", str(metrics.unique_tool_pairs))
    table.add_row("Unique Domains", str(metrics.unique_domains_used))
    table.add_row("Mean Quality Score", str(metrics.mean_quality_score))
    table.add_row("Pattern Distribution", json.dumps(metrics.pattern_distribution, indent=2))
    console.print(table)


@main.command("run-experiment")
@click.option("--seed", default=42, help="Random seed")
@click.option("--num-conversations", "-n", default=50, help="Conversations per run")
@click.option("--max-tools", default=None, type=int, help="Limit tools loaded")
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Category to include. Can be passed multiple times.",
)
def run_experiment(
    seed: int,
    num_conversations: int,
    max_tools: int | None,
    categories: tuple[str, ...],
) -> None:
    """Run the diversity experiment (Run A vs Run B)."""
    settings = get_settings()

    pipeline = Pipeline(settings)
    pipeline.load_registry(
        max_tools=max_tools,
        categories=list(categories) if categories else None,
    )
    pipeline.build_graph()

    results = pipeline.run_diversity_experiment(
        num_conversations=num_conversations,
        seed=seed,
    )

    # Print comparison table
    table = Table(title="Diversity Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Run A (no steering)", style="yellow")
    table.add_column("Run B (steering)", style="green")

    table.add_row(
        "Tool Combo Entropy",
        str(results.run_a.tool_combination_entropy),
        str(results.run_b.tool_combination_entropy),
    )
    table.add_row(
        "Domain Coverage CV",
        str(results.run_a.domain_coverage_cv),
        str(results.run_b.domain_coverage_cv),
    )
    table.add_row(
        "Unique Tool Pairs",
        str(results.run_a.unique_tool_pairs),
        str(results.run_b.unique_tool_pairs),
    )
    table.add_row(
        "Mean Quality",
        str(results.run_a.mean_quality_score),
        str(results.run_b.mean_quality_score),
    )
    console.print(table)


@main.command()
@click.option("--max-tools", default=None, type=int, help="Limit tools loaded")
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Category to include. Can be passed multiple times.",
)
def info(max_tools: int | None, categories: tuple[str, ...]) -> None:
    """Show registry and graph statistics."""
    settings = get_settings()
    pipeline = Pipeline(settings)

    with console.status("Loading registry..."):
        registry = pipeline.load_registry(
            max_tools=max_tools,
            categories=list(categories) if categories else None,
        )

    stats = registry.stats()

    table = Table(title="Registry Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Tools", str(stats.total_tools))
    table.add_row("Total Endpoints", str(stats.total_endpoints))
    table.add_row("Categories", str(stats.total_categories))
    table.add_row("Avg Params/Endpoint", str(stats.avg_params_per_endpoint))
    table.add_row("With Response Schema", str(stats.endpoints_with_response_schema))
    console.print(table)

    # Top categories
    cat_table = Table(title="Top Categories")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Endpoints", style="green")
    for cat, count in sorted(stats.categories.items(), key=lambda x: -x[1])[:15]:
        cat_table.add_row(cat, str(count))
    console.print(cat_table)


def _print_summary(conversations: list[Conversation]) -> None:
    """Print a summary table of generated conversations."""
    if not conversations:
        console.print("[yellow]No conversations generated.[/yellow]")
        return

    total = len(conversations)
    multi_step = sum(
        1 for c in conversations
        if c.metadata.num_tool_calls >= 3 and c.metadata.num_distinct_tools >= 2
    )
    avg_turns = sum(c.metadata.num_turns for c in conversations) / total
    avg_tools = sum(c.metadata.num_tool_calls for c in conversations) / total
    scored = [c for c in conversations if c.judge_scores]
    avg_score = (
        sum(c.judge_scores.overall for c in scored) / len(scored)
        if scored
        else 0.0
    )
    repaired = sum(1 for c in conversations if c.metadata.repair_attempts > 0)

    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Conversations", str(total))
    table.add_row("Multi-step + Multi-tool", f"{multi_step} ({multi_step/total*100:.1f}%)")
    table.add_row("Avg Turns", f"{avg_turns:.1f}")
    table.add_row("Avg Tool Calls", f"{avg_tools:.1f}")
    table.add_row("Avg Quality Score", f"{avg_score:.2f}")
    table.add_row("Repaired", str(repaired))
    console.print(table)


def _format_models(models: list[str]) -> str:
    if len(models) == 1:
        return models[0]
    return "random: " + ", ".join(models)


def _format_mode(provider: str, offline: bool) -> str:
    if offline:
        return "offline deterministic"
    if provider.lower() == "auto":
        return "live auto"
    return f"live {provider}"


def _secret_state(value: str) -> str:
    return "set" if value else "missing"


def _relevant_shell_keys() -> list[str]:
    keys = []
    for key in os.environ:
        if key.startswith("TOOLGEN_") or key in {"GEMINI_API_KEY", "GROQ_API_KEY"}:
            keys.append(key)
    return sorted(keys, key=_shell_key_sort)


def _shell_key_sort(key: str) -> tuple[int, str]:
    return (1 if any(fragment in key.upper() for fragment in SECRET_KEY_FRAGMENTS) else 0, key)


def _parse_status_label(label: str) -> tuple[str | None, str, str]:
    parts = [part.strip() for part in str(label).split("|")]
    conv_id = _extract_conversation_id(parts[0]) if parts else None
    if conv_id:
        raw_stage = parts[1] if len(parts) > 1 else "status"
        detail = " | ".join(parts[2:]) if len(parts) > 2 else ""
    else:
        raw_stage = parts[0] if parts else "status"
        detail = " | ".join(parts[1:]) if len(parts) > 1 else ""
    return conv_id, _normalize_stage(raw_stage, detail), detail


def _extract_conversation_id(label: str) -> str | None:
    match = re.search(r"\bconv_\d{4}\b", str(label))
    return match.group(0) if match else None


def _normalize_stage(stage: str, detail: str = "") -> str:
    stage_text = stage.strip().lower()
    detail_text = detail.strip().lower()

    if stage_text.startswith("turn"):
        if detail_text.startswith("tool call"):
            return "tool_call"
        if detail_text.startswith("executing"):
            return "execute"
        if detail_text.startswith("result"):
            return "result"
        if detail_text.startswith("assistant summary"):
            return "summary"
        if detail_text.startswith("assistant text"):
            return "assistant_text"
        return "assistant"

    if stage_text.startswith("offline step"):
        if detail_text.startswith("building args"):
            return "args"
        if detail_text.startswith("executing"):
            return "execute"
        if detail_text.startswith("result"):
            return "result"
        return "offline_step"

    if stage_text == "offline" and detail_text.startswith("planning"):
        return "planner"
    if stage_text == "chain ready":
        return "chain"
    return stage_text.replace(" ", "_") or "status"


def _show_conversation_progress(stage: str) -> bool:
    return stage not in {"steering", "sampling", "chain", "skipped"}


def _format_compact_event(event_count: int, label: str, live_calls: int) -> str:
    conv_id, stage, detail = _parse_status_label(label)
    event = f"event={event_count:04d}" if event_count else "event=----"
    if conv_id:
        pieces = [event, conv_id, stage]
    else:
        pieces = [event, stage]
    if detail and not conv_id:
        pieces.append(_short_progress_label(detail, 42))
    pieces.append(f"live_calls={live_calls}")
    return " | ".join(pieces)


def _format_conversation_progress(
    conv_id: str,
    event_count: int,
    stage: str,
    detail: str,
) -> str:
    suffix = f" | {_short_progress_label(detail, 70)}" if detail else ""
    return f"{conv_id} | event={event_count:04d} | {stage}{suffix}"


def _short_progress_label(label: str, max_chars: int = 150) -> str:
    normalized = " ".join(str(label).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _count_jsonl_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
