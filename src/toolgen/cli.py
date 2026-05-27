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
    table.add_row("Judge models", _format_models(settings.judge_models))
    table.add_row("Require live LLM", str(settings.require_live_llm))
    table.add_row("Requests/minute", str(settings.llm_requests_per_minute))
    table.add_row("Gemini key", _secret_state(settings.gemini_api_key))
    table.add_row("Groq key", _secret_state(settings.groq_api_key))
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
def generate(
    seed: int,
    num_conversations: int,
    no_cross_conversation_steering: bool,
    no_repair: bool,
    max_tools: int | None,
    categories: tuple[str, ...],
    output: str | None,
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
    console.print(f"  Judge models: {_format_models(settings.judge_models)}")
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
        task_id = progress.add_task("Generating chats", total=num_conversations)

        def update_progress(current: int, total: int, label: str) -> None:
            progress.update(
                task_id,
                total=total,
                completed=current,
                description=f"Generating chats | {label}",
            )

        conversations = pipeline.generate(
            num_conversations=num_conversations,
            seed=seed,
            enable_steering=not no_cross_conversation_steering,
            enable_repair=not no_repair,
            output_path=output_path,
            progress_callback=update_progress,
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
