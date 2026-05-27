"""End-to-end pipeline orchestration.

Ties all components together:
  Registry → Graph → Sampler → Agents → Judge → Repair → Output
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolgen.agents.llm_client import LLMClient, SharedRateLimiter
from toolgen.agents.orchestrator import generate_conversation, generate_offline_conversation
from toolgen.config import ToolGenSettings
from toolgen.executor.mock_executor import MockExecutor
from toolgen.graph.builder import build_tool_graph, get_graph_stats
from toolgen.graph.sampler import sample_chain
from toolgen.judge.scorer import passes_quality_threshold, score_conversation
from toolgen.memory.steering import DiversitySteerer
from toolgen.models import (
    Conversation,
    DiversityMetrics,
    ExperimentResults,
    SamplerConstraints,
    ToolChain,
)
from toolgen.registry.registry import ToolRegistry
from toolgen.repair.repairer import repair_conversation

logger = logging.getLogger(__name__)
_MAX_ENDPOINT_DIAGRAM_NODES = 80
_MAX_EDGE_DIAGRAM_EDGES = 120


@dataclass(frozen=True)
class _GenerationTask:
    index: int
    chain: ToolChain
    steering: Any


class Pipeline:
    """End-to-end conversation generation pipeline."""

    def __init__(self, settings: ToolGenSettings):
        self.settings = settings
        self._registry: ToolRegistry | None = None
        self._graph = None

    def load_registry(
        self,
        max_tools: int | None = None,
        categories: list[str] | None = None,
    ) -> ToolRegistry:
        """Load the tool registry from disk."""
        logger.info("Loading tool registry from %s", self.settings.toolenv_dir)
        self._registry = ToolRegistry.from_directory(
            self.settings.toolenv_dir,
            max_tools=max_tools,
            categories=categories,
        )
        logger.info("Registry: %s", self._registry)
        return self._registry

    def build_graph(self) -> Any:
        """Build the tool graph from the registry."""
        if self._registry is None:
            raise RuntimeError("Registry not loaded. Call load_registry() first.")
        logger.info("Building tool graph...")
        self._graph = build_tool_graph(self._registry)
        stats = get_graph_stats(self._graph)
        logger.info("Graph stats: %s", stats)
        return self._graph

    def build_artifacts(
        self,
        max_tools: int | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Path]:
        """Build registry/graph artifacts used for inspection and reproducibility."""
        registry = self.load_registry(max_tools=max_tools, categories=categories)
        graph = self.build_graph()

        artifacts_dir = self.settings.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        registry_stats_path = artifacts_dir / "registry_stats.json"
        graph_stats_path = artifacts_dir / "graph_stats.json"
        endpoints_path = artifacts_dir / "endpoints.jsonl"
        edges_path = artifacts_dir / "graph_edges.jsonl"
        registry_stats_md_path = artifacts_dir / "registry_stats.md"
        graph_stats_md_path = artifacts_dir / "graph_stats.md"
        endpoints_md_path = artifacts_dir / "endpoints.md"
        edges_md_path = artifacts_dir / "graph_edges.md"

        registry_stats = registry.stats().model_dump(mode="json")
        graph_stats = get_graph_stats(graph)
        endpoints = [
            endpoint.model_dump(mode="json")
            for endpoint in registry.all_endpoints()
        ]
        edges = [
            {"source": source, "target": target, **data}
            for source, target, data in graph.edges(data=True)
        ]

        registry_stats_path.write_text(
            json.dumps(registry_stats, indent=2),
            encoding="utf-8",
        )
        graph_stats_path.write_text(
            json.dumps(graph_stats, indent=2),
            encoding="utf-8",
        )
        with endpoints_path.open("w", encoding="utf-8") as f:
            for endpoint in endpoints:
                f.write(json.dumps(endpoint) + "\n")
        with edges_path.open("w", encoding="utf-8") as f:
            for edge in edges:
                f.write(json.dumps(edge, default=str) + "\n")

        registry_stats_md_path.write_text(
            _registry_stats_markdown(registry_stats),
            encoding="utf-8",
        )
        graph_stats_md_path.write_text(
            _graph_stats_markdown(graph_stats),
            encoding="utf-8",
        )
        endpoints_md_path.write_text(
            _endpoints_markdown(endpoints),
            encoding="utf-8",
        )
        edges_md_path.write_text(
            _graph_edges_markdown(edges),
            encoding="utf-8",
        )

        return {
            "registry_stats": registry_stats_path,
            "graph_stats": graph_stats_path,
            "endpoints": endpoints_path,
            "edges": edges_path,
            "registry_stats_diagram": registry_stats_md_path,
            "graph_stats_diagram": graph_stats_md_path,
            "endpoints_diagram": endpoints_md_path,
            "edges_diagram": edges_md_path,
        }

    def generate(
        self,
        num_conversations: int = 100,
        seed: int = 42,
        enable_steering: bool = True,
        enable_repair: bool = True,
        output_path: Path | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        live_call_callback: Callable[[int], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> list[Conversation]:
        """Run the full generation pipeline.

        Returns list of generated conversations and writes to JSONL.
        """
        if self._registry is None or self._graph is None:
            raise RuntimeError("Registry and graph must be built first.")

        # Initialize components
        offline = self.settings.use_offline_llm
        if self.settings.require_live_llm and offline:
            raise RuntimeError(
                "TOOLGEN_REQUIRE_LIVE_LLM=true requires a configured live LLM provider"
            )
        live_profile = self.settings.normalized_live_profile
        judge_models = self.settings.judge_models
        role_model_pools = self._role_model_pools()
        generation_model_label = _format_role_model_label(role_model_pools)
        rate_limiter = (
            None
            if offline
            else SharedRateLimiter(
                self.settings.llm_requests_per_minute,
                on_request_start=live_call_callback,
            )
        )
        role_clients, gen_client = self._build_generation_clients(
            role_model_pools,
            seed=seed,
            rate_limiter=rate_limiter,
        )
        judge_client = self._build_client(
            judge_models,
            seed=seed + 10_000,
            rate_limiter=rate_limiter,
        )
        executor = MockExecutor(seed=seed)
        steerer = DiversitySteerer(enabled=enable_steering)

        if output_path is None:
            output_path = self.settings.output_dir / f"conversations_seed{seed}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        parallelism = self.settings.normalized_max_parallel_conversations
        if parallelism > 1 and num_conversations > 1:
            return self._generate_parallel(
                num_conversations=num_conversations,
                seed=seed,
                enable_steering=enable_steering,
                enable_repair=enable_repair,
                output_path=output_path,
                progress_callback=progress_callback,
                offline=offline,
                live_profile=live_profile,
                role_model_pools=role_model_pools,
                judge_models=judge_models,
                generation_model_label=generation_model_label,
                rate_limiter=rate_limiter,
                steerer=steerer,
                parallelism=parallelism,
                status_callback=status_callback,
            )

        # Generate conversations
        conversations: list[Conversation] = []
        rng = random.Random(seed)
        multi_step_count = 0
        target_multi_step = int(num_conversations * 0.55)

        for i in range(num_conversations):
            progress_label = f"conv_{i:04d}"
            _emit_status(status_callback, f"{progress_label} | steering | selecting guidance")
            # Get steering guidance before sampling so it can influence the chain.
            steering = steerer.get_steering_guidance(
                available_domains=self._registry.list_categories()
            ) if enable_steering else None

            _emit_status(status_callback, f"{progress_label} | sampling | constrained tool chain")
            need_multi_step = multi_step_count < target_multi_step
            constraints = SamplerConstraints(
                min_steps=3 if need_multi_step else 1,
                max_steps=5 if need_multi_step else rng.choice([1, 2]),
                min_distinct_tools=2 if need_multi_step else 1,
                required_domains=(
                    [rng.choice(steering.prefer_domains)]
                    if steering and steering.prefer_domains
                    else None
                ),
            )

            chain = self._sample_chain_with_fallbacks(constraints, rng)
            if chain is None:
                logger.warning("Skipping conversation %d; sampler found no valid chain", i)
                _emit_status(status_callback, f"{progress_label} | skipped | no valid chain")
                if progress_callback:
                    progress_callback(i + 1, num_conversations, f"{progress_label} skipped")
                continue

            if chain.num_steps >= 3 and chain.num_distinct_tools >= 2:
                multi_step_count += 1

            logger.info(
                "Generating conversation %d/%d (chain: %s)",
                i + 1, num_conversations, chain.endpoint_ids,
            )
            _emit_status(
                status_callback,
                f"{progress_label} | chain ready | {_format_chain_label(chain)}",
            )

            # Generate
            if offline:
                conv = generate_offline_conversation(
                    chain=chain,
                    executor=executor,
                    conversation_index=i,
                    seed=seed,
                    max_turns=self.settings.max_turns_per_conversation,
                    steering=steering,
                    model_name=generation_model_label,
                    event_callback=status_callback,
                )
            else:
                conv = generate_conversation(
                    client=gen_client,
                    chain=chain,
                    executor=executor,
                    conversation_index=i,
                    seed=seed,
                    max_turns=self.settings.max_turns_per_conversation,
                    steering=steering,
                    model_name=generation_model_label,
                    strict_live=self.settings.require_live_llm,
                    live_profile=live_profile,
                    role_clients=role_clients,
                    event_callback=status_callback,
                )

            # Score
            _emit_status(status_callback, f"{progress_label} | judge | scoring conversation")
            scores = score_conversation(
                judge_client,
                conv,
                strict_live=self.settings.require_live_llm,
            )
            conv.judge_scores = scores

            # Repair if needed
            if enable_repair and not passes_quality_threshold(
                scores, self.settings.quality_threshold
            ):
                logger.info(
                    "Conv %d scored %.2f (below %.2f), attempting repair...",
                    i, scores.overall, self.settings.quality_threshold,
                )
                _emit_status(
                    status_callback,
                    f"{progress_label} | repair | score={scores.overall:.2f} below "
                    f"{self.settings.quality_threshold:.2f}",
                )
                conv, _ = repair_conversation(
                    conversation=conv,
                    scores=scores,
                    gen_client=gen_client,
                    judge_client=judge_client,
                    chain=chain,
                    executor=executor,
                    max_attempts=self.settings.max_repair_attempts,
                    quality_threshold=self.settings.quality_threshold,
                    seed=seed,
                    steering=steering,
                    strict_live=self.settings.require_live_llm,
                    live_profile=live_profile,
                    role_clients=role_clients,
                )

            # Record for steering
            steerer.record_generation(conv)
            conversations.append(conv)
            _emit_status(status_callback, f"{progress_label} | write | appending JSONL record")
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")
            if progress_callback:
                label = (
                    f"{conv.conversation_id} score={conv.judge_scores.overall:.2f}"
                    if conv.judge_scores
                    else conv.conversation_id
                )
                progress_callback(
                    i + 1,
                    num_conversations,
                    label,
                )

        # Write output
        with output_path.open("w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")

        logger.info(
            "Generated %d conversations → %s",
            len(conversations),
            output_path,
        )

        # Log summary stats
        self._log_generation_summary(conversations, steerer)

        return conversations

    def evaluate(
        self,
        input_path: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Conversation]:
        """Re-score conversations from an existing JSONL file."""
        if self.settings.require_live_llm and self.settings.use_offline_llm:
            raise RuntimeError(
                "TOOLGEN_REQUIRE_LIVE_LLM=true requires a configured live LLM provider"
            )
        judge_models = self.settings.judge_models
        judge_client = LLMClient(
            api_key=self.settings.gemini_api_key,
            api_keys=self.settings.api_keys,
            provider=self.settings.llm_provider,
            model=judge_models[0],
            model_pool=judge_models,
            requests_per_minute=self.settings.llm_requests_per_minute,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            max_output_tokens=self.settings.llm_max_output_tokens,
            seed=self.settings.default_seed + 10_000,
        )

        conversations: list[Conversation] = []
        total = _count_jsonl_records(input_path)
        with input_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                conv = Conversation(**data)
                scores = score_conversation(
                    judge_client,
                    conv,
                    strict_live=self.settings.require_live_llm,
                )
                conv.judge_scores = scores
                conversations.append(conv)
                if progress_callback:
                    progress_callback(
                        len(conversations),
                        total,
                        f"{conv.conversation_id} score={scores.overall:.2f}",
                    )

        return conversations

    def run_diversity_experiment(
        self,
        num_conversations: int = 50,
        seed: int = 42,
    ) -> ExperimentResults:
        """Run the diversity experiment: Run A (no steering) vs Run B (steering).

        Both runs use the same seed for reproducibility.
        """
        logger.info("=== Diversity Experiment: Run A (no steering) ===")
        output_a = self.settings.output_dir / f"run_a_seed{seed}.jsonl"
        convs_a = self.generate(
            num_conversations=num_conversations,
            seed=seed,
            enable_steering=False,
            output_path=output_a,
        )

        logger.info("=== Diversity Experiment: Run B (with steering) ===")
        output_b = self.settings.output_dir / f"run_b_seed{seed}.jsonl"
        convs_b = self.generate(
            num_conversations=num_conversations,
            seed=seed,
            enable_steering=True,
            output_path=output_b,
        )

        # Compute metrics
        metrics_a = compute_diversity_metrics(convs_a)
        metrics_b = compute_diversity_metrics(convs_b)

        results = ExperimentResults(
            run_a=metrics_a,
            run_b=metrics_b,
            seed=seed,
            num_conversations=num_conversations,
        )

        # Write analysis
        analysis_path = self.settings.output_dir / "diversity_analysis.json"
        with analysis_path.open("w", encoding="utf-8") as f:
            json.dump(results.model_dump(), f, indent=2)

        logger.info("Diversity experiment complete → %s", analysis_path)
        return results

    def _generate_parallel(
        self,
        num_conversations: int,
        seed: int,
        enable_steering: bool,
        enable_repair: bool,
        output_path: Path,
        progress_callback: Callable[[int, int, str], None] | None,
        offline: bool,
        live_profile: str,
        role_model_pools: dict[str, list[str]],
        judge_models: list[str],
        generation_model_label: str,
        rate_limiter: SharedRateLimiter | None,
        steerer: DiversitySteerer,
        parallelism: int,
        status_callback: Callable[[str], None] | None,
    ) -> list[Conversation]:
        """Generate conversations concurrently while preserving deterministic sampling.

        Sampling remains single-threaded so seed behavior and steering constraints stay
        easy to audit. Each worker gets fresh LLM clients and a fresh mock executor;
        all live clients share one rate limiter to respect the configured RPM.
        """
        tasks, skipped = self._prepare_generation_tasks(
            num_conversations=num_conversations,
            seed=seed,
            enable_steering=enable_steering,
            steerer=steerer,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        if not tasks:
            return []

        conversations_by_index: dict[int, Conversation] = {}
        completed = skipped
        max_workers = min(parallelism, len(tasks))
        logger.info("Generating with %d parallel workers", max_workers)
        _emit_status(status_callback, f"parallel | starting {max_workers} worker(s)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor_pool:
            futures = {
                executor_pool.submit(
                    self._generate_task,
                    task=task,
                    seed=seed,
                    offline=offline,
                    live_profile=live_profile,
                    enable_repair=enable_repair,
                    role_model_pools=role_model_pools,
                    judge_models=judge_models,
                    generation_model_label=generation_model_label,
                    rate_limiter=rate_limiter,
                    status_callback=status_callback,
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                conv = future.result()
                conversations_by_index[task.index] = conv
                steerer.record_generation(conv)
                completed += 1
                _emit_status(
                    status_callback,
                    f"{conv.conversation_id} | write | appending JSONL record",
                )
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")
                if progress_callback:
                    label = (
                        f"{conv.conversation_id} score={conv.judge_scores.overall:.2f}"
                        if conv.judge_scores
                        else conv.conversation_id
                    )
                    progress_callback(
                        completed,
                        num_conversations,
                        label,
                    )

        conversations = [
            conversations_by_index[index]
            for index in sorted(conversations_by_index)
        ]
        with output_path.open("w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")

        logger.info("Generated %d conversations → %s", len(conversations), output_path)
        self._log_generation_summary(conversations, steerer)
        return conversations

    def _prepare_generation_tasks(
        self,
        num_conversations: int,
        seed: int,
        enable_steering: bool,
        steerer: DiversitySteerer,
        progress_callback: Callable[[int, int, str], None] | None,
        status_callback: Callable[[str], None] | None,
    ) -> tuple[list[_GenerationTask], int]:
        if self._registry is None:
            raise RuntimeError("Registry must be loaded first.")

        tasks: list[_GenerationTask] = []
        rng = random.Random(seed)
        multi_step_count = 0
        skipped = 0
        target_multi_step = int(num_conversations * 0.55)

        for i in range(num_conversations):
            progress_label = f"conv_{i:04d}"
            _emit_status(status_callback, f"{progress_label} | sampling | preparing chain")
            steering = steerer.get_steering_guidance(
                available_domains=self._registry.list_categories()
            ) if enable_steering else None

            need_multi_step = multi_step_count < target_multi_step
            constraints = SamplerConstraints(
                min_steps=3 if need_multi_step else 1,
                max_steps=5 if need_multi_step else rng.choice([1, 2]),
                min_distinct_tools=2 if need_multi_step else 1,
                required_domains=(
                    [rng.choice(steering.prefer_domains)]
                    if steering and steering.prefer_domains
                    else None
                ),
            )

            chain = self._sample_chain_with_fallbacks(constraints, rng)
            if chain is None:
                logger.warning("Skipping conversation %d; sampler found no valid chain", i)
                skipped += 1
                _emit_status(status_callback, f"{progress_label} | skipped | no valid chain")
                if progress_callback:
                    progress_callback(skipped, num_conversations, f"{progress_label} skipped")
                continue

            if chain.num_steps >= 3 and chain.num_distinct_tools >= 2:
                multi_step_count += 1
            _emit_status(
                status_callback,
                f"{progress_label} | chain ready | {_format_chain_label(chain)}",
            )
            tasks.append(_GenerationTask(index=i, chain=chain, steering=steering))

        return tasks, skipped

    def _generate_task(
        self,
        task: _GenerationTask,
        seed: int,
        offline: bool,
        live_profile: str,
        enable_repair: bool,
        role_model_pools: dict[str, list[str]],
        judge_models: list[str],
        generation_model_label: str,
        rate_limiter: SharedRateLimiter | None,
        status_callback: Callable[[str], None] | None,
    ) -> Conversation:
        _emit_status(status_callback, f"conv_{task.index:04d} | worker | building LLM clients")
        role_clients, gen_client = self._build_generation_clients(
            role_model_pools,
            seed=seed + task.index,
            rate_limiter=rate_limiter,
        )
        judge_client = self._build_client(
            judge_models,
            seed=seed + task.index + 10_000,
            rate_limiter=rate_limiter,
        )
        executor = MockExecutor(seed=seed + task.index)

        if offline:
            _emit_status(status_callback, f"conv_{task.index:04d} | generate | offline state machine")
            conv = generate_offline_conversation(
                chain=task.chain,
                executor=executor,
                conversation_index=task.index,
                seed=seed,
                max_turns=self.settings.max_turns_per_conversation,
                steering=task.steering,
                model_name=generation_model_label,
                event_callback=status_callback,
            )
        else:
            _emit_status(status_callback, f"conv_{task.index:04d} | generate | live state machine")
            conv = generate_conversation(
                client=gen_client,
                chain=task.chain,
                executor=executor,
                conversation_index=task.index,
                seed=seed,
                max_turns=self.settings.max_turns_per_conversation,
                steering=task.steering,
                model_name=generation_model_label,
                strict_live=self.settings.require_live_llm,
                live_profile=live_profile,
                role_clients=role_clients,
                event_callback=status_callback,
            )

        _emit_status(status_callback, f"conv_{task.index:04d} | judge | scoring conversation")
        scores = score_conversation(
            judge_client,
            conv,
            strict_live=self.settings.require_live_llm,
        )
        conv.judge_scores = scores

        if enable_repair and not passes_quality_threshold(
            scores, self.settings.quality_threshold
        ):
            _emit_status(
                status_callback,
                f"conv_{task.index:04d} | repair | score={scores.overall:.2f} below "
                f"{self.settings.quality_threshold:.2f}",
            )
            conv, _ = repair_conversation(
                conversation=conv,
                scores=scores,
                gen_client=gen_client,
                judge_client=judge_client,
                chain=task.chain,
                executor=executor,
                max_attempts=self.settings.max_repair_attempts,
                quality_threshold=self.settings.quality_threshold,
                seed=seed,
                steering=task.steering,
                strict_live=self.settings.require_live_llm,
                live_profile=live_profile,
                role_clients=role_clients,
            )

        return conv

    def _build_generation_clients(
        self,
        role_model_pools: dict[str, list[str]],
        seed: int,
        rate_limiter: SharedRateLimiter | None,
    ) -> tuple[dict[str, LLMClient], LLMClient]:
        offsets = {
            "planner": 101,
            "assistant": 202,
            "user": 303,
            "summary": 404,
        }
        clients = {
            role: self._build_client(
                models,
                seed=seed + offsets.get(role, 0),
                rate_limiter=rate_limiter,
            )
            for role, models in role_model_pools.items()
        }
        return clients, clients.get("assistant") or next(iter(clients.values()))

    def _build_client(
        self,
        models: list[str],
        seed: int,
        rate_limiter: SharedRateLimiter | None,
    ) -> LLMClient:
        model_pool = models or ["offline-deterministic"]
        return LLMClient(
            api_key=self.settings.gemini_api_key,
            api_keys=self.settings.api_keys,
            provider=self.settings.llm_provider,
            model=model_pool[0],
            model_pool=model_pool,
            requests_per_minute=self.settings.llm_requests_per_minute,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            max_output_tokens=self.settings.llm_max_output_tokens,
            seed=seed,
            rate_limiter=rate_limiter,
        )

    def _role_model_pools(self) -> dict[str, list[str]]:
        return {
            "planner": self.settings.planner_models,
            "assistant": self.settings.assistant_models,
            "user": self.settings.user_models,
            "summary": self.settings.summary_models,
        }

    def _log_generation_summary(
        self,
        conversations: list[Conversation],
        steerer: DiversitySteerer,
    ) -> None:
        """Log summary statistics for the generation run."""
        total = len(conversations)
        if total == 0:
            return

        multi_step = sum(
            1 for c in conversations
            if c.metadata.num_tool_calls >= 3 and c.metadata.num_distinct_tools >= 2
        )
        avg_score = sum(
            c.judge_scores.overall for c in conversations
            if c.judge_scores
        ) / max(1, sum(1 for c in conversations if c.judge_scores))
        repaired = sum(1 for c in conversations if c.metadata.repair_attempts > 0)

        logger.info(
            "Summary: %d conversations, %.1f%% multi-step+multi-tool, "
            "avg score %.2f, %d repaired",
            total,
            multi_step / total * 100,
            avg_score,
            repaired,
        )

        if steerer.enabled:
            logger.info("Steering summary: %s", steerer.get_summary())

    def _sample_chain_with_fallbacks(
        self,
        constraints: SamplerConstraints,
        rng: random.Random,
    ) -> ToolChain | None:
        """Sample with progressively relaxed constraints.

        Full ToolBench slices can be sparse after category filtering. This keeps
        generation productive while still trying the assignment constraints first.
        """
        if self._registry is None or self._graph is None:
            raise RuntimeError("Registry and graph must be built first.")

        candidates = [constraints]
        if constraints.required_domains:
            candidates.append(constraints.model_copy(update={"required_domains": None}))
        candidates.append(
            constraints.model_copy(
                update={
                    "min_steps": 1,
                    "min_distinct_tools": 1,
                    "required_domains": constraints.required_domains,
                }
            )
        )
        if constraints.required_domains:
            candidates.append(
                constraints.model_copy(
                    update={
                        "min_steps": 1,
                        "min_distinct_tools": 1,
                        "required_domains": None,
                    }
                )
            )

        for candidate in candidates:
            chain = sample_chain(
                self._graph,
                self._registry,
                constraints=candidate,
                rng=rng,
                max_attempts=100,
            )
            if chain is not None:
                return chain
        return None


def _format_model_label(models: list[str]) -> str:
    """Compact metadata label for single-model and randomized runs."""
    if len(models) == 1:
        return models[0]
    return f"random[{', '.join(models)}]"


def _format_role_model_label(role_model_pools: dict[str, list[str]]) -> str:
    """Compact metadata label for role-routed model configurations."""
    labels = {role: _format_model_label(models) for role, models in role_model_pools.items()}
    unique_labels = set(labels.values())
    if len(unique_labels) == 1:
        return next(iter(unique_labels))
    return "; ".join(f"{role}={label}" for role, label in labels.items())


def _emit_status(callback: Callable[[str], None] | None, label: str) -> None:
    if callback:
        callback(label)


def _format_chain_label(chain: ToolChain) -> str:
    endpoints = " -> ".join(chain.endpoint_ids)
    if len(endpoints) > 140:
        endpoints = endpoints[:139].rstrip() + "…"
    return (
        f"{chain.num_steps} step(s), {chain.num_distinct_tools} tool(s), "
        f"pattern={chain.pattern.value}, path={endpoints}"
    )


def _registry_stats_markdown(stats: dict[str, Any]) -> str:
    categories = _sorted_count_items(stats.get("categories", {}))
    flowchart = [
        "flowchart TB",
        (
            f'  R["Tool Registry<br/>{stats.get("total_tools", 0)} tools<br/>'
            f'{stats.get("total_endpoints", 0)} endpoints<br/>'
            f'{stats.get("total_categories", 0)} categories"]'
        ),
        (
            f'  R --> S["Response schemas<br/>'
            f'{stats.get("endpoints_with_response_schema", 0)} endpoints"]'
        ),
        f'  R --> P["Avg params/endpoint<br/>{stats.get("avg_params_per_endpoint", 0)}"]',
    ]
    for index, (category, count) in enumerate(categories):
        flowchart.append(
            f'  R --> C{index}["{_mermaid_label(category)}<br/>{count} endpoints"]'
        )

    return "\n".join(
        [
            "# Registry Stats Diagram",
            "",
            "Source artifact: `registry_stats.json`",
            "",
            "```mermaid",
            *flowchart,
            "```",
            "",
            *(_mermaid_pie("Endpoints by Category", categories)),
            "",
        ]
    )


def _graph_stats_markdown(stats: dict[str, Any]) -> str:
    edge_types = _sorted_count_items(stats.get("edge_type_counts", {}))
    flowchart = [
        "flowchart TB",
        (
            f'  G["Tool Graph<br/>{stats.get("num_nodes", 0)} nodes<br/>'
            f'{stats.get("num_edges", 0)} edges"]'
        ),
        f'  G --> C["Connected components<br/>{stats.get("num_connected_components", 0)}"]',
        f'  G --> L["Largest component<br/>{stats.get("largest_component_size", 0)} nodes"]',
        f'  G --> D["Average degree<br/>{stats.get("avg_degree", 0)}"]',
        '  G --> E["Edge types"]',
    ]
    for index, (edge_type, count) in enumerate(edge_types):
        flowchart.append(
            f'  E --> ET{index}["{_mermaid_label(edge_type)}<br/>{count} edges"]'
        )

    return "\n".join(
        [
            "# Graph Stats Diagram",
            "",
            "Source artifact: `graph_stats.json`",
            "",
            "```mermaid",
            *flowchart,
            "```",
            "",
            *(_mermaid_pie("Edges by Type", edge_types)),
            "",
        ]
    )


def _endpoints_markdown(endpoints: list[dict[str, Any]]) -> str:
    sorted_endpoints = sorted(
        endpoints,
        key=lambda row: (
            str(row.get("category", "")),
            str(row.get("tool_name", "")),
            str(row.get("endpoint_name", "")),
        ),
    )
    displayed = sorted_endpoints[:_MAX_ENDPOINT_DIAGRAM_NODES]
    omitted = max(0, len(sorted_endpoints) - len(displayed))

    flowchart = ["flowchart TB", f'  R["Endpoint Registry<br/>{len(endpoints)} endpoints"]']
    category_ids: dict[str, str] = {}
    tool_ids: dict[tuple[str, str], str] = {}

    for endpoint in displayed:
        category = str(endpoint.get("category") or "Uncategorized")
        tool_name = str(endpoint.get("tool_name") or "unknown_tool")
        endpoint_name = str(endpoint.get("endpoint_name") or endpoint.get("endpoint_id") or "endpoint")

        if category not in category_ids:
            category_id = f"C{len(category_ids)}"
            category_ids[category] = category_id
            category_count = sum(
                1 for row in endpoints if str(row.get("category") or "Uncategorized") == category
            )
            flowchart.append(
                f'  R --> {category_id}["{_mermaid_label(category)}<br/>'
                f'{category_count} endpoints"]'
            )

        tool_key = (category, tool_name)
        if tool_key not in tool_ids:
            tool_id = f"T{len(tool_ids)}"
            tool_ids[tool_key] = tool_id
            flowchart.append(
                f'  {category_ids[category]} --> {tool_id}["{_mermaid_label(tool_name)}"]'
            )

        endpoint_id = f"E{len(flowchart)}"
        parameters = endpoint.get("parameters") or []
        required_count = sum(1 for parameter in parameters if parameter.get("required"))
        label = (
            f"{endpoint_name}<br/>{endpoint.get('method', 'GET')} | "
            f"{len(parameters)} args | {required_count} required"
        )
        flowchart.append(f'  {tool_ids[tool_key]} --> {endpoint_id}["{_mermaid_label(label)}"]')

    note = (
        f"\nDiagram capped at {_MAX_ENDPOINT_DIAGRAM_NODES} endpoints; "
        f"{omitted} omitted for readability.\n"
        if omitted
        else ""
    )

    return "\n".join(
        [
            "# Endpoints Diagram",
            "",
            "Source artifact: `endpoints.jsonl`",
            note,
            "```mermaid",
            *flowchart,
            "```",
            "",
        ]
    )


def _graph_edges_markdown(edges: list[dict[str, Any]]) -> str:
    edge_priority = {
        "io_chain": 0,
        "complementary": 1,
        "same_tool": 2,
        "same_category": 3,
    }
    sorted_edges = sorted(
        edges,
        key=lambda row: (
            edge_priority.get(str(row.get("edge_type", "")), 99),
            str(row.get("source", "")),
            str(row.get("target", "")),
        ),
    )
    displayed = sorted_edges[:_MAX_EDGE_DIAGRAM_EDGES]
    omitted = max(0, len(sorted_edges) - len(displayed))

    node_ids: dict[str, str] = {}
    flowchart = ["flowchart LR"]
    for edge in displayed:
        source = str(edge.get("source", "unknown_source"))
        target = str(edge.get("target", "unknown_target"))
        source_id = node_ids.setdefault(source, f"N{len(node_ids)}")
        target_id = node_ids.setdefault(target, f"N{len(node_ids)}")
        flowchart.append(f'  {source_id}["{_mermaid_label(source)}"]')
        flowchart.append(f'  {target_id}["{_mermaid_label(target)}"]')
        edge_type = str(edge.get("edge_type", "edge"))
        weight = edge.get("weight", "")
        label = f"{edge_type} {weight}".strip()
        flowchart.append(f"  {source_id} -->|{_mermaid_label(label)}| {target_id}")

    edge_counts = Counter(str(edge.get("edge_type", "unknown")) for edge in edges)
    note = (
        f"\nDiagram capped at {_MAX_EDGE_DIAGRAM_EDGES} edges; "
        f"{omitted} omitted for readability.\n"
        if omitted
        else ""
    )

    return "\n".join(
        [
            "# Graph Edges Diagram",
            "",
            "Source artifact: `graph_edges.jsonl`",
            note,
            "```mermaid",
            *flowchart,
            "```",
            "",
            *(_mermaid_pie("Edges by Type", _sorted_count_items(edge_counts))),
            "",
        ]
    )


def _mermaid_pie(title: str, items: list[tuple[str, int]]) -> list[str]:
    if not items:
        return []
    lines = ["```mermaid", "pie showData", f"  title {_mermaid_label(title)}"]
    for label, value in items:
        lines.append(f'  "{_mermaid_label(label)}" : {int(value)}')
    lines.append("```")
    return lines


def _sorted_count_items(counts: dict[str, Any] | Counter) -> list[tuple[str, int]]:
    return sorted(
        ((str(key), int(value)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _mermaid_label(value: Any, max_chars: int = 80) -> str:
    text = " ".join(str(value).replace("\n", " ").split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return (
        text.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
    )


def _count_jsonl_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def compute_diversity_metrics(conversations: list[Conversation]) -> DiversityMetrics:
    """Compute diversity metrics for a set of conversations.

    Metrics:
    1. Tool combination entropy: Shannon entropy over tool-pair co-occurrences
    2. Domain coverage CV: Coefficient of variation of conversations per domain
    """
    if not conversations:
        return DiversityMetrics()

    # Tool pair counts
    pair_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    pattern_counts: Counter = Counter()
    all_domains: set[str] = set()

    for conv in conversations:
        tools = sorted(conv.metadata.tools_used)
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                pair_counts[(tools[i], tools[j])] += 1

        for domain in conv.metadata.category_domains:
            domain_counts[domain] += 1
            all_domains.add(domain)

        pattern_counts[conv.metadata.pattern] += 1

    # 1. Tool combination entropy
    total_pairs = sum(pair_counts.values())
    entropy = 0.0
    if total_pairs > 0:
        for count in pair_counts.values():
            p = count / total_pairs
            if p > 0:
                entropy -= p * math.log2(p)

    # 2. Domain coverage CV
    domain_values = list(domain_counts.values())
    if domain_values:
        mean_domain = sum(domain_values) / len(domain_values)
        if mean_domain > 0:
            variance = sum((v - mean_domain) ** 2 for v in domain_values) / len(domain_values)
            std_dev = math.sqrt(variance)
            domain_cv = std_dev / mean_domain
        else:
            domain_cv = 0.0
    else:
        domain_cv = 0.0

    # Quality scores
    scores = {}
    total_score = 0.0
    score_count = 0
    for conv in conversations:
        if conv.judge_scores:
            total_score += conv.judge_scores.overall
            score_count += 1
    mean_quality = total_score / score_count if score_count > 0 else 0.0

    if score_count > 0:
        scores = {
            "tool_correctness": sum(
                c.judge_scores.tool_correctness.score
                for c in conversations if c.judge_scores
            ) / score_count,
            "naturalness": sum(
                c.judge_scores.naturalness.score
                for c in conversations if c.judge_scores
            ) / score_count,
            "task_completion": sum(
                c.judge_scores.task_completion.score
                for c in conversations if c.judge_scores
            ) / score_count,
        }

    return DiversityMetrics(
        tool_combination_entropy=round(entropy, 4),
        domain_coverage_cv=round(domain_cv, 4),
        unique_tool_pairs=len(pair_counts),
        unique_domains_used=len(all_domains),
        total_domains_available=len(all_domains),
        pattern_distribution=dict(pattern_counts),
        mean_quality_score=round(mean_quality, 2),
        quality_scores={k: round(v, 2) for k, v in scores.items()},
    )
