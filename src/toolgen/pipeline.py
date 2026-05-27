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
from pathlib import Path
from typing import Any

from toolgen.agents.llm_client import LLMClient
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

        registry_stats_path.write_text(
            json.dumps(registry.stats().model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        graph_stats_path.write_text(
            json.dumps(get_graph_stats(graph), indent=2),
            encoding="utf-8",
        )
        with endpoints_path.open("w", encoding="utf-8") as f:
            for endpoint in registry.all_endpoints():
                f.write(json.dumps(endpoint.model_dump(mode="json")) + "\n")
        with edges_path.open("w", encoding="utf-8") as f:
            for source, target, data in graph.edges(data=True):
                row = {"source": source, "target": target, **data}
                f.write(json.dumps(row, default=str) + "\n")

        return {
            "registry_stats": registry_stats_path,
            "graph_stats": graph_stats_path,
            "endpoints": endpoints_path,
            "edges": edges_path,
        }

    def generate(
        self,
        num_conversations: int = 100,
        seed: int = 42,
        enable_steering: bool = True,
        enable_repair: bool = True,
        output_path: Path | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
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
        generation_models = self.settings.generation_models
        judge_models = self.settings.judge_models
        generation_model_label = _format_model_label(generation_models)
        gen_client = LLMClient(
            api_key=self.settings.gemini_api_key,
            groq_api_key=self.settings.groq_api_key,
            api_keys=self.settings.api_keys,
            provider=self.settings.llm_provider,
            model=generation_models[0],
            model_pool=generation_models,
            requests_per_minute=self.settings.llm_requests_per_minute,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            seed=seed,
        )
        judge_client = LLMClient(
            api_key=self.settings.gemini_api_key,
            groq_api_key=self.settings.groq_api_key,
            api_keys=self.settings.api_keys,
            provider=self.settings.llm_provider,
            model=judge_models[0],
            model_pool=judge_models,
            requests_per_minute=self.settings.llm_requests_per_minute,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            seed=seed + 10_000,
        )
        executor = MockExecutor(seed=seed)
        steerer = DiversitySteerer(enabled=enable_steering)

        if output_path is None:
            output_path = self.settings.output_dir / f"conversations_seed{seed}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        # Generate conversations
        conversations: list[Conversation] = []
        rng = random.Random(seed)
        multi_step_count = 0
        target_multi_step = int(num_conversations * 0.55)

        for i in range(num_conversations):
            progress_label = f"conv_{i:04d}"
            # Get steering guidance before sampling so it can influence the chain.
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
                if progress_callback:
                    progress_callback(i + 1, num_conversations, f"{progress_label} skipped")
                continue

            if chain.num_steps >= 3 and chain.num_distinct_tools >= 2:
                multi_step_count += 1

            logger.info(
                "Generating conversation %d/%d (chain: %s)",
                i + 1, num_conversations, chain.endpoint_ids,
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
                )

            # Score
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
                )

            # Record for steering
            steerer.record_generation(conv)
            conversations.append(conv)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(conv.to_output_dict(), default=str) + "\n")
            if progress_callback:
                progress_callback(
                    i + 1,
                    num_conversations,
                    f"{conv.conversation_id} score={conv.judge_scores.overall:.2f}"
                    if conv.judge_scores
                    else conv.conversation_id,
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
            groq_api_key=self.settings.groq_api_key,
            api_keys=self.settings.api_keys,
            provider=self.settings.llm_provider,
            model=judge_models[0],
            model_pool=judge_models,
            requests_per_minute=self.settings.llm_requests_per_minute,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
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
