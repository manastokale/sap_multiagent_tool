"""Cross-conversation diversity steering via mem0.

Uses mem0's vector memory to track what has been generated and steer
future generations toward diverse, balanced coverage. This avoids
repetitive tool combinations, domains, and patterns.

CLI flag: --no-cross-conversation-steering disables this entirely (for Run A).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from toolgen.models import (
    Conversation,
    ConversationMetadata,
    SteeringGuidance,
)

logger = logging.getLogger(__name__)


class DiversitySteerer:
    """Cross-conversation steering using mem0 for vector-backed memory.

    When enabled, stores summaries of generated conversations and queries
    them to produce diversity guidance for the planner.
    When disabled (Run A), returns empty guidance.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._memory = None
        self._generation_log: list[ConversationMetadata] = []
        self._tool_pair_counts: Counter = Counter()
        self._domain_counts: Counter = Counter()
        self._pattern_counts: Counter = Counter()

        if enabled:
            try:
                from mem0 import Memory
                self._memory = Memory()
                logger.info("mem0 memory initialized for cross-conversation steering")
            except Exception as e:
                logger.info(
                    "mem0 unavailable: %s. Falling back to counter-based steering.",
                    e,
                )
                self._memory = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_generation(self, conversation: Conversation) -> None:
        """Store what was just generated for future steering."""
        if not self._enabled:
            return

        meta = conversation.metadata
        self._generation_log.append(meta)

        # Update counters
        tools = sorted(meta.tools_used)
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                self._tool_pair_counts[(tools[i], tools[j])] += 1

        for domain in meta.category_domains:
            self._domain_counts[domain] += 1

        self._pattern_counts[meta.pattern] += 1

        # Store in mem0 if available
        if self._memory:
            try:
                summary = (
                    f"Generated conversation using tools: {', '.join(meta.tools_used)}. "
                    f"Domains: {', '.join(meta.category_domains)}. "
                    f"Pattern: {meta.pattern}. "
                    f"Quality score: {conversation.judge_scores.overall if conversation.judge_scores else 'N/A'}."
                )
                self._memory.add(
                    summary,
                    user_id="toolgen_steering",
                    metadata={
                        "conversation_id": conversation.conversation_id,
                        "tools": meta.tools_used,
                        "domains": meta.category_domains,
                        "pattern": meta.pattern,
                    },
                )
            except Exception as e:
                logger.debug("mem0 store failed (non-critical): %s", e)

    def get_steering_guidance(
        self,
        available_domains: list[str] | None = None,
    ) -> SteeringGuidance:
        """Query past generations and return guidance for diversity."""
        if not self._enabled:
            return SteeringGuidance()

        # Find over-represented tool pairs
        avoid_combos: list[list[str]] = []
        if self._tool_pair_counts:
            max_count = max(self._tool_pair_counts.values())
            threshold = max(2, max_count * 0.7)  # Avoid pairs used more than 70% of max
            for pair, count in self._tool_pair_counts.most_common(5):
                if count >= threshold:
                    avoid_combos.append(list(pair))

        # Find under-represented domains
        prefer_domains: list[str] = []
        if available_domains and self._domain_counts:
            avg_count = sum(self._domain_counts.values()) / len(available_domains) if available_domains else 0
            for domain in available_domains:
                if self._domain_counts.get(domain, 0) < avg_count * 0.5:
                    prefer_domains.append(domain)
            # If all domains are under-represented, pick the least used
            if not prefer_domains:
                prefer_domains = [
                    d for d, _ in sorted(
                        [(d, self._domain_counts.get(d, 0)) for d in available_domains],
                        key=lambda x: x[1],
                    )[:3]
                ]

        # Complexity suggestion based on pattern distribution
        complexity_suggestion = ""
        if self._pattern_counts:
            total = sum(self._pattern_counts.values())
            multi_step_ratio = (
                self._pattern_counts.get("multi_step", 0) +
                self._pattern_counts.get("search_and_act", 0)
            ) / total
            if multi_step_ratio < 0.5:
                complexity_suggestion = "Prefer multi-step scenarios"
            elif multi_step_ratio > 0.7:
                complexity_suggestion = "Allow simpler single-step scenarios"

        # Build rationale from mem0 if available
        rationale = ""
        if self._memory:
            try:
                results = self._memory.search(
                    "What conversations have been generated so far?",
                    user_id="toolgen_steering",
                    limit=5,
                )
                if results and hasattr(results, '__iter__'):
                    recent = []
                    for r in results:
                        if hasattr(r, 'get'):
                            recent.append(r.get("memory", str(r)))
                        else:
                            recent.append(str(r))
                    rationale = f"Recent generations: {'; '.join(recent[:3])}"
            except Exception as e:
                logger.debug("mem0 search failed (non-critical): %s", e)

        if not rationale:
            total = len(self._generation_log)
            top_domains = self._domain_counts.most_common(3)
            rationale = (
                f"{total} conversations generated so far. "
                f"Most used domains: {', '.join(f'{d}({c})' for d, c in top_domains)}"
            )

        return SteeringGuidance(
            avoid_tool_combinations=avoid_combos,
            prefer_domains=prefer_domains,
            complexity_suggestion=complexity_suggestion,
            rationale=rationale,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of steering state."""
        return {
            "enabled": self._enabled,
            "total_conversations": len(self._generation_log),
            "domain_distribution": dict(self._domain_counts),
            "pattern_distribution": dict(self._pattern_counts),
            "top_tool_pairs": [
                {"pair": list(pair), "count": count}
                for pair, count in self._tool_pair_counts.most_common(10)
            ],
        }
