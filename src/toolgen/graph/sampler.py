"""Constrained tool-chain sampler.

Samples realistic tool chains from the tool graph using weighted random walks
with constraint satisfaction. This is the core mechanism that ensures the
generator uses graph structure (not hardcoded lists) to produce tool sequences.

Sampling strategies:
  1. Weighted random walk: follow edges weighted by type, prefer io_chain
  2. Pattern-driven: start from a pattern template and fill with matching endpoints
  3. Constraint satisfaction: retry walks until constraints are met
"""

from __future__ import annotations

import logging
import random

import networkx as nx

from toolgen.models import (
    APIEndpoint,
    ChainPattern,
    EdgeType,
    SamplerConstraints,
    ToolChain,
)
from toolgen.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Edge type preferences for weighted selection
_EDGE_TYPE_MULTIPLIERS = {
    EdgeType.IO_CHAIN.value: 3.0,       # Strongly prefer IO chains
    EdgeType.COMPLEMENTARY.value: 2.0,  # Good for realistic sequences
    EdgeType.SAME_TOOL.value: 1.0,      # Decent fallback
    EdgeType.SAME_CATEGORY.value: 0.5,  # Weakest preference
}


def _select_weighted_neighbor(
    G: nx.DiGraph,
    node: str,
    visited: set[str],
    rng: random.Random,
    exclude: set[str] | None = None,
) -> str | None:
    """Select a neighbor using edge-type-weighted random selection."""
    exclude = exclude or set()
    candidates: list[tuple[str, float]] = []

    for _, neighbor, data in G.edges(node, data=True):
        if neighbor in visited or neighbor in exclude:
            continue
        edge_type = data.get("edge_type", EdgeType.SAME_CATEGORY.value)
        weight = data.get("weight", 0.5) * _EDGE_TYPE_MULTIPLIERS.get(edge_type, 1.0)
        candidates.append((neighbor, weight))

    if not candidates:
        return None

    # Weighted random selection
    total_weight = sum(w for _, w in candidates)
    if total_weight == 0:
        return rng.choice([c for c, _ in candidates])

    r = rng.uniform(0, total_weight)
    cumulative = 0.0
    for neighbor, weight in candidates:
        cumulative += weight
        if r <= cumulative:
            return neighbor

    return candidates[-1][0]


def _classify_chain_pattern(chain: list[APIEndpoint]) -> ChainPattern:
    """Classify a chain into a pattern type based on its endpoint characteristics."""
    if len(chain) == 1:
        return ChainPattern.SINGLE_STEP

    endpoint_names = [ep.endpoint_name.lower() for ep in chain]

    # Check for CRUD cycle
    has_create = any("create" in n or "add" in n or "book" in n for n in endpoint_names)
    has_read = any("get" in n or "list" in n or "search" in n for n in endpoint_names)
    has_update = any("update" in n or "modify" in n for n in endpoint_names)
    has_delete = any("delete" in n or "remove" in n or "cancel" in n for n in endpoint_names)

    crud_count = sum([has_create, has_read, has_update, has_delete])
    if crud_count >= 3:
        return ChainPattern.CRUD_CYCLE

    # Check for search-and-act
    if has_read and (has_create or has_update or has_delete):
        return ChainPattern.SEARCH_AND_ACT

    return ChainPattern.MULTI_STEP


def _check_constraints(
    chain: ToolChain,
    constraints: SamplerConstraints,
) -> tuple[bool, list[str]]:
    """Check if a chain satisfies all constraints. Returns (satisfied, reasons)."""
    satisfied: list[str] = []
    violations: list[str] = []

    # Step count
    if constraints.min_steps <= chain.num_steps <= constraints.max_steps:
        satisfied.append(f"steps={chain.num_steps}")
    else:
        violations.append(
            f"steps={chain.num_steps} not in [{constraints.min_steps}, {constraints.max_steps}]"
        )

    # Distinct tools
    if chain.num_distinct_tools >= constraints.min_distinct_tools:
        satisfied.append(f"distinct_tools={chain.num_distinct_tools}")
    else:
        violations.append(
            f"distinct_tools={chain.num_distinct_tools} < {constraints.min_distinct_tools}"
        )

    # Required domains
    if constraints.required_domains:
        chain_cats = chain.categories
        for domain in constraints.required_domains:
            if domain in chain_cats:
                satisfied.append(f"domain={domain}")
            else:
                violations.append(f"missing domain={domain}")

    # Excluded endpoints
    if constraints.exclude_endpoints:
        chain_ids = set(chain.endpoint_ids)
        excluded_used = chain_ids & constraints.exclude_endpoints
        if excluded_used:
            violations.append(f"used excluded endpoints: {excluded_used}")
        else:
            satisfied.append("no excluded endpoints")

    # Required patterns
    if constraints.required_patterns:
        if chain.pattern in constraints.required_patterns:
            satisfied.append(f"pattern={chain.pattern.value}")
        else:
            required = [p.value for p in constraints.required_patterns]
            violations.append(f"pattern={chain.pattern.value} not in {required}")

    is_ok = len(violations) == 0
    return is_ok, satisfied if is_ok else violations


def sample_chain(
    G: nx.DiGraph,
    registry: ToolRegistry,
    constraints: SamplerConstraints | None = None,
    rng: random.Random | None = None,
    max_attempts: int = 50,
) -> ToolChain | None:
    """Sample a single tool chain from the graph.

    Uses weighted random walks with constraint satisfaction.
    Returns None if no valid chain can be found after max_attempts.
    """
    if rng is None:
        rng = random.Random()
    if constraints is None:
        constraints = SamplerConstraints()

    nodes = list(G.nodes())
    if not nodes:
        return None

    for attempt in range(max_attempts):
        # Pick start node
        if constraints.required_domains:
            # Start from a node in a required domain
            domain = rng.choice(constraints.required_domains)
            domain_nodes = [
                n for n in nodes
                if G.nodes[n].get("category") == domain
                and n not in constraints.exclude_endpoints
            ]
            if not domain_nodes:
                # Fallback to any node
                domain_nodes = [n for n in nodes if n not in constraints.exclude_endpoints]
            if not domain_nodes:
                continue
            start = rng.choice(domain_nodes)
        else:
            eligible = [n for n in nodes if n not in constraints.exclude_endpoints]
            if not eligible:
                return None
            start = rng.choice(eligible)

        # Random walk
        path = [start]
        visited = {start}
        target_len = rng.randint(constraints.min_steps, constraints.max_steps)

        for _ in range(target_len - 1):
            next_node = _select_weighted_neighbor(
                G, path[-1], visited, rng, constraints.exclude_endpoints
            )
            if next_node is None:
                break
            path.append(next_node)
            visited.add(next_node)

        # Resolve to endpoints
        endpoints: list[APIEndpoint] = []
        for node_id in path:
            ep = registry.get_endpoint(node_id)
            if ep:
                endpoints.append(ep)

        if not endpoints:
            continue

        pattern = _classify_chain_pattern(endpoints)
        chain = ToolChain(
            endpoints=endpoints,
            pattern=pattern,
        )

        ok, reasons = _check_constraints(chain, constraints)
        if ok:
            chain.constraints_satisfied = reasons
            return chain

    logger.warning(
        "Failed to sample chain after %d attempts with constraints: %s",
        max_attempts,
        constraints.model_dump(),
    )
    return None


def sample_chains(
    G: nx.DiGraph,
    registry: ToolRegistry,
    num_chains: int,
    constraints: SamplerConstraints | None = None,
    seed: int = 42,
    target_multi_step_ratio: float = 0.55,
) -> list[ToolChain]:
    """Sample multiple chains with target distribution.

    The PDF requires 50-60% of conversations to have multi-step (≥3 tool calls)
    AND multi-tool (≥2 distinct tools) traces. This function steers sampling
    toward that distribution.
    """
    rng = random.Random(seed)
    chains: list[ToolChain] = []
    target_multi_step = int(num_chains * target_multi_step_ratio)
    multi_step_count = 0

    for i in range(num_chains):
        # Steer toward the required 50-60% multi-step + multi-tool slice.
        need_multi_step = multi_step_count < target_multi_step
        c = constraints or SamplerConstraints()
        if need_multi_step:
            # Force multi-step chain
            c = c.model_copy(update={
                "min_steps": max(c.min_steps, 3),
                "max_steps": max(c.max_steps, 5),
                "min_distinct_tools": max(c.min_distinct_tools, 2),
            })
        else:
            # Allow simpler chains for variety
            c = c.model_copy(update={
                "min_steps": 1,
                "max_steps": rng.choice([1, 2, 3, 4, 5]),
                "min_distinct_tools": min(c.min_distinct_tools, 1),
            })

        chain = sample_chain(G, registry, c, rng)
        if chain:
            chains.append(chain)
            if chain.num_steps >= 3 and chain.num_distinct_tools >= 2:
                multi_step_count += 1

    actual_ratio = multi_step_count / len(chains) if chains else 0
    logger.info(
        "Sampled %d chains. Multi-step+multi-tool ratio: %.1f%% (target: %.1f%%)",
        len(chains),
        actual_ratio * 100,
        target_multi_step_ratio * 100,
    )
    return chains
