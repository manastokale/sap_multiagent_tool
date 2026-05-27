"""Tool Graph construction.

Builds a directed graph where nodes are API endpoints and edges represent
relationships useful for sampling realistic tool chains.

Edge types and weights:
  - io_chain (1.0): Output of A can feed as input to B
  - same_tool (0.3): Both belong to the same tool
  - same_category (0.5): Both belong to the same ToolBench category
  - complementary (0.8): Semantically related (search→book, create→delete)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx

from toolgen.models import APIEndpoint, EdgeType
from toolgen.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Verb groups for complementary detection
_VERB_PATTERNS: dict[str, re.Pattern] = {
    "search": re.compile(r"^(search|find|list|query|browse|lookup|get_all|fetch_all)", re.I),
    "get": re.compile(r"^(get|retrieve|fetch|show|view|read|detail)", re.I),
    "create": re.compile(r"^(create|add|new|register|post|submit|book|reserve)", re.I),
    "update": re.compile(r"^(update|modify|edit|change|patch|set)", re.I),
    "delete": re.compile(r"^(delete|remove|cancel|revoke|unregister)", re.I),
}

# Expected verb sequences for complementary edges
_COMPLEMENTARY_SEQUENCES = [
    ("search", "get"),
    ("search", "create"),
    ("search", "book"),
    ("get", "update"),
    ("get", "delete"),
    ("create", "get"),
    ("create", "update"),
    ("create", "delete"),
    ("list", "get"),
]

_EDGE_PRIORITY = {
    EdgeType.IO_CHAIN.value: 4,
    EdgeType.COMPLEMENTARY.value: 3,
    EdgeType.SAME_CATEGORY.value: 2,
    EdgeType.SAME_TOOL.value: 1,
}

_FULL_MESH_CATEGORY_LIMIT = 80
_MAX_SAME_CATEGORY_NEIGHBORS = 30


def _add_typed_edge(
    G: nx.DiGraph,
    source: str,
    target: str,
    edge_type: EdgeType,
    weight: float,
) -> bool:
    """Add an edge without losing stronger semantic relationships.

    NetworkX DiGraph stores one edge per source/target pair. A pair can satisfy
    multiple relationship heuristics, so keep the highest-priority type in
    ``edge_type`` and preserve all matches in ``edge_types`` for diagnostics.
    """
    edge_type_value = edge_type.value
    if G.has_edge(source, target):
        data = G[source][target]
        edge_types = set(data.get("edge_types", [data.get("edge_type")]))
        edge_types.add(edge_type_value)
        data["edge_types"] = sorted(edge_types)
        data["weight"] = max(float(data.get("weight", 0.0)), weight)

        current_type = data.get("edge_type", EdgeType.SAME_CATEGORY.value)
        if _EDGE_PRIORITY[edge_type_value] > _EDGE_PRIORITY.get(current_type, 0):
            data["edge_type"] = edge_type_value
        return False

    G.add_edge(
        source,
        target,
        edge_type=edge_type_value,
        edge_types=[edge_type_value],
        weight=weight,
    )
    return True


def _classify_verb(endpoint_name: str) -> str | None:
    """Classify an endpoint name into a verb group."""
    for verb, pattern in _VERB_PATTERNS.items():
        if pattern.search(endpoint_name):
            return verb
    return None


def _extract_noun(endpoint_name: str) -> str:
    """Extract the noun part of an endpoint name (e.g., 'search_hotels' → 'hotel')."""
    # Remove common prefixes
    name = endpoint_name.lower()
    for prefix in ("get_", "search_", "find_", "list_", "create_", "add_",
                    "update_", "delete_", "remove_", "book_", "fetch_",
                    "retrieve_", "cancel_", "post_", "put_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Singularize (simple heuristic)
    if name.endswith("ies"):
        name = name[:-3] + "y"
    elif name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]

    return name


def _detect_io_chain(source: APIEndpoint, target: APIEndpoint) -> bool:
    """Check if output of source could feed as input to target.

    Heuristics:
    - Source is a search/list/get → returns objects with IDs
    - Target has a parameter whose name matches a likely output field
      (e.g., target takes "hotel_id" and source searches hotels)
    """
    source_verb = _classify_verb(source.endpoint_name)
    source_noun = _extract_noun(source.endpoint_name)

    # Source should be something that produces data
    if source_verb not in ("search", "get", "create", None):
        # update/delete are less likely to produce useful chaining data
        pass  # Still allow, but less common

    target_param_names = {p.name.lower() for p in target.parameters}

    # Check if any target param looks like it references source's output
    # Common patterns: source noun + "_id", source noun + "_name", just "id"
    id_patterns = [
        f"{source_noun}_id",
        f"{source_noun}id",
        f"{source_noun}_name",
        "id",
        "item_id",
        "resource_id",
    ]

    for pattern in id_patterns:
        if pattern in target_param_names:
            return True

    # Also check if source and target share parameter names (suggesting data flow)
    source_param_names = {p.name.lower() for p in source.parameters}
    shared = source_param_names & target_param_names
    # If they share params and source is a read operation, there's likely a chain
    if len(shared) >= 2 and source_verb in ("search", "get", "list"):
        return True

    return False


def _detect_complementary(source: APIEndpoint, target: APIEndpoint) -> bool:
    """Check if source and target are semantically complementary."""
    if source.tool_name != target.tool_name:
        return False

    source_verb = _classify_verb(source.endpoint_name)
    target_verb = _classify_verb(target.endpoint_name)

    if source_verb is None or target_verb is None:
        return False

    source_noun = _extract_noun(source.endpoint_name)
    target_noun = _extract_noun(target.endpoint_name)

    # Same noun (or close) and complementary verbs
    if source_noun == target_noun or source_noun in target_noun or target_noun in source_noun:
        pair = (source_verb, target_verb)
        if pair in _COMPLEMENTARY_SEQUENCES:
            return True

    return False


def _same_category_neighbors(source: APIEndpoint, candidates: list[APIEndpoint]) -> list[APIEndpoint]:
    """Pick a bounded deterministic set of same-category fallback neighbors."""
    if len(candidates) <= _MAX_SAME_CATEGORY_NEIGHBORS:
        return candidates

    source_params = {param.name.lower() for param in source.parameters}
    source_noun = _extract_noun(source.endpoint_name)
    scored: list[tuple[float, str, APIEndpoint]] = []
    for candidate in candidates:
        candidate_params = {param.name.lower() for param in candidate.parameters}
        candidate_noun = _extract_noun(candidate.endpoint_name)
        shared_params = len(source_params & candidate_params)
        score = float(shared_params)
        if source_noun and candidate_noun and source_noun == candidate_noun:
            score += 2.0
        if _detect_io_chain(source, candidate):
            score += 5.0
        scored.append((-score, candidate.endpoint_id, candidate))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in scored[:_MAX_SAME_CATEGORY_NEIGHBORS]]


def build_tool_graph(registry: ToolRegistry) -> nx.DiGraph:
    """Construct the tool graph from a registry.

    Returns a NetworkX DiGraph where:
      - Nodes: endpoint_id with endpoint data as attributes
      - Edges: typed and weighted relationships
    """
    G = nx.DiGraph()
    endpoints = registry.all_endpoints()

    # Add nodes
    for ep in endpoints:
        G.add_node(
            ep.endpoint_id,
            tool_name=ep.tool_name,
            endpoint_name=ep.endpoint_name,
            category=ep.category,
            description=ep.description,
            method=ep.method.value,
            num_params=len(ep.parameters),
        )

    logger.info("Added %d nodes to tool graph", len(endpoints))

    edge_counts: dict[str, int] = {e.value: 0 for e in EdgeType}

    # Add edges — iterate all pairs (expensive but necessary for IO detection)
    # Optimization: group by category first to reduce pairs
    category_endpoints: dict[str, list[APIEndpoint]] = {}
    for ep in endpoints:
        cat = ep.category
        if cat not in category_endpoints:
            category_endpoints[cat] = []
        category_endpoints[cat].append(ep)

    # Same-tool edges (cheap, within each tool)
    tool_endpoints: dict[str, list[APIEndpoint]] = {}
    for ep in endpoints:
        tn = ep.tool_name
        if tn not in tool_endpoints:
            tool_endpoints[tn] = []
        tool_endpoints[tn].append(ep)

    for tool_name, eps in tool_endpoints.items():
        for i, a in enumerate(eps):
            for b in eps[i + 1:]:
                # Same-tool edges (bidirectional)
                _add_typed_edge(G, a.endpoint_id, b.endpoint_id, EdgeType.SAME_TOOL, 0.3)
                _add_typed_edge(G, b.endpoint_id, a.endpoint_id, EdgeType.SAME_TOOL, 0.3)
                edge_counts[EdgeType.SAME_TOOL.value] += 2

                # Check complementary (within same tool)
                if _detect_complementary(a, b):
                    _add_typed_edge(
                        G, a.endpoint_id, b.endpoint_id, EdgeType.COMPLEMENTARY, 0.8
                    )
                    edge_counts[EdgeType.COMPLEMENTARY.value] += 1

                if _detect_complementary(b, a):
                    _add_typed_edge(
                        G, b.endpoint_id, a.endpoint_id, EdgeType.COMPLEMENTARY, 0.8
                    )
                    edge_counts[EdgeType.COMPLEMENTARY.value] += 1

                # Check IO chain (within same tool)
                if _detect_io_chain(a, b):
                    _add_typed_edge(G, a.endpoint_id, b.endpoint_id, EdgeType.IO_CHAIN, 1.0)
                    edge_counts[EdgeType.IO_CHAIN.value] += 1

                if _detect_io_chain(b, a):
                    _add_typed_edge(G, b.endpoint_id, a.endpoint_id, EdgeType.IO_CHAIN, 1.0)
                    edge_counts[EdgeType.IO_CHAIN.value] += 1

    # Same-category edges and cross-tool IO chains. For large real ToolBench
    # categories, keep fallback edges sparse so graph construction remains usable.
    for cat, eps in category_endpoints.items():
        if len(eps) <= _FULL_MESH_CATEGORY_LIMIT:
            pairs = (
                (a, b)
                for i, a in enumerate(eps)
                for b in eps[i + 1:]
                if a.tool_name != b.tool_name
            )
            for a, b in pairs:
                _add_same_category_relationships(G, a, b, edge_counts)
                _add_same_category_relationships(G, b, a, edge_counts)
        else:
            for a in eps:
                candidates = [b for b in eps if a.tool_name != b.tool_name]
                for b in _same_category_neighbors(a, candidates):
                    _add_same_category_relationships(G, a, b, edge_counts)

    logger.info(
        "Built tool graph: %d nodes, %d edges. Edge type counts: %s",
        G.number_of_nodes(),
        G.number_of_edges(),
        edge_counts,
    )
    return G


def _add_same_category_relationships(
    G: nx.DiGraph,
    source: APIEndpoint,
    target: APIEndpoint,
    edge_counts: dict[str, int],
) -> None:
    _add_typed_edge(
        G,
        source.endpoint_id,
        target.endpoint_id,
        EdgeType.SAME_CATEGORY,
        0.5,
    )
    edge_counts[EdgeType.SAME_CATEGORY.value] += 1

    if _detect_io_chain(source, target):
        _add_typed_edge(G, source.endpoint_id, target.endpoint_id, EdgeType.IO_CHAIN, 1.0)
        edge_counts[EdgeType.IO_CHAIN.value] += 1


def get_graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Compute summary statistics for the tool graph."""
    edge_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1

    # Connected components (treat as undirected for this)
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "edge_type_counts": edge_types,
        "num_connected_components": len(components),
        "largest_component_size": max(len(c) for c in components) if components else 0,
        "avg_degree": (
            round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2)
            if G.number_of_nodes() > 0
            else 0
        ),
    }
