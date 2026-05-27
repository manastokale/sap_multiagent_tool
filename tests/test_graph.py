"""Tests for tool graph construction and sampling."""


from toolgen.graph.builder import (
    build_tool_graph,
    get_graph_stats,
    _classify_verb,
    _extract_noun,
    _detect_io_chain,
    _detect_complementary,
)
from toolgen.graph.sampler import (
    sample_chain,
    sample_chains,
    _classify_chain_pattern,
)
from toolgen.models import (
    APIEndpoint,
    ChainPattern,
    Parameter,
    ParameterType,
    SamplerConstraints,
)


class TestVerbClassification:
    def test_search_verbs(self):
        assert _classify_verb("search_hotels") == "search"
        assert _classify_verb("find_restaurants") == "search"
        assert _classify_verb("list_flights") == "search"

    def test_get_verbs(self):
        assert _classify_verb("get_details") == "get"
        assert _classify_verb("retrieve_info") == "get"
        assert _classify_verb("fetch_data") == "get"

    def test_create_verbs(self):
        assert _classify_verb("create_booking") == "create"
        assert _classify_verb("book_hotel") == "create"
        assert _classify_verb("add_item") == "create"

    def test_update_verbs(self):
        assert _classify_verb("update_profile") == "update"
        assert _classify_verb("modify_order") == "update"

    def test_delete_verbs(self):
        assert _classify_verb("delete_booking") == "delete"
        assert _classify_verb("cancel_reservation") == "delete"

    def test_unknown(self):
        assert _classify_verb("process_data") is None


class TestNounExtraction:
    def test_basic(self):
        assert _extract_noun("search_hotels") == "hotel"
        assert _extract_noun("get_flight_details") == "flight_detail"

    def test_plurals(self):
        assert _extract_noun("list_categories") == "category"
        assert _extract_noun("search_buses") == "buse"  # Simple heuristic


class TestIOChainDetection:
    def test_id_chaining(self):
        """search_hotels → book_hotel should chain via hotel_id."""
        search = APIEndpoint(
            tool_name="hotel_api",
            endpoint_name="search_hotels",
            parameters=[Parameter(name="city", type=ParameterType.STRING)],
        )
        book = APIEndpoint(
            tool_name="hotel_api",
            endpoint_name="book_hotel",
            parameters=[Parameter(name="hotel_id", type=ParameterType.STRING, required=True)],
        )
        assert _detect_io_chain(search, book)

    def test_no_chain(self):
        """Two unrelated endpoints should not chain."""
        weather = APIEndpoint(
            tool_name="weather_api",
            endpoint_name="get_weather",
            parameters=[Parameter(name="city", type=ParameterType.STRING)],
        )
        stock = APIEndpoint(
            tool_name="stock_api",
            endpoint_name="get_stock_price",
            parameters=[Parameter(name="symbol", type=ParameterType.STRING)],
        )
        assert not _detect_io_chain(weather, stock)


class TestComplementaryDetection:
    def test_search_get(self):
        search = APIEndpoint(
            tool_name="api",
            endpoint_name="search_hotels",
        )
        get = APIEndpoint(
            tool_name="api",
            endpoint_name="get_hotel",
        )
        assert _detect_complementary(search, get)

    def test_different_tools(self):
        """Complementary requires same tool."""
        a = APIEndpoint(tool_name="api_a", endpoint_name="search_hotels")
        b = APIEndpoint(tool_name="api_b", endpoint_name="get_hotel")
        assert not _detect_complementary(a, b)


class TestBuildToolGraph:
    def test_basic_graph(self, sample_registry):
        G = build_tool_graph(sample_registry)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_edge_types_present(self, sample_registry):
        G = build_tool_graph(sample_registry)
        stats = get_graph_stats(G)
        edge_types = stats["edge_type_counts"]
        # Should have at least same_tool edges
        assert "same_tool" in edge_types

    def test_graph_stats(self, sample_registry):
        G = build_tool_graph(sample_registry)
        stats = get_graph_stats(G)
        assert stats["num_nodes"] == 5  # 3 hotel + 2 flight
        assert stats["num_edges"] > 0
        assert stats["num_connected_components"] >= 1


class TestSampler:
    def test_sample_single_chain(self, sample_registry):
        G = build_tool_graph(sample_registry)
        chain = sample_chain(G, sample_registry)
        assert chain is not None
        assert chain.num_steps >= 1

    def test_constrained_sampling(self, sample_registry):
        G = build_tool_graph(sample_registry)
        constraints = SamplerConstraints(min_steps=2, max_steps=4)
        chain = sample_chain(G, sample_registry, constraints)
        if chain:  # May fail if graph is too small
            assert chain.num_steps >= 2
            assert chain.num_steps <= 4

    def test_sample_multiple_chains(self, sample_registry):
        G = build_tool_graph(sample_registry)
        chains = sample_chains(G, sample_registry, num_chains=5, seed=42)
        assert len(chains) > 0

    def test_deterministic_with_seed(self, sample_registry):
        G = build_tool_graph(sample_registry)
        chains_a = sample_chains(G, sample_registry, num_chains=5, seed=42)
        chains_b = sample_chains(G, sample_registry, num_chains=5, seed=42)
        # Same seed should produce same chains
        assert len(chains_a) == len(chains_b)
        for a, b in zip(chains_a, chains_b):
            assert a.endpoint_ids == b.endpoint_ids


class TestChainPatternClassification:
    def test_single_step(self, sample_endpoints):
        pattern = _classify_chain_pattern([sample_endpoints[0]])
        assert pattern == ChainPattern.SINGLE_STEP

    def test_search_and_act(self, sample_endpoints):
        pattern = _classify_chain_pattern(sample_endpoints)
        assert pattern in (ChainPattern.SEARCH_AND_ACT, ChainPattern.MULTI_STEP)
