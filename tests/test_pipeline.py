"""Integration tests for the end-to-end pipeline."""

from pathlib import Path

from toolgen.config import get_settings
from toolgen.models import Conversation, ConversationMetadata, JudgeScore, JudgeScores
from toolgen.pipeline import Pipeline, compute_diversity_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_TOOLENV = PROJECT_ROOT / "data" / "toolenv" / "tools"


def _settings(tmp_path):
    return get_settings(
        llm_provider="offline",
        generation_model="offline-deterministic",
        judge_model="offline-heuristic",
        toolenv_dir=BUNDLED_TOOLENV,
        output_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )


def test_build_artifacts_writes_registry_and_graph_files(tmp_path):
    pipeline = Pipeline(_settings(tmp_path))

    paths = pipeline.build_artifacts()

    assert paths["registry_stats"].exists()
    assert paths["graph_stats"].exists()
    assert paths["endpoints"].exists()
    assert paths["edges"].exists()
    assert paths["registry_stats_diagram"].exists()
    assert paths["graph_stats_diagram"].exists()
    assert paths["endpoints_diagram"].exists()
    assert paths["edges_diagram"].exists()
    for key in (
        "registry_stats_diagram",
        "graph_stats_diagram",
        "endpoints_diagram",
        "edges_diagram",
    ):
        assert "```mermaid" in paths[key].read_text(encoding="utf-8")


def test_compute_diversity_metrics():
    conversations = [
        Conversation(
            conversation_id="a",
            judge_scores=JudgeScores(
                tool_correctness=JudgeScore(score=4.0),
                naturalness=JudgeScore(score=4.0),
                task_completion=JudgeScore(score=5.0),
            ),
            metadata=ConversationMetadata(
                tools_used=["hotel/search", "flight/book"],
                category_domains=["Travel"],
                pattern="multi_step",
            ),
        ),
        Conversation(
            conversation_id="b",
            judge_scores=JudgeScores(
                tool_correctness=JudgeScore(score=5.0),
                naturalness=JudgeScore(score=4.0),
                task_completion=JudgeScore(score=4.0),
            ),
            metadata=ConversationMetadata(
                tools_used=["product/search", "payment/create"],
                category_domains=["Commerce"],
                pattern="search_and_act",
            ),
        ),
    ]

    metrics = compute_diversity_metrics(conversations)

    assert metrics.unique_tool_pairs == 2
    assert metrics.unique_domains_used == 2
    assert metrics.mean_quality_score > 4.0
    assert metrics.pattern_distribution["multi_step"] == 1


def test_offline_pipeline_generates_100_sample_dataset(tmp_path):
    pipeline = Pipeline(_settings(tmp_path))
    registry = pipeline.load_registry()
    pipeline.build_graph()
    output_path = tmp_path / "offline_100.jsonl"

    conversations = pipeline.generate(
        num_conversations=100,
        seed=42,
        enable_steering=True,
        enable_repair=True,
        output_path=output_path,
    )

    assert len(conversations) == 100
    assert output_path.exists()
    assert len(output_path.read_text().splitlines()) == 100

    multi_step_multi_tool = [
        conv
        for conv in conversations
        if conv.metadata.num_tool_calls >= 3 and conv.metadata.num_distinct_tools >= 2
    ]
    ratio = len(multi_step_multi_tool) / len(conversations)
    assert 0.50 <= ratio <= 0.65

    mean_score = sum(conv.judge_scores.overall for conv in conversations) / len(conversations)
    assert mean_score >= 4.0
    assert any(
        msg.role == "assistant" and isinstance(msg.content, str) and "?" in msg.content
        for conv in conversations
        for msg in conv.messages
    )
    assert all(len(conv.step_trace) == conv.metadata.num_tool_calls for conv in conversations)
    assert any(
        any(
            source.source == "previous_tool_result"
            for source in step.argument_sources.values()
        )
        for conv in conversations
        for step in conv.step_trace
    )
    for conv in conversations:
        visible_text = "\n".join(
            str(msg.content)
            for msg in conv.messages
            if msg.role in {"user", "assistant"} and msg.content
        ).lower()
        for endpoint_id in conv.metadata.tools_used:
            tool_name, _, endpoint_name = endpoint_id.partition("/")
            forbidden = {
                endpoint_id.lower(),
                tool_name.lower(),
                endpoint_name.lower(),
                endpoint_name.replace("_", " ").lower(),
            }
            assert not any(token and token in visible_text for token in forbidden)

    metrics = compute_diversity_metrics(conversations)
    assert metrics.unique_domains_used >= 3
    assert metrics.unique_tool_pairs > 5
    assert registry.stats().total_endpoints >= 20


def test_pipeline_progress_callbacks_for_generate_and_evaluate(tmp_path):
    pipeline = Pipeline(_settings(tmp_path))
    pipeline.load_registry()
    pipeline.build_graph()
    output_path = tmp_path / "progress.jsonl"
    generate_updates = []

    conversations = pipeline.generate(
        num_conversations=3,
        seed=7,
        enable_steering=False,
        enable_repair=False,
        output_path=output_path,
        progress_callback=lambda current, total, label: generate_updates.append(
            (current, total, label)
        ),
    )

    assert len(conversations) == 3
    assert generate_updates[-1][0] == 3
    assert generate_updates[-1][1] == 3
    assert "conv_" in generate_updates[-1][2]

    evaluate_updates = []
    rescored = pipeline.evaluate(
        output_path,
        progress_callback=lambda current, total, label: evaluate_updates.append(
            (current, total, label)
        ),
    )

    assert len(rescored) == 3
    assert evaluate_updates[-1][0] == 3
    assert evaluate_updates[-1][1] == 3
    assert "score=" in evaluate_updates[-1][2]


def test_parallel_offline_generation_preserves_output_order(tmp_path):
    settings = _settings(tmp_path).model_copy(update={"max_parallel_conversations": 3})
    pipeline = Pipeline(settings)
    pipeline.load_registry()
    pipeline.build_graph()
    output_path = tmp_path / "parallel.jsonl"

    conversations = pipeline.generate(
        num_conversations=8,
        seed=11,
        enable_steering=False,
        enable_repair=False,
        output_path=output_path,
    )

    assert [conv.conversation_id for conv in conversations] == [
        f"conv_{i:04d}" for i in range(8)
    ]
    assert output_path.read_text().splitlines()[0].startswith('{"conversation_id": "conv_0000"')
