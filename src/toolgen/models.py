"""Pydantic data models for the ToolGen pipeline.

Design decisions (documented in DESIGN.md):
- endpoint_id as "{tool_name}/{endpoint_name}" provides a unique, human-readable key
  for graph nodes and cross-referencing.
- ParameterType enum with fallback handles ToolBench's inconsistent type annotations.
- response_schema is optional because many ToolBench entries lack it — the mock
  executor infers schemas heuristically when missing.
- Category is denormalized onto endpoints for fast graph construction and filtering.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ParameterType(str, enum.Enum):
    """Normalized parameter types. ToolBench uses inconsistent type strings."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"

    @classmethod
    def from_raw(cls, raw: str | None) -> ParameterType:
        """Parse a raw type string from ToolBench, handling inconsistencies."""
        if raw is None:
            return cls.STRING  # Default assumption for missing types
        normalized = raw.strip().lower()
        mapping = {
            "str": cls.STRING,
            "string": cls.STRING,
            "text": cls.STRING,
            "num": cls.NUMBER,
            "number": cls.NUMBER,
            "float": cls.NUMBER,
            "double": cls.NUMBER,
            "int": cls.INTEGER,
            "integer": cls.INTEGER,
            "long": cls.INTEGER,
            "bool": cls.BOOLEAN,
            "boolean": cls.BOOLEAN,
            "list": cls.ARRAY,
            "array": cls.ARRAY,
            "dict": cls.OBJECT,
            "object": cls.OBJECT,
            "json": cls.OBJECT,
        }
        return mapping.get(normalized, cls.UNKNOWN)


class HTTPMethod(str, enum.Enum):
    """HTTP methods supported by API endpoints."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

    @classmethod
    def from_raw(cls, raw: str | None) -> HTTPMethod:
        if raw is None:
            return cls.GET
        normalized = raw.strip().upper()
        try:
            return cls(normalized)
        except ValueError:
            return cls.GET


class ChainPattern(str, enum.Enum):
    """Types of tool-calling patterns in a conversation."""

    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"
    PARALLEL = "parallel"
    CRUD_CYCLE = "crud_cycle"
    SEARCH_AND_ACT = "search_and_act"


class EdgeType(str, enum.Enum):
    """Types of edges in the tool graph."""

    IO_CHAIN = "io_chain"
    SAME_TOOL = "same_tool"
    SAME_CATEGORY = "same_category"
    COMPLEMENTARY = "complementary"


# ---------------------------------------------------------------------------
# Tool Registry Models
# ---------------------------------------------------------------------------

class Parameter(BaseModel):
    """A single API endpoint parameter."""

    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[str] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return str(v)


class APIEndpoint(BaseModel):
    """A single API endpoint (function) within a tool."""

    tool_name: str
    endpoint_name: str
    endpoint_id: str = ""  # Computed: "{tool_name}/{endpoint_name}"
    description: str = ""
    method: HTTPMethod = HTTPMethod.GET
    category: str = ""
    parameters: list[Parameter] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.endpoint_id:
            self.endpoint_id = f"{self.tool_name}/{self.endpoint_name}"

    @property
    def required_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if not p.required]

    @property
    def param_names(self) -> set[str]:
        return {p.name for p in self.parameters}


class Tool(BaseModel):
    """A tool (collection of related API endpoints)."""

    name: str
    description: str = ""
    category: str = ""
    endpoints: list[APIEndpoint] = Field(default_factory=list)

    @property
    def endpoint_ids(self) -> list[str]:
        return [ep.endpoint_id for ep in self.endpoints]


class RegistryStats(BaseModel):
    """Summary statistics for the tool registry."""

    total_tools: int = 0
    total_endpoints: int = 0
    total_categories: int = 0
    categories: dict[str, int] = Field(default_factory=dict)  # category -> endpoint count
    endpoints_with_response_schema: int = 0
    avg_params_per_endpoint: float = 0.0
    parse_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Graph Models
# ---------------------------------------------------------------------------

class ToolEdge(BaseModel):
    """An edge in the tool graph."""

    source: str  # endpoint_id
    target: str  # endpoint_id
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolChain(BaseModel):
    """A sampled sequence of endpoints forming a tool-calling chain."""

    endpoints: list[APIEndpoint]
    pattern: ChainPattern = ChainPattern.MULTI_STEP
    constraints_satisfied: list[str] = Field(default_factory=list)

    @property
    def endpoint_ids(self) -> list[str]:
        return [ep.endpoint_id for ep in self.endpoints]

    @property
    def num_steps(self) -> int:
        return len(self.endpoints)

    @property
    def num_distinct_tools(self) -> int:
        return len({ep.tool_name for ep in self.endpoints})

    @property
    def categories(self) -> set[str]:
        return {ep.category for ep in self.endpoints}


class SamplerConstraints(BaseModel):
    """Constraints for the tool chain sampler."""

    min_steps: int = 1
    max_steps: int = 5
    min_distinct_tools: int = 1
    required_domains: list[str] | None = None
    required_patterns: list[ChainPattern] | None = None
    exclude_endpoints: set[str] = Field(default_factory=set)
    prefer_io_chains: bool = True


# ---------------------------------------------------------------------------
# Conversation Models
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    endpoint: str  # endpoint_id
    arguments: dict[str, Any] = Field(default_factory=dict)


class ArgumentSource(BaseModel):
    """Auditable source for one tool-call argument."""

    source: str = "unknown"
    value: Any = None
    evidence: str = ""
    source_endpoint: str | None = None


class ToolStepTrace(BaseModel):
    """Trace-first reasoning metadata for one tool step.

    This is not hidden chain-of-thought. It records observable planning facts:
    which endpoint was used, what earlier tool calls it depends on, where each
    argument came from, and which reference ids the tool returned.
    """

    step: int
    endpoint: str
    goal: str = ""
    depends_on: list[str] = Field(default_factory=list)
    argument_sources: dict[str, ArgumentSource] = Field(default_factory=dict)
    output_refs: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "user", "assistant", "tool", "system"
    content: str | dict | None = None
    tool_calls: list[ToolCall] | None = None


class JudgeScore(BaseModel):
    """Score from LLM-as-judge for a single dimension."""

    score: float = Field(ge=1.0, le=5.0)
    rationale: str = ""


class JudgeScores(BaseModel):
    """Full scoring from LLM-as-judge."""

    tool_correctness: JudgeScore = Field(
        default_factory=lambda: JudgeScore(score=1.0)
    )
    naturalness: JudgeScore = Field(
        default_factory=lambda: JudgeScore(score=1.0)
    )
    task_completion: JudgeScore = Field(
        default_factory=lambda: JudgeScore(score=1.0)
    )

    @property
    def overall(self) -> float:
        scores = [
            self.tool_correctness.score,
            self.naturalness.score,
            self.task_completion.score,
        ]
        return round(sum(scores) / len(scores), 2)


class ScenarioPlan(BaseModel):
    """Structured output from the Planner agent."""

    scenario: str = ""
    user_persona: str = ""
    expected_tool_sequence: list[str] = Field(default_factory=list)
    disambiguation_points: list[str] = Field(default_factory=list)
    complexity: str = "multi_step"


class ConversationMetadata(BaseModel):
    """Metadata for a generated conversation."""

    seed: int = 42
    conversation_index: int = 0
    tools_used: list[str] = Field(default_factory=list)
    num_turns: int = 0
    num_tool_calls: int = 0
    num_distinct_tools: int = 0
    pattern: str = ""
    category_domains: list[str] = Field(default_factory=list)
    chain_source: str = "graph_sampler"
    repair_attempts: int = 0
    steering_enabled: bool = True
    generation_timestamp: str = ""
    model: str = ""
    generation_profile: str = ""
    planner_scenario: ScenarioPlan | None = None


class Conversation(BaseModel):
    """A complete generated conversation record."""

    conversation_id: str = ""
    messages: list[Message] = Field(default_factory=list)
    step_trace: list[ToolStepTrace] = Field(default_factory=list)
    judge_scores: JudgeScores | None = None
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)

    def to_output_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        d = self.model_dump(mode="json")
        if self.judge_scores:
            d["judge_scores"]["overall"] = self.judge_scores.overall
        return d


# ---------------------------------------------------------------------------
# Diversity / Steering Models
# ---------------------------------------------------------------------------

class SteeringGuidance(BaseModel):
    """Guidance from the cross-conversation steering system."""

    avoid_tool_combinations: list[list[str]] = Field(default_factory=list)
    prefer_domains: list[str] = Field(default_factory=list)
    prefer_patterns: list[str] = Field(default_factory=list)
    complexity_suggestion: str = ""
    rationale: str = ""


class DiversityMetrics(BaseModel):
    """Quantitative diversity metrics for a generated dataset."""

    tool_combination_entropy: float = 0.0
    domain_coverage_cv: float = 0.0  # Coefficient of variation
    unique_tool_pairs: int = 0
    unique_domains_used: int = 0
    total_domains_available: int = 0
    pattern_distribution: dict[str, int] = Field(default_factory=dict)
    mean_quality_score: float = 0.0
    quality_scores: dict[str, float] = Field(default_factory=dict)


class ExperimentResults(BaseModel):
    """Results from the diversity experiment (Run A vs Run B)."""

    run_a: DiversityMetrics = Field(default_factory=DiversityMetrics)
    run_b: DiversityMetrics = Field(default_factory=DiversityMetrics)
    seed: int = 42
    num_conversations: int = 50
    analysis: str = ""
