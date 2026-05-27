"""Planner agent — uses structured JSON output.

This agent satisfies the PDF requirement that at least one agent must use
structured output rather than free-text LLM generation. It takes a sampled
ToolChain and diversity guidance, and produces a deterministic scenario plan
that guides the rest of the conversation.
"""

from __future__ import annotations

import logging

from toolgen.agents.llm_client import LLMClient
from toolgen.models import (
    ScenarioPlan,
    SteeringGuidance,
    ToolChain,
)

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_INSTRUCTION = """\
Create compact scenario plans for synthetic tool-use chats.

Return only valid JSON:
{
  "scenario": "1 short sentence in ordinary user language",
  "user_persona": "brief user style",
  "expected_tool_sequence": ["tool/endpoint1", "tool/endpoint2", ...],
  "disambiguation_points": ["missing detail to ask about"],
  "complexity": "one of: single_step, multi_step, multi_step_with_disambiguation"
}

Rules:
- Keep scenario/persona/disambiguation concise.
- expected_tool_sequence must match the provided tools.
- Do not mention API/tool/endpoint names or raw parameter names in user-facing fields.
"""


def plan_scenario(
    client: LLMClient,
    chain: ToolChain,
    steering: SteeringGuidance | None = None,
    strict_live: bool = False,
) -> ScenarioPlan:
    """Generate a scenario plan using structured JSON output.

    This is the key structured-output agent in the system.
    """
    tool_descriptions = []
    for ep in chain.endpoints:
        params = ", ".join(
            f"{p.name}:{p.type.value}" + (" required" if p.required else "")
            for p in ep.parameters
        )
        tool_descriptions.append(
            f"- {ep.endpoint_id}: {_shorten(ep.description or 'No description', 120)}; "
            f"args: {params or 'none'}"
        )

    tools_text = "\n".join(tool_descriptions)

    # Build steering guidance if provided
    steering_text = ""
    if steering and steering.rationale:
        avoid = ", ".join(
            f"({', '.join(combo)})" for combo in steering.avoid_tool_combinations
        ) if steering.avoid_tool_combinations else "none"
        prefer = ", ".join(steering.prefer_domains) if steering.prefer_domains else "none"
        steering_text = (
            "\nDiversity guidance:\n"
            f"avoid similar: {_shorten(steering.rationale, 160)}\n"
            f"avoid combos: {avoid}\n"
            f"prefer domains: {prefer}\n"
            f"complexity: {steering.complexity_suggestion or 'vary'}\n"
        )

    prompt = (
        f"Tools:\n{tools_text}\n"
        f"Pattern: {chain.pattern.value}\n"
        f"Categories: {', '.join(chain.categories)}\n"
        f"{steering_text}\n"
        "Generate the compact scenario plan."
    )

    response = client.generate_json(
        prompt,
        system_instruction=_PLANNER_SYSTEM_INSTRUCTION,
        temperature=0.4,
    )

    if response.error or not response.parsed:
        if strict_live:
            raise RuntimeError(f"Planner live LLM failed: {response.error}")
        logger.warning("Planner failed: %s. Using fallback.", response.error)
        return _fallback_plan(chain)

    try:
        plan = ScenarioPlan(**response.parsed)
        # Ensure expected_tool_sequence matches the chain
        if not plan.expected_tool_sequence:
            plan.expected_tool_sequence = chain.endpoint_ids
        return plan
    except Exception as e:
        if strict_live:
            raise RuntimeError(f"Planner live LLM output failed schema validation: {e}") from e
        logger.warning("Failed to parse planner output: %s", e)
        return _fallback_plan(chain)


def _fallback_plan(chain: ToolChain) -> ScenarioPlan:
    """Generate a simple fallback plan when the LLM fails."""
    categories = ", ".join(chain.categories)
    tools = [ep.endpoint_id for ep in chain.endpoints]
    return ScenarioPlan(
        scenario=f"User needs help with {categories}-related tasks involving {len(tools)} steps.",
        user_persona="A straightforward user who provides information when asked.",
        expected_tool_sequence=tools,
        disambiguation_points=["specific preferences", "date or time constraints"],
        complexity="multi_step" if len(tools) > 1 else "single_step",
    )


def _shorten(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."
