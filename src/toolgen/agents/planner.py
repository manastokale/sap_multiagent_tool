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
You are a scenario planner for synthetic conversation generation.
Your job is to create realistic user scenarios that exercise specific API tools.

You MUST output valid JSON matching this exact schema:
{
  "scenario": "A 1-2 sentence description of the user's goal",
  "user_persona": "Brief description of the user's personality and communication style",
  "expected_tool_sequence": ["tool/endpoint1", "tool/endpoint2", ...],
  "disambiguation_points": ["What info is missing that the assistant should ask about"],
  "complexity": "one of: single_step, multi_step, multi_step_with_disambiguation"
}

Rules:
- The scenario must be realistic and natural — something a real person would ask
- Include 1-2 disambiguation points where the user's request is vague
- The persona should influence how the user communicates (formal, casual, impatient, etc.)
- The expected_tool_sequence should match the tools provided
- Vary complexity: some scenarios should be straightforward, others complex
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
    # Build the tool description for the prompt
    tool_descriptions = []
    for ep in chain.endpoints:
        params = ", ".join(
            f"{p.name}: {p.type.value}" + (" (required)" if p.required else "")
            for p in ep.parameters
        )
        tool_descriptions.append(
            f"- {ep.endpoint_id}: {ep.description or 'No description'}\n"
            f"  Method: {ep.method.value}\n"
            f"  Parameters: {params or 'none'}"
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
            f"\nDiversity guidance:\n"
            f"- Avoid similar scenarios to: {steering.rationale}\n"
            f"- Avoid tool combinations: {avoid}\n"
            f"- Prefer domains: {prefer}\n"
            f"- Complexity suggestion: {steering.complexity_suggestion or 'vary'}\n"
        )

    prompt = (
        f"Create a realistic scenario plan for a conversation that uses these tools:\n\n"
        f"{tools_text}\n\n"
        f"The conversation pattern is: {chain.pattern.value}\n"
        f"Categories involved: {', '.join(chain.categories)}\n"
        f"{steering_text}\n"
        f"Generate the scenario plan as JSON."
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
