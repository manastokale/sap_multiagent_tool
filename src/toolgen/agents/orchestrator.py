"""Orchestrator — deterministic state machine that drives conversations.

This is NOT an LLM agent. It's the control loop that coordinates:
  Planner → User Sim → Assistant → Mock Executor → ...
and manages conversation state, turn limits, and structural validation.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

from toolgen.agents.llm_client import LLMClient
from toolgen.agents.planner import plan_scenario
from toolgen.agents.user_sim import (
    generate_initial_message,
    generate_response,
    should_end_conversation,
)
from toolgen.agents.assistant import generate_assistant_turn
from toolgen.executor.mock_executor import MockExecutor
from toolgen.models import (
    APIEndpoint,
    Conversation,
    ConversationMetadata,
    Message,
    ScenarioPlan,
    SteeringGuidance,
    ToolCall,
    ToolChain,
)

logger = logging.getLogger(__name__)

_CITY_VALUES = ["Paris", "Tokyo", "Chicago", "Berlin", "Singapore", "Madrid"]
_TOPIC_VALUES = ["quarterly planning", "a customer visit", "team travel", "a product launch"]


def generate_conversation(
    client: LLMClient,
    chain: ToolChain,
    executor: MockExecutor,
    conversation_index: int = 0,
    seed: int = 42,
    max_turns: int = 15,
    steering: SteeringGuidance | None = None,
    model_name: str = "",
    strict_live: bool = False,
    live_profile: str = "full",
) -> Conversation:
    """Generate a single conversation from a tool chain.

    This is the main conversation loop:
    1. Planner creates scenario
    2. User sim generates initial message
    3. Loop: Assistant → (tool call → executor → tool result) or text → User sim
    4. Conversation ends when user is satisfied or max turns reached
    """
    # Reset executor state for this conversation
    executor.reset()
    hybrid_live = _is_hybrid_live(live_profile)
    rng = random.Random(seed + conversation_index)

    # Step 1: Plan the scenario
    logger.info("Generating conversation %d with chain: %s", conversation_index, chain.endpoint_ids)
    plan = plan_scenario(client, chain, steering, strict_live=strict_live)
    logger.debug("Scenario: %s", plan.scenario)

    # Initialize conversation
    messages: list[Message] = []
    tools_used: list[str] = []
    num_tool_calls = 0
    available_tools = chain.endpoints

    # Step 2: User's opening message
    initial_msg = (
        _offline_initial_request(plan, chain, rng)
        if hybrid_live
        else generate_initial_message(client, plan, strict_live=strict_live)
    )
    messages.append(Message(role="user", content=initial_msg))
    logger.debug("User: %s", initial_msg[:100])

    # Step 3: Conversation loop
    for turn in range(max_turns):
        # Assistant's turn
        text_response, tool_call = generate_assistant_turn(
            client,
            messages,
            available_tools,
            session_context=executor.get_session_summary(),
            strict_live=strict_live,
        )

        if tool_call:
            # Record the tool call
            messages.append(Message(
                role="assistant",
                content=None,
                tool_calls=[tool_call],
            ))
            tools_used.append(tool_call.endpoint)
            num_tool_calls += 1

            # Execute the tool
            endpoint = None
            for ep in available_tools:
                if ep.endpoint_id == tool_call.endpoint:
                    endpoint = ep
                    break

            if endpoint:
                result = executor.execute(endpoint, tool_call.arguments)
            else:
                result = {"error": f"Unknown endpoint: {tool_call.endpoint}"}

            # Add tool result
            messages.append(Message(role="tool", content=result))
            logger.debug("Tool %s → %s", tool_call.endpoint, str(result)[:100])

            # Let assistant summarize the result. Hybrid mode uses a deterministic
            # summary so only the actual tool-decision turns consume LLM calls.
            if hybrid_live:
                summary_text = _hybrid_tool_summary(
                    tool_call.endpoint,
                    result,
                    num_tool_calls,
                    len(chain.endpoints),
                )
            else:
                summary_text, _ = generate_assistant_turn(
                    client,
                    messages,
                    available_tools,
                    session_context=executor.get_session_summary(),
                    strict_live=strict_live,
                )
            if summary_text:
                messages.append(Message(role="assistant", content=summary_text))
                logger.debug("Assistant: %s", summary_text[:100])

        elif text_response:
            messages.append(Message(role="assistant", content=text_response))
            logger.debug("Assistant: %s", text_response[:100])

        # Check if conversation should end
        if should_end_conversation(plan, messages, num_tool_calls):
            logger.debug("Conversation %d ending (task complete)", conversation_index)
            break

        # User's response (if assistant didn't end)
        if turn < max_turns - 1:
            # Build tool results summary for user context
            tool_summary = None
            if messages and messages[-1].role == "assistant" and messages[-1].content:
                tool_summary = messages[-1].content

            user_msg = (
                _hybrid_user_response(
                    plan,
                    chain,
                    num_tool_calls,
                    rng,
                    messages[-1].content if messages and messages[-1].content else None,
                )
                if hybrid_live
                else generate_response(
                    client,
                    plan,
                    messages,
                    tool_results_summary=tool_summary,
                    strict_live=strict_live,
                )
            )
            messages.append(Message(role="user", content=user_msg))
            logger.debug("User: %s", user_msg[:100])

    # Build conversation record
    distinct_tools = len(set(tools_used))
    categories = list(chain.categories)

    conversation = Conversation(
        conversation_id=f"conv_{conversation_index:04d}",
        messages=messages,
        metadata=ConversationMetadata(
            seed=seed,
            conversation_index=conversation_index,
            tools_used=tools_used,
            num_turns=len(messages),
            num_tool_calls=num_tool_calls,
            num_distinct_tools=distinct_tools,
            pattern=chain.pattern.value,
            category_domains=categories,
            chain_source="graph_sampler",
            steering_enabled=steering is not None,
            generation_timestamp=datetime.now().isoformat(),
            model=model_name,
            generation_profile="hybrid" if hybrid_live else "full",
            planner_scenario=plan,
        ),
    )

    logger.info(
        "Generated conv_%04d: %d turns, %d tool calls, %d distinct tools",
        conversation_index,
        len(messages),
        num_tool_calls,
        distinct_tools,
    )

    return conversation


def generate_offline_conversation(
    chain: ToolChain,
    executor: MockExecutor,
    conversation_index: int = 0,
    seed: int = 42,
    max_turns: int = 15,
    steering: SteeringGuidance | None = None,
    model_name: str = "offline-deterministic",
) -> Conversation:
    """Generate a deterministic conversation without remote LLM calls.

    This path keeps the same agent protocol as the LLM-backed path:
    a structured planner record, user turns, assistant tool calls, offline tool
    execution, and a final assistant answer. It is intentionally simple so tests
    and reviewers can run the full pipeline without credentials.
    """
    executor.reset()
    rng = random.Random(seed + conversation_index)
    plan = _offline_plan(chain, steering)

    messages: list[Message] = [
        Message(role="user", content=_offline_initial_request(plan, chain, rng))
    ]

    first_required = _visible_required_params(chain.endpoints[0]) if chain.endpoints else []
    if first_required:
        messages.append(Message(role="assistant", content=_clarifying_question(first_required)))
        messages.append(Message(role="user", content=_clarifying_answer(first_required, rng)))

    tools_used: list[str] = []
    value_context: dict[str, Any] = {}
    results_seen: list[dict[str, Any]] = []

    for step_index, endpoint in enumerate(chain.endpoints[:max_turns]):
        args = _build_arguments(endpoint, value_context, rng, step_index)
        tool_call = ToolCall(endpoint=endpoint.endpoint_id, arguments=args)
        messages.append(Message(role="assistant", content=None, tool_calls=[tool_call]))
        tools_used.append(endpoint.endpoint_id)

        result = executor.execute(endpoint, args)
        messages.append(Message(role="tool", content=result))
        results_seen.append(result)
        _remember_result(endpoint, result, value_context)

    messages.append(
        Message(
            role="assistant",
            content=_offline_final_answer(chain, results_seen, value_context),
        )
    )

    return Conversation(
        conversation_id=f"conv_{conversation_index:04d}",
        messages=messages,
        metadata=ConversationMetadata(
            seed=seed,
            conversation_index=conversation_index,
            tools_used=tools_used,
            num_turns=len(messages),
            num_tool_calls=len(tools_used),
            num_distinct_tools=len(set(ep.split("/")[0] for ep in tools_used)),
            pattern=chain.pattern.value,
            category_domains=list(chain.categories),
            chain_source="graph_sampler",
            steering_enabled=steering is not None,
            generation_timestamp=datetime.now().isoformat(),
            model=model_name,
            generation_profile="offline",
            planner_scenario=plan,
        ),
    )


def _is_hybrid_live(live_profile: str) -> bool:
    return live_profile.strip().lower().replace("-", "_") in {
        "hybrid",
        "hybrid_live",
        "quota_saver",
        "fast",
    }


def _hybrid_user_response(
    plan: ScenarioPlan,
    chain: ToolChain,
    num_tool_calls_made: int,
    rng: random.Random,
    latest_assistant_text: str | None,
) -> str:
    if latest_assistant_text and "?" in latest_assistant_text:
        next_index = min(num_tool_calls_made, max(len(chain.endpoints) - 1, 0))
        params = _visible_required_params(chain.endpoints[next_index]) if chain.endpoints else []
        if params:
            return _clarifying_answer(params, rng)
        return "Use the most relevant available option and continue."

    if num_tool_calls_made < len(plan.expected_tool_sequence):
        return "Yes, continue with the next step using that result."
    return "Thanks, that covers what I needed."


def _hybrid_tool_summary(
    endpoint_id: str,
    result: dict[str, Any],
    num_tool_calls_made: int,
    expected_tool_calls: int,
) -> str:
    reference = _first_reference(result) or "the returned result"
    endpoint_name = endpoint_id.split("/")[-1]
    if num_tool_calls_made >= expected_tool_calls:
        return (
            f"Done. I completed {endpoint_name} and used reference {reference} "
            "for the final result."
        )
    return (
        f"I received the result from {endpoint_name} and will use reference "
        f"{reference} for the next step."
    )


def _first_reference(value: Any) -> str | None:
    if isinstance(value, dict):
        for preferred_key in ("confirmation_id", "booking_id", "id"):
            if preferred_key in value:
                return str(value[preferred_key])
        for key, child in value.items():
            if key.lower().endswith("_id"):
                return str(child)
            nested = _first_reference(child)
            if nested:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _first_reference(item)
            if nested:
                return nested
    return None


def _offline_plan(chain: ToolChain, steering: SteeringGuidance | None) -> ScenarioPlan:
    domains = ", ".join(sorted(chain.categories)) or "general"
    first = chain.endpoints[0] if chain.endpoints else None
    task = first.endpoint_name.replace("_", " ") if first else "complete a task"
    rationale = ""
    if steering and steering.prefer_domains:
        rationale = f" Prefer under-covered domains such as {', '.join(steering.prefer_domains[:2])}."

    return ScenarioPlan(
        scenario=f"The user needs help with a {domains.lower()} workflow that starts with {task}.",
        user_persona="A concise business user who gives missing details when asked.",
        expected_tool_sequence=chain.endpoint_ids,
        disambiguation_points=["required search criteria", "date or budget constraints"],
        complexity="multi_step_with_disambiguation"
        if chain.num_steps >= 3
        else ("multi_step" if chain.num_steps > 1 else "single_step"),
    ).model_copy(update={"scenario": f"The user needs help with a {domains.lower()} workflow that starts with {task}.{rationale}"})


def _offline_initial_request(plan: ScenarioPlan, chain: ToolChain, rng: random.Random) -> str:
    domain = next(iter(chain.categories), "this")
    topic = rng.choice(_TOPIC_VALUES)
    return f"I need help with {domain.lower()} for {topic}. Can you take care of it?"


def _visible_required_params(endpoint: APIEndpoint) -> list[str]:
    """Required fields worth asking the user about before the first tool call."""
    visible = []
    for param in endpoint.required_parameters:
        name = param.name.lower()
        if name.endswith("_id") or name in {"id", "item_id", "resource_id"}:
            continue
        visible.append(param.name)
    return visible[:3]


def _clarifying_question(params: list[str]) -> str:
    readable = ", ".join(p.replace("_", " ") for p in params)
    return f"What {readable} should I use?"


def _clarifying_answer(params: list[str], rng: random.Random) -> str:
    values = [_value_for_name(param, rng, step_index=0) for param in params]
    details = ", ".join(f"{param.replace('_', ' ')}: {value}" for param, value in zip(params, values))
    return f"Use {details}."


def _build_arguments(
    endpoint: APIEndpoint,
    value_context: dict[str, Any],
    rng: random.Random,
    step_index: int,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    params = endpoint.required_parameters + [
        param for param in endpoint.optional_parameters if _include_optional(param.name)
    ]

    for param in params:
        value = _grounded_value(param.name, value_context)
        if value is None:
            value = _value_for_name(param.name, rng, step_index, param.type.value)
        args[param.name] = value

    return args


def _include_optional(name: str) -> bool:
    lowered = name.lower()
    return lowered in {
        "max_price",
        "currency",
        "guests",
        "quantity",
        "limit",
        "date",
        "time",
        "party_size",
    }


def _grounded_value(name: str, value_context: dict[str, Any]) -> Any | None:
    lowered = name.lower()
    if lowered in value_context:
        return value_context[lowered]
    if lowered.endswith("_id"):
        entity = lowered[:-3]
        if f"{entity}_id" in value_context:
            return value_context[f"{entity}_id"]
        if lowered == "booking_id" and "confirmation_id" in value_context:
            return value_context["confirmation_id"]
        return value_context.get("last_id")
    if lowered in {"id", "item_id", "resource_id"}:
        return value_context.get("last_id")
    return None


def _value_for_name(
    name: str,
    rng: random.Random,
    step_index: int,
    param_type: str = "string",
) -> Any:
    lowered = name.lower()
    if "city" in lowered or "location" in lowered:
        return rng.choice(_CITY_VALUES)
    if lowered == "origin":
        return rng.choice(["Chicago", "New York", "San Francisco", "Dallas"])
    if lowered == "destination":
        return rng.choice(["Paris", "Tokyo", "London", "Seattle"])
    if "date" in lowered or lowered in {"check_in", "check_out"}:
        day = 10 + step_index + rng.randint(0, 8)
        return f"2026-06-{day:02d}"
    if "time" in lowered:
        return rng.choice(["18:30", "09:00", "13:15"])
    if "price" in lowered or "budget" in lowered:
        return 200 + rng.randint(0, 200)
    if "currency" in lowered:
        return "USD"
    if "guest" in lowered or "party" in lowered or "quantity" in lowered:
        return rng.randint(1, 4)
    if "email" in lowered:
        return "customer@example.com"
    if "query" in lowered or "search" in lowered:
        return rng.choice(["analytics dashboard", "team dinner", "travel options"])
    if "status" in lowered:
        return "confirmed"
    if param_type in {"integer", "number"}:
        return rng.randint(1, 10)
    if param_type == "boolean":
        return True
    return rng.choice(["standard", "business", "preferred"])


def _remember_result(
    endpoint: APIEndpoint,
    result: dict[str, Any],
    value_context: dict[str, Any],
) -> None:
    entity = _entity_name(endpoint)

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lowered = key.lower()
                if isinstance(value, (dict, list)):
                    visit(value)
                    continue
                if lowered == "id":
                    value_context["last_id"] = value
                    value_context[f"{entity}_id"] = value
                elif lowered.endswith("_id") or lowered == "confirmation_id":
                    value_context[lowered] = value
                    if lowered == "confirmation_id":
                        value_context["booking_id"] = value
                elif lowered == "name":
                    value_context[f"{entity}_name"] = value
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(result)


def _entity_name(endpoint: APIEndpoint) -> str:
    name = endpoint.endpoint_name.lower()
    for prefix in (
        "search_",
        "find_",
        "list_",
        "get_",
        "create_",
        "book_",
        "reserve_",
        "update_",
        "delete_",
        "cancel_",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in ("_details", "_detail"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.endswith("ies"):
        name = name[:-3] + "y"
    elif name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]
    return name or "item"


def _offline_final_answer(
    chain: ToolChain,
    results_seen: list[dict[str, Any]],
    value_context: dict[str, Any],
) -> str:
    if not results_seen:
        return "I could not complete the task because no tools returned results."

    confirmation = (
        value_context.get("confirmation_id")
        or value_context.get("booking_id")
        or value_context.get("last_id")
        or "available"
    )
    final_endpoint = chain.endpoint_ids[-1] if chain.endpoint_ids else "the requested workflow"
    return (
        f"Done. I completed the workflow through {final_endpoint} and used the returned "
        f"reference {confirmation} for the final step."
    )
