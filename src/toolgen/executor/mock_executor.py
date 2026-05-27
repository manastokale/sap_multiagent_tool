"""Offline mock tool executor.

Produces mock responses consistent with endpoint schemas while maintaining
session state so that multi-step traces chain correctly. Key design insight:
if a hotel search returns htl_881, a subsequent booking must accept that ID.

Session state tracks:
  - Generated entity IDs (keyed by entity type)
  - Entity details (for subsequent lookups)
  - Operation results (bookings, confirmations)
"""

from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime
from typing import Any

from toolgen.models import APIEndpoint, ParameterType

logger = logging.getLogger(__name__)

# Common entity nouns and their ID prefixes
_ENTITY_PREFIXES: dict[str, str] = {
    "hotel": "htl",
    "flight": "flt",
    "restaurant": "rst",
    "car": "car",
    "booking": "bk",
    "reservation": "rsv",
    "order": "ord",
    "user": "usr",
    "product": "prd",
    "event": "evt",
    "ticket": "tkt",
    "review": "rev",
    "payment": "pmt",
    "location": "loc",
    "route": "rte",
    "movie": "mov",
    "song": "sng",
    "album": "alb",
    "artist": "art",
    "recipe": "rcp",
    "news": "nws",
    "weather": "wthr",
    "stock": "stk",
    "crypto": "cry",
}

# Sample data for realistic responses
_SAMPLE_NAMES: dict[str, list[str]] = {
    "hotel": ["Grand Palace Hotel", "Seaside Resort", "Urban Boutique Inn",
              "Mountain View Lodge", "City Center Suites"],
    "flight": ["United UA-2431", "Delta DL-891", "Emirates EK-305",
               "Lufthansa LH-720", "BA-142"],
    "restaurant": ["Le Petit Bistro", "Sakura Japanese", "The Golden Fork",
                    "Mama's Kitchen", "Azure Rooftop"],
    "product": ["Premium Widget Pro", "Ultra Gadget X", "Smart Device 3000",
                "Classic Edition", "Essential Kit"],
    "city": ["Paris", "Tokyo", "New York", "London", "Sydney", "Berlin",
             "Rome", "Barcelona", "Singapore", "Dubai"],
    "default": ["Item Alpha", "Item Beta", "Item Gamma", "Item Delta", "Item Epsilon"],
}


def _generate_id(entity_type: str, seed_data: str, rng: random.Random) -> str:
    """Generate a deterministic but realistic-looking entity ID."""
    prefix = _ENTITY_PREFIXES.get(entity_type, "itm")
    hash_part = hashlib.md5(
        f"{entity_type}:{seed_data}:{rng.randint(0, 9999)}".encode()
    ).hexdigest()[:4]
    return f"{prefix}_{hash_part}"


def _infer_entity_type(endpoint: APIEndpoint) -> str:
    """Infer the entity type from the endpoint name/description."""
    name_lower = endpoint.endpoint_name.lower()
    desc_lower = endpoint.description.lower()
    combined = f"{name_lower} {desc_lower}"

    for entity in _ENTITY_PREFIXES:
        if entity in combined:
            return entity

    return "item"


def _get_sample_name(entity_type: str, rng: random.Random) -> str:
    """Get a realistic sample name for an entity type."""
    names = _SAMPLE_NAMES.get(entity_type, _SAMPLE_NAMES["default"])
    return rng.choice(names)


def _generate_value_for_type(param_type: ParameterType, rng: random.Random) -> Any:
    """Generate a realistic mock value for a given parameter type."""
    if param_type == ParameterType.STRING:
        return rng.choice(["sample_value", "test_string", "example_text", "data_point"])
    elif param_type == ParameterType.NUMBER:
        return round(rng.uniform(1.0, 1000.0), 2)
    elif param_type == ParameterType.INTEGER:
        return rng.randint(1, 100)
    elif param_type == ParameterType.BOOLEAN:
        return rng.choice([True, False])
    elif param_type == ParameterType.ARRAY:
        return []
    elif param_type == ParameterType.OBJECT:
        return {}
    return "unknown"


class MockExecutor:
    """Schema-aware mock tool executor with session state.

    Maintains state across calls within a conversation so that multi-step
    traces chain correctly (IDs, entities, results persist).
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._session_state: dict[str, Any] = {}
        # Tracks generated entities: {entity_type: [{id, name, ...}]}
        self._entities: dict[str, list[dict[str, Any]]] = {}
        # Tracks operations: [{operation, entity_id, result}]
        self._operations: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset session state for a new conversation."""
        self._session_state.clear()
        self._entities.clear()
        self._operations.clear()

    def execute(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a mock tool call and return a schema-consistent response.

        The response is:
        1. Consistent with the endpoint's implied schema
        2. Linked to session state (IDs chain correctly)
        3. Deterministic given the same seed + arguments
        """
        entity_type = _infer_entity_type(endpoint)
        verb = self._classify_operation(endpoint)

        logger.debug(
            "Mock executing %s (verb=%s, entity=%s, args=%s)",
            endpoint.endpoint_id, verb, entity_type, arguments,
        )

        if verb == "search":
            return self._handle_search(endpoint, arguments, entity_type)
        elif verb == "get":
            return self._handle_get(endpoint, arguments, entity_type)
        elif verb == "create":
            return self._handle_create(endpoint, arguments, entity_type)
        elif verb == "update":
            return self._handle_update(endpoint, arguments, entity_type)
        elif verb == "delete":
            return self._handle_delete(endpoint, arguments, entity_type)
        else:
            return self._handle_generic(endpoint, arguments, entity_type)

    def _classify_operation(self, endpoint: APIEndpoint) -> str:
        """Classify the endpoint into an operation type."""
        name = endpoint.endpoint_name.lower()
        if any(v in name for v in ("search", "find", "list", "query", "browse", "get_all")):
            return "search"
        elif any(v in name for v in ("get", "retrieve", "fetch", "detail", "show", "view")):
            return "get"
        elif any(v in name for v in ("delete", "remove", "cancel", "revoke")):
            return "delete"
        elif any(v in name for v in ("create", "add", "new", "register", "book", "reserve",
                                      "post", "submit")):
            return "create"
        elif any(v in name for v in ("update", "modify", "edit", "change", "patch", "set")):
            return "update"
        return "generic"

    def _handle_search(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle search/list operations — returns list of entities."""
        num_results = self._rng.randint(2, 5)
        results = []

        for i in range(num_results):
            seed_data = f"{endpoint.endpoint_id}:{i}:{str(sorted(arguments.items()))}"
            entity_id = _generate_id(entity_type, seed_data, self._rng)
            entity_name = _get_sample_name(entity_type, self._rng)

            entity = {
                "id": entity_id,
                "name": entity_name,
            }

            # Add type-specific fields
            if entity_type == "hotel":
                entity.update({
                    "price": self._rng.randint(50, 500),
                    "rating": round(self._rng.uniform(3.0, 5.0), 1),
                    "location": self._rng.choice(_SAMPLE_NAMES["city"]),
                })
            elif entity_type == "flight":
                entity.update({
                    "price": self._rng.randint(100, 2000),
                    "departure": "08:30",
                    "arrival": "14:45",
                    "duration": f"{self._rng.randint(1, 12)}h {self._rng.randint(0, 59)}m",
                })
            elif entity_type == "restaurant":
                entity.update({
                    "cuisine": self._rng.choice(["Italian", "Japanese", "French", "Mexican"]),
                    "price_range": self._rng.choice(["$", "$$", "$$$"]),
                    "rating": round(self._rng.uniform(3.5, 5.0), 1),
                })
            elif entity_type == "product":
                entity.update({
                    "price": round(self._rng.uniform(10.0, 999.0), 2),
                    "in_stock": self._rng.choice([True, True, True, False]),
                    "category": endpoint.category,
                })
            else:
                entity.update({
                    "description": f"A {entity_type} matching your search",
                    "score": round(self._rng.uniform(0.5, 1.0), 2),
                })

            results.append(entity)

        # Store in session state
        if entity_type not in self._entities:
            self._entities[entity_type] = []
        self._entities[entity_type].extend(results)

        return {
            "results": results,
            "total_count": self._rng.randint(num_results, num_results * 10),
            "page": 1,
        }

    def _handle_get(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle get/detail operations — returns single entity."""
        if entity_type == "weather":
            return {
                "temperature": self._rng.randint(-10, 40),
                "condition": self._rng.choice(["sunny", "cloudy", "rainy", "snowy"]),
                "humidity": self._rng.randint(20, 90),
                "wind_speed": round(self._rng.uniform(0, 50), 1),
            }

        # Try to find the entity by ID in session state
        entity_id = None
        for key in ("id", f"{entity_type}_id", "item_id", "resource_id"):
            if key in arguments:
                entity_id = arguments[key]
                break

        # Look up in session state
        if entity_id and entity_type in self._entities:
            for entity in self._entities[entity_type]:
                if entity.get("id") == entity_id:
                    return {**entity, "details": f"Full details for {entity.get('name', entity_id)}"}

        # Generate a new entity if not found
        entity = {
            "id": entity_id or _generate_id(entity_type, str(arguments), self._rng),
            "name": _get_sample_name(entity_type, self._rng),
            "description": f"Detailed {entity_type} information",
            "created_at": datetime.now().isoformat(),
        }
        return entity

    def _handle_create(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle create/book operations — returns confirmation."""
        entity_id = _generate_id(entity_type, str(arguments), self._rng)
        confirmation_id = _generate_id("booking", str(arguments), self._rng)

        result = {
            f"{entity_type}_id": entity_id,
            "confirmation_id": confirmation_id,
            "status": "confirmed",
            "created_at": datetime.now().isoformat(),
        }

        # Include relevant arguments in the result
        for key, value in arguments.items():
            if key not in result:
                result[key] = value

        self._operations.append({
            "operation": "create",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "result": result,
        })

        return result

    def _handle_update(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle update operations — returns updated entity."""
        entity_id = arguments.get("id") or arguments.get(f"{entity_type}_id", "unknown")

        result = {
            "id": entity_id,
            "status": "updated",
            "updated_at": datetime.now().isoformat(),
            "changes": {k: v for k, v in arguments.items() if k != "id"},
        }

        self._operations.append({
            "operation": "update",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "result": result,
        })

        return result

    def _handle_delete(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle delete/cancel operations."""
        entity_id = arguments.get("id") or arguments.get(f"{entity_type}_id", "unknown")

        result = {
            "id": entity_id,
            "status": "deleted",
            "deleted_at": datetime.now().isoformat(),
        }

        self._operations.append({
            "operation": "delete",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "result": result,
        })

        return result

    def _handle_generic(
        self,
        endpoint: APIEndpoint,
        arguments: dict[str, Any],
        entity_type: str,
    ) -> dict[str, Any]:
        """Handle generic operations with schema-aware response generation."""
        result: dict[str, Any] = {
            "status": "success",
            "endpoint": endpoint.endpoint_name,
        }

        # Generate response fields based on endpoint name heuristics
        name_lower = endpoint.endpoint_name.lower()
        if "weather" in name_lower or "forecast" in name_lower:
            result.update({
                "temperature": self._rng.randint(-10, 40),
                "condition": self._rng.choice(["sunny", "cloudy", "rainy", "snowy"]),
                "humidity": self._rng.randint(20, 90),
                "wind_speed": round(self._rng.uniform(0, 50), 1),
            })
        elif "price" in name_lower or "quote" in name_lower:
            result.update({
                "price": round(self._rng.uniform(1.0, 10000.0), 2),
                "currency": self._rng.choice(["USD", "EUR", "GBP"]),
            })
        elif "review" in name_lower or "rating" in name_lower:
            result.update({
                "average_rating": round(self._rng.uniform(1.0, 5.0), 1),
                "total_reviews": self._rng.randint(1, 5000),
                "reviews": [
                    {"text": "Great experience!", "rating": 5},
                    {"text": "Decent but could improve", "rating": 3},
                ],
            })
        else:
            # Fallback: echo back arguments with a response wrapper
            result["data"] = arguments

        return result

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session state."""
        return {
            "entities": {
                entity_type: [e.get("id") for e in entities]
                for entity_type, entities in self._entities.items()
            },
            "operations": len(self._operations),
            "operation_types": [op["operation"] for op in self._operations],
        }
