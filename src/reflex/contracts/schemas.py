"""Exploratory finding and persistent JSON Schema registry contracts."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field

from reflex.core.models.base import FrozenModel


class ExploratoryObservation(FrozenModel):
    """An evidence-linked suggestion that is not yet a verified fact."""

    subject: str
    predicate: str
    object: Any
    evidence: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    proposed_classification: str


class SchemaRegistry:
    """Persistent schema registry with explicit, auditable promotion."""

    EXPLORATORY_REF = "reflex.exploratory.observation@1.0"

    def __init__(self, state: Any | None = None):
        self.state = state
        self._schemas: dict[str, dict[str, Any]] = {
            "reflex.observation@1.0": {
                "schema_id": "reflex.observation", "version": "1.0", "status": "registered",
                "json_schema": {"type": "object", "required": ["value", "window_start", "window_end"],
                                "properties": {"value": {}, "window_start": {"type": "string"},
                                               "window_end": {"type": "string"},
                                               "state_snapshot": {"type": "object"}, "quality": {"type": "object"}}},
            },
            "reflex.action-result@1.0": {
                "schema_id": "reflex.action-result", "version": "1.0", "status": "registered",
                "json_schema": {"type": "object"},
            },
            self.EXPLORATORY_REF: {
                "schema_id": "reflex.exploratory.observation", "version": "1.0",
                "status": "exploratory", "json_schema": {
                    "type": "object",
                    "required": ["subject", "predicate", "object", "evidence", "confidence", "provenance", "proposed_classification"],
                    "properties": {
                        "subject": {"type": "string"}, "predicate": {"type": "string"},
                        "object": {}, "evidence": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "provenance": {"type": "object"}, "proposed_classification": {"type": "string"},
                    },
                },
            }
        }
        if state is not None:
            for key, value in state.snapshot("knowledge").items():
                if key.startswith("schema:") and isinstance(value, dict):
                    self._schemas[key.removeprefix("schema:")] = value

    def register(self, schema_id: str, version: str, json_schema: dict[str, Any], *, approved_by: str | None = None) -> str:
        if not schema_id or not version:
            raise ValueError("schema_id and version are required")
        ref = f"{schema_id}@{version}"
        if schema_id.startswith("reflex.exploratory") and not approved_by:
            raise ValueError("exploratory schemas may only be promoted with explicit approval")
        if json_schema.get("type") != "object" or not isinstance(json_schema.get("properties"), dict):
            raise ValueError("schema must describe an object and its properties")
        _validate_schema_shape(json_schema, path="$schema")
        record = {"schema_id": schema_id, "version": version, "status": "registered",
                  "json_schema": json_schema, "approved_by": approved_by}
        prior = self._schemas.get(ref)
        if prior is not None and prior != record:
            raise ValueError(f"schema {ref!r} is already registered with different content")
        self._schemas[ref] = record
        if self.state is not None and prior is None:
            self.state.set("knowledge", f"schema:{ref}", record, f"register schema {ref}")
        return ref

    def promote_exploratory(self, schema_id: str, version: str, json_schema: dict[str, Any], *, approved_by: str) -> str:
        if not approved_by.strip():
            raise ValueError("promotion requires a named approver")
        ref = self.register(schema_id, version, json_schema, approved_by=approved_by)
        if self.state is not None:
            self.state.set("knowledge", f"schema-promotion:{ref}", {
                "schema_ref": ref, "source": self.EXPLORATORY_REF, "approved_by": approved_by,
                "policy_authorized": False,
            }, f"explicit promotion of exploratory schema {ref}")
        return ref

    def schema(self, schema_ref: str) -> dict[str, Any]:
        try:
            return self._schemas[schema_ref]
        except KeyError as error:
            raise KeyError(f"unregistered schema {schema_ref!r}") from error

    def validate(self, schema_ref: str, payload: dict[str, Any]) -> None:
        record = self.schema(schema_ref)
        _validate_json_schema(payload, record["json_schema"], path="$payload")

    def record_exploratory(self, observation: ExploratoryObservation) -> dict[str, Any]:
        payload = observation.model_dump(mode="json")
        self.validate(self.EXPLORATORY_REF, payload)
        return payload


def _validate_json_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    """Validate the small JSON Schema subset used by runtime event contracts."""
    expected = schema.get("type")
    checks: dict[str, Callable[[Any], bool]] = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, (list, tuple)),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected and expected in checks and not checks[expected](value):
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not one of the allowed values")
    if expected == "object":
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(extra))}")
        for key, item in value.items():
            if key in properties:
                _validate_json_schema(item, properties[key], path=f"{path}.{key}")
    elif expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], path=f"{path}[{index}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above the maximum")


def _validate_schema_shape(schema: dict[str, Any], *, path: str) -> None:
    """Reject schema declarations outside the small supported JSON Schema subset."""
    allowed = {"object", "array", "string", "number", "integer", "boolean", "null"}
    expected = schema.get("type")
    if expected is not None and expected not in allowed:
        raise ValueError(f"{path} uses unsupported type {expected!r}")
    if expected == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError(f"{path} has invalid object properties or required fields")
        if any(key not in properties for key in required):
            raise ValueError(f"{path} requires a field not declared in properties")
        for key, nested in properties.items():
            if not isinstance(nested, dict):
                raise ValueError(f"{path}.properties.{key} must be a schema object")
            _validate_schema_shape(nested, path=f"{path}.properties.{key}")
    elif expected == "array" and "items" in schema:
        if not isinstance(schema["items"], dict):
            raise ValueError(f"{path}.items must be a schema object")
        _validate_schema_shape(schema["items"], path=f"{path}.items")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum must be a list")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError(f"{path} has a minimum above its maximum")
