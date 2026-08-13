"""Small deterministic JSON Schema validator for frozen MCP tool contracts.

The benchmark cannot rely on an optional runtime package at its trust boundary.
This module implements the assertion vocabulary emitted by the AIVideo MCP
Pydantic schemas and fails closed when a schema uses an unsupported keyword.
It is intentionally general enough for every tool in a frozen surface rather
than encoding tool-specific argument rules.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any


PlaceholderPredicate = Callable[[Any], bool]

_ANNOTATION_KEYS = frozenset(
    {
        "$id",
        "$schema",
        "default",
        "deprecated",
        "description",
        "discriminator",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_ASSERTION_KEYS = frozenset(
    {
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "prefixItems",
        "properties",
        "required",
        "type",
        "uniqueItems",
    }
)
_STRUCTURAL_KEYS = frozenset({"$defs", "definitions"})
_SUPPORTED_KEYS = _ANNOTATION_KEYS | _ASSERTION_KEYS | _STRUCTURAL_KEYS


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same(left: Any, right: Any) -> bool:
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        return False


def _resolve_pointer(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise ValueError(f"only local JSON Schema references are supported: {reference}")
    current: Any = root
    for encoded in reference[2:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"unresolvable local JSON Schema reference: {reference}")
        current = current[part]
    return current


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(float(instance))
        )
    if expected == "string":
        return isinstance(instance, str)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "object":
        return isinstance(instance, Mapping)
    return False


def _schema_keyword_errors(schema: Any, *, path: str = "$schema") -> list[str]:
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, Mapping):
        return [f"{path} must be an object or boolean schema"]
    errors = [
        f"{path} uses unsupported JSON Schema keyword {key!r}"
        for key in schema
        if key not in _SUPPORTED_KEYS
    ]
    for container in ("$defs", "definitions", "properties"):
        children = schema.get(container)
        if children is not None and not isinstance(children, Mapping):
            errors.append(f"{path}.{container} must be an object")
        elif isinstance(children, Mapping):
            for name, child in children.items():
                errors.extend(
                    _schema_keyword_errors(child, path=f"{path}.{container}.{name}")
                )
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        errors.extend(
            _schema_keyword_errors(additional, path=f"{path}.additionalProperties")
        )
    items = schema.get("items")
    if isinstance(items, (Mapping, bool)):
        errors.extend(_schema_keyword_errors(items, path=f"{path}.items"))
    for container in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(container)
        if children is not None and not isinstance(children, list):
            errors.append(f"{path}.{container} must be a list")
        elif isinstance(children, list):
            for index, child in enumerate(children):
                errors.extend(
                    _schema_keyword_errors(child, path=f"{path}.{container}[{index}]")
                )
    if "not" in schema:
        errors.extend(_schema_keyword_errors(schema["not"], path=f"{path}.not"))
    return errors


def validate_schema_vocabulary(schema: Any) -> list[str]:
    """Reject schemas whose assertion semantics this validator cannot enforce."""

    return _schema_keyword_errors(schema)


def validate_json_schema_instance(
    instance: Any,
    schema: Any,
    *,
    root_schema: Mapping[str, Any] | None = None,
    path: str = "$",
    placeholder_predicate: PlaceholderPredicate | None = None,
) -> list[str]:
    """Return deterministic validation errors for one instance and schema."""

    if placeholder_predicate is not None and placeholder_predicate(instance):
        return []
    if schema is True:
        return []
    if schema is False:
        return [f"{path} is rejected by the false schema"]
    if not isinstance(schema, Mapping):
        return [f"{path} schema is malformed"]
    root = root_schema or schema
    if "$ref" in schema:
        try:
            resolved = _resolve_pointer(root, str(schema["$ref"]))
        except ValueError as error:
            return [f"{path} {error}"]
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        errors = validate_json_schema_instance(
            instance,
            resolved,
            root_schema=root,
            path=path,
            placeholder_predicate=placeholder_predicate,
        )
        if siblings:
            errors.extend(
                validate_json_schema_instance(
                    instance,
                    siblings,
                    root_schema=root,
                    path=path,
                    placeholder_predicate=placeholder_predicate,
                )
            )
        return errors

    errors: list[str] = []
    for keyword in ("allOf",):
        for child in schema.get(keyword, []):
            errors.extend(
                validate_json_schema_instance(
                    instance,
                    child,
                    root_schema=root,
                    path=path,
                    placeholder_predicate=placeholder_predicate,
                )
            )
    for keyword in ("anyOf", "oneOf"):
        if keyword not in schema:
            continue
        branch_results = [
            validate_json_schema_instance(
                instance,
                child,
                root_schema=root,
                path=path,
                placeholder_predicate=placeholder_predicate,
            )
            for child in schema[keyword]
        ]
        passes = sum(not branch for branch in branch_results)
        if (keyword == "anyOf" and passes == 0) or (keyword == "oneOf" and passes != 1):
            errors.append(f"{path} does not satisfy {keyword} ({passes} branches matched)")
    if "not" in schema:
        rejected = validate_json_schema_instance(
            instance,
            schema["not"],
            root_schema=root,
            path=path,
            placeholder_predicate=placeholder_predicate,
        )
        if not rejected:
            errors.append(f"{path} matches a prohibited schema")

    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if isinstance(expected_types, list) and not any(
        isinstance(expected, str) and _type_matches(instance, expected)
        for expected in expected_types
    ):
        errors.append(f"{path} has wrong type; expected {expected_types}")
        return errors
    if "const" in schema and not _same(instance, schema["const"]):
        errors.append(f"{path} does not equal const value")
    if "enum" in schema and not any(_same(instance, value) for value in schema["enum"]):
        errors.append(f"{path} is outside enum")

    if isinstance(instance, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path} is missing required property {key!r}")
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, Mapping) else {}
        for key, value in instance.items():
            child_path = f"{path}.{key}"
            if key in properties:
                errors.extend(
                    validate_json_schema_instance(
                        value,
                        properties[key],
                        root_schema=root,
                        path=child_path,
                        placeholder_predicate=placeholder_predicate,
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path} has unknown property {key!r}")
            elif isinstance(schema.get("additionalProperties"), Mapping):
                errors.extend(
                    validate_json_schema_instance(
                        value,
                        schema["additionalProperties"],
                        root_schema=root,
                        path=child_path,
                        placeholder_predicate=placeholder_predicate,
                    )
                )
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path} has fewer than minProperties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            errors.append(f"{path} has more than maxProperties")

    if isinstance(instance, list):
        prefix = schema.get("prefixItems", [])
        for index, child in enumerate(prefix[: len(instance)]):
            errors.extend(
                validate_json_schema_instance(
                    instance[index],
                    child,
                    root_schema=root,
                    path=f"{path}[{index}]",
                    placeholder_predicate=placeholder_predicate,
                )
            )
        items = schema.get("items")
        if isinstance(items, (Mapping, bool)):
            start = len(prefix)
            for index, value in enumerate(instance[start:], start=start):
                errors.extend(
                    validate_json_schema_instance(
                        value,
                        items,
                        root_schema=root,
                        path=f"{path}[{index}]",
                        placeholder_predicate=placeholder_predicate,
                    )
                )
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path} has fewer than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path} has more than maxItems")
        if schema.get("uniqueItems") is True:
            canonical = [_canonical(value) for value in instance]
            if len(canonical) != len(set(canonical)):
                errors.append(f"{path} violates uniqueItems")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path} is longer than maxLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            errors.append(f"{path} does not match pattern")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        value = float(instance)
        if "minimum" in schema and value < float(schema["minimum"]):
            errors.append(f"{path} is below minimum")
        if "maximum" in schema and value > float(schema["maximum"]):
            errors.append(f"{path} is above maximum")
        if "exclusiveMinimum" in schema and value <= float(schema["exclusiveMinimum"]):
            errors.append(f"{path} is not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and value >= float(schema["exclusiveMaximum"]):
            errors.append(f"{path} is not below exclusiveMaximum")
        if "multipleOf" in schema:
            divisor = float(schema["multipleOf"])
            quotient = value / divisor
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                errors.append(f"{path} is not a multipleOf value")
    return errors
