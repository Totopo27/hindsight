"""The metadata schema a knowledge bank extracts, and the model the LLM fills in.

A schema is a flat map of property name -> spec, one for documents and one for chunks:

    {"doc_type": {"type": "string", "values": ["invoice", "contract"],
                  "description": "What kind of document this is"},
     "total":    {"type": "number", "description": "Total amount"},
     "parties":  {"type": "array", "items": "string"},
     "signed":   {"type": "boolean"},
     "signed_on":{"type": "date"},
     "terms":    {"type": "object"}}

``values`` is how classification is expressed: the LLM picks one of the listed values or
leaves the property out. Every property is optional in the extraction model — "not in this
document" has to be expressible, or the model invents values to fill the shape.

A property's ``source`` says who fills it: ``extract`` (the LLM reads the document, the
default) or ``request`` (the caller supplies it on write, and the LLM never sees it). Either
way the value lands in the same object, so a query and a filter cannot tell them apart —
and a write may also supply a value for an ``extract`` property, which skips the call for
that document. A write whose properties cover the whole schema costs no LLM call at all.
"""

from __future__ import annotations

import datetime as _datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, create_model

#: Property types a schema may use. ``date``/``datetime`` are strings the LLM must format
#: as ISO-8601, validated by pydantic on the way in; ``object`` is a free-form JSON object,
#: which is the escape hatch for anything the flat types cannot say.
PropertyType = Literal["string", "integer", "number", "boolean", "date", "datetime", "array", "object"]
PROPERTY_TYPES: tuple[str, ...] = get_args(PropertyType)

#: Element types an ``array`` property may hold. Arrays of objects are not a v1 shape:
#: that is a structured record, which is what v3 is for.
ITEM_TYPES: tuple[str, ...] = ("string", "integer", "number", "boolean", "date", "datetime")

#: Who fills a property in.
SOURCES: tuple[str, ...] = ("extract", "request")

MAX_PROPERTIES = 50
MAX_VALUES = 200

_SCALARS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "date": _datetime.date,
    "datetime": _datetime.datetime,
    "object": dict[str, Any],
}


class MetadataSchemaError(ValueError):
    """A schema a caller cannot have meant: unknown type, empty name, too many values."""


def validate_schema(raw: Any, *, level: str) -> dict[str, dict[str, Any]]:
    """Check one level (document or chunk) of a schema and return it normalised."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MetadataSchemaError(f"{level} schema must be an object of property name -> spec")
    if len(raw) > MAX_PROPERTIES:
        raise MetadataSchemaError(f"{level} schema has {len(raw)} properties; at most {MAX_PROPERTIES}")

    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise MetadataSchemaError(f"{level} schema has a property with an empty name")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise MetadataSchemaError(f"property {name!r} must be alphanumeric with _ or -")
        if not isinstance(spec, dict):
            raise MetadataSchemaError(f"property {name!r} must be an object, e.g. {{'type': 'string'}}")

        property_type = spec.get("type", "string")
        if property_type not in PROPERTY_TYPES:
            raise MetadataSchemaError(f"property {name!r} has unknown type {property_type!r}; one of {PROPERTY_TYPES}")

        source = spec.get("source", "extract")
        if source not in SOURCES:
            raise MetadataSchemaError(f"property {name!r}: source must be one of {SOURCES}")
        normalised: dict[str, Any] = {"type": property_type, "source": source}
        description = spec.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise MetadataSchemaError(f"property {name!r}: description must be a string")
            normalised["description"] = description

        if property_type == "array":
            item_type = spec.get("items", "string")
            if item_type not in ITEM_TYPES:
                raise MetadataSchemaError(f"property {name!r}: items must be one of {ITEM_TYPES}")
            normalised["items"] = item_type

        values = spec.get("values")
        if values is not None:
            if property_type in ("object",):
                raise MetadataSchemaError(f"property {name!r}: an object property cannot have fixed values")
            if not isinstance(values, list) or not values:
                raise MetadataSchemaError(f"property {name!r}: values must be a non-empty list")
            if len(values) > MAX_VALUES:
                raise MetadataSchemaError(f"property {name!r}: at most {MAX_VALUES} values")
            if len({str(v) for v in values}) != len(values):
                raise MetadataSchemaError(f"property {name!r}: values must be unique")
            normalised["values"] = values

        out[name] = normalised
    return out


def _annotation(spec: dict[str, Any]) -> Any:
    """The Python type for one property, as the extraction model should see it."""
    property_type = spec["type"]
    values = spec.get("values")
    if property_type == "array":
        inner = Literal[tuple(values)] if values else _SCALARS[spec.get("items", "string")]  # type: ignore[valid-type]
        return list[inner]  # type: ignore[valid-type]
    if values:
        # A fixed set of values IS the classification: let the schema enforce it rather
        # than asking the model nicely and validating afterwards.
        return Literal[tuple(values)]  # type: ignore[valid-type]
    return _SCALARS[property_type]


def extraction_model(schema: dict[str, dict[str, Any]], *, name: str) -> type[BaseModel] | None:
    """Build the pydantic model the LLM fills for this schema, or None if it is empty.

    Every field is optional and defaults to None: a document that says nothing about a
    property must be able to come back with nothing for it.
    """
    if not schema:
        return None
    fields: dict[str, Any] = {}
    for property_name, spec in schema.items():
        description = spec.get("description") or f"{property_name}, if the text says it"
        if spec.get("values"):
            description += ". Use one of the allowed values, or leave it out."
        fields[property_name] = (
            _annotation(spec) | None,
            Field(default=None, description=description),
        )
    return create_model(name, **fields)


def jsonable(values: dict[str, Any]) -> dict[str, Any]:
    """Extracted values as JSON: dates become ISO strings, empty answers are dropped."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or value == [] or value == {}:
            continue
        if isinstance(value, _datetime.datetime | _datetime.date):
            out[key] = value.isoformat()
        elif isinstance(value, list):
            out[key] = [v.isoformat() if isinstance(v, _datetime.datetime | _datetime.date) else v for v in value]
        else:
            out[key] = value
    return out


def extract_only(schema: dict[str, dict[str, Any]], supplied: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The properties the LLM still has to read, given what the write already supplied.

    A ``request`` property is never extracted, and a value supplied for an ``extract``
    property overrides it for that document — so a caller who already knows the answer does
    not pay for a call to rediscover it.
    """
    supplied_names = set(supplied or {})
    return {
        name: spec
        for name, spec in schema.items()
        if spec.get("source", "extract") == "extract" and name not in supplied_names
    }


def validate_values(schema: dict[str, dict[str, Any]], values: dict[str, Any], *, level: str) -> dict[str, Any]:
    """Coerce caller-supplied property values against the schema, or say what is wrong.

    The same model the LLM fills validates them, so a supplied value and an extracted one
    are held to exactly one definition of the property — including its allowed values.
    """
    if not values:
        return {}
    unknown = sorted(set(values) - set(schema))
    if unknown:
        raise MetadataSchemaError(f"{level} properties not in the schema: {unknown}")
    model = extraction_model({name: schema[name] for name in values}, name="SuppliedMetadata")
    if model is None:
        return {}
    try:
        validated = model.model_validate(values)
    except Exception as e:
        raise MetadataSchemaError(f"{level} properties do not match the schema: {e}") from e
    return jsonable(validated.model_dump())
