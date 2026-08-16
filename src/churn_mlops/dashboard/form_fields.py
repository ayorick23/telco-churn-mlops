"""Deriva la especificación del formulario de predicción directamente de
`CustomerFeatures` (Pydantic, `serving/api/schemas.py`) en vez de declarar los
37 campos una segunda vez a mano en el dashboard — misma fuente de verdad que
usa la API para validar el request."""

import types
import typing
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from churn_mlops.serving.api.schemas import CustomerFeatures

FieldKind = Literal["select", "int", "float", "text"]

_UNION_ORIGINS = (typing.Union, types.UnionType)


@dataclass
class FieldSpec:
    name: str  # atributo Python de CustomerFeatures (snake_case)
    label: str  # alias = nombre de columna original (con espacios)
    kind: FieldKind
    choices: tuple[str, ...] = ()
    nullable: bool = False


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in _UNION_ORIGINS:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0], True
    return annotation, False


def build_field_specs() -> list[FieldSpec]:
    specs = []
    for name, field in CustomerFeatures.model_fields.items():
        annotation, nullable = _unwrap_optional(field.annotation)
        label = str(field.alias) if field.alias else name

        if get_origin(annotation) is Literal:
            specs.append(
                FieldSpec(name, label, "select", get_args(annotation), nullable)
            )
        elif annotation is int:
            specs.append(FieldSpec(name, label, "int", nullable=nullable))
        elif annotation is float:
            specs.append(FieldSpec(name, label, "float", nullable=nullable))
        else:
            specs.append(FieldSpec(name, label, "text", nullable=nullable))
    return specs
