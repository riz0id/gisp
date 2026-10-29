from dataclasses import dataclass

from Syntax.Type.Type import Type

__all__: list[str] = ["ListType"]


@dataclass(frozen=True, slots=True)
class ListType(Type):
    # Written `(@list t)` at the type-level, and as `(list x ...)` at the
    # term-level.

    # The `Type` of the list's elements.
    element: Type
