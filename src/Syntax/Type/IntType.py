from dataclasses import dataclass

from Syntax.Type.Type import Type

__all__: list[str] = ["IntType"]


@dataclass(frozen=True, slots=True)
class IntType(Type):
    # Written `@int` at the type-level, and as an integer literal at the
    # term-level.
    pass
