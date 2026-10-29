from dataclasses import dataclass

from Syntax.Type.Type import Type

__all__: list[str] = ["CharType"]


@dataclass(frozen=True, slots=True)
class CharType(Type):
    # Written `@char` at the type-level, and as a character literal at the
    # term-level.
    pass
