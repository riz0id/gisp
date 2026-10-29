from dataclasses import dataclass

from Syntax.Type.Type import Type

__all__: list[str] = ["VariadicStarType", "VariadicType"]


@dataclass(frozen=True, slots=True)
class VariadicType(Type):
    # Written `(@variadic t)` at the type-level: zero or more arguments, each
    # of the one type `t`. A variadic type describes the argument run of a
    # function type; it classifies no terms of its own.

    # The `Type` of every argument in the run.
    argument: Type


@dataclass(frozen=True, slots=True)
class VariadicStarType(Type):
    # Written `(@variadic* t)` at the type-level: one or more arguments, each
    # of the one type `t`. A variadic type describes the argument run of a
    # function type; it classifies no terms of its own.

    # The `Type` of every argument in the run.
    argument: Type
