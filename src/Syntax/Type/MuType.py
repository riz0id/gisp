from dataclasses import dataclass

from Syntax import Symbol
from Syntax.Type.Type import Type

__all__: list[str] = ["MuType"]


@dataclass(frozen=True, slots=True)
class MuType(Type):
    # Written `(@mu a t)` at the type-level: the type `t` with `a` standing for
    # the whole of `(@mu a t)` within it. Equirecursive, and with no term-level
    # fold or unfold — a recursive type is taken apart as its unfolding, and
    # `TypeCheck` does that wherever it looks inside a type.

    # The type variable bound in the body.
    variable: Symbol

    # The `Type` the variable stands for this whole type within.
    body: Type
