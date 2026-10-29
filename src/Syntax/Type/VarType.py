from dataclasses import dataclass

from Syntax import Symbol
from Syntax.Type.Type import Type

__all__: list[str] = ["VarType"]


@dataclass(frozen=True, slots=True)
class VarType(Type):
    # A type variable, written as its own name at the type-level: the `a` in
    # `(@mu a t)`. It stands for the `@mu` that binds it and classifies no
    # terms of its own, so nothing reads a variable as a type: `unfold`
    # removes it, and the queries in `Syntax.Type` answer for whatever the
    # context binds it to.

    binder: Symbol
