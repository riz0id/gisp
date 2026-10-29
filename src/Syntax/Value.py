from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from Syntax.Expr import Expr

if TYPE_CHECKING:
    from Evaluate.Environment import Environment


@dataclass(frozen=True, slots=True)
class Closure:
    parameters: tuple[str, ...]
    body: Expr
    environment: Environment

    @override
    def __str__(self: Self) -> str:
        return f"#<closure {len(self.parameters)}>"


@dataclass(frozen=True, slots=True)
class Primitive:
    # Arity lives on the value so application sites check it the same way
    # they check closures; `operate` therefore assumes the exact count.
    name: str
    arity: int
    operate: Callable[[list[Value]], Value]

    @override
    def __str__(self: Self) -> str:
        return f"#<primitive {self.name}>"


type Value = Expr | Closure | Primitive
