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
    #
    # `None` where the primitive reads any number of arguments, which is the
    # reading `ConstructorArity` already has of a constructor written with an
    # ellipsis. `prod#` is the one: it builds a product of whatever width it is
    # applied at, and the type checker has fixed that width against the product
    # being constructed before anything runs.
    name: str
    arity: int | None
    operate: Callable[[list[Value]], Value]

    @override
    def __str__(self: Self) -> str:
        return f"#<primitive {self.name}>"


@dataclass(frozen=True, slots=True)
class Residual:
    # A term that reduced no further: it needed the value of a free variable,
    # and a free variable has none. The answer is in weak head normal form, so
    # nothing below the stuck head has been reduced either.
    #
    # A sort of its own rather than the bare `Expr` the union already admits,
    # because the two are not the same answer: a quoted list evaluates *to* an
    # `SList`, and a stuck one is an `SList` that was never reduced. Only the
    # marker tells a primitive which of the two it has been handed.
    form: Expr

    @override
    def __str__(self: Self) -> str:
        # Printed as what it is: the term, with the free variables still in it.
        return str(self.form)


type Value = Expr | Closure | Primitive | Residual
