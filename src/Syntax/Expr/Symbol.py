from dataclasses import dataclass, field
from typing import Callable, Self, override

from Syntax.Expr.Base import Expr
from Syntax.Scope import Scope


@dataclass(frozen=False, slots=True)
class Symbol(Expr):
    identifier: str
    scopes: set[Scope] = field(default_factory=set)

    def __post_init__(self: Self) -> None:
        if self.identifier == "":
            assert self.scopes == set()

    @override
    def __eq__(self: Self, other: object) -> bool:
        return (
            self.identifier == other.identifier and self.scopes == other.scopes
            if isinstance(other, Symbol)
            else False
        )

    @override
    def __hash__(self: Self) -> int:
        return hash((self.identifier, frozenset(self.scopes)))

    @override
    def __str__(self: Self) -> str:
        return self.identifier

    @staticmethod
    def toplevel() -> Symbol:
        # The special symbol that is assigned to the "top-level" scope. The
        # top-level scope exists, but:
        #   1. You cannot refer to it in code. Note this is consistent with the
        #      `synonyms` value for the top-level since no name is a valid
        #      parser production for `Symbol`.
        #   2. It can be referred to by all other scopes. Note this is
        #      consistent with the property that `set()` is a subset of all
        #      other sets (of scopes).
        return Symbol(identifier="", scopes=set())

    @override
    def with_scopes(self: Self, op: Callable[[set[Scope]], set[Scope]]) -> None:
        self.scopes = op(self.scopes)
