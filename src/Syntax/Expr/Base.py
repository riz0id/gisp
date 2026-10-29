from abc import ABC, abstractmethod
from typing import Callable, Self, override

from Syntax import Scope

__all__: list[str] = ["Expr"]


class Expr(ABC):
    __slots__ = ()

    @override
    @abstractmethod
    def __str__(self: Self) -> str: ...

    @override
    @abstractmethod
    def __eq__(self: Self, other: object) -> bool: ...

    def with_scopes(self: Self, op: Callable[[set[Scope]], set[Scope]]) -> None:
        return None

    def add(self: Self, scope: Scope | set[Scope] | frozenset[Scope]) -> None:
        added: set[Scope] | frozenset[Scope] = (
            scope if isinstance(scope, set | frozenset) else {scope}
        )
        self.with_scopes(op=lambda scopes: scopes | added)

    def flip(self: Self, scope: Scope | set[Scope] | frozenset[Scope]) -> None:
        flipped: set[Scope] | frozenset[Scope] = (
            scope if isinstance(scope, set | frozenset) else {scope}
        )
        self.with_scopes(op=lambda scopes: scopes ^ flipped)
