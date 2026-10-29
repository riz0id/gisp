from dataclasses import dataclass
from typing import Self, override

from Syntax import Symbol
from Syntax.Type.Type import Type

__all__: list[str] = ["SumType"]


@dataclass(frozen=True, slots=True)
class SumType(Type):
    # Written `(@sum t_0 t_1 ... t_n)` at the type-level.

    # A mapping from the `Symbol` of the constructor, to the `Type` of the
    # constructor.
    cons: dict[Symbol, Type]

    def __post_init__(self: Self) -> None:
        if len(self.cons) < 1:
            raise ValueError("SumType.cons must have one or more entries.")
        # Copied so the frozen value cannot be mutated through the caller's
        # mapping, which would silently change its hash.
        object.__setattr__(self, "cons", dict(self.cons))

    @override
    def __hash__(self: Self) -> int:
        return hash(frozenset(self.cons.items()))
