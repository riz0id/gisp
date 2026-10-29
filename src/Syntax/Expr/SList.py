from dataclasses import dataclass
from typing import Callable, Iterator, Self, override

from Syntax.Expr.Base import Expr
from Syntax.Expr.Char import Char
from Syntax.Expr.Symbol import Symbol
from Syntax.Scope import Scope


@dataclass(frozen=True, slots=True)
class SList(Expr):
    items: list[Expr]

    def __add__(self: Self, other: SList) -> SList:
        return SList(items=self.items + other.items)

    def __iter__(self: Self) -> Iterator[Expr]:
        return iter(self.items)

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.items == other.items if isinstance(other, SList) else False

    @override
    def __str__(self: Self) -> str:
        # A non-empty list of characters is a string and prints in string
        # notation; the empty string is the empty list and prints as `()`.
        chars: list[Char] = [item for item in self.items if isinstance(item, Char)]

        if chars and len(chars) == len(self.items):
            escaped: str = (
                "".join(chr(char.value) for char in chars)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\r", "\\r")
            )
            return f'"{escaped}"'

        return f"({' '.join(str(item) for item in self.items)})"

    @staticmethod
    def empty() -> SList:
        return SList(items=[])

    @override
    def with_scopes(self: Self, op: Callable[[set[Scope]], set[Scope]]) -> None:
        match self.items:
            case [Symbol() as head, *_] if head.identifier == "quote":
                pass
            case _:
                for item in self:
                    item.with_scopes(op)
