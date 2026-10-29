from dataclasses import dataclass
from typing import Self, override

from Syntax.Expr.Base import Expr


@dataclass(frozen=True, slots=True)
class Boolean(Expr):
    value: bool

    def __add__(self: Self, other: Boolean) -> Boolean:
        return Boolean(value=self.value or other.value)

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.value == other.value if isinstance(other, Boolean) else False

    def __int__(self: Self) -> int:
        return 1 if self.value else 0

    @override
    def __str__(self: Self) -> str:
        return "#t" if self.value else "#f"

    @staticmethod
    def bounds() -> tuple[Boolean, Boolean]:
        return (Boolean.false(), Boolean.true())

    @staticmethod
    def true() -> Boolean:
        return Boolean(True)

    @staticmethod
    def false() -> Boolean:
        return Boolean(False)
