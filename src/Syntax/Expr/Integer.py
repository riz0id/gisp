from dataclasses import dataclass
from typing import Self, override

from Syntax.Expr.Base import Expr


@dataclass(frozen=True, slots=True)
class Integer(Expr):
    value: int

    def __add__(self: Self, other: Integer) -> Integer:
        return Integer(value=self.value + other.value)

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.value == other.value if isinstance(other, Integer) else False

    def __int__(self: Self) -> int:
        return self.value

    @override
    def __str__(self: Self) -> str:
        return str(self.value)

    @staticmethod
    def bounds() -> tuple[int, int]:
        return (-(2**31), 2**31 - 1)

    @staticmethod
    def zero() -> Integer:
        return Integer(value=0)
