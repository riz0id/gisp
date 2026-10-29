from dataclasses import dataclass
from typing import Self, override

from Syntax.Expr.Base import Expr


@dataclass(frozen=True, slots=True)
class Nil(Expr):
    def __add__(self: Self, other: Nil) -> Nil:
        return Nil()

    @override
    def __eq__(self: Self, other: object) -> bool:
        return True if isinstance(other, Nil) else False

    def __int__(self: Self) -> int:
        return 0

    @override
    def __str__(self: Self) -> str:
        return "(prod#)"

    @staticmethod
    def bounds() -> tuple[Nil, Nil]:
        return (Nil(), Nil())

    @staticmethod
    def empty() -> Nil:
        return Nil()
