from dataclasses import dataclass
from typing import Self, override

from Syntax.Expr.Base import Expr


@dataclass(frozen=True, slots=True)
class Char(Expr):
    value: int

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.value == other.value if isinstance(other, Char) else False

    @override
    def __str__(self: Self) -> str:
        character: str = chr(self.value)
        if character.isprintable() and not character.isspace():
            return f"#\\{character}"
        else:
            return f"#\\x{self.value:x}"

    @staticmethod
    def bounds() -> tuple[Char, Char]:
        return (Char.zero(), Char(255))

    @staticmethod
    def zero() -> Char:
        return Char(value=0)
