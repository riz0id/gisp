from dataclasses import dataclass

from Syntax.Type.Type import Type

__all__: list[str] = ["BoolType"]


@dataclass(frozen=True, slots=True)
class BoolType(Type):
    # Written `@bool` at the type-level, and as a boolean literal (either `#t`
    # and `#f`) at the term-level.
    pass
