from enum import Enum

from Syntax.Expr.Base import Expr
from Syntax.Expr.Boolean import Boolean
from Syntax.Expr.Char import Char
from Syntax.Expr.Integer import Integer
from Syntax.Expr.Keyword import Keyword
from Syntax.Expr.Nil import Nil
from Syntax.Expr.SList import SList
from Syntax.Expr.Symbol import Symbol

__all__: list[str] = [
    "Symbol",
    "Integer",
    "Char",
    "Boolean",
    "Keyword",
    "Nil",
    "SList",
    "Expr",
]


class DatumType(Enum):
    INTEGER = Integer
    CHAR = Char
    BOOLEAN = Boolean
    NIL = Nil
    SLIST = SList
