from dataclasses import dataclass

from Syntax.Type.ProdType import ProdType
from Syntax.Type.Type import Type
from Syntax.Type.VariadicType import VariadicStarType, VariadicType

__all__: list[str] = ["FunType"]


@dataclass(frozen=True, slots=True)
class FunType(Type):
    return_type: Type
    argument_type: ProdType | VariadicType | VariadicStarType = ProdType()
