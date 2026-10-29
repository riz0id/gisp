from abc import ABC
from dataclasses import dataclass

from Resolve import BindingTable, Unbound, resolve
from Syntax import Symbol

__all__: list[str] = ["ConstructorArity", "Type", "meaning"]

type ConstructorArity = int | None


def meaning(symbol: Symbol, table: BindingTable) -> Symbol:
    # The binder a reference resolves to, or its bare spelling when the table
    # binds nothing: an unbound head is read as the built-in it spells. This
    # duplicates `Analysis.meaning` because the analysis sits above the syntax.
    try:
        return resolve(symbol, table).binder
    except Unbound:
        return Symbol(symbol.identifier)


@dataclass(frozen=True, slots=True)
class Type(ABC):
    # The base every sort shares, and nothing more: the four queries over a
    # type — `type_constructor_arity`, `type_constructor_symbol`,
    # `constructor_arity`, and `is_constructor` — are functions in
    # `Syntax.Type` that `match` on the sort rather than methods declared here.
    pass
