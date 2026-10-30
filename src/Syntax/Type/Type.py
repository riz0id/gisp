from dataclasses import dataclass

__all__: list[str] = ["ConstructorArity", "Type"]

type ConstructorArity = int | None


@dataclass(frozen=True, slots=True)
class Type:
    # The base every sort shares, and nothing more: the four queries over a
    # type — `type_constructor_arity`, `type_constructor_symbol`,
    # `constructor_arity`, and `is_constructor` — are functions in
    # `Syntax.Type` that `match` on the sort rather than methods declared here.
    pass
