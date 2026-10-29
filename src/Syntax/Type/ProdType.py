from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self, override

from Syntax import Symbol
from Syntax.Type.Type import Type

__all__: list[str] = ["ProdType"]


@dataclass(frozen=True, slots=True)
class ProdType(Type):
    # Written `(@prod t_0 t_1 ... t_n)` for a tuple and
    # `(@prod (x_0 t_0) ... (x_n t_n))` for a record at the type-level.
    fields: dict[Symbol, Type] | list[Type] = field(default_factory=list)

    def __post_init__(self: Self) -> None:
        # Copied so the frozen value cannot be mutated through the caller's
        # collection, which would silently change its hash.
        match self.fields:
            case dict() as named if named:
                object.__setattr__(self, "fields", dict(named))
            case dict():
                # The nullary product is the empty tuple however it was
                # written: a record is named fields, and this one has none to
                # name.
                object.__setattr__(self, "fields", [])
            case anonymous:
                object.__setattr__(self, "fields", list(anonymous))

    @override
    def __hash__(self: Self) -> int:
        match self.fields:
            case dict() as named:
                return hash(frozenset(named.items()))
            case anonymous:
                return hash(tuple(anonymous))

    def is_record(self: Self) -> bool:
        return isinstance(self.fields, dict)

    def names(self: Self) -> tuple[Symbol, ...] | None:
        # `None` for a tuple, whose fields have no names — which is not the
        # same as a record that has none.
        match self.fields:
            case dict() as named:
                return tuple(named)
            case _:
                return None

    def types(self: Self) -> tuple[Type, ...]:
        # The field types in order, which is the whole of a product to the
        # positional reading every judgment over one makes.
        match self.fields:
            case dict() as named:
                return tuple(named.values())
            case anonymous:
                return tuple(anonymous)

    def with_types(self: Self, types: Sequence[Type]) -> ProdType:
        # This product's shape with its field types replaced, so a
        # transformation rewriting types need not know which shape it rebuilds.
        match self.names():
            case None:
                if len(types) != len(self.fields):
                    raise ValueError(
                        f"cannot rebuild a product of {len(self.fields)} fields"
                        f" from {len(types)} types."
                    )

                return ProdType(list(types))
            case names:
                return ProdType(dict(zip(names, types, strict=True)))
