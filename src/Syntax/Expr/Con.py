from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Self, override

from Syntax.Expr.Base import Expr
from Syntax.Expr.Char import Char
from Syntax.Expr.SList import SList
from Syntax.Expr.Symbol import Symbol

__all__: list[str] = [
    "ConExpr",
    "ProductConExpr",
    "RecordConExpr",
    "SumConExpr",
    "TupleConExpr",
    "as_data",
    "as_syntax",
    "boolean",
    "consed",
    "elements",
    "listed",
    "product",
    "spread",
    "unit",
]

# Read as identifiers and not as the `Symbol` constants `Syntax.Type` defines,
# because that module imports this one. `SList.with_scopes` reads `"quote"` the
# same way.
CONS: Final[str] = "cons"
NIL: Final[str] = "nil"
PRODUCT: Final[str] = "prod#"


class ConExpr(Expr):
    # Constructed data, represented as the construction that builds it: a
    # boolean is `(#t (prod#))`, the one value of `(@prod)` is `(prod#)`, and a
    # list is a nest of `cons` cells ending in `nil`.
    __slots__ = ()

    @override
    @abstractmethod
    def __str__(self: Self) -> str: ...

    @override
    @abstractmethod
    def __eq__(self: Self, other: object) -> bool: ...


class ProductConExpr(ConExpr):
    # The accessors mirror `ProdType.names` and `ProdType.types`, so a product
    # value and the product type classifying it are read the same way.
    __slots__ = ()

    @abstractmethod
    def names(self: Self) -> tuple[Symbol, ...] | None: ...

    @abstractmethod
    def values(self: Self) -> tuple[Expr, ...]: ...


@dataclass(frozen=True, slots=True)
class SumConExpr(ConExpr):
    # One payload and not a run of operands, because `SumType.cons` maps a
    # constructor to a single `Type`. Where that type is a product the payload
    # is a product value, and `spread` is what reads its fields.
    constructor: Symbol
    payload: Expr

    @override
    def __eq__(self: Self, other: object) -> bool:
        return (
            self.constructor == other.constructor and self.payload == other.payload
            if isinstance(other, SumConExpr)
            else False
        )

    @override
    def __str__(self: Self) -> str:
        # A list prints in the notation the reader has for one, and every other
        # construction prints as itself: `'(1 2 3)` reads back as this list and
        # `"abc"` as this string, while nothing reads back as `(#t (prod#))`.
        # The two branches below are `SList.__str__`'s.
        items: list[Expr] | None = elements(self)

        if items is None:
            return f"({self.constructor} {self.payload})"

        chars: list[Char] = [item for item in items if isinstance(item, Char)]

        if chars and len(chars) == len(items):
            escaped: str = (
                "".join(chr(char.value) for char in chars)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\r", "\\r")
            )
            return f'"{escaped}"'

        return f"({' '.join(str(item) for item in items)})"


@dataclass(frozen=True, slots=True)
class TupleConExpr(ProductConExpr):
    fields: list[Expr]

    @override
    def names(self: Self) -> tuple[Symbol, ...] | None:
        return None

    @override
    def values(self: Self) -> tuple[Expr, ...]:
        return tuple(self.fields)

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.fields == other.fields if isinstance(other, TupleConExpr) else False

    @override
    def __str__(self: Self) -> str:
        return f"({' '.join([PRODUCT, *(str(field) for field in self.fields)])})"


@dataclass(frozen=True, slots=True)
class RecordConExpr(ProductConExpr):
    # At least one field, for the reason `ProdType.__post_init__` collapses an
    # empty record to the empty tuple: the nullary product is the empty tuple
    # however it was written, so `unit()` is the one value of `(@prod)`.
    fields: dict[Symbol, Expr]

    def __post_init__(self: Self) -> None:
        if len(self.fields) < 1:
            raise ValueError("RecordConExpr.fields must have one or more entries.")

        object.__setattr__(self, "fields", dict(self.fields))

    @override
    def names(self: Self) -> tuple[Symbol, ...] | None:
        return tuple(self.fields)

    @override
    def values(self: Self) -> tuple[Expr, ...]:
        return tuple(self.fields.values())

    @override
    def __eq__(self: Self, other: object) -> bool:
        # Never equal to a tuple of the same fields, as `ProdType({a: t})` is
        # never equal to `ProdType([t])`.
        return (
            self.fields == other.fields if isinstance(other, RecordConExpr) else False
        )

    @override
    def __str__(self: Self) -> str:
        written: list[str] = [
            f"({name} {value})" for name, value in self.fields.items()
        ]

        return f"({' '.join([PRODUCT, *written])})"


def spread(payload: Expr) -> list[Expr]:
    # A product value's fields, and the value itself where it is not a product.
    # `TypeCheck.Patterns.product_fields` over values rather than types, and
    # read for the same reason: a cons cell's payload holds the head and the
    # rest whichever of the two product shapes it was built as.
    match payload:
        case ProductConExpr() as fields:
            return list(fields.values())
        case _:
            return [payload]


def unit() -> TupleConExpr:
    return TupleConExpr([])


def product(fields: Sequence[Expr]) -> TupleConExpr:
    # A tuple and not a record, because `(prod# 1 2)` names no field and
    # nothing reading a value has a type to take names from.
    return TupleConExpr(list(fields))


def boolean(which: bool) -> SumConExpr:
    return SumConExpr(Symbol("#t" if which else "#f"), unit())


def consed(first: Expr, rest: Expr) -> SumConExpr:
    return SumConExpr(Symbol(CONS), product([first, rest]))


def listed(items: Sequence[Expr]) -> SumConExpr:
    # Built from the back, so the tail each cell carries is the list already
    # made.
    rest: SumConExpr = SumConExpr(Symbol(NIL), unit())

    for item in reversed(items):
        rest = consed(item, rest)

    return rest


def elements(value: Expr) -> list[Expr] | None:
    # The items of a list, or `None` where the value is not one. Iterative
    # because a list is as long as the program made it.
    items: list[Expr] = []

    while True:
        match value:
            case SumConExpr(constructor=con, payload=payload) if con.identifier == CONS:
                match spread(payload):
                    case [first, rest]:
                        items.append(first)
                        value = rest
                    case _:
                        return None
            case SumConExpr(constructor=con) if con.identifier == NIL:
                return items
            case _:
                return None


def as_data(datum: Expr) -> Expr:
    match datum:
        case SList(items):
            return listed([as_data(item) for item in items])
        case _:
            return datum


def as_syntax(value: Expr) -> Expr:
    # The inverse of `as_data`, for the builtins written over syntax. Partial
    # in principle — nothing tells a one-field tuple from a singleton list —
    # and total on quoted syntax, which holds only symbols, integers,
    # characters, and lists.
    items: list[Expr] | None = elements(value)

    if items is None:
        return value

    return SList([as_syntax(item) for item in items])
