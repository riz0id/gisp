from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Final, Self, override

from Analysis.Support import Support
from Resolve import BindingTable, meaning
from Syntax import (
    Char,
    Expr,
    Integer,
    ProductConExpr,
    SList,
    SumConExpr,
    Symbol,
    TupleConExpr,
    as_data,
    binder_name,
)
from Syntax.Type import (
    PRODUCT,
    CharType,
    Context,
    IntType,
    MuType,
    ProdType,
    Type,
    VarType,
    constructors,
    unfold,
)
from TypeCheck.Patterns import WILDCARD, operand_types

__all__: list[str] = [
    "Admission",
    "CallShape",
    "ConShape",
    "GroundShape",
    "OpaqueShape",
    "ProductShape",
    "Shape",
    "Verdict",
    "admits",
    "classified",
    "columns_of",
    "construction",
    "naming",
    "opened",
    "operand_shapes",
    "resolved",
    "shaped",
    "topped",
    "unknown",
    "weakest",
]


def naming(start: int = 0) -> Callable[[], Symbol]:
    # A fresh name per ground unknown.
    #
    # Minted rather than taken from the pattern binder that binds it, because a
    # recursive procedure binds one spelling once per call: `pass` reads `x` at
    # every depth, and those are different elements of the one list. A name
    # here is an identity and not a spelling, and nothing resolves one against
    # a table, so none of these can collide with what a program wrote.
    uids = count(start)

    return lambda: Symbol(f"x{next(uids)}")


class Shape(ABC):
    # An abstract value: what is known about a value without having it.
    #
    # `Evaluate.Residual` is the same idea with nothing known — it is the form
    # that could not be reduced, and a `match` on one gets stuck because an
    # unknown value selects no clause. A shape knows its *type*, which is what
    # lets a `match` on one fan out instead: the constructors to name come from
    # the type, and `Analysis.PatternCoverage` is what says that naming them all
    # covers it.
    __slots__ = ()

    # The type the shape is at, where the analysis knows it.
    #
    # Carried rather than derived, because a fan-out is a question about the
    # type and `OpaqueShape` has nothing else to answer it with — so that one
    # always has a type, as does a `GroundShape`.
    #
    # `None` on a construction built by the interpreter, and necessarily:
    # `cons` and `nil` carry no ascription — a list is checked against the type
    # the position wants, never synthesized from the construction — so building
    # `(nil (prod#))` tells nobody which list it is. Nothing reads the type of a
    # construction either: `opened` consults a type only for `OpaqueShape`,
    # because a shape that already names its constructor has nothing left to
    # open.
    ty: Type | None

    @abstractmethod
    def to_pattern(self: Self) -> Expr: ...

    @abstractmethod
    def to_value(self: Self) -> Expr: ...


@dataclass(frozen=True, slots=True)
class ConShape(Shape):
    # A value known to be built by `constructor`, mirroring `Syntax.SumConExpr`.
    #
    # One payload and not a run of operands, for the reason `SumType.cons` maps
    # a constructor to a single type: where the payload is a product it is a
    # `ProductShape`, and that is what holds the fields.
    constructor: Symbol
    payload: Shape
    ty: Type | None = None

    @override
    def to_pattern(self: Self) -> Expr:
        return SList([self.constructor, self.payload.to_pattern()])

    @override
    def to_value(self: Self) -> Expr:
        return SumConExpr(self.constructor, self.payload.to_value())


@dataclass(frozen=True, slots=True)
class ProductShape(Shape):
    # A value known to be a product, mirroring `Syntax.TupleConExpr`.
    fields: tuple[Shape, ...]
    ty: Type | None = None

    @override
    def to_pattern(self: Self) -> Expr:
        return SList([PRODUCT, *(field.to_pattern() for field in self.fields)])

    @override
    def to_value(self: Self) -> Expr:
        return TupleConExpr([field.to_value() for field in self.fields])


@dataclass(frozen=True, slots=True)
class GroundShape(Shape):
    # A value of a sort with no closed set of constructors — an `@int` or a
    # `@char` — known by the set it lies in and, where it is unknown, by a name.
    #
    # The one named shape, and necessarily: a comparison is a constraint
    # *between* two values, so the measure can only speak of values it can
    # refer to. Two anonymous unknowns at one type would be equal under `==`
    # and so indistinguishable, which is why the name is minted when the shape
    # is made rather than when something first constrains it.
    #
    # `variable is None` is a literal, which needs no name because it is one.
    variable: Symbol | None
    support: Support
    ty: Type

    @override
    def to_pattern(self: Self) -> Expr:
        if self.variable is not None:
            return self.variable

        match self.support.intervals:
            case ((value, same),) if value == same:
                return Integer(value)
            case _:
                return WILDCARD

    @override
    def to_value(self: Self) -> Expr:
        return self.to_pattern()


@dataclass(frozen=True, slots=True)
class OpaqueShape(Shape):
    # ⊤: a value of a type and nothing more.
    #
    # The only unbounded shape, and expanded only by `opened` — never up front.
    # A `(@list @int)` has no finite expansion, and this is unfolded exactly as
    # far as some clause wrote a pattern, which is the argument that makes
    # `TypeCheck.Patterns.missing` terminate over a `@mu`.
    ty: Type

    @override
    def to_pattern(self: Self) -> Expr:
        return WILDCARD

    @override
    def to_value(self: Self) -> Expr:
        return WILDCARD


@dataclass(frozen=True, slots=True)
class CallShape(Shape):
    # Whatever `procedure` answers when its argument is one of `classes`.
    #
    # The one shape standing for a call rather than for a value, and the whole
    # of why nothing here is unrolled: a recursive call is *this* — a reference
    # to the procedure's own walks at a class — and not a second traversal of
    # its body. So a cycle in a program's calls comes out as a cycle among
    # these, rather than as a recursion the analysis has to be stopped from
    # following.
    # One set of class indices per parameter, as `Walk.Call` carries them.
    procedure: Symbol
    classes: tuple[tuple[int, ...], ...]
    ty: Type | None = None

    @override
    def to_pattern(self: Self) -> Expr:
        return WILDCARD

    @override
    def to_value(self: Self) -> Expr:
        # Written for a reader and not for a table. Spelled as one symbol
        # because a shape's leaves are read back as syntax, and nothing
        # resolves this against a binding — the reason a minted element name is
        # safe is the reason this is.
        written: str = " ".join(
            ",".join(str(index) for index in indices) for indices in self.classes
        )

        return Symbol(f"<{self.procedure}@{written}>")


def resolved(ty: Type, store: Context) -> Type:
    # `ty` with the forms standing for another type read through, which is the
    # resolution `operand_types` and `constructors` each make before they
    # dispatch. Read here so a caller can ask *which* sort is in hand — neither
    # of those answers that, and `construction` has to know whether it is
    # writing a product.
    #
    # Total where those two raise: an unbound type variable is answered as
    # itself, because a shape is built for a program the analysis was handed
    # and not only for one the checker approved.
    match ty:
        case MuType() as recursive:
            return resolved(unfold(recursive), store)
        case VarType(binder=binder) if binder not in store.rigid:
            try:
                return resolved(store.variable(binder), store)
            except ValueError:
                return ty
        case _:
            return ty


def unknown(ty: Type, store: Context, fresh: Callable[[], Symbol]) -> Shape:
    # ⊤ at a type, named where that type is ground.
    match resolved(ty, store):
        case IntType() | CharType():
            return GroundShape(fresh(), Support.integers(), ty)
        case _:
            return OpaqueShape(ty)


def columns_of(constructor: Symbol, ty: Type, store: Context) -> list[Type] | None:
    # The types a `(constructor ...)` pattern's operands are read against at
    # `ty`, or `None` where no value of `ty` is built by that constructor.
    #
    # `constructors` and `operand_types` asked together, which is the one rule
    # about taking a constructor apart. The checker reads it through
    # `TypeChecker.pattern`, `missing` reads it to walk a matrix, and this is
    # the third reader — so a shape is opened into exactly the columns the other
    # two agree on.
    cons: dict[Symbol, Type] | None = constructors(ty, store)

    if cons is None or constructor not in cons:
        return None

    return operand_types(ty, cons[constructor], store)


def construction(
    constructor: Symbol, parts: list[Shape], ty: Type, store: Context
) -> Shape:
    # A shape built by `constructor` over `parts`, written as the construction
    # it is: a product as its fields, and every other constructor as its one
    # payload. `operand_types`' reading of a type, over shapes.
    if isinstance(resolved(ty, store), ProdType):
        return ProductShape(tuple(parts), ty)

    [payload] = parts

    return ConShape(constructor, payload, ty)


def operand_shapes(shape: Shape) -> list[Shape] | None:
    # The shapes a constructor pattern's operands are read against, or `None`
    # where the shape is not a construction.
    #
    # `Syntax.Expr.Con.spread` over shapes, and the inverse of `construction`.
    match shape:
        case ProductShape(fields=fields):
            return list(fields)
        case ConShape(payload=payload):
            return [payload]
        case _:
            return None


def rebuilt(whole: Shape, parts: list[Shape]) -> Shape:
    # `whole` with its operands replaced, which is `operand_shapes` inverted.
    # Needs no store, unlike `construction`: the sort is already decided by the
    # shape in hand.
    match whole:
        case ProductShape(ty=ty):
            return ProductShape(tuple(parts), ty)
        case ConShape(constructor=constructor, ty=ty):
            [payload] = parts

            return ConShape(constructor, payload, ty)
        case _:
            return whole


def opened(
    constructor: Symbol, shape: Shape, store: Context, fresh: Callable[[], Symbol]
) -> tuple[Shape, bool] | None:
    # `shape` made concrete enough to read a `(constructor ...)` pattern
    # against, with whether committing to that constructor is *certain* — or
    # `None` where no value the shape stands for is built by it.
    #
    # The algebraic step, and the whole of what pattern coverage licenses: ⊤ at
    # a type is opened into one constructor of that type with a fresh ⊤ per
    # operand, and coverage is what says that naming every constructor covers
    # the type. `columns_of` is where the operands come from, so this opens a
    # shape into the columns `missing` walks.
    match shape:
        case ConShape(constructor=named):
            return (shape, True) if named == constructor else None
        case ProductShape():
            return (shape, True) if constructor == PRODUCT else None
        case GroundShape():
            # A sort with no closed set of constructors, so no constructor
            # pattern matches one of its values.
            return None
        case OpaqueShape(ty=ty):
            pass
        case CallShape(ty=Type() as ty):
            # What a call returns, opened by its *type* and not by the callee's
            # branches. Nothing is learned about the result here — every
            # constructor of the type remains possible — which is the price of
            # reading a call by reference, and is why a test on one is reported
            # with no probability.
            pass
        case _:
            return None

    columns: list[Type] | None = columns_of(constructor, ty, store)

    if columns is None:
        return None

    # Certain exactly where the type has no other constructor to be. A product
    # has one, so `(prod# x xs)` refutes nothing and always matches; a sum of
    # two is a coin a clause names one side of.
    cons: dict[Symbol, Type] | None = constructors(ty, store)
    certain: bool = cons is not None and len(cons) == 1

    return (
        construction(
            constructor,
            [unknown(column, store, fresh) for column in columns],
            ty,
            store,
        ),
        certain,
    )


class Verdict(Enum):
    # Whether a pattern matches the values a shape stands for: none of them,
    # some of them, or all of them.
    NO = auto()
    MAYBE = auto()
    YES = auto()


# Ordered weakest first. `NO` absorbs and `YES` is the unit, so a pattern is
# certain exactly where every operand of it is, and refuted where any one is.
ORDER: Final[tuple[Verdict, ...]] = (Verdict.NO, Verdict.MAYBE, Verdict.YES)


def weakest(verdicts: Iterable[Verdict]) -> Verdict:
    return min(verdicts, key=ORDER.index, default=Verdict.YES)


@dataclass(frozen=True, slots=True)
class Admission:
    # What reading one pattern against one shape found.
    verdict: Verdict

    # The scrutinee as the commitment leaves it. A clause body may refer to the
    # scrutinee by name — `((nil (prod#)) xs)` answers with the very list it
    # just learned is empty — so the narrowing is worth keeping and not only the
    # binders.
    refined: Shape

    # What the pattern's binders name, by binder.
    bound: dict[Symbol, Shape]

    # The unary facts the commitment establishes, by ground variable. An
    # integer *pattern* narrows the element it is read against, which is the
    # pattern-side counterpart of what a comparison guard contributes.
    narrowed: dict[Symbol, Support]


def refuted(shape: Shape) -> Admission:
    return Admission(Verdict.NO, shape, {}, {})


def literal(value: Integer | Char, shape: Shape) -> Admission:
    # An integer or character pattern, which matches the one value it is.
    #
    # A character is read through its code point, so one `Support` serves both
    # ground sorts: `@char` is ground for the reason `@int` is — a constructor
    # per value, and no run of literals enumerates them.
    point: int = value.value

    match shape:
        case GroundShape(variable=variable, support=support, ty=ty):
            if point not in support:
                return refuted(shape)

            narrowed: Support = Support.point(point)

            return Admission(
                Verdict.YES if support.cardinality() == 1 else Verdict.MAYBE,
                GroundShape(variable, narrowed, ty),
                {},
                {} if variable is None else {variable: narrowed},
            )
        case OpaqueShape(ty=ty):
            return Admission(
                Verdict.MAYBE, GroundShape(None, Support.point(point), ty), {}, {}
            )
        case _:
            # No construction is a literal.
            return refuted(shape)


def constructed(
    constructor: Symbol,
    operands: list[Expr],
    shape: Shape,
    table: BindingTable,
    store: Context,
    fresh: Callable[[], Symbol],
) -> Admission:
    # A constructor pattern, read against the shape opened at its constructor.
    found: tuple[Shape, bool] | None = opened(constructor, shape, store, fresh)

    if found is None:
        return refuted(shape)

    whole, certain = found
    parts: list[Shape] | None = operand_shapes(whole)

    if parts is None or len(parts) != len(operands):
        # How many operands a constructor is written with is the checker's to
        # refuse, and `PatternCoverage.coverage` declines a `match` whose
        # clauses it refused. Answered `MAYBE` so a caller arriving another way
        # is not handed a decision made on a pattern this could not read.
        return Admission(Verdict.MAYBE, shape, {}, {})

    verdicts: list[Verdict] = [Verdict.YES if certain else Verdict.MAYBE]
    refined: list[Shape] = []
    bound: dict[Symbol, Shape] = {}
    narrowed: dict[Symbol, Support] = {}

    for operand, part in zip(operands, parts, strict=True):
        inner: Admission = admits(operand, part, table, store, fresh)

        if inner.verdict is Verdict.NO:
            # One refuted operand refutes the pattern, and nothing the others
            # bound is reachable.
            return refuted(shape)

        verdicts.append(inner.verdict)
        refined.append(inner.refined)
        bound.update(inner.bound)
        narrowed.update(inner.narrowed)

    return Admission(weakest(verdicts), rebuilt(whole, refined), bound, narrowed)


def admits(
    pattern: Expr,
    shape: Shape,
    table: BindingTable,
    store: Context,
    fresh: Callable[[], Symbol],
) -> Admission:
    # Whether `pattern` matches the values `shape` stands for, with the shape
    # the commitment leaves and what the pattern's binders name.
    #
    # `Evaluate.matched` made three-valued. That one answers bound-or-`None`
    # because it holds a value; this one holds a shape, and `MAYBE` is the
    # answer that makes a clause a branch rather than a step. A walk set is
    # exactly the `MAYBE`s taken both ways.
    named: Symbol | None = binder_name(pattern)

    if named is not None:
        # A binder matches every value: nothing to narrow, nothing to decide,
        # and the binder names the shape as it stands.
        return Admission(Verdict.YES, shape, {meaning(named, table): shape}, {})

    match pattern:
        case Integer() | Char() as value:
            return literal(value, shape)
        case SList([Symbol() as head, *operands]):
            return constructed(
                meaning(head, table), operands, shape, table, store, fresh
            )
        case _:
            # Not a pattern this reads. Answered `MAYBE` rather than refused,
            # because a clause is called unreachable only where it was read and
            # found to be.
            return Admission(Verdict.MAYBE, shape, {}, {})


def shaped(
    pattern: Expr,
    ty: Type,
    store: Context,
    table: BindingTable,
    fresh: Callable[[], Symbol],
) -> Shape:
    # The shape standing for exactly the values `pattern` matches: its
    # constructors known, its binder positions unknown.
    #
    # The inverse of `admits`, and what makes an input *class* something the
    # analysis can walk a body under. A class is a pattern — one of the ways an
    # argument can be written — and this is that pattern read as the argument
    # it stands for.
    #
    # Built from `columns_of`, so a class is taken apart into the same columns
    # `missing` walks and `TypeChecker.pattern` checks against. A binder
    # position becomes `unknown`, which names it where the position is ground —
    # and that name is what lets the measure say which of two elements is
    # larger.
    if binder_name(pattern) is not None:
        return unknown(ty, store, fresh)

    match pattern:
        case Integer() | Char() as value:
            return GroundShape(None, Support.point(value.value), ty)
        case SList([Symbol() as head, *operands]):
            constructor: Symbol = meaning(head, table)
            columns: list[Type] | None = columns_of(constructor, ty, store)

            if columns is None or len(columns) != len(operands):
                # Not a pattern for this type, which the checker refuses. ⊤ is
                # the answer that says nothing rather than one that says
                # something wrong.
                return unknown(ty, store, fresh)

            return construction(
                constructor,
                [
                    shaped(operand, column, store, table, fresh)
                    for operand, column in zip(operands, columns, strict=True)
                ],
                ty,
                store,
            )
        case _:
            return unknown(ty, store, fresh)


def topped(
    datum: Expr,
    ty: Type,
    store: Context,
    fresh: Callable[[], Symbol],
) -> Shape:
    # A quoted datum read *one level*: its top constructor, with ⊤ beneath it.
    #
    # One level and no deeper, which is the whole of what a class asks. Reading
    # the spine would make the answer depend on the datum's length, and a
    # length is what this analysis does not read — `'(1)` and
    # `'(3 1 4 1 5 9 2 6)` are both a `cons`, and that is all either of them
    # says about which walk it takes.
    #
    # The same one level a type's own split gives, so nothing is learned here
    # that `splits` would not have offered anyway; what is learned is which of
    # those splits this datum is.
    #
    # `as_data` is the reading `Evaluate.evaluate`'s `quote` arm makes, so a
    # datum is the `cons` nest that is its value before its head is read.
    value: Expr = as_data(datum)
    constructor: Symbol

    match value:
        case Integer() | Char() as written:
            return GroundShape(None, Support.point(written.value), ty)
        case SumConExpr(constructor=constructor):
            pass
        case ProductConExpr():
            constructor = PRODUCT
        case _:
            # A symbol or a procedure, neither of which is data this reads.
            return OpaqueShape(ty)

    columns: list[Type] | None = columns_of(constructor, ty, store)

    if columns is None:
        return unknown(ty, store, fresh)

    return construction(
        constructor,
        [unknown(column, store, fresh) for column in columns],
        ty,
        store,
    )


def classified(
    shape: Shape,
    classes: tuple[Expr, ...],
    store: Context,
    table: BindingTable,
    fresh: Callable[[], Symbol],
) -> tuple[int, ...]:
    # Which of `classes` a shape may fall in.
    #
    # The transition a call makes, and the step that replaces entering a
    # callee's body. `(cons (prod# y ys))` with `ys` unknown is refuted by the
    # `nil` class and admitted by both `cons` classes, so it falls in two — and
    # a walk that reaches it branches into the callee's walks at each, without
    # any of them being traversed here.
    #
    # Classes are complete over the type, so this is never empty for a shape of
    # that type: some class admits every value, which is what
    # `PatternCoverage` is relied on for.
    return tuple(
        index
        for index, written in enumerate(classes)
        if admits(written, shape, table, store, fresh).verdict is not Verdict.NO
    )
