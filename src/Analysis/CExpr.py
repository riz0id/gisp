from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Self, override

from Expander import KEYWORDS
from Resolve import BindingTable, Unbound, meaning, resolve
from Syntax import Char, Expr, Integer, SList, Symbol, binder_name
from Syntax.Type import CONS, FALSE, NIL, PRODUCT, TRUE

__all__: list[str] = [
    "CExpr",
    "CExprError",
    "VarCExpr",
    "LitCExpr",
    "IntegerCExpr",
    "CharCExpr",
    "ListCExpr",
    "AppCExpr",
    "ConCExpr",
    "ProductCExpr",
    "ConjunctionCExpr",
    "DisjunctionCExpr",
    "ComparisonCExpr",
    "NegationCExpr",
    "EquivalenceCExpr",
]


class CExprError(Exception):
    pass


# "CExpr" is the shortened name for "conditional expression".
class CExpr(ABC):
    @classmethod
    def from_expr(cls, expr: Expr, store: BindingTable | None = None) -> CExpr:
        if store is None:
            store = BindingTable()

        return convert(
            expr=expr,
            store=BindingTable(
                [*store, *map(Symbol, OPERATORS), *map(Symbol, UNARY_OPERATORS)]
            ),
        )

    @abstractmethod
    def to_expr(self: Self) -> Expr: ...


# 'VarCExpr(s)' corresponds to the S-expression 's'.
@dataclass()
class VarCExpr(CExpr):
    symbol: Symbol

    @override
    def to_expr(self: Self) -> Expr:
        return self.symbol


# The abstract base class for all 'CExpr' (conditional expression) literals.
class LitCExpr(CExpr):
    pass


# 'IntegerCExpr(n)' corresponds to a literal integer value.
@dataclass()
class IntegerCExpr(LitCExpr):
    value: Integer

    @override
    def to_expr(self: Self) -> Expr:
        return self.value


# 'CharCExpr(c)' corresponds to a literal character value.
@dataclass()
class CharCExpr(LitCExpr):
    value: Char

    @override
    def to_expr(self: Self) -> Expr:
        return self.value


# 'ListCExpr(a, b, ...)' corresponds to the S-expression '(a b ...)' whose head
# is not a symbol, which is the one list this language has nothing to say about.
# Its items are held as syntax and not converted, because there is no head to
# read them against.
@dataclass()
class ListCExpr(CExpr):
    items: list[Expr]

    @override
    def to_expr(self: Self) -> Expr:
        return SList(items=self.items)


# 'AppCExpr(f, [a, b, ...])' corresponds to the S-expression '(f a b ...)' where
# `f` names neither an operator below nor a constructor: a procedure applied to
# its arguments, and the one headed form whose value this language cannot read.
#
# Told apart from a construction, which is what `ConCExpr` and `ProductCExpr`
# are: `(cons (prod# x xs))` builds a value out of its operands, while `(f x)`
# runs a procedure over them. Nothing could tell the two apart while both were
# the same node, which is why `Analysis.ascription` had to re-exclude a
# constructor by name before reading an application's return type.
@dataclass()
class AppCExpr(CExpr):
    head: Symbol
    operands: list[CExpr]

    @override
    def to_expr(self: Self) -> Expr:
        return SList([self.head, *(operand.to_expr() for operand in self.operands)])


# 'ConCExpr(c, p)' corresponds to the S-expression '(c p)': a sum constructor
# applied to its one payload, mirroring `Syntax.SumConExpr`.
#
# One payload and not a run of operands, for the reason `SumType.cons` maps a
# constructor to a single type: where that payload is a product, the pattern or
# construction written for it is a `ProductCExpr`, and that is what splits it
# into fields. The reading `TypeCheck.Patterns.operand_types` makes of a type,
# over the term.
@dataclass()
class ConCExpr(CExpr):
    constructor: Symbol
    payload: CExpr

    @override
    def to_expr(self: Self) -> Expr:
        return SList([self.constructor, self.payload.to_expr()])


# 'ProductCExpr(prod#, [a, b, ...])' corresponds to '(prod# a b ...)': the
# product constructor applied to its fields, mirroring `Syntax.TupleConExpr`.
#
# The head is kept as written rather than rebuilt from `PRODUCT`, as
# `ConCExpr`'s is, so that `to_expr` answers the form that was read.
@dataclass()
class ProductCExpr(CExpr):
    constructor: Symbol
    fields: list[CExpr]

    @override
    def to_expr(self: Self) -> Expr:
        return SList([self.constructor, *(field.to_expr() for field in self.fields)])


# 'ConjunctionCExpr(a, b)' corresponds to the S-expression '(and a b)'.
@dataclass()
class ConjunctionCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("and"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'DisjunctionCExpr(a, b)' corresponds to the S-expression '(or a b)'.
@dataclass()
class DisjunctionCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("or"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'ComparisonCExpr(a, b)' corresponds to the S-expression '(lt a b)'.
@dataclass()
class ComparisonCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("lt"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


# 'NegationCExpr(a)' corresponds to the S-expression '(not a)'.
@dataclass()
class NegationCExpr(CExpr):
    operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("not"),
                self.operand.to_expr(),
            ]
        )


# 'EquivalenceCExpr(a, b)' corresponds to the S-expression '(eq a b)'.
@dataclass()
class EquivalenceCExpr(CExpr):
    left_operand: CExpr
    right_operand: CExpr

    @override
    def to_expr(self: Self):
        return SList(
            [
                Symbol("eq"),
                self.left_operand.to_expr(),
                self.right_operand.to_expr(),
            ]
        )


OPERATORS: Final[dict[str, Callable[[CExpr, CExpr], CExpr]]] = {
    "and": ConjunctionCExpr,
    "or": DisjunctionCExpr,
    "lt": ComparisonCExpr,
    "eq": EquivalenceCExpr,
}

UNARY_OPERATORS: Final[dict[str, Callable[[CExpr], CExpr]]] = {
    "not": NegationCExpr,
}


def constructor[T](
    head: Symbol, store: BindingTable, operators: dict[str, T]
) -> T | None:
    try:
        binder: Symbol = resolve(head, store).binder

        # Every built-in binder is bare, so a binder carrying scopes is one the
        # program introduced that merely shares a built-in's spelling.
        return None if binder.scopes else operators.get(binder.identifier)
    except Unbound:
        # Not a failure here: every head that is neither a built-in nor
        # program-bound arrives here, and all of them are applications.
        return None


# The constructors a head can be read as one of with no type in hand, which is
# all `convert` is given. Each is a built-in whose spelling the parse phase
# re-emits unresolved, so a bare head naming one of these is that constructor.
#
# Short of what `Syntax.Type.constructors` enumerates, and necessarily: a sum a
# program wrote names its own constructors, and knowing `(foo (prod#))` builds
# one takes the type `foo` belongs to. Such a head reads as an application here,
# which is the honest answer where no type is in hand — `Analysis.Shape` is the
# reader that has one, and it recognises a construction by the type it is at.
#
# `ABSTRACTION` is absent though `CONSTRUCTORS` names it: a `lambda` builds a
# procedure, and no procedure is data this language reads.
DATA: Final[frozenset[Symbol]] = frozenset({CONS, NIL, PRODUCT, TRUE, FALSE})


def headed(head: Symbol, operands: list[Expr], store: BindingTable) -> CExpr:
    # A list with a symbol head, read as the first of the three things it can
    # be: an operator form, a construction, or an application.
    #
    # The order is the dispatch. An operator is tried first and by arity, so an
    # over- or under-applied `and` is not one; a construction is tried next,
    # because a constructor is never also an operator; and an application is
    # what is left, which is why it needs no test of its own.
    if head.identifier in KEYWORDS:
        # A core form is none of the three. It is read as the opaque list it is,
        # which is what a `match` or a `letrec` standing in a scrutinee position
        # has always read as — and the alternative would be to call `match` a
        # procedure applied to its clauses, which it is not.
        #
        # Recognised by spelling, as `Expander.transformer` recognises one: a
        # keyword is unbindable, so no resolution can make a head stop being
        # one. `forall` is in the set and never arrives here, having been read
        # by `convert` already.
        return ListCExpr([head, *operands])

    match operands:
        case [operand]:
            build_unary: Callable[[CExpr], CExpr] | None = constructor(
                head, store, UNARY_OPERATORS
            )

            if build_unary is not None:
                return build_unary(convert(operand, store))
        case [left, right]:
            build: Callable[[CExpr, CExpr], CExpr] | None = constructor(
                head, store, OPERATORS
            )

            if build is not None:
                return build(convert(left, store), convert(right, store))

    named: Symbol = meaning(head, store)

    # A binder carrying scopes is one the program introduced that merely shares
    # a built-in's spelling, as `constructor` reads it above.
    if not named.scopes and named in DATA:
        if named == PRODUCT:
            return ProductCExpr(head, [convert(field, store) for field in operands])

        match operands:
            case [payload]:
                return ConCExpr(head, convert(payload, store))

    return AppCExpr(head, [convert(operand, store) for operand in operands])


def convert(expr: Expr, store: BindingTable) -> CExpr:
    match expr:
        case Symbol() as symbol:
            return VarCExpr(symbol)
        case Integer() as value:
            return IntegerCExpr(value)
        case Char() as value:
            return CharCExpr(value)
        case SList([Symbol() as head, operand]) if (head.identifier == "forall") and (
            bound := binder_name(operand)
        ) is not None:
            # Read before the headed arm below, because the form is itself a
            # two-item list and an application of `forall` is a condition
            # nothing can read.
            return VarCExpr(bound)
        case SList([Symbol() as head, *operands]):
            return headed(head, operands, store)
        case SList(items):
            return ListCExpr(items)
        case _:
            raise CExprError(f"not a convertible expression: {expr}")
