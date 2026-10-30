from Syntax import Expr, SList, Symbol
from Syntax.Type import (
    BOOL,
    CharType,
    FunType,
    IntType,
    MuType,
    ProdType,
    Scheme,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    VarType,
    list_of,
)

__all__: list[str] = ["ElaborateError", "elaborate", "elaborate_type"]


class ElaborateError(Exception):
    pass


# A written type read as the type it denotes. The input is what the parse phase
# emitted: every form is already known to be written at the right arity, and
# every type variable is already resolved to its binder. So the arms below
# dispatch on a form to build it, and never to decide whether it is one.
#
# Two rules are this module's own, because each is about what a type means
# rather than how it is written:
#
#   * rank-1 — a quantifier may be read only at the outermost position, which
#     is what `elaborate` reads and `elaborate_type` refuses. `Scheme` is
#     deliberately not a `Type`, so the restriction is held by the
#     representation and this is only where a violation is reported.
#   * the argument run — a `@fun`'s first operand must *denote* a product or a
#     variadic run, which is a fact about the operand's meaning and not its
#     shape.
#
# The `ElaborateError` arms that are not one of those two are unreachable from
# parse output, and exist because a `match` over syntax has to be total.


def argument_run(
    argument_type: Expr,
) -> ProdType | VariadicType | VariadicStarType:
    # The argument type of a `(@fun a r)`: the product or the variadic run a
    # function is applied to, and nothing else. These are the three sorts
    # `FunType.argument_type` admits and the three `substitute_arguments`
    # rebuilds — a type that classifies values describes no argument list.
    match elaborate_type(argument_type):
        case ProdType() | VariadicType() | VariadicStarType() as run:
            return run
        case _:
            raise ElaborateError(f"not an argument type: {argument_type}")


def named_types(pairs: list[Expr]) -> dict[Symbol, Type]:
    # The ⟨name, type⟩ of every `(name type)` pair, which is the shape a
    # record's fields and a sum's constructors share. The parse phase has
    # already refused a pair that is not one, and a name repeated among them.
    named: dict[Symbol, Type] = {}

    for pair in pairs:
        match pair:
            case SList([Symbol() as name, ty]):
                named[name] = elaborate_type(ty)
            case _:
                raise ElaborateError(f"not a named type: {pair}")

    return named


def product(fields: list[Expr]) -> ProdType:
    # `(@prod t ...)` is a tuple and `(@prod (x t) ...)` a record. The parse
    # phase has already refused a product that is named in part, so the first
    # field decides which this is.
    #
    # The two are told apart by the leading `@`, as the parse phase tells them
    # apart: a field is a `(name type)` pair exactly when its head names no type
    # constructor. That test is written in both places, which is the price of a
    # phase that hands on syntax rather than something tagged — here it reads a
    # guarantee the parse phase has already enforced, rather than deciding it.
    match fields:
        case [SList([Symbol() as name, _]), *_] if not (
            name.identifier.startswith("@")
        ):
            return ProdType(named_types(fields))
        case []:
            # The nullary product is the empty tuple however it was written,
            # which is what `ProdType` reads an empty record as.
            return ProdType()
        case _:
            return ProdType([elaborate_type(field) for field in fields])


def elaborate_type(expr: Expr) -> Type:
    match expr:
        case Symbol(identifier="@int"):
            return IntType()
        case Symbol(identifier="@char"):
            return CharType()
        case Symbol(identifier="@bool"):
            # The spelling is the built-in; the sum is what it denotes.
            return BOOL
        case Symbol() as symbol if symbol.identifier.startswith("@"):
            raise ElaborateError(f"not a type: {symbol}")
        case Symbol() as symbol:
            # A type variable, read as the binder the parse phase resolved it
            # to. Binder and occurrence are therefore one `Symbol` value, which
            # is what makes `substitute` and `unfold` meet.
            return VarType(symbol)
        case SList([Symbol() as head, element]) if head.identifier == "@list":
            return list_of(elaborate_type(element))
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic":
            return VariadicType(elaborate_type(argument))
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic*":
            return VariadicStarType(elaborate_type(argument))
        case SList([Symbol() as head, *fields]) if head.identifier == "@prod":
            return product(fields)
        case SList([Symbol() as head, *constructors]) if head.identifier == "@sum":
            return SumType(named_types(constructors))
        case SList([Symbol() as head, argument_type, return_type]) if (
            head.identifier == "@fun"
        ):
            return FunType(
                return_type=elaborate_type(return_type),
                argument_type=argument_run(argument_type),
            )
        case SList([Symbol() as head, Symbol() as variable, body]) if (
            head.identifier == "@mu"
        ):
            # `(@mu a t)` binds `a` within `t`. Guardedness is not read here: it
            # is the precondition `Context.bind` documents, and nothing enforces
            # it, so `(@mu a a)` elaborates.
            return MuType(variable, elaborate_type(body))
        case SList([Symbol() as head, *_]) if head.identifier == "forall#":
            # The rank-1 restriction. A quantifier is written at the outermost
            # position of a type and nowhere within one, so reaching this arm
            # means one was written inside a type — which parses, and is refused
            # here.
            raise ElaborateError(f"a quantifier is only written outermost: {expr}")
        case _:
            raise ElaborateError(f"not a type: {expr}")


def elaborate(expr: Expr) -> Scheme:
    # A written type read as what a binder is ascribed, which is a scheme
    # whether or not a quantifier was written: a type written on its own is the
    # scheme quantifying nothing, so every ascription reads the same way and the
    # monomorphic ones take the path a monotype always took.
    #
    # The one position a `forall#` may occupy. `elaborate_type` refuses it
    # everywhere else, which is what keeps the quantifier from nesting.
    match expr:
        case SList([Symbol() as head, SList(variables), body]) if (
            head.identifier == "forall#"
        ):
            bound: list[Symbol] = []

            for variable in variables:
                match variable:
                    case Symbol() as name:
                        bound.append(name)
                    case _:
                        raise ElaborateError(f"not a type variable: {variable}")

            if not bound:
                # A quantifier over nothing is refused rather than read as the
                # trivial scheme: it is written where a reader expects a
                # variable to be bound, so an empty binder list is far more
                # likely a mistake than an intent.
                raise ElaborateError(
                    f"a quantifier binds one or more variables: {expr}"
                )

            return Scheme(variables=tuple(bound), body=elaborate_type(body))
        case _:
            return Scheme(variables=(), body=elaborate_type(expr))
