from Resolve import BindingTable, Phase, Unbound, resolve
from Syntax import (
    Char,
    Expr,
    Integer,
    SList,
    Symbol,
    ascription,
    heads_ascribe,
)

__all__: list[str] = [
    "ParseError",
    "parse",
    "strip",
]


class ParseError(Exception):
    pass


def strip(expr: Expr) -> Expr:
    match expr:
        case Symbol() as symbol:
            return Symbol(symbol.identifier)
        case SList(items):
            return SList([strip(item) for item in items])
        case _:
            return expr


def binder(symbol: Symbol, table: BindingTable, phase: Phase | None = None) -> Symbol:
    try:
        return resolve(symbol, table, phase).binder
    except Unbound:
        raise ParseError(f"binder missing from the store: {symbol}") from None


def reference(
    symbol: Symbol, table: BindingTable, phase: Phase | None = None
) -> Symbol:
    try:
        return resolve(symbol, table, phase).binder
    except Unbound:
        return Symbol(symbol.identifier)


def binder_position(
    parameter: Expr, table: BindingTable, phase: Phase | None = None
) -> Expr:
    match parameter:
        case SList([head, *_]) if (ascribed := ascription(parameter)) is not None:
            name, ty = ascribed
            return SList(
                [
                    strip(head),
                    binder(name, table, phase),
                    type_syntax(ty, table, phase),
                ]
            )
        case _ if heads_ascribe(parameter):
            raise ParseError(f"malformed ascribe#: {parameter}")
        case Symbol() as symbol:
            return binder(symbol, table, phase)
        case _:
            raise ParseError(f"binder is not a symbol: {parameter}")


def pattern_syntax(
    pattern: Expr, table: BindingTable, phase: Phase | None = None
) -> Expr:
    match pattern:
        case SList([]):
            raise ParseError(f"not a pattern: {pattern}")
        case SList([constructor, *operands]) if not heads_ascribe(pattern):
            match constructor:
                case Symbol() as head:
                    pass
                case _:
                    raise ParseError(f"pattern head is not a constructor: {pattern}")

            return SList(
                [
                    reference(head, table, phase),
                    *[pattern_syntax(operand, table, phase) for operand in operands],
                ]
            )
        case Integer() | Char():
            return pattern
        case _:
            return binder_position(pattern, table, phase)


def pattern_clause(
    clause: Expr, table: BindingTable, phase: Phase | None = None
) -> Expr:
    match clause:
        case SList([pattern, body]):
            return SList(
                [
                    pattern_syntax(pattern, table, phase),
                    parse(body, table, phase),
                ]
            )
        case _:
            raise ParseError(f"malformed match clause: {clause}")


def type_variable(
    symbol: Symbol, table: BindingTable, phase: Phase | None = None
) -> Symbol:
    try:
        return resolve(symbol, table, phase).binder
    except Unbound:
        raise ParseError(f"unbound type variable: {symbol}") from None


def names_a_type(field: Expr) -> bool:
    match field:
        case SList([Symbol() as name, _]):
            return not name.identifier.startswith("@")
        case _:
            return False


def named_types(
    pairs: list[Expr], table: BindingTable, phase: Phase | None = None
) -> list[Expr]:
    seen: set[str] = set()
    named: list[Expr] = []

    for pair in pairs:
        match pair:
            case SList([Symbol() as written, ty]) if not (
                written.identifier.startswith("@")
            ):
                pass
            case _:
                raise ParseError(f"not a named type: {pair}")

        if written.identifier in seen:
            raise ParseError(f"duplicate name in one type: {written}")

        seen.add(written.identifier)
        named.append(SList([Symbol(written.identifier), type_syntax(ty, table, phase)]))

    return named


def product_fields(
    fields: list[Expr], table: BindingTable, phase: Phase | None = None
) -> list[Expr]:
    match fields:
        case []:
            return []
        case [first, *_] if names_a_type(first):
            # `named_types` refuses a field that is not a pair, so a product
            # that starts named cannot finish positional.
            return named_types(fields, table, phase)
        case _:
            for field in fields:
                if names_a_type(field):
                    raise ParseError(
                        f"a product is named throughout or not at all: {field}"
                    )

            return [type_syntax(field, table, phase) for field in fields]


def type_syntax(expr: Expr, table: BindingTable, phase: Phase | None = None) -> Expr:
    match expr:
        case Symbol(identifier="@int" | "@char" | "@bool") as symbol:
            return Symbol(symbol.identifier)
        case Symbol() as symbol if symbol.identifier.startswith("@"):
            # Every other `@` identifier names a type constructor, and a
            # constructor is written as a list: `@list` is not a type, and
            # `(@list @int)` is.
            raise ParseError(f"not a type: {symbol}")
        case Symbol() as symbol:
            return type_variable(symbol, table, phase)
        case SList([Symbol() as head, element]) if head.identifier == "@list":
            return SList([Symbol(head.identifier), type_syntax(element, table, phase)])
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic":
            return SList([Symbol(head.identifier), type_syntax(argument, table, phase)])
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic*":
            return SList([Symbol(head.identifier), type_syntax(argument, table, phase)])
        case SList([Symbol() as head, *fields]) if head.identifier == "@prod":
            return SList(
                [
                    Symbol(head.identifier),
                    *product_fields(fields, table, phase),
                ]
            )
        case SList([Symbol() as head, *constructors]) if head.identifier == "@sum":
            if not constructors:
                # Refused here so the diagnosis is this phase's, and not the
                # `ValueError` `SumType` raises on an empty mapping.
                raise ParseError(f"a sum has one or more constructors: {expr}")

            return SList(
                [
                    Symbol(head.identifier),
                    *named_types(constructors, table, phase),
                ]
            )
        case SList([Symbol() as head, argument_type, return_type]) if (
            head.identifier == "@fun"
        ):
            return SList(
                [
                    Symbol(head.identifier),
                    type_syntax(argument_type, table, phase),
                    type_syntax(return_type, table, phase),
                ]
            )
        case SList([Symbol() as head, Symbol() as variable, body]) if (
            head.identifier == "@mu"
        ):
            return SList(
                [
                    Symbol(head.identifier),
                    binder(variable, table, phase),
                    type_syntax(body, table, phase),
                ]
            )
        case SList([Symbol() as head, SList(variables), body]) if (
            head.identifier == "forall#"
        ):
            return SList(
                [
                    Symbol(head.identifier),
                    SList(
                        [binder_of(variable, table, phase) for variable in variables]
                    ),
                    type_syntax(body, table, phase),
                ]
            )
        case SList([Symbol() as head, *_]) if head.identifier == "forall#":
            raise ParseError(f"malformed forall#: {expr}")
        case SList([Symbol() as head, *_]) if head.identifier.startswith("@"):
            # A type constructor's identifier written at the wrong arity, or an
            # identifier that names no constructor at all.
            raise ParseError(f"malformed or unknown type form: {expr}")
        case _:
            # A datum, the empty list, and a list headed by anything but a type
            # constructor's identifier. A list headed by a name is a
            # `(name type)` pair, read only where a product's fields and a sum's
            # constructors are.
            raise ParseError(f"not a type: {expr}")


def binder_of(
    variable: Expr, table: BindingTable, phase: Phase | None = None
) -> Symbol:
    # A `forall#`'s binder position. Loosely matched and then narrowed, so a
    # non-symbol reads as the malformed quantifier it is rather than failing the
    # arm and being reported as an unknown type form.
    match variable:
        case Symbol() as symbol:
            return binder(symbol, table, phase)
        case _:
            raise ParseError(f"not a type variable: {variable}")


def parse(expr: Expr, table: BindingTable, phase: Phase | None = None) -> Expr:
    match expr:
        case Symbol() as symbol:
            return reference(symbol, table, phase)
        case Integer() | Char():
            return expr
        case SList([Symbol() as head, SList(parameters), body]) if (
            head.identifier == "lambda"
        ):
            binders: list[Expr] = [
                binder_position(parameter, table, phase) for parameter in parameters
            ]
            return SList([Symbol("lambda"), SList(binders), parse(body, table, phase)])
        case SList([Symbol() as head, SList(pairs), body]) if (
            head.identifier == "letrec"
        ):
            bindings: list[Expr] = []
            for pair in pairs:
                match pair:
                    case SList([name, initializer]):
                        # `name` is matched loosely: a binder position is a
                        # bare symbol or an `ascribe#` wrapper, and
                        # `binder_position` is what tells those apart.
                        bindings.append(
                            SList(
                                [
                                    binder_position(name, table, phase),
                                    parse(initializer, table, phase),
                                ]
                            )
                        )
                    case _:
                        raise ParseError(f"malformed letrec binding: {pair}")
            return SList([Symbol("letrec"), SList(bindings), parse(body, table, phase)])
        case SList([Symbol() as head, scrutinee, *clauses]) if (
            head.identifier == "match"
        ) and clauses:
            return SList(
                [
                    Symbol("match"),
                    parse(scrutinee, table, phase),
                    *[pattern_clause(clause, table, phase) for clause in clauses],
                ]
            )
        case SList([Symbol() as head, *_]) if head.identifier == "match":
            # A `match` with no clause, or written with no scrutinee at all.
            raise ParseError(f"malformed match: {expr}")
        case SList([Symbol() as head, datum]) if head.identifier == "quote":
            return SList([Symbol("quote"), strip(datum)])
        case SList([Symbol() as head, operand]) if head.identifier == "forall":
            return SList([Symbol("forall"), binder_position(operand, table, phase)])
        case SList([Symbol() as head, *_]) if head.identifier == "forall":
            raise ParseError(f"malformed forall: {expr}")
        case SList([Symbol() as head, name, init]) if head.identifier == "define":
            return SList(
                [
                    Symbol("define"),
                    binder_position(name, table, phase),
                    parse(init, table, phase),
                ]
            )
        case SList([Symbol() as head, *_]) if head.identifier in {
            "lambda",
            "letrec",
            "letrec-syntax",
            "syntax-rules",
            "match",
            "quote",
            "define",
            "ascribe#",
            "forall#",
        }:
            raise ParseError(f"malformed or un-expanded core form: {expr}")
        case SList([]):
            raise ParseError(f"empty combination: {expr}")
        case SList(items):
            return SList([parse(item, table, phase) for item in items])
        case _:
            raise ParseError(f"not a parsable expression: {expr}")
