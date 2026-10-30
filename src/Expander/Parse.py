from Resolve import BindingTable, Phase, Unbound, resolve
from Syntax import (
    Boolean,
    Char,
    Expr,
    Integer,
    Nil,
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
    # The post-expansion shape of a binder position: a bare symbol, or an
    # `ascribe#` wrapper whose name is one. The wrapper is kept, because the
    # ascription is what the type checker reads the binder's type off of.
    #
    # The keyword is re-emitted with an empty scope set, as `parse` rebuilds
    # `lambda`: it names no value and resolves against nothing. The type operand
    # is read as a type — checked for its written form and resolved against the
    # same table every other identifier is.
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


def type_variable(
    symbol: Symbol, table: BindingTable, phase: Phase | None = None
) -> Symbol:
    # A type variable occurrence, resolved to its binder exactly as `reference`
    # resolves a term occurrence — except that an unresolved one is refused
    # rather than read as itself. A term identifier the table does not bind is
    # a builtin's; a type identifier nothing binds stands for no type.
    try:
        return resolve(symbol, table, phase).binder
    except Unbound:
        raise ParseError(f"unbound type variable: {symbol}") from None


def names_a_type(field: Expr) -> bool:
    # Whether a product's field or a sum's constructor is written as a
    # `(name type)` pair rather than as a type on its own. The two are told
    # apart by the head: every type constructor's identifier begins with `@`, so
    # a list headed by anything else names what follows it.
    match field:
        case SList([Symbol() as name, _]):
            return not name.identifier.startswith("@")
        case _:
            return False


def named_types(
    pairs: list[Expr], table: BindingTable, phase: Phase | None = None
) -> list[Expr]:
    # Every `(name type)` pair, which is the shape a record's fields and a sum's
    # constructors share. A name is neither a binder nor an occurrence — it
    # labels a position — so it is re-emitted with an empty scope set, as a core
    # form's identifier is.
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
            # Refused rather than collapsed: one of the two would be left with
            # no position, and which one would depend on the reading order.
            raise ParseError(f"duplicate name in one type: {written}")

        seen.add(written.identifier)
        named.append(SList([Symbol(written.identifier), type_syntax(ty, table, phase)]))

    return named


def product_fields(
    fields: list[Expr], table: BindingTable, phase: Phase | None = None
) -> list[Expr]:
    # `(@prod t ...)` is a tuple and `(@prod (x t) ...)` a record. A product is
    # one shape or the other throughout: `ProdType` keeps names for a record
    # only, so a mixed one has no reading that keeps every field.
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
    # A written type, checked for its written form and resolved against the one
    # binding table. The result is still syntax: this phase builds no `Type`,
    # because what a type denotes is not a question about how it is written.
    #
    # The two sorts of identifier are read as `parse` reads the two sorts in a
    # term. An identifier beginning with `@` is a type constructor's and is
    # re-emitted with an empty scope set, as a core form's is. Any other
    # identifier is a variable and resolves to its binder, as a reference does.
    #
    # Not read here: the rank-1 restriction, so a `forall#` written inside a
    # type resolves and is emitted; whether a `@fun`'s first operand denotes an
    # argument run; and whether a `@mu` is guarded. Each is about meaning, and
    # each belongs to the phase that builds the `Type`.
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
        case Integer() | Boolean() | Char() | Nil():
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
        case SList([Symbol() as head, test, consequent, alternative]) if (
            head.identifier == "if"
        ):
            return SList(
                [
                    Symbol("if"),
                    parse(test, table, phase),
                    parse(consequent, table, phase),
                    parse(alternative, table, phase),
                ]
            )
        case SList([Symbol() as head, datum]) if head.identifier == "quote":
            return SList([Symbol("quote"), strip(datum)])
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
            "if",
            "quote",
            "define",
            "ascribe#",
            "forall#",
        }:
            # `forall#` is listed for the reason `ascribe#` is: neither names a
            # value, so a form headed by one at expression position is a
            # misplaced type and not an application of something unbound.
            raise ParseError(f"malformed or un-expanded core form: {expr}")
        case SList([]):
            raise ParseError(f"empty combination: {expr}")
        case SList(items):
            return SList([parse(item, table, phase) for item in items])
        case _:
            raise ParseError(f"not a parsable expression: {expr}")
