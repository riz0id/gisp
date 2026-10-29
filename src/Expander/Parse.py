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
from Syntax.Type import (
    BoolType,
    CharType,
    FunType,
    IntType,
    ListType,
    MuType,
    ProdType,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    VarType,
)

__all__: list[str] = [
    "ParseError",
    "parse",
    "parse_type",
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
    # Both the keyword and the type operand are re-emitted stripped, exactly
    # as a quoted datum is, and for the same reason `parse` rebuilds `lambda`
    # as a bare symbol: the type denotes a type and resolves against nothing,
    # so reading it as a reference would search for a binder that was never
    # registered.
    match parameter:
        case SList([head, *_]) if (ascribed := ascription(parameter)) is not None:
            name, ty = ascribed
            return SList([strip(head), binder(name, table, phase), strip(ty)])
        case _ if heads_ascribe(parameter):
            raise ParseError(f"malformed ascribe#: {parameter}")
        case Symbol() as symbol:
            return binder(symbol, table, phase)
        case _:
            raise ParseError(f"binder is not a symbol: {parameter}")


def names_a_type(field: Expr) -> bool:
    # Whether a product's field or a sum's constructor is written as a
    # `(name type)` pair rather than as a type on its own. The two are told
    # apart by the head: every type constructor is spelled `@...`, so a list
    # headed by anything else names what follows it.
    match field:
        case SList([Symbol() as name, _]):
            return not name.identifier.startswith("@")
        case _:
            return False


def named_types(pairs: list[Expr], variables: frozenset[str]) -> dict[Symbol, Type]:
    # The ⟨name, type⟩ of every `(name type)` pair, which is the shape a
    # record's fields and a sum's constructors share. Names are emitted bare: a
    # type carries no scopes once it is read, and `is_constructor` looks a
    # constructor up by the bare symbol `meaning` answers with.
    named: dict[Symbol, Type] = {}

    for pair in pairs:
        match pair:
            case SList([Symbol() as written, ty]) if not (
                written.identifier.startswith("@")
            ):
                name: Symbol = Symbol(written.identifier)
            case _:
                raise ParseError(f"not a named type: {pair}")

        if name in named:
            # Refused rather than collapsed: the mapping would keep one of the
            # two and leave the other with no position, which is the
            # duplicate-binder refusal `checked` makes of a form's binders.
            raise ParseError(f"duplicate name in one type: {name}")

        named[name] = parse_type(ty, variables)

    return named


def product(fields: list[Expr], variables: frozenset[str]) -> ProdType:
    # `(@prod t ...)` is a tuple and `(@prod (x t) ...)` a record. A product is
    # one shape or the other throughout: `ProdType` keeps names for a record
    # only, so a mixed one has no reading that keeps every field.
    match fields:
        case []:
            # The nullary product is the empty tuple however it is written,
            # which is what `ProdType` reads an empty record as.
            return ProdType()
        case [first, *_] if names_a_type(first):
            # `named_types` refuses a field that is not a pair, so a product
            # that starts named cannot finish positional.
            return ProdType(named_types(fields, variables))
        case _:
            for field in fields:
                if names_a_type(field):
                    raise ParseError(
                        f"a product is named throughout or not at all: {field}"
                    )

            return ProdType([parse_type(field, variables) for field in fields])


def argument_run(
    argument_type: Expr, variables: frozenset[str]
) -> ProdType | VariadicType | VariadicStarType:
    # The argument type of a `(@fun a r)`: the product or the variadic run a
    # function is applied to, and nothing else. These are the three sorts
    # `FunType.argument_type` admits and the three `substitute_arguments`
    # rebuilds — a type that classifies values describes no argument list.
    match parse_type(argument_type, variables):
        case ProdType() | VariadicType() | VariadicStarType() as run:
            return run
        case _:
            raise ParseError(f"not an argument type: {argument_type}")


def recursive(variable: Symbol, body: Expr, variables: frozenset[str]) -> MuType:
    # `(@mu a t)` binds `a` within `t`, so the body is read with that spelling
    # added. An inner `@mu` of the same spelling needs nothing of its own: the
    # inner binding is the one its body is read under, which is how `substitute`
    # and `Context.bind` already read a rebound variable.
    #
    # Guardedness is not read here, so `(@mu a a)` parses. It is the
    # precondition `Context.bind` documents and nothing enforces.
    if variable.identifier.startswith("@"):
        # Every `@` spelling is a type constructor's. A variable stands for a
        # type rather than naming one, so it is spelled as an ordinary name.
        raise ParseError(f"not a type variable: {variable}")

    bound: Symbol = Symbol(variable.identifier)

    return MuType(bound, parse_type(body, variables | {bound.identifier}))


def parse_type(expr: Expr, variables: frozenset[str] = frozenset()) -> Type:
    # A written type read as the type itself: the inverse of the spellings
    # `type_constructor_symbol` reports, and the one place surface syntax
    # becomes a `Type`. `variables` is the spellings the enclosing `@mu`s bind,
    # carried as spellings rather than as `Symbol`s because `Symbol` equality
    # covers scopes — the reading `checked` makes of a form's binders.
    #
    # Nothing of the type checker is read here. A `Context` answers what a
    # variable stands for and what a binder is ascribed, and both are facts
    # about a judgment that runs after this phase, over what it produced.
    match expr:
        case Symbol(identifier="@int"):
            return IntType()
        case Symbol(identifier="@char"):
            return CharType()
        case Symbol(identifier="@bool"):
            return BoolType()
        case Symbol() as symbol if symbol.identifier.startswith("@"):
            # Every other `@` spelling names a type constructor, and a
            # constructor is written as a list: `@list` is not a type, and
            # `(@list @int)` is.
            raise ParseError(f"not a type: {symbol}")
        case Symbol() as symbol if symbol.identifier in variables:
            # A type variable, emitted bare for the reason its binder is.
            return VarType(Symbol(symbol.identifier))
        case Symbol() as symbol:
            # Nothing binds the spelling, so it stands for no type. Refused
            # here rather than built into a `VarType` that every later reading
            # would have to refuse instead.
            raise ParseError(f"unbound type variable: {symbol}")
        case SList([Symbol() as head, element]) if head.identifier == "@list":
            return ListType(parse_type(element, variables))
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic":
            return VariadicType(parse_type(argument, variables))
        case SList([Symbol() as head, argument]) if head.identifier == "@variadic*":
            return VariadicStarType(parse_type(argument, variables))
        case SList([Symbol() as head, *fields]) if head.identifier == "@prod":
            return product(fields, variables)
        case SList([Symbol() as head, *constructors]) if head.identifier == "@sum":
            if not constructors:
                # Refused here so the diagnosis is this phase's, and not the
                # `ValueError` `SumType` raises on an empty mapping.
                raise ParseError(f"a sum has one or more constructors: {expr}")

            return SumType(named_types(constructors, variables))
        case SList([Symbol() as head, argument_type, return_type]) if (
            head.identifier == "@fun"
        ):
            return FunType(
                return_type=parse_type(return_type, variables),
                argument_type=argument_run(argument_type, variables),
            )
        case SList([Symbol() as head, Symbol() as variable, body]) if (
            head.identifier == "@mu"
        ):
            return recursive(variable, body, variables)
        case SList([Symbol() as head, *_]) if head.identifier.startswith("@"):
            # A type constructor's spelling written at the wrong arity, or a
            # spelling that names no constructor at all.
            raise ParseError(f"malformed or unknown type form: {expr}")
        case _:
            # A datum, the empty list, and a list headed by anything but a type
            # constructor's spelling. A list headed by a name is a `(name type)`
            # pair, which is read only where a product's fields and a sum's
            # constructors are.
            raise ParseError(f"not a type: {expr}")


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
        }:
            raise ParseError(f"malformed or un-expanded core form: {expr}")
        case SList([]):
            raise ParseError(f"empty combination: {expr}")
        case SList(items):
            return SList([parse(item, table, phase) for item in items])
        case _:
            raise ParseError(f"not a parsable expression: {expr}")
