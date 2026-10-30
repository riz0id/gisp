from typing import Final

from Evaluate.Error import EvalError
from Expander import ExpandError, ParseError
from Expander import expand as expand_syntax
from Expander import parse as parse_syntax
from Resolve import Ambiguous, bindings
from Syntax import (
    Expr,
    Integer,
    SumConExpr,
    Value,
    as_data,
    as_syntax,
    consed,
    elements,
    listed,
    spread,
    unit,
)
from Syntax import (
    boolean as boolean_value,
)
from Syntax import (
    product as product_value,
)
from Syntax.Value import Primitive

__all__: list[str] = ["BUILTINS"]


def cell(value: Value) -> tuple[Expr, Expr] | None:
    # The head and the rest a cons cell carries, or `None` where the value is
    # not a cell. The empty list is not one, which is what the selectors below
    # refuse.
    match value:
        case SumConExpr(constructor=con, payload=payload) if con.identifier == "cons":
            match spread(payload):
                case [first, rest]:
                    return (first, rest)

    return None


def head(arguments: list[Value]) -> Value:
    [value] = arguments

    match cell(value):
        case (first, _):
            return first
        case _ if elements(value) == []:
            raise EvalError("head: empty list")
        case _:
            raise EvalError(f"head: not a list: {value}")


def tail(arguments: list[Value]) -> Value:
    [value] = arguments

    match cell(value):
        case (_, rest):
            return rest
        case _ if elements(value) == []:
            raise EvalError("tail: empty list")
        case _:
            raise EvalError(f"tail: not a list: {value}")


def cons(arguments: list[Value]) -> Value:
    [payload] = arguments

    match payload:
        case Expr() as fields:
            match spread(fields):
                case [first, rest] if elements(rest) is not None:
                    return consed(first, rest)
                case [_, rest]:
                    raise EvalError(f"cons: not a list: {rest}")

    raise EvalError(f"cons: not a cons cell's payload: {payload}")


def nil(arguments: list[Value]) -> Value:
    [payload] = arguments

    if payload == unit():
        return listed([])

    raise EvalError(f"nil: not the empty product: {payload}")


def le(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return boolean_value(a <= b)
        case (Integer(), _):
            raise EvalError(f"le: not an integer: {other}")
        case _:
            raise EvalError(f"le: not an integer: {value}")


def lt(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return boolean_value(a < b)
        case (Integer(), _):
            raise EvalError(f"lt: not an integer: {other}")
        case _:
            raise EvalError(f"lt: not an integer: {value}")


def add(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return Integer(a + b)
        case (Integer(), _):
            raise EvalError(f"+: not an integer: {other}")
        case _:
            raise EvalError(f"+: not an integer: {value}")


def sub(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return Integer(a - b)
        case (Integer(), _):
            raise EvalError(f"-: not an integer: {other}")
        case _:
            raise EvalError(f"-: not an integer: {value}")


def mul(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return Integer(a * b)
        case (Integer(), _):
            raise EvalError(f"*: not an integer: {other}")
        case _:
            raise EvalError(f"*: not an integer: {value}")


def mod(arguments: list[Value]) -> Value:
    # Python remainder semantics: the result takes the divisor's sign. A zero
    # divisor is refused like any other ill-typed argument — the evaluator's
    # convention is to refuse loudly; only the stochastic analysis is total.
    [value, other] = arguments
    match value, other:
        case (Integer(_), Integer(0)):
            raise EvalError("mod: division by zero")
        case (Integer(a), Integer(b)):
            return Integer(a % b)
        case (Integer(), _):
            raise EvalError(f"mod: not an integer: {other}")
        case _:
            raise EvalError(f"mod: not an integer: {value}")


def eq(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Expr(), Expr()):
            return boolean_value(value == other)
        case (Expr(), _):
            raise EvalError(f"eq: a procedure cannot be compared: {other}")
        case _:
            raise EvalError(f"eq: a procedure cannot be compared: {value}")


def length(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case Expr() as datum if (items := elements(datum)) is not None:
            return Integer(len(items))
        case _:
            raise EvalError(f"len: not a list: {value}")


def boolean(name: str, which: bool) -> Primitive:
    def operate(arguments: list[Value]) -> Value:
        [payload] = arguments

        if payload == unit():
            return boolean_value(which)

        raise EvalError(f"{name}: not the empty product: {payload}")

    return Primitive(name, 1, operate)


def product(arguments: list[Value]) -> Value:
    fields: list[Expr] = []

    for value in arguments:
        match value:
            case Expr() as datum:
                fields.append(datum)
            case _:
                raise EvalError(
                    f"prod#: a procedure cannot be a product's field: {value}"
                )

    return product_value(fields)


def expand_program(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case Expr() as program:
            try:
                return as_data(expand_syntax(as_syntax(program)))
            except ExpandError as error:
                raise EvalError(f"expand: {error}") from error
        case _:
            raise EvalError(f"expand: not syntax: {value}")


def parse_program(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case Expr() as program:
            syntax: Expr = as_syntax(program)

            try:
                return as_data(parse_syntax(syntax, bindings(syntax)))
            except (Ambiguous, ParseError) as error:
                raise EvalError(f"parse: {error}") from error
        case _:
            raise EvalError(f"parse: not syntax: {value}")


BUILTINS: Final[dict[str, Value]] = {
    "#t": boolean("#t", True),
    "#f": boolean("#f", False),
    "nil": Primitive("nil", 1, nil),
    "cons": Primitive("cons", 1, cons),
    "prod#": Primitive("prod#", None, product),
    "head": Primitive("head", 1, head),
    "tail": Primitive("tail", 1, tail),
    "le": Primitive("le", 2, le),
    "lt": Primitive("lt", 2, lt),
    "+": Primitive("+", 2, add),
    "-": Primitive("-", 2, sub),
    "*": Primitive("*", 2, mul),
    "mod": Primitive("mod", 2, mod),
    "eq": Primitive("eq", 2, eq),
    "len": Primitive("len", 1, length),
    "expand": Primitive("expand", 1, expand_program),
    "parse": Primitive("parse", 1, parse_program),
}
