from typing import Final

from Evaluate.Error import EvalError
from Expander import ExpandError, ParseError
from Expander import expand as expand_syntax
from Expander import parse as parse_syntax
from Resolve import Ambiguous, bindings
from Syntax import Boolean, Expr, Integer, SList, Value
from Syntax.Value import Primitive

__all__: list[str] = ["BUILTINS"]


def head(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case SList([first, *_]):
            return first
        case SList([]):
            raise EvalError("head: empty list")
        case _:
            raise EvalError(f"head: not a list: {value}")


def tail(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case SList([_, *rest]):
            return SList(rest)
        case SList([]):
            raise EvalError("tail: empty list")
        case _:
            raise EvalError(f"tail: not a list: {value}")


def cons(arguments: list[Value]) -> Value:
    [value, rest] = arguments

    match value, rest:
        case (Expr() as datum, SList(items)):
            return SList([datum, *items])
        case (Expr(), _):
            raise EvalError(f".: not a list: {rest}")
        case _:
            raise EvalError(f".: a procedure cannot be list data: {value}")


def le(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return Boolean(a <= b)
        case (Integer(), _):
            raise EvalError(f"le: not an integer: {other}")
        case _:
            raise EvalError(f"le: not an integer: {value}")


def lt(arguments: list[Value]) -> Value:
    [value, other] = arguments

    match value, other:
        case (Integer(a), Integer(b)):
            return Boolean(a < b)
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
            return Boolean(value == other)
        case (Expr(), _):
            raise EvalError(f"eq: a procedure cannot be compared: {other}")
        case _:
            raise EvalError(f"eq: a procedure cannot be compared: {value}")


def length(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case SList(items):
            return Integer(len(items))
        case _:
            raise EvalError(f"len: not a list: {value}")


def expand_program(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case Expr() as program:
            try:
                return expand_syntax(program)
            except ExpandError as error:
                raise EvalError(f"expand: {error}") from error
        case _:
            raise EvalError(f"expand: not syntax: {value}")


def parse_program(arguments: list[Value]) -> Value:
    [value] = arguments

    match value:
        case Expr() as program:
            try:
                return parse_syntax(program, bindings(program))
            except (Ambiguous, ParseError) as error:
                raise EvalError(f"parse: {error}") from error
        case _:
            raise EvalError(f"parse: not syntax: {value}")


BUILTINS: Final[dict[str, Value]] = {
    "head": Primitive("head", 1, head),
    "tail": Primitive("tail", 1, tail),
    ".": Primitive(".", 2, cons),
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
