from Evaluate.Builtins import BUILTINS
from Evaluate.Environment import Environment
from Evaluate.Error import EvalError, UnboundSymbol
from Syntax import (
    Boolean,
    Char,
    Closure,
    Expr,
    Integer,
    Nil,
    Primitive,
    SList,
    Symbol,
    Value,
    binder_name,
)

__all__ = [
    "BUILTINS",
    "Environment",
    "EvalError",
    "UnboundSymbol",
    "evaluate",
    "evaluate_toplevel",
    "apply_procedure",
]


def name(symbol: Symbol) -> str:
    # The environment is keyed by spelling and the evaluator is scope-blind,
    # so a symbol's runtime name is its identifier. An anonymous symbol
    # carries no spelling and has no runtime meaning.
    if not symbol.identifier:
        raise EvalError(f"no unique spelling: {symbol!r}")
    return symbol.identifier


def parameter_names(parameters: list[Expr]) -> tuple[str, ...] | None:
    # The runtime names of a lambda's parameters, or `None` when any parameter
    # is not a binder position. An ascription is erased here: the type names no
    # value, and a closure's arity is the count of these names, so folding the
    # type in would give the closure an extra parameter.
    names: list[str] = []

    for parameter in parameters:
        bound: Symbol | None = binder_name(parameter)

        if bound is None:
            return None

        names.append(name(bound))

    return tuple(names)


def letrec_bindings(bindings: list[Expr]) -> list[tuple[Symbol, Expr]] | None:
    # The ⟨binder, initializer⟩ of every letrec binding, or `None` when any
    # binding is not one. The arm guard and its body read the shape once
    # between them through this, so no binding can pass the guard and then be
    # silently dropped by a second, stricter match.
    pairs: list[tuple[Symbol, Expr]] = []

    for binding in bindings:
        match binding:
            case SList([binder, init]) if (bound := binder_name(binder)) is not None:
                pairs.append((bound, init))
            case _:
                return None

    return pairs


def evaluate_toplevel(form: Expr, environment: Environment) -> Value | None:
    # A program is a sequence of top-level forms, and this function is where
    # that sequence lives: define is a declaration against the environment the
    # drivers thread through it, not an expression, so it produces no value.
    # A lambda init captures that same mutable environment and the binding
    # lands before any call can look it up, which is why a defined procedure
    # is self-recursive with no pre-binding step.
    match form:
        case SList([Symbol() as head, *rest]) if head.identifier == "define":
            match rest:
                case [binder, init] if (bound := binder_name(binder)) is not None:
                    environment.define(name(bound), evaluate(init, environment))
                    return None
                case _:
                    raise EvalError(f"malformed define: {form}")
        case _:
            return evaluate(form, environment)


def evaluate(expr: Expr, environment: Environment) -> Value:
    match expr:
        case Integer() | Char() | Boolean() | Nil():
            return expr
        case Symbol() as symbol:
            return environment.lookup(name(symbol))
        case SList([]):
            raise EvalError("the empty combination cannot be evaluated")
        case SList([Symbol() as head, datum]) if head.identifier == "quote":
            # Quotation is the one runtime constructor of symbol and list
            # data: the datum is the value, verbatim.
            return datum
        case SList([Symbol() as head, test, consequent, alternative]) if (
            head.identifier == "if"
        ):
            if evaluate(test, environment) == Boolean(True):
                return evaluate(consequent, environment)
            else:
                return evaluate(alternative, environment)
        case SList([Symbol() as head, SList(bindings), body]) if (
            head.identifier == "letrec"
        ) and (pairs := letrec_bindings(bindings)) is not None:
            frame = Environment(bindings={}, parent=environment)
            for bound, _ in pairs:
                frame.define(name(bound), Nil())
            for bound, init in pairs:
                frame.assign(name(bound), evaluate(init, frame))
            return evaluate(body, frame)
        case SList([Symbol() as head, SList(parameters), body]) if (
            head.identifier == "lambda"
        ) and (names := parameter_names(parameters)) is not None:
            return Closure(names, body, environment)
        case SList([Symbol() as head, *_]) if head.identifier == "define":
            # Only the top-level dispatch above gives define a meaning; here
            # the form would otherwise fall into the application arm and die
            # as an unbound symbol, hiding what actually went wrong.
            raise EvalError("define: only allowed at the top level")
        case SList(items):
            return apply_procedure(
                operator=evaluate(items[0], environment),
                arguments=[evaluate(item, environment) for item in items[1:]],
            )
        case _:
            raise EvalError(f"cannot evaluate: {expr!r}")


def apply_procedure(operator: Value, arguments: list[Value]) -> Value:
    match operator:
        case Closure(parameters, body, environment):
            if len(parameters) != len(arguments):
                raise EvalError(
                    f"expected {len(parameters)} arguments, got {len(arguments)}"
                )
            else:
                frame: Environment = Environment(
                    bindings=dict(zip(parameters, arguments, strict=True)),
                    parent=environment,
                )

                return evaluate(body, frame)
        case Primitive(arity=arity, operate=operate):
            if len(arguments) != arity:
                raise EvalError(f"expected {arity} arguments, got {len(arguments)}")
            else:
                return operate(arguments)
        case _:
            raise EvalError(f"not a procedure: {operator!r}")
