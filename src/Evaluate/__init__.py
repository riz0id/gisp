from Evaluate.Builtins import BUILTINS
from Evaluate.Environment import Environment
from Evaluate.Error import EvalError, UnboundSymbol
from Syntax import (
    Char,
    Closure,
    Expr,
    Integer,
    Primitive,
    ProductConExpr,
    Residual,
    SList,
    SumConExpr,
    Symbol,
    Value,
    as_data,
    binder_name,
    spread,
    unit,
)

__all__ = [
    "BUILTINS",
    "Environment",
    "EvalError",
    "Residual",
    "UnboundSymbol",
    "evaluate",
    "evaluate_toplevel",
    "apply_procedure",
]


def name(symbol: Symbol) -> str:
    if not symbol.identifier:
        raise EvalError(f"no unique spelling: {symbol!r}")
    return symbol.identifier


def parameter_names(parameters: list[Expr]) -> tuple[str, ...] | None:
    names: list[str] = []

    for parameter in parameters:
        bound: Symbol | None = binder_name(parameter)

        if bound is None:
            return None

        names.append(name(bound))

    return tuple(names)


def letrec_bindings(bindings: list[Expr]) -> list[tuple[Symbol, Expr]] | None:
    pairs: list[tuple[Symbol, Expr]] = []

    for binding in bindings:
        match binding:
            case SList([binder, init]) if (bound := binder_name(binder)) is not None:
                pairs.append((bound, init))
            case _:
                return None

    return pairs


def destructure(pattern: Expr, value: Value) -> dict[str, Value] | None:
    bound: dict[str, Value] = {}

    return bound if matched(pattern, value, bound) else None


def matched(pattern: Expr, value: Value, bound: dict[str, Value]) -> bool:
    named: Symbol | None = binder_name(pattern)

    if named is not None:
        bound[name(named)] = value
        return True

    match pattern:
        case Integer() | Char():
            return pattern == value
        case SList([Symbol() as head, *operands]):
            return constructed(head, operands, value, bound)
        case _:
            raise EvalError(f"cannot match: {pattern!r}")


def constructed(
    head: Symbol, operands: list[Expr], value: Value, bound: dict[str, Value]
) -> bool:
    match value:
        case ProductConExpr() as fields if head.identifier == "prod#":
            # A product names its own fields, so the operands are read against
            # them directly; every other constructor carries a payload, and
            # `spread` is what reads through it.
            matching: list[Expr] = list(fields.values())
        case SumConExpr(constructor=con, payload=payload):
            if con.identifier != head.identifier:
                return False

            matching = spread(payload)
        case _:
            raise EvalError(f"cannot match ({head} ...) against {value}")

    if len(matching) != len(operands):
        raise EvalError(f"cannot match ({head} ...) against {value}")

    return all(
        matched(operand, field, bound)
        for operand, field in zip(operands, matching, strict=True)
    )


def unreduced(value: Value) -> Expr:
    match value:
        case Residual(form):
            return form
        case Closure() | Primitive():
            raise EvalError(f"no term denotes the procedure {value}")
        case _:
            return value


def stuck(arguments: list[Value]) -> bool:
    return any(isinstance(argument, Residual) for argument in arguments)


def evaluate_toplevel(form: Expr, environment: Environment) -> Value | None:
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
        case Integer() | Char():
            return expr
        case Symbol() as symbol:
            return environment.lookup(name(symbol))
        case SList([]):
            raise EvalError("the empty combination cannot be evaluated")
        case SList([Symbol() as head, datum]) if head.identifier == "quote":
            # The one place a datum becomes a value. No value is an `SList`,
            # so every list within is read here as the `cons` nest that is
            # one; an atom is already the value it denotes.
            return as_data(datum)
        case SList([Symbol() as head, _]) if head.identifier == "forall":
            return Residual(expr)
        case SList([Symbol() as head, scrutinee, *clauses]) if (
            head.identifier == "match"
        ) and clauses:
            decided: Value = evaluate(scrutinee, environment)

            if isinstance(decided, Residual):
                return Residual(SList([head, decided.form, *clauses]))

            for clause in clauses:
                match clause:
                    case SList([pattern, body]):
                        pass
                    case _:
                        raise EvalError(f"malformed match clause: {clause}")

                bound: dict[str, Value] | None = destructure(pattern, decided)

                if bound is not None:
                    return evaluate(
                        body, Environment(bindings=bound, parent=environment)
                    )

            raise EvalError(f"no clause matches: {decided}")
        case SList([Symbol() as head, SList(bindings), body]) if (
            head.identifier == "letrec"
        ) and (pairs := letrec_bindings(bindings)) is not None:
            frame = Environment(bindings={}, parent=environment)
            for bound, _ in pairs:
                frame.define(name(bound), unit())
            for bound, init in pairs:
                frame.assign(name(bound), evaluate(init, frame))
            return evaluate(body, frame)
        case SList([Symbol() as head, SList(parameters), body]) if (
            head.identifier == "lambda"
        ) and (names := parameter_names(parameters)) is not None:
            return Closure(names, body, environment)
        case SList([Symbol() as head, *_]) if head.identifier == "define":
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
        case Residual(form):
            return Residual(SList([form, *map(unreduced, arguments)]))
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
        case Primitive(name=primitive, arity=arity, operate=operate):
            if arity is not None and len(arguments) != arity:
                raise EvalError(f"expected {arity} arguments, got {len(arguments)}")
            elif stuck(arguments):
                return Residual(SList([Symbol(primitive), *map(unreduced, arguments)]))
            else:
                return operate(arguments)
        case _:
            raise EvalError(f"not a procedure: {operator!r}")
