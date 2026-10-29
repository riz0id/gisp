from typing import Mapping

from Resolve.Binding import Binding
from Resolve.BindingTable import BindingTable
from Resolve.Phase import Phase
from Syntax import Expr, SList, Symbol, binder_name

__all__: list[str] = [
    "Ambiguous",
    "Binding",
    "BindingTable",
    "Phase",
    "ResolveError",
    "Unbound",
    "bindings",
    "resolve",
    "same_variable",
]


class ResolveError(Exception):
    pass


class Unbound(ResolveError):
    pass


class Ambiguous(ResolveError):
    pass


def bindings(expr: Expr, phase: Phase | None = None) -> BindingTable:
    binders: list[Symbol] = []
    stack: list[Expr] = [expr]

    while stack:
        current: Expr = stack.pop()

        match current:
            case SList([Symbol() as head, SList(parameters), _]) if (
                head.identifier == "lambda"
            ):
                binders.extend(
                    name
                    for parameter in parameters
                    if (name := binder_name(parameter)) is not None
                )

                for item in reversed(current.items):
                    stack.append(item)
            case SList([Symbol() as head, SList(pairs), _]) if (
                head.identifier == "letrec"
            ):
                for pair in pairs:
                    match pair:
                        case SList([name, _]):
                            bound: Symbol | None = binder_name(name)
                            if bound is not None:
                                binders.append(bound)

                for item in reversed(current.items):
                    stack.append(item)
            case SList([Symbol() as head, name, _]) if head.identifier == "define":
                # Position 1 is a binder position; the initializer below is
                # still walked, so nested binders inside it are not lost.
                declared: Symbol | None = binder_name(name)

                if declared is not None:
                    binders.append(declared)

                for item in reversed(current.items):
                    stack.append(item)
            case SList(items):
                for item in reversed(items):
                    stack.append(item)

    return BindingTable(binders, phase)


def resolve(ref: Symbol, table: BindingTable, phase: Phase | None = None) -> Binding:
    layer: Mapping[Symbol, Binding] = table.at(
        Phase.runtime() if phase is None else phase
    )
    candidates: list[Symbol] = [
        b for b in layer if b.identifier == ref.identifier and b.scopes <= ref.scopes
    ]

    if candidates:
        best: Symbol = max(candidates, key=lambda b: len(b.scopes))

        if any(not (b.scopes <= best.scopes) for b in candidates):
            raise Ambiguous(str(ref))

        return layer[best]
    else:
        raise Unbound(str(ref))


def same_variable(
    a: Symbol, b: Symbol, table: BindingTable, phase: Phase | None = None
) -> bool:
    return resolve(a, table, phase) == resolve(b, table, phase)
