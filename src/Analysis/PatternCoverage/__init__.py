from dataclasses import dataclass
from typing import Self

from Resolve import BindingTable
from Syntax import Expr, SList, Symbol
from Syntax.Type import Context, Type
from TypeCheck import TypeChecker, TypeCheckError
from TypeCheck.Patterns import missing, useful

__all__: list[str] = [
    "Coverage",
    "CoverageError",
    "clause_body",
    "clause_pattern",
    "clauses",
    "coverage",
    "probing",
]


class CoverageError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Coverage:
    # What the clauses of one `match` cover of the type its scrutinee has.
    #
    # The analysis reads a clause as a fact about the scrutinee: reaching
    # clause `k` means the clauses before it failed and that one matched. That
    # describes the program only where some clause always matches, so this is
    # the precondition `refinements` reads a guard under — and the two readings
    # it is are one walk, because a clause covers nothing new exactly when a
    # row of wildcards would still be useful without it.
    scrutinee: Type

    # A pattern matching some value no clause matches, or `None` where the
    # clauses leave nothing unmatched. A pattern and not a value: adding it as
    # a clause is what repairs the `match` it is reported about.
    witness: Expr | None

    # The clauses matching nothing an earlier clause did not, by position. A
    # clause here is reached by no value at all, which is a stronger reason
    # than the contradictory guard `Analysis.unreachable` already reads: that
    # one is about what a variable may be, and this is about the shape alone.
    redundant: tuple[int, ...]

    def complete(self: Self) -> bool:
        return self.witness is None


def clause_pattern(clause: Expr) -> Expr:
    match clause:
        case SList([pattern, _]):
            return pattern
        case _:
            return clause


def clause_body(clause: Expr) -> Expr:
    match clause:
        case SList([_, body]):
            return body
        case _:
            return clause


def clauses(form: Expr) -> tuple[Expr, list[Expr]] | None:
    # A `match`'s scrutinee and its clauses, or `None` where the form is not
    # one. A `match` with no clause is refused by `parse`, so one answered here
    # always has a clause to read.
    match form:
        case SList([Symbol() as head, scrutinee, *written]) if (
            head.identifier == "match"
        ) and written:
            return scrutinee, list(written)
        case _:
            return None


def probing(store: Context) -> Context:
    # `store` with its two mutable namespaces copied, so reading a type cannot
    # write one.
    #
    # `type_of` and `pattern` register the binders they read through — a
    # pattern's binders are what its clause's body is checked under — and the
    # analysis is asking a question, not checking a program. The scoped
    # namespaces are shared by value already, so only the two dictionaries have
    # to be copied.
    return Context(
        dict(store.ascriptions),
        store.variables,
        dict(store.schemes),
        store.rigid,
    )


def coverage(form: Expr, table: BindingTable, store: Context) -> Coverage | None:
    # What one `match` covers, or `None` where coverage is not a question that
    # can be asked of it.
    #
    # Two readings answer `None`, and both are about the question being the
    # wrong one rather than about the answer being bad:
    #
    #   * A scrutinee whose type cannot be read. Coverage is a question about a
    #     type — which constructors there are to name — so a `match` on an
    #     unascribed binder or on a `lambda` has nothing to be covered.
    #   * A clause whose pattern the checker refuses for that type. A `#t`
    #     pattern on an `@int` scrutinee is a type error, and reporting it as a
    #     coverage gap would have the analysis refuse programs on the type
    #     checker's behalf — in a phase that runs after it, and in words about
    #     the wrong thing.
    #
    # The second is read by running each pattern through `TypeChecker.pattern`,
    # so the rule is the checker's own and not a second opinion about it. That
    # is also what makes the rows safe to walk: `missing` reads a constructor
    # pattern's operands as the constructor's, and a pattern written with the
    # wrong count of them is refused here before it is counted there.
    read: tuple[Expr, list[Expr]] | None = clauses(form)

    if read is None:
        return None

    scrutinee, written = read
    checker: TypeChecker = TypeChecker(table=table, store=probing(store))
    patterns: list[Expr] = [clause_pattern(clause) for clause in written]

    try:
        ty: Type = checker.type_of(scrutinee)

        for pattern in patterns:
            checker.pattern(pattern, ty)
    except TypeCheckError:
        return None

    rows: list[list[Expr]] = [[pattern] for pattern in patterns]
    found: list[Expr] | None = missing(rows, [ty], checker.store, table)

    return Coverage(
        scrutinee=ty,
        # One column was asked about, so a witness row is the one pattern a
        # clause would have to add.
        witness=None if found is None else found[0],
        redundant=tuple(
            index
            for index in range(len(rows))
            if not useful(rows[:index], rows[index], [ty], checker.store, table)
        ),
    )
