from typing import Final

from Resolve import BindingTable, meaning
from Syntax import Expr, SList, Symbol, binder_name
from Syntax.Type import Context, ProdType, Type, constructors

__all__: list[str] = [
    "ANY",
    "exhaustive",
    "irrefutable",
    "pattern_head",
    "pattern_operands",
    "payload_fields",
]

# The pattern that matches every value, used where the reading below has to
# stand a wildcard in for the fields of a constructor a clause did not name.
#
# A bare symbol already *is* an irrefutable pattern, so no new sort is needed —
# and this is the one symbol no reference resolves to, which is what keeps it
# from being mistaken for a binder a program wrote.
ANY: Final[Symbol] = Symbol.toplevel()


def payload_fields(payload: Type) -> list[Type]:
    # The types a constructor pattern's operands are read against: the
    # payload's fields when the payload is a product, and the payload itself
    # when it is not.
    #
    # This one rule is what makes the four sorts read uniformly. `@bool`'s
    # constructors carry the empty product, so `(#t)` is written with no
    # operands; a cons cell carries a two-field product, so `(. x xs)` is
    # written with two; and a sum constructor over a bare `@int` carries no
    # product at all, so `(succ n)` is written with one.
    match payload:
        case ProdType() as product:
            return list(product.types())
        case _:
            return [payload]


def irrefutable(pattern: Expr) -> bool:
    # Whether a pattern matches every value of its type. A binder position
    # does, and nothing else does: a literal refutes every other value, and a
    # constructor pattern refutes every other constructor.
    return binder_name(pattern) is not None


def pattern_head(pattern: Expr, table: BindingTable) -> Symbol | None:
    # The constructor a pattern names, or `None` where it names none — a binder
    # position and a literal both.
    #
    # The binder position is read first and not last: an `(ascribe# x t)` is a
    # list headed by a symbol, and reading the head would make it a constructor
    # pattern named `ascribe#`.
    if binder_name(pattern) is not None:
        return None

    match pattern:
        case SList([Symbol() as head, *_]):
            return meaning(head, table)
        case _:
            return None


def pattern_operands(pattern: Expr) -> list[Expr]:
    # A constructor pattern's operands, which are patterns of their own.
    match pattern:
        case SList([_, *operands]):
            return list(operands)
        case _:
            return []


def exhaustive(
    matrix: list[list[Expr]],
    types: list[Type],
    store: Context,
    table: BindingTable,
) -> bool:
    # Whether the clauses of a `match` leave no value of the scrutinee's type
    # unmatched, read as a matrix: one row per clause, one column per value
    # being discriminated on. A `match` starts as one column — the scrutinee —
    # and a constructor pattern replaces its column with one per field.
    #
    # Counting the constructors a clause names is not enough, which is why this
    # is a matrix and not a set. `((list) a) ((. x (list)) b)` names both
    # constructors of a list and still misses every list of two or more, so
    # coverage of a constructor is a question about its fields, asked
    # recursively.
    #
    # The precondition is that every pattern has already been checked against
    # the type of the position it occupies, so a row's length matches `types`
    # and a constructor's operand count matches its payload. `matching` checks
    # every clause before asking this.
    if not types:
        # Nothing left to discriminate on, so any row at all matches — and no
        # row means nothing does.
        return len(matrix) > 0

    head_type: Type = types[0]
    rest: list[Type] = types[1:]
    cons: dict[Symbol, Type] | None = constructors(head_type, store)
    present: set[Symbol] = {
        named for row in matrix if (named := pattern_head(row[0], table)) is not None
    }

    if cons is not None and set(cons) <= present:
        # Every constructor of this column is named by some row, so the column
        # is covered exactly when each constructor's own fields are. A row
        # naming a different constructor cannot match here and drops out; an
        # irrefutable row matches whatever the constructor is, and stands a
        # wildcard in for each of its fields.
        for constructor, payload in cons.items():
            fields: list[Type] = payload_fields(payload)
            specialized: list[list[Expr]] = []

            for row in matrix:
                if irrefutable(row[0]):
                    specialized.append([*[ANY] * len(fields), *row[1:]])
                elif pattern_head(row[0], table) == constructor:
                    specialized.append([*pattern_operands(row[0]), *row[1:]])

            if not exhaustive(specialized, [*fields, *rest], store, table):
                return False

        return True

    # Some constructor of this column is named by no row — or the column has no
    # constructors to name, which is every `@int` and `@char`. Either way only
    # an irrefutable row can cover what is left, so the column is dropped and
    # the rows that match everything in it are what remain.
    #
    # This is also what makes the reading terminate over a recursive type. The
    # branch above introduces wildcard columns, and a matrix of nothing but
    # wildcards never completes a signature — so `(@list t)` is unfolded only
    # as far as some clause actually wrote a pattern.
    defaulted: list[list[Expr]] = [row[1:] for row in matrix if irrefutable(row[0])]

    return exhaustive(defaulted, rest, store, table)
