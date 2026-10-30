from typing import Final

from Resolve import BindingTable, meaning
from Syntax import Expr, SList, Symbol, binder_name
from Syntax.Type import (
    Context,
    MuType,
    ProdType,
    Type,
    VarType,
    constructors,
    unfold,
)

__all__: list[str] = [
    "ANY",
    "WILDCARD",
    "defaulted",
    "exhaustive",
    "heads",
    "irrefutable",
    "missing",
    "operand_types",
    "pattern_head",
    "pattern_operands",
    "product_fields",
    "specialized",
    "useful",
]

# The pattern that matches every value, used where the reading below has to
# stand a wildcard in for the operands of a constructor a clause did not name.
#
# A bare symbol already *is* an irrefutable pattern, so no new sort is needed —
# and this is the one symbol no reference resolves to, which is what keeps it
# from being mistaken for a binder a program wrote.
ANY: Final[Symbol] = Symbol.toplevel()

# The wildcard a witness is written with, which `ANY` cannot be: `ANY` is the
# symbol with no name and prints as nothing, while a witness is read by whoever
# the refusal is addressed to.
#
# Spelled as a name a program may write, because that is what makes a witness a
# pattern: `_` in a clause is a binder, and a binder matches every value — so
# the witness a refusal prints is the clause that repairs it.
WILDCARD: Final[Symbol] = Symbol("_")


def product_fields(ty: Type) -> list[Type]:
    # A product's field types, and the type itself where it is not a product.
    #
    # Says nothing about how a pattern is written — `operand_types` below is
    # that rule — and is read where a product's fields are wanted whatever
    # stands in the position: `quotation` reads a cons cell's payload through
    # this to name the types its element and its rest are checked at.
    match ty:
        case ProdType() as product:
            return list(product.types())
        case _:
            return [ty]


def operand_types(ty: Type, payload: Type, store: Context) -> list[Type]:
    # The types a constructor pattern's operands are read against, which are
    # the types `construction` reads a construction's operands against: a
    # product is written as its fields, and every other constructor as its one
    # payload. So a pattern is written exactly as the construction it inverts,
    # and `(cons (prod# x xs))` takes apart what `(cons (prod# 1 ()))` builds.
    #
    # Dispatching on the type taken apart and not on the payload is the whole
    # of it. A sum constructor carries one payload however that payload is
    # written, so a product among them is one operand here and is spread only
    # by the pattern written for it — where reading the payload instead would
    # spread it unasked, and leave `#t` a constructor written with no operands
    # at all because the product it carries has no fields.
    #
    # Read through the forms that stand for another type, as `constructors`
    # reads them: a `@mu` as its unfolding, and a variable as what is bound to
    # it. The two must agree about which sort is in hand, and every list type
    # is written as a `@mu`.
    match ty:
        case MuType() as recursive:
            return operand_types(unfold(recursive), payload, store)
        case VarType(binder=binder) if binder not in store.rigid:
            return operand_types(store.variable(binder), payload, store)
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


def heads(matrix: list[list[Expr]], table: BindingTable) -> set[Symbol]:
    # Every constructor some row names in the first column. A row naming none —
    # a binder position, or a literal — contributes nothing, which is what
    # keeps a column of wildcards from completing a signature.
    return {
        named for row in matrix if (named := pattern_head(row[0], table)) is not None
    }


def specialized(
    constructor: Symbol,
    columns: list[Type],
    matrix: list[list[Expr]],
    table: BindingTable,
) -> list[list[Expr]]:
    # The rows that can match where the first column is `constructor`, with
    # that column replaced by one per operand the constructor is written with.
    #
    # A row naming a different constructor cannot match here and drops out; an
    # irrefutable row matches whatever the constructor is, and stands a
    # wildcard in for each of its operands. `columns` is what `operand_types`
    # answers for the constructor, so the count of wildcards is the count of
    # patterns a row naming it would have written.
    rets: list[list[Expr]] = []

    for row in matrix:
        if irrefutable(row[0]):
            rets.append([*[ANY] * len(columns), *row[1:]])
        elif pattern_head(row[0], table) == constructor:
            rets.append([*pattern_operands(row[0]), *row[1:]])

    return rets


def defaulted(matrix: list[list[Expr]]) -> list[list[Expr]]:
    # The rows that match whatever stands in the first column, with the column
    # dropped. Read where some constructor of the column is named by no row, or
    # where the column has no constructors to name: either way only an
    # irrefutable row covers what is left.
    return [row[1:] for row in matrix if irrefutable(row[0])]


def uncovered(
    ty: Type,
    cons: dict[Symbol, Type] | None,
    present: set[Symbol],
    store: Context,
) -> Expr:
    # A pattern for a value of `ty` that no row naming a constructor in
    # `present` matches: a constructor the set does not name, written with a
    # wildcard for each of its operands.
    #
    # A bare wildcard where there is no such constructor to name, which is the
    # honest answer and not a precise one. `@int` and `@char` have a
    # constructor per value and no run of literals enumerates them, so what a
    # clause leaves unmatched there is "some other value" — and `_` is how that
    # is written.
    unnamed: list[Symbol] = [
        constructor for constructor in (cons or {}) if constructor not in present
    ]

    if cons is None or not unnamed:
        return WILDCARD

    constructor: Symbol = unnamed[0]
    columns: list[Type] = operand_types(ty, cons[constructor], store)

    return SList([constructor, *[WILDCARD] * len(columns)])


def missing(
    matrix: list[list[Expr]],
    types: list[Type],
    store: Context,
    table: BindingTable,
) -> list[Expr] | None:
    # A row of patterns matching some value no row of `matrix` matches, or
    # `None` where the matrix leaves nothing unmatched. The clauses of a
    # `match` read as a matrix: one row per clause, one column per value being
    # discriminated on. A `match` starts as one column — the scrutinee — and a
    # constructor pattern replaces its column with one per operand it is
    # written with, which `operand_types` names.
    #
    # Counting the constructors a clause names is not enough, which is why this
    # is a matrix and not a set. `((nil (prod#)) a) ((cons (prod# x (nil
    # (prod#)))) b)` names both constructors of a list and still misses every
    # list of two or more, so coverage of a constructor is a question about its
    # operands, asked recursively.
    #
    # A sum constructor contributes the one column its payload is, and the
    # product pattern written for that payload is what splits it into the
    # fields. So a list is unfolded two levels per cell rather than one, and
    # terminates for the same reason either way: the wildcard columns
    # `specialized` introduces never complete a signature.
    #
    # The precondition is that every pattern has already been checked against
    # the type of the position it occupies, so a row's length matches `types`
    # and a constructor's operand count matches the constructor. `matching`
    # checks every clause before asking this.
    if not types:
        # Nothing left to discriminate on, so any row at all matches — and
        # where no row remains, what is unmatched is the row of no patterns.
        return None if matrix else []

    head_type: Type = types[0]
    rest: list[Type] = types[1:]
    cons: dict[Symbol, Type] | None = constructors(head_type, store)
    present: set[Symbol] = heads(matrix, table)

    if cons is not None and set(cons) <= present:
        # Every constructor of this column is named by some row, so the column
        # is covered exactly when each constructor's own operands are.
        for constructor, payload in cons.items():
            columns: list[Type] = operand_types(head_type, payload, store)
            witness: list[Expr] | None = missing(
                specialized(constructor, columns, matrix, table),
                [*columns, *rest],
                store,
                table,
            )

            if witness is not None:
                # A witness over this constructor's operands followed by the
                # columns after it, so the operands are read back off the front
                # and written as the construction they take apart.
                return [
                    SList([constructor, *witness[: len(columns)]]),
                    *witness[len(columns) :],
                ]

        return None

    # Some constructor of this column is named by no row — or the column has no
    # constructors to name, which is every `@int` and `@char`. Either way the
    # column is dropped, and what is unmatched is a value no row names in it
    # followed by whatever the remaining columns leave.
    #
    # This is also what makes the reading terminate over a recursive type. The
    # branch above introduces wildcard columns, and a matrix of nothing but
    # wildcards never completes a signature — so `(@list t)` is unfolded only
    # as far as some clause actually wrote a pattern.
    witness = missing(defaulted(matrix), rest, store, table)

    if witness is None:
        return None

    return [uncovered(head_type, cons, present, store), *witness]


def exhaustive(
    matrix: list[list[Expr]],
    types: list[Type],
    store: Context,
    table: BindingTable,
) -> bool:
    # Whether the clauses of a `match` leave no value of the scrutinee's type
    # unmatched. `missing` is the reading and this is the question asked of it,
    # so there is one walk and the refusal can name what it found.
    return missing(matrix, types, store, table) is None


def useful(
    matrix: list[list[Expr]],
    row: list[Expr],
    types: list[Type],
    store: Context,
    table: BindingTable,
) -> bool:
    # Whether `row` matches some value no row of `matrix` matches — the
    # relation `missing` is the constructive half of. A clause is worth writing
    # exactly when it is useful against the clauses before it, and a `match` is
    # complete exactly when a row of wildcards is useful against none of them.
    #
    # Read over the same columns as `missing` and through the same
    # specialization, so the two agree about what a constructor pattern takes
    # apart.
    if not types:
        # The row of no patterns matches the value of no columns, which every
        # row of the matrix matches too — so it is useful exactly where no row
        # remains.
        return not matrix

    head: Expr = row[0]
    head_type: Type = types[0]
    rest: list[Type] = types[1:]
    cons: dict[Symbol, Type] | None = constructors(head_type, store)
    named: Symbol | None = pattern_head(head, table)
    columns: list[Type]

    if named is not None:
        if cons is None or named not in cons:
            # A head naming no constructor of the column's type is a pattern
            # the checker refuses, and nothing is known about the values it
            # matches. Answered as useful, because a clause is called redundant
            # only where it was read and found to be.
            return True

        columns = operand_types(head_type, cons[named], store)

        return useful(
            specialized(named, columns, matrix, table),
            [*pattern_operands(head), *row[1:]],
            [*columns, *rest],
            store,
            table,
        )

    if not irrefutable(head):
        # A literal, which matches the one value it is. The rows that can match
        # that value are those irrefutable in this column and those written as
        # the same literal, and the column is then dropped — the reading
        # `defaulted` makes, narrowed by an equality.
        #
        # The arm `missing` has no use for: a literal never completes a
        # signature, so coverage drops it, and it is only here that a run of
        # literals is read against itself. It is what makes a repeated literal
        # clause the redundancy it is.
        narrowed: list[list[Expr]] = [
            other[1:] for other in matrix if irrefutable(other[0]) or other[0] == head
        ]

        return useful(narrowed, row[1:], rest, store, table)

    if cons is not None and set(cons) <= heads(matrix, table):
        # An irrefutable pattern over a column the rows' constructors cover is
        # useful where it is useful for some one of them, with a wildcard
        # standing in for each of that constructor's operands.
        for constructor, payload in cons.items():
            columns = operand_types(head_type, payload, store)

            if useful(
                specialized(constructor, columns, matrix, table),
                [*[ANY] * len(columns), *row[1:]],
                [*columns, *rest],
                store,
                table,
            ):
                return True

        return False

    return useful(defaulted(matrix), row[1:], rest, store, table)
