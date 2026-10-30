from dataclasses import dataclass, field
from math import factorial
from typing import Final, Self

from Analysis.Support import Support
from Syntax import Symbol

__all__: list[str] = [
    "CEILING",
    "Constraints",
    "Edge",
    "box",
    "extensions",
    "probability",
    "total",
]

# The most ground unknowns a linear extension is counted over. The count is
# subset DP, so the work is `2^n · n` — at sixteen that is a million steps, and
# each further element doubles it. A wider constraint set is answered as
# unmeasured rather than waited on.
CEILING: Final[int] = 16


@dataclass(frozen=True, slots=True)
class Edge:
    # `lesser` before `greater`, strictly where `strict`.
    #
    # Written as one relation rather than as `lt` and `le` apart, because the
    # measure below counts orderings of distinct elements and a `≤` between two
    # distinct values *is* a `<`. The flag is kept because it is a fact about
    # the guard that produced the edge, and `infeasible` reads it.
    lesser: Symbol
    greater: Symbol
    strict: bool


@dataclass(frozen=True, slots=True)
class Constraints:
    # What a walk committed its ground unknowns to.
    #
    # Two sorts of fact, kept apart because they are measured differently. A
    # support is unary — it is about one value, and counting the assignments
    # that satisfy a set of them is a product of cardinalities. An edge is
    # relational, and counting is counting linear extensions. No single number
    # is both, which is why `box` and `extensions` are two functions and not
    # one.
    supports: dict[Symbol, Support] = field(default_factory=dict)
    order: tuple[Edge, ...] = ()

    def support(self: Self, variable: Symbol) -> Support:
        # What a variable is known to lie in, which is every integer until
        # something says otherwise.
        return self.supports.get(variable, Support.integers())

    def narrowing(self: Self, variable: Symbol, support: Support) -> Constraints:
        # These constraints with `variable` also known to lie in `support`.
        # Intersected and not replaced: a walk only ever learns more.
        return Constraints(
            {**self.supports, variable: self.support(variable).intersection(support)},
            self.order,
        )

    def ordering(self: Self, edge: Edge) -> Constraints:
        return Constraints(self.supports, (*self.order, edge))

    def variables(self: Self) -> set[Symbol]:
        return {
            *self.supports,
            *(edge.lesser for edge in self.order),
            *(edge.greater for edge in self.order),
        }

    def infeasible(self: Self) -> bool:
        # Whether nothing satisfies these constraints at all.
        #
        # An empty support is the unary reason and `Support.is_empty` is it. The
        # relational reason is a circle: `x < y` and `y ≤ x` together say
        # `x < x`, and no value is. This is the reason `Analysis.unreachable`
        # has no way to see — it reads what one variable may be, and a circle is
        # about two.
        #
        # A walk found infeasible is dropped where it is formed. It weighs
        # nothing, so no sum changes, and following it further only produces
        # more walks that weigh nothing — which is most of what a `sort` under
        # a budget would otherwise spend the budget on.
        if any(support.is_empty() for support in self.supports.values()):
            return True

        return cyclic(self.order)


def reachability(order: tuple[Edge, ...]) -> dict[Symbol, set[Symbol]]:
    # What each variable is transitively below, by closing the edges.
    #
    # Written as a closure rather than as a search per edge because every edge
    # is asked about, and an ordering is small: a comparison per guard, and a
    # walk carries as many guards as it has branches.
    above: dict[Symbol, set[Symbol]] = {}

    for edge in order:
        above.setdefault(edge.lesser, set()).add(edge.greater)
        above.setdefault(edge.greater, set())

    for pivot in list(above):
        for lower in above:
            if pivot in above[lower]:
                above[lower] |= above[pivot]

    return above


def cyclic(order: tuple[Edge, ...]) -> bool:
    # Whether the ordering runs round in a circle.
    #
    # Any circle and not only a strict one, because the measure counts
    # orderings of *distinct* keys: a circle of `≤` forces everything on it
    # equal, and no ordering of distinct keys does that. So this and
    # `extensions` answering zero are one fact, which is what lets a walk be
    # pruned the moment it is formed rather than followed, weighed, and found
    # to have been about nothing.
    #
    # `Edge.strict` is recorded all the same. It is what the guard actually
    # said, and a measure admitting ties would have to tell the two circles
    # apart; this one has no need to.
    above: dict[Symbol, set[Symbol]] = reachability(order)

    return any(edge.lesser in above.get(edge.greater, set()) for edge in order)


def box(constraints: Constraints, over: tuple[Symbol, ...]) -> int:
    # How many assignments to `over` satisfy the unary facts: the product of
    # the cardinalities, which is exact where there is no relation to count.
    #
    # The unary reading, and `Support.cardinality`'s caller. Not comparable with
    # `extensions` below and never summed with it: this counts assignments and
    # that counts orderings.
    rets: int = 1

    for variable in over:
        rets *= constraints.support(variable).cardinality()

    return rets


def extensions(constraints: Constraints, over: tuple[Symbol, ...]) -> int | None:
    # How many orderings of `over` satisfy the walk's comparisons — the number
    # of linear extensions of the partial order they describe.
    #
    # The measure a sorting procedure asks for. An input of `n` distinct keys
    # arrives in one of `n!` orders; a walk's comparisons admit some of them,
    # and this is how many. So a walk's share of the inputs is this over `n!`,
    # and the shares of a form's walks sum to one — which is the law that makes
    # this a measure and the test that catches a branch missed or counted twice.
    #
    # Distinct keys is the model and is why every edge counts as strict: two
    # distinct values related by `≤` are related by `<`. A cycle then admits no
    # ordering, so an infeasible walk counts zero without being special-cased.
    #
    # `None` where the question is not this one:
    #
    #   * an edge touching a variable outside `over`. The count would no longer
    #     be a count of *input* orderings, and the shares would stop summing to
    #     one.
    #   * a variable of `over` narrowed by a unary fact. An ordering says which
    #     element is larger and cannot say that one of them is three.
    #   * more than `CEILING` variables, which is the work and not the meaning.
    if len(over) > CEILING:
        return None

    placed: dict[Symbol, int] = {name: index for index, name in enumerate(over)}

    if any(
        endpoint not in placed
        for edge in constraints.order
        for endpoint in (edge.lesser, edge.greater)
    ):
        return None

    if any(
        not constraints.support(variable) == Support.integers() for variable in over
    ):
        return None

    width: int = len(over)
    below: list[int] = [0] * width

    for edge in constraints.order:
        below[placed[edge.greater]] |= 1 << placed[edge.lesser]

    # `counted[chosen]` is how many orderings place exactly the elements of
    # `chosen` first. An element may be placed once everything below it has
    # been, so a cycle leaves some element forever unplaceable and the full set
    # is never reached.
    counted: list[int] = [0] * (1 << width)
    counted[0] = 1

    for chosen in range(1 << width):
        if not counted[chosen]:
            continue

        for index in range(width):
            if chosen & (1 << index):
                continue

            if below[index] & ~chosen:
                continue

            counted[chosen | (1 << index)] += counted[chosen]

    return counted[(1 << width) - 1]


def total(over: tuple[Symbol, ...]) -> int:
    # How many orderings `over` has at all, which is what a walk's count is a
    # share of.
    return factorial(len(over))


def probability(weight: int | None, over: tuple[Symbol, ...]) -> float | None:
    if weight is None:
        return None

    return weight / total(over)
