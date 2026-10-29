from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self, override

from Syntax import Integer

__all__: list[str] = [
    "Support",
]


# A support is the set of values an integer variable may take at some point in
# a program, represented as a finite union of disjoint closed intervals kept in
# a normal form: sorted ascending, inside the ground range, and with at least
# one missing integer between neighbours. The normal form is what makes the
# structural `__eq__` a set equality.
@dataclass(frozen=True, slots=True)
class Support:
    intervals: tuple[tuple[int, int], ...]

    def __post_init__(self: Self) -> None:
        lower, upper = Integer.bounds()
        previous: int | None = None

        for lo, hi in self.intervals:
            if not lower <= lo <= hi <= upper:
                raise ValueError(f"malformed interval [{lo}, {hi}] in {self}")

            # Disjoint is not enough: two abutting intervals denote the same
            # set as their merge, so abutment is refused along with overlap.
            if previous is not None and lo <= previous + 1:
                raise ValueError(f"interval [{lo}, {hi}] is out of order in {self}")

            previous = hi

    @classmethod
    def of(cls, intervals: Iterable[tuple[int, int]]) -> Support:
        lower, upper = Integer.bounds()
        clipped: list[tuple[int, int]] = sorted(
            (max(lo, lower), min(hi, upper)) for lo, hi in intervals
        )
        rets: list[tuple[int, int]] = []

        for lo, hi in clipped:
            if lo > hi:
                continue

            if rets and lo <= rets[-1][1] + 1:
                rets[-1] = (rets[-1][0], max(rets[-1][1], hi))
            else:
                rets.append((lo, hi))

        return cls(tuple(rets))

    @classmethod
    def empty(cls) -> Support:
        return cls(())

    @classmethod
    def integers(cls) -> Support:
        return cls((Integer.bounds(),))

    @classmethod
    def less_than(cls, bound: int) -> Support:
        return cls.of([(Integer.bounds()[0], bound - 1)])

    @classmethod
    def point(cls, value: int) -> Support:
        return cls.of([(value, value)])

    def cardinality(self: Self) -> int:
        return sum(hi - lo + 1 for lo, hi in self.intervals)

    def is_empty(self: Self) -> bool:
        return not self.intervals

    def complement(self: Self) -> Support:
        lower, upper = Integer.bounds()
        rets: list[tuple[int, int]] = []
        cursor: int = lower

        for lo, hi in self.intervals:
            if cursor <= lo - 1:
                rets.append((cursor, lo - 1))

            cursor = hi + 1

        if cursor <= upper:
            rets.append((cursor, upper))

        return Support(tuple(rets))

    def intersection(self: Self, other: Support) -> Support:
        rets: list[tuple[int, int]] = []
        ours: int = 0
        theirs: int = 0

        while ours < len(self.intervals) and theirs < len(other.intervals):
            our_lo, our_hi = self.intervals[ours]
            their_lo, their_hi = other.intervals[theirs]

            lo: int = max(our_lo, their_lo)
            hi: int = min(our_hi, their_hi)

            if lo <= hi:
                rets.append((lo, hi))

            if our_hi < their_hi:
                ours += 1
            else:
                theirs += 1

        return Support(tuple(rets))

    def union(self: Self, other: Support) -> Support:
        return Support.of([*self.intervals, *other.intervals])

    def __contains__(self: Self, value: int) -> bool:
        return any(lo <= value <= hi for lo, hi in self.intervals)

    @override
    def __str__(self: Self) -> str:
        if not self.intervals:
            return "#<support>"

        body: str = " ∪ ".join(f"[{lo}, {hi}]" for lo, hi in self.intervals)
        return f"#<support {body}>"
