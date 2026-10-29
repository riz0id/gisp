from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Self, overload, override

from Analysis.Path.Step import Step
from Syntax import Expr, SList

__all__: list[str] = [
    "Path",
]


@dataclass(frozen=True, slots=True)
class Path(Sequence[int]):
    steps: list[int] = field(default_factory=list)

    def child(self: Self, position: int) -> Path:
        return Path([*self.steps, position])

    def parent(self: Self) -> Path | None:
        return Path(self.steps[:-1]) if self.steps else None

    def is_prefix_of(self: Self, other: Path) -> bool:
        return self.steps == other.steps[: len(self.steps)]

    def trail(self: Self, expr: Expr) -> Iterator[Step]:
        for depth, position in enumerate(self.steps):
            if not isinstance(expr, SList):
                raise ValueError(f"step {depth} of {self} descends into {expr}")

            if position >= len(expr.items):
                raise ValueError(f"step {depth} of {self} has no child in {expr}")

            step: Step = Step(parent=expr, position=position)
            yield step
            expr: Expr = step.target

    def follow(self: Self, expr: Expr) -> Expr:
        for step in self.trail(expr):
            expr: Expr = step.target

        return expr

    @override
    def __len__(self: Self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self: Self, index: int) -> int: ...

    @overload
    def __getitem__(self: Self, index: slice) -> Path: ...

    @override
    def __getitem__(self: Self, index: int | slice) -> int | Path:
        if isinstance(index, slice):
            return Path(self.steps[index])

        return self.steps[index]

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.steps == other.steps if isinstance(other, Path) else False

    @override
    def __hash__(self: Self) -> int:
        return hash(tuple(self.steps))

    def __lt__(self: Self, other: Path) -> bool:
        return self.steps < other.steps

    @override
    def __str__(self: Self) -> str:
        if not self.steps:
            return "#<path>"

        return f"#<path {'.'.join(str(position) for position in self.steps)}>"
