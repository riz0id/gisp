from dataclasses import dataclass
from typing import Self

from Syntax import Expr, SList

__all__: list[str] = [
    "Step",
]


@dataclass(frozen=True, slots=True)
class Step:
    parent: SList
    position: int

    def __post_init__(self: Self) -> None:
        if self.position < 0:
            raise ValueError(
                f"Step.step must be >=0, but Step.position = {self.position}"
            )

    @property
    def target(self: Self) -> Expr:
        return self.parent.items[self.position]
