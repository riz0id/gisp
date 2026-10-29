from dataclasses import dataclass
from itertools import count
from typing import Self, override

__all__: list[str] = [
    "Scope",
]

uids: count[int] = count()


@dataclass(frozen=True, slots=True)
class Scope:
    uid: int

    @classmethod
    def fresh(cls) -> Self:
        return cls(uid=next(uids))

    @override
    def __str__(self: Self) -> str:
        return f"#<scope {self.uid}>"
