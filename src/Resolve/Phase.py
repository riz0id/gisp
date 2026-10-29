from dataclasses import dataclass
from typing import Self, override

__all__: list[str] = [
    "Phase",
]


@dataclass(frozen=True, slots=True)
class Phase:
    level: int

    @classmethod
    def runtime(cls) -> Self:
        return cls(level=0)

    @override
    def __str__(self: Self) -> str:
        return f"#<phase {self.level}>"
