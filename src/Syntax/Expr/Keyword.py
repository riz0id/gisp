from dataclasses import dataclass
from typing import Self, override

from Syntax.Expr.Base import Expr


@dataclass(frozen=True, slots=True)
class Keyword(Expr):
    # Written `#:name`. A keyword is a datum, not a reference: it names an
    # argument position rather than denoting a binding, so it carries no
    # scopes and hygiene does not apply to it. That is what distinguishes it
    # from a `Symbol` whose spelling happens to begin `#:`.
    name: str

    def __post_init__(self: Self) -> None:
        assert self.name != ""

    @override
    def __eq__(self: Self, other: object) -> bool:
        return self.name == other.name if isinstance(other, Keyword) else False

    @override
    def __str__(self: Self) -> str:
        return f"#:{self.name}"
