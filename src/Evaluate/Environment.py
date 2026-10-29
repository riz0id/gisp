from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Self

from Evaluate.Error import UnboundSymbol

if TYPE_CHECKING:
    from Syntax.Value import Value


class Environment:
    def __init__(
        self: Self, bindings: Mapping[str, Value], parent: Environment | None = None
    ) -> None:
        self.bindings: dict[str, Value] = dict(bindings)
        self.parent = parent

    def __iter__(self) -> Iterator[Environment]:
        # The frame chain, innermost first: the same order lookup and
        # assignment search, so a linear scan over this iterator is the
        # resolution order of the language.
        frame: Environment | None = self

        while frame is not None:
            yield frame
            frame = frame.parent

    def lookup(self, name: str) -> Value:
        for frame in self:
            if name in frame.bindings:
                return frame.bindings[name]

        raise UnboundSymbol(f"unbound symbol {name}")

    def define(self, name: str, value: Value) -> None:
        self.bindings[name] = value

    def assign(self, name: str, value: Value) -> None:
        for frame in self:
            if name in frame.bindings:
                frame.bindings[name] = value
                return

        raise UnboundSymbol(f"unbound symbol {name}")
