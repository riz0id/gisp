from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from Resolve.Binding import Binding
from Resolve.Phase import Phase
from Syntax import Symbol


@dataclass(frozen=True, slots=True, init=False)
class BindingTable:
    binders: dict[Phase, dict[Symbol, Binding]]

    def __init__(
        self, binders: Iterable[Symbol] = (), phase: Phase | None = None
    ) -> None:
        if phase is None:
            phase = Phase.runtime()

        object.__setattr__(
            self,
            "binders",
            {phase: {binder: Binding(binder, phase) for binder in binders}},
        )

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.at(Phase.runtime()))

    def __contains__(self, binder: object) -> bool:
        return binder in self.at(Phase.runtime())

    def __len__(self) -> int:
        return len(self.at(Phase.runtime()))

    def __getitem__(self, binder: Symbol) -> Binding:
        return self.at(Phase.runtime())[binder]

    def at(self, phase: Phase) -> Mapping[Symbol, Binding]:
        return MappingProxyType(self.binders.get(phase, {}))

    def register(self, binder: Symbol, phase: Phase | None = None) -> Binding:
        at: Phase = Phase.runtime() if phase is None else phase
        layer: dict[Symbol, Binding] = self.binders.setdefault(at, {})

        if binder in layer:
            return layer[binder]
        else:
            binding = Binding(binder, at)
            layer[binder] = binding
            return binding

    @classmethod
    def default(cls) -> BindingTable:
        return cls(
            [
                Symbol("#t"),
                Symbol("#f"),
                Symbol("prod#"),
            ]
        )
