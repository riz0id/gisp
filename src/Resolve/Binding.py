from dataclasses import dataclass, field

from Resolve.Phase import Phase
from Syntax import Symbol


@dataclass(frozen=True, slots=True)
class Binding:
    binder: Symbol
    phase: Phase = field(default_factory=Phase.runtime)
