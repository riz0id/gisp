from dataclasses import dataclass

from Syntax import Symbol
from Syntax.Type.Type import Type

__all__: list[str] = ["Scheme"]


@dataclass(frozen=True, slots=True)
class Scheme:
    # A rank-1 polymorphic type: `body` with each of `variables` standing for
    # whatever type a use is instantiated at. Written `(forall# (a ...) t)`, and
    # only at the outermost position of a written type.
    #
    # Deliberately not a `Type`. Rank-1 is the whole of the restriction, and
    # keeping the quantifier outside the sorts is what holds it: every `match`
    # over `Type` in `Syntax.Type` stays exhaustive as written, and no query has
    # to answer for a quantifier it cannot be handed. A scheme is what a binder
    # is ascribed, never what a term inhabits — which is also why `Context`
    # keeps schemes in a field of their own rather than widening `ascriptions`.

    # The variables bound in the body, in the order they were written.
    variables: tuple[Symbol, ...]

    # The type those variables are bound within.
    body: Type

    def monomorphic(self) -> bool:
        # Whether this scheme quantifies nothing, in which case it is its body.
        # `parse_scheme` reads a type written without `forall#` as one of these,
        # so every ascription is a scheme and the ones binding no variable take
        # the path a monotype always took.
        return not self.variables
