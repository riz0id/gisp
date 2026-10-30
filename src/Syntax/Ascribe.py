from typing import Final

from Syntax.Expr import Expr, SList, Symbol

__all__: list[str] = ["ascription", "binder_name", "heads_ascribe"]

# The core form that annotates a binder with its type, written `(ascribe# x t)`
# wherever a binder position would accept a bare symbol. It is spelled once
# here because an ascription changes the shape of a binder position, and every
# reader of that shape has to agree on what the shape is.
#
# Module-internal: callers go through the predicates below rather than matching
# the spelling for themselves.
ASCRIBE: Final[str] = "ascribe#"


def heads_ascribe(parameter: Expr) -> bool:
    # Whether a binder position is an `ascribe#` form of any arity. The
    # contract callers rely on is
    #
    #     heads_ascribe(p) and ascription(p) is None
    #
    # which says `p` is a *malformed* ascription rather than an ordinary
    # binder — so a wrong-arity or non-symbol triple can be refused as the bad
    # ascription it is, and not as "not a symbol".
    match parameter:
        case SList([Symbol() as head, *_]):
            return head.identifier == ASCRIBE
        case _:
            return False


def ascription(parameter: Expr) -> tuple[Symbol, Expr] | None:
    # The ⟨name, type⟩ a well-formed ascribed binder position carries, or
    # `None` when the position is not one. A bare binder and a malformed
    # ascription both answer `None`; `heads_ascribe` tells those apart.
    #
    # The head is read by identifier and never by `Symbol` equality: a
    # macro-introduced ascription carries the scopes of wherever it was
    # written, because the scope passes descend into every form but `quote`.
    #
    # The type operand is any expression and not just a symbol. Only a ground
    # type is written as one, and a compound type — or a `forall#` over one —
    # is written as a list, so reading the shape here would be reading the
    # type-level grammar in a module that has no business knowing it.
    # `Expander.parse_scheme` is what decides whether the operand names a type.
    match parameter:
        case SList([Symbol() as head, Symbol() as name, ty]) if (
            head.identifier == ASCRIBE
        ):
            return name, ty
        case _:
            return None


def binder_name(parameter: Expr) -> Symbol | None:
    # The symbol a binder position binds: the position itself when it is a bare
    # symbol, or the name an ascription wraps. The type operand names no value,
    # which is what makes an ascription erasable by every reader that needs
    # only the name.
    #
    # `None` when the position is neither shape, which each caller answers for
    # itself: a reader of already-expanded syntax may skip it, and one reading
    # source must refuse it.
    match parameter:
        case Symbol() as symbol:
            return symbol
        case _:
            ascribed: tuple[Symbol, Expr] | None = ascription(parameter)
            return None if ascribed is None else ascribed[0]
