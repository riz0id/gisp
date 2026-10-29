from collections.abc import Callable
from dataclasses import dataclass, field

from Expander.Parse import ParseError, parse, parse_type, strip
from Resolve import Binding, BindingTable, Phase, Unbound, resolve
from Syntax import (
    Boolean,
    Char,
    Expr,
    Integer,
    Nil,
    Scope,
    SList,
    Symbol,
    ascription,
    heads_ascribe,
)

__all__: list[str] = [
    "ExpandError",
    "ParseError",
    "expand",
    "parse",
    "parse_type",
    "strip",
]

KEYWORDS: frozenset[str] = frozenset(
    {
        "lambda",
        "letrec",
        "letrec-syntax",
        "syntax-rules",
        "if",
        "choice",
        "quote",
        "define",
        "ascribe#",
    }
)


class ExpandError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Macro:
    # The compile-time meaning of a macro binding: its rewrite rules, with
    # pattern and template already carrying the definition context so
    # template identifiers resolve where the macro was defined.
    rules: list[tuple[SList, Expr]]


@dataclass(slots=True)
class Expansion:
    # The binding table answers "which binding"; the transformer environment
    # answers "what it means" — a binding absent from the environment is a
    # variable. Keeping the two apart keeps the table purely syntactic, so
    # analyses can rebuild it from the expanded tree alone.
    table: BindingTable
    transformers: dict[Binding, Macro]
    fresh: Callable[[], Scope]
    # The phase every binder of this expansion is registered at and every
    # reference resolved at. Nothing shifts it: no form expands its subforms at
    # another phase, so one expansion lives wholly at one phase.
    phase: Phase = field(default_factory=Phase.runtime)


def expand(
    expr: Expr,
    context: frozenset[Scope] = frozenset(),
    fresh: Callable[[], Scope] = Scope.fresh,
    phase: Phase | None = None,
) -> Expr:
    # The single writer of scope sets. Each binding form creates one scope
    # and adds it to its binders and region before descending; a macro
    # invocation additionally creates a use-site scope and an introduction
    # scope. Scope passes mutate the tree in place, and symbols acquire
    # scopes by union, so the pass composes with already-scoped input.
    expr.add(context)

    state = Expansion(
        table=BindingTable(),
        transformers={},
        fresh=fresh,
        phase=Phase.runtime() if phase is None else phase,
    )
    return expand_toplevel(expr, state)


def expand_toplevel(expr: Expr, state: Expansion) -> Expr:
    # A top-level form. `define` is a declaration and means something only
    # here, which is what makes "the scopes its binder already carries" the
    # top-level scopes — `expand_expr` is recursive and has no notion of
    # depth, so a `define` arm there would be reachable at any depth and a
    # nested one would register a binder indistinguishable from the enclosing
    # form's. This split mirrors `evaluate` and `evaluate_toplevel`.
    match expr:
        case SList([Symbol() as head, *_]) if head.identifier == "define":
            return expand_define(expr, state)
        case _:
            return expand_expr(expr, state)


def expand_expr(expr: Expr, state: Expansion) -> Expr:
    match expr:
        case Symbol() as symbol:
            return reference(symbol, state)
        case SList([Symbol() as head, *_]) if head.identifier == "lambda":
            return expand_lambda(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier == "letrec":
            return expand_letrec(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier == "letrec-syntax":
            return expand_letrec_syntax(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier == "syntax-rules":
            # Rules forms are consumed at their letrec-syntax binding; one
            # loose in expression position is structure out of place, not an
            # application.
            raise ExpandError(f"syntax-rules outside letrec-syntax: {expr}")
        case SList([Symbol() as head, *_]) if head.identifier == "define":
            # A declaration is meaningful only at the top level, where
            # `expand_toplevel` takes it. One here would otherwise register a
            # binder carrying the enclosing form's scopes.
            raise ExpandError(f"define outside the top level: {expr}")
        case SList([Symbol() as head, *_]) if head.identifier == "if":
            return expand_if(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier == "choice":
            return expand_choice(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier in {"lchoice", "rchoice"}:
            # Marker spellings are reserved for the Taylor expansion's own
            # output; a source list wearing one would be misread as a marker
            # by every recognizer downstream.
            raise ExpandError(f"reserved marker spelling: {expr}")
        case SList() if heads_ascribe(expr):
            # Ascription annotates a binder and is consumed at the binding
            # form that holds it, so one reaching expression position is
            # structure out of place. Refused before the application arm
            # below, which would otherwise resolve the type spelling as
            # though it were a variable.
            raise ExpandError(f"ascribe# outside a binder position: {expr}")
        case SList([Symbol() as head, *_]) if head.identifier == "quote":
            return expand_quote(expr)
        case SList([Symbol() as head, *_]) as form:
            # A non-keyword symbol head is the one position where a macro
            # binding means something: resolve it, and only when it maps to a
            # transformer does the form stop being a plain application.
            macro: Macro | None = transformer(head, state)
            if macro is None:
                return SList([expand_expr(item, state) for item in form.items])
            else:
                return invoke(form, macro, state)
        case SList(items):
            # Any other list is an application; every child lives in the
            # enclosing region, whose scopes it already carries.
            return SList([expand_expr(item, state) for item in items])
        case Integer() | Char() | Boolean() | Nil():
            return expr
        case _:
            raise ExpandError(f"not an expandable expression: {expr}")


def reference(symbol: Symbol, state: Expansion) -> Symbol:
    if symbol.identifier:
        # A macro name is compile-time structure with no value, so an occurrence
        # outside operator position has nothing to mean.
        if transformer(symbol, state) is None:
            return symbol
        else:
            raise ExpandError(f"macro name used outside operator position: {symbol}")
    else:
        raise ExpandError("anonymous symbol cannot be scoped")


def transformer(symbol: Symbol, state: Expansion) -> Macro | None:
    # Keywords are recognized by spelling before resolution runs and are
    # unbindable, so a core form can never be shadowed into a macro. An
    # unbound spelling is an ordinary free symbol. Ambiguity propagates:
    # it is a resolution fact, not an expansion one.
    if symbol.identifier in KEYWORDS:
        return None

    try:
        binding: Binding = resolve(symbol, state.table, state.phase)
    except Unbound:
        return None

    return state.transformers.get(binding)


def checked(symbol: Expr, seen: set[str]) -> Symbol:
    # The refusals every binder position shares, with the spelling recorded as
    # seen. Binders within one form share that form's single scope, so the
    # ⟨spelling, scope set⟩ pair stays a unique identity only if no spelling
    # repeats inside the form — a repeat would make two binding sites
    # indistinguishable, so it is refused rather than silently collapsed.
    #
    # Minting is the caller's: a region-opening form joins its own scope,
    # while a `define` keeps the scopes the name already carries.
    if not isinstance(symbol, Symbol):
        raise ExpandError(f"binder is not a symbol: {symbol}")

    if not symbol.identifier:
        raise ExpandError("anonymous symbol cannot be scoped")

    if symbol.identifier in KEYWORDS:
        raise ExpandError(f"binder shadows a core form: {symbol}")

    if symbol.identifier in seen:
        raise ExpandError(f"duplicate binder in one form: {symbol}")

    seen.add(symbol.identifier)
    return symbol


def binder(symbol: Expr, scope: Scope, seen: set[str], state: Expansion) -> Symbol:
    # A binder of a form that opens a region: the form's one scope joins the
    # binder's own, which is what tells two same-spelled binders in nested
    # forms apart.
    name: Symbol = checked(symbol, seen)
    scoped = Symbol(name.identifier, name.scopes | {scope})
    state.table.register(scoped, state.phase)
    return scoped


def toplevel_binder(symbol: Expr, seen: set[str], state: Expansion) -> Symbol:
    # A `define` binder. It takes the same refusals as any other, but mints no
    # scope: a declaration is visible to the whole program, and a scope minted
    # here would be absent from every later top-level form's references, so
    # none of them could resolve the name.
    name: Symbol = checked(symbol, seen)
    bound = Symbol(name.identifier, set(name.scopes))
    state.table.register(bound, state.phase)
    return bound


def binder_position(parameter: Expr, register: Callable[[Expr], Symbol]) -> Expr:
    # A binder position is either a bare symbol or an `ascribe#`-wrapped one.
    # Only the name is a binder: the type operand is an opaque type-level
    # spelling and not a program identifier, so it never reaches `register`,
    # acquires no scope, and is registered nowhere. `ascribe#` is a keyword,
    # so the wrapper cannot be mistaken for a binder a macro built.
    #
    # How a binder is minted is the caller's business, passed in as
    # `register`, so every binding form reads this one shape in this one
    # place.
    match parameter:
        case SList([head, *_]) if (ascribed := ascription(parameter)) is not None:
            name, ty = ascribed
            # The head is re-emitted exactly as it arrived, scopes and all: a
            # macro-introduced ascription carries the scopes of wherever it
            # was written, and rebuilding it bare would discard them.
            return SList([head, register(name), ty])
        case _ if heads_ascribe(parameter):
            # Caught here rather than below so a wrong-arity or non-symbol
            # triple reads as the malformed ascription it is.
            raise ExpandError(f"malformed ascribe#: {parameter}")
        case _:
            return register(parameter)


def expand_lambda(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, SList(parameters), body]:
            pass
        case _:
            raise ExpandError(f"malformed lambda: {expr}")

    # One scope for the whole form: all parameters are binding positions and
    # the body is the region where they apply, so both acquire the scope
    # before descent. The keyword is an identifier of the enclosing region —
    # it carries only the enclosing forms' scopes, never its own form's.
    scope: Scope = state.fresh()
    seen: set[str] = set()
    body.add(scope)

    return SList(
        [
            head,
            SList(
                [
                    binder_position(parameter, lambda s: binder(s, scope, seen, state))
                    for parameter in parameters
                ]
            ),
            expand_expr(body, state),
        ]
    )


def expand_letrec(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, SList(pairs), body]:
            pass
        case _:
            raise ExpandError(f"malformed letrec: {expr}")

    # One scope for the whole form, and the region where letrec bindings
    # apply is every right-hand side as well as the body: each bound name
    # must be visible from every initializer for the bindings to be
    # mutually recursive.
    scope: Scope = state.fresh()
    seen: set[str] = set()
    bindings: list[Expr] = []

    for pair in pairs:
        match pair:
            case SList([name, initializer]):
                initializer.add(scope)
                bindings.append(
                    SList(
                        [
                            binder_position(
                                name, lambda s: binder(s, scope, seen, state)
                            ),
                            expand_expr(initializer, state),
                        ]
                    )
                )
            case _:
                raise ExpandError(f"malformed letrec binding: {pair}")

    body.add(scope)
    return SList([head, SList(bindings), expand_expr(body, state)])


def expand_letrec_syntax(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol(), SList(pairs), body]:
            pass
        case _:
            raise ExpandError(f"malformed letrec-syntax: {expr}")

    if not pairs:
        raise ExpandError(f"malformed letrec-syntax: {expr}")

    # One scope for the whole form, added to the macro names, every rules
    # form, and the body: a template can therefore mention its own or a
    # sibling's macro name and resolve to it at transcription time —
    # letrec-style recursion for macros.
    scope: Scope = state.fresh()
    seen: set[str] = set()

    for pair in pairs:
        match pair:
            case SList([name, rules_form]):
                bound: Symbol = binder(name, scope, seen, state)
                rules_form.add(scope)
                ruleset: list[tuple[SList, Expr]] = rules(rules_form)
                binding: Binding = state.table.at(state.phase)[bound]
                state.transformers[binding] = Macro(ruleset)
            case _:
                raise ExpandError(f"malformed letrec-syntax binding: {pair}")

    # Macro bindings are compile-time structure with no residue: the
    # expansion of the body is the whole result, so the analyses never see
    # the form.
    body.add(scope)
    return expand_expr(body, state)


def rules(form: Expr) -> list[tuple[SList, Expr]]:
    match form:
        case SList([Symbol() as head, SList(literals), *pairs]) if (
            head.identifier == "syntax-rules"
        ):
            pass
        case _:
            raise ExpandError(f"macro binding is not a syntax-rules form: {form}")

    if literals:
        raise ExpandError(f"syntax-rules literals are not supported: {form}")

    if not pairs:
        raise ExpandError(f"syntax-rules needs at least one rule: {form}")

    collected: list[tuple[SList, Expr]] = []
    for pair in pairs:
        match pair:
            case SList([SList([Symbol(), *_]) as pattern, template]):
                pattern_variables(pattern)
                refuse_ellipsis(template)
                collected.append((pattern, template))
            case _:
                raise ExpandError(f"malformed syntax-rules rule: {pair}")

    return collected


def pattern_variables(pattern: SList) -> list[Symbol]:
    # Pattern variables are every non-wildcard symbol below the head, which
    # is structure on both sides of a match. A spelling binds at most once
    # per pattern — a nonlinear pattern would make two match positions
    # indistinguishable, the duplicate-binder refusal again — and ellipses
    # are deferred.
    variables: list[Symbol] = []

    def walk(form: Expr) -> None:
        match form:
            case Symbol() as symbol if symbol.identifier == "...":
                raise ExpandError(f"ellipsis patterns are not supported: {pattern}")
            case Symbol() as symbol if symbol.identifier == "_":
                pass
            case Symbol() as symbol:
                if any(
                    symbol.identifier == variable.identifier for variable in variables
                ):
                    raise ExpandError(f"duplicate pattern variable: {symbol}")
                variables.append(symbol)
            case _ if heads_ascribe(form):
                # A pattern variable binds matched syntax, not a value, so it
                # has no type to be ascribed. A pattern and the ascription a
                # macro might want to destructure are written identically, and
                # `syntax-rules` has no literals to tell them apart — so a
                # template may construct an ascription, but a pattern may not
                # match one.
                raise ExpandError(f"a pattern variable takes no ascription: {pattern}")
            case SList(items):
                for item in items:
                    walk(item)
            case _:
                pass

    for item in pattern.items[1:]:
        walk(item)

    return variables


def refuse_ellipsis(template: Expr) -> None:
    match template:
        case Symbol() as symbol if symbol.identifier == "...":
            raise ExpandError(f"ellipsis templates are not supported: {template}")
        case SList(items):
            for item in items:
                refuse_ellipsis(item)
        case _:
            pass


def invoke(form: SList, macro: Macro, state: Expansion) -> Expr:
    # Two scopes per invocation, the use-site scope minted first (tests pin
    # uids on that order). Adding the use-site scope marks what came from the
    # use; flipping the introduction scope around transcription leaves it on
    # exactly the material the template introduced (use-site material gains
    # it here and loses it on the second flip). The asymmetry is what keeps a
    # use-site binder from capturing a template reference and a template
    # binder from capturing a use-site reference.
    sc_use: Scope = state.fresh()
    sc_intro: Scope = state.fresh()

    form.add(sc_use)
    form.flip(sc_intro)

    for pattern, template in macro.rules:
        matches: dict[Symbol, Expr] | None = match(pattern, form)

        if matches is not None:
            # Re-expansion is unbounded: a template that mentions its own
            # macro name unconditionally loops forever, exactly as the
            # rewrite semantics says it must.
            transcribed: Expr = transcribe(template, matches)
            transcribed.flip(sc_intro)
            return expand_expr(transcribed, state)

    raise ExpandError(f"no rule matches: {form}")


def match(pattern: SList, form: Expr) -> dict[Symbol, Expr] | None:
    # Index 0 is structure on both sides: the pattern's head stands for the
    # macro name, whatever either is spelled.
    if not isinstance(form, SList) or not form.items:
        return None

    matches: dict[Symbol, Expr] = {}
    if match_into(pattern.items[1:], form.items[1:], matches):
        return matches
    return None


def match_into(
    patterns: list[Expr], forms: list[Expr], matches: dict[Symbol, Expr]
) -> bool:
    if len(patterns) != len(forms):
        return False

    for pattern, form in zip(patterns, forms, strict=True):
        match pattern:
            case Symbol() as symbol if symbol.identifier == "_":
                continue
            case Symbol() as symbol:
                matches[symbol] = form
            case SList(subpatterns):
                if not isinstance(form, SList) or not match_into(
                    subpatterns, form.items, matches
                ):
                    return False
            case _:
                # A literal pattern matches by value, nothing to bind.
                if pattern != form:
                    return False

    return True


def transcribe(template: Expr, matches: dict[Symbol, Expr]) -> Expr:
    match template:
        case Symbol() as symbol:
            # The lookup is bound-identifier=?: pattern and template carry
            # the identical definition context, so a template occurrence of
            # a pattern variable is exactly equal to it, and nothing else
            # can be. An unmatched symbol is emitted as a copy: the scope
            # passes mutate in place, so emitting the stored rule's own
            # symbol would let one invocation's scopes leak into the
            # template and corrupt every later use of the macro.
            bound: Expr | None = matches.get(symbol)
            if bound is None:
                return Symbol(symbol.identifier, set(symbol.scopes))
            return bound
        case SList([Symbol() as head, *_]) if head.identifier == "quote":
            # Quoted template data carries no scopes, so its symbols can
            # never equal a pattern variable: substitution cannot reach into
            # data, and the form is copied verbatim.
            return template
        case SList(items):
            return SList([transcribe(item, matches) for item in items])
        case _:
            return template


def expand_if(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, test, consequent, alternative]:
            pass
        case _:
            raise ExpandError(f"malformed if: {expr}")

    # Not a binding form: there is nothing for a new scope to bind, so all
    # three subforms stay in the enclosing region unchanged.
    return SList(
        [
            head,
            expand_expr(test, state),
            expand_expr(consequent, state),
            expand_expr(alternative, state),
        ]
    )


def expand_choice(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, Integer() as p, Integer() as q, left, right]:
            pass
        case _:
            raise ExpandError(f"malformed choice: {expr}")

    # Not a binding form, like if: both branches stay in the enclosing
    # region. The numerals are notation read by the Taylor expansion, not
    # data flowing through the program, so they pass through unexpanded.
    return SList(
        [
            head,
            p,
            q,
            expand_expr(left, state),
            expand_expr(right, state),
        ]
    )


def expand_quote(expr: SList) -> Expr:
    match expr.items:
        case [Symbol(), _]:
            return expr
        case _:
            raise ExpandError(f"malformed quote: {expr}")


def expand_define(expr: Expr, state: Expansion) -> Expr:
    match expr:
        case SList([Symbol() as head, name, init]):
            pass
        case _:
            raise ExpandError(f"malformed define: {expr}")

    # Not a region-opening form: a declaration's name is visible to the whole
    # program, so it mints no scope of its own. The binder is registered
    # before the initializer expands, which is what lets a defined procedure
    # refer to itself.
    seen: set[str] = set()
    bound: Expr = binder_position(name, lambda s: toplevel_binder(s, seen, state))

    return SList([head, bound, expand_expr(init, state)])
