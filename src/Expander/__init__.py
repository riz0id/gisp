from collections.abc import Callable
from dataclasses import dataclass, field

from Expander.Parse import ParseError, parse, strip, type_syntax
from Resolve import Binding, BindingTable, Phase, Unbound, resolve
from Syntax import (
    Char,
    Expr,
    Integer,
    Scope,
    SList,
    Symbol,
    ascription,
    heads_ascribe,
)

__all__: list[str] = [
    "KEYWORDS",
    "ExpandError",
    "ParseError",
    "expand",
    "parse",
    "strip",
    "type_syntax",
]

KEYWORDS: frozenset[str] = frozenset(
    {
        "lambda",
        "letrec",
        "letrec-syntax",
        "syntax-rules",
        "match",
        "quote",
        "define",
        "forall",
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
        case SList([Symbol() as head, *_]) if head.identifier == "match":
            return expand_match(expr, state)
        case SList([Symbol() as head, *_]) if head.identifier == "forall":
            return expand_forall(expr, state)
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
        case Integer() | Char():
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


def opaque_binder(symbol: Expr, seen: set[str], state: Expansion) -> Symbol:
    # A `forall`'s binder: the free variable it stands for. It mints no scope,
    # as a `define` binder does not, and keeps none either — the scopes a
    # written occurrence carries are the enclosing regions', and two
    # occurrences of one opaque variable at two depths would become binders
    # with incomparable scope sets, which `resolve` refuses as ambiguous
    # rather than reading as the one variable they are meant to be.
    #
    # Minting the same symbol twice is the point and not a conflict:
    # `BindingTable.register` answers the binding already registered for an
    # equal symbol, so a second occurrence binds nothing new.
    name: Symbol = checked(symbol, seen)
    bound = Symbol(name.identifier)
    state.table.register(bound, state.phase)
    return bound


def type_variable(
    symbol: Expr, scope: Scope, seen: set[str], state: Expansion
) -> Symbol:
    # A type variable bound by `@mu` or `forall#`, minted through `binder` like
    # every other binder: one minting path, so a type binder takes the refusals
    # a term binder takes and is registered the same way.
    #
    # The one refusal of its own precedes that, because it has to: every `@`
    # identifier is a type constructor's, so one never names a variable, and
    # catching it here keeps a binder from being registered for it.
    match symbol:
        case Symbol() as written if written.identifier.startswith("@"):
            raise ExpandError(f"not a type variable: {symbol}")
        case _:
            return binder(symbol, scope, seen, state)


def type_form(ty: Expr, state: Expansion) -> Expr:
    # A written type, descended into for its binding positions alone. `@mu` and
    # `forall#` bind type variables, so each opens a region and mints one scope
    # for it exactly as `expand_lambda` and `expand_letrec` do.
    #
    # Nothing here reads a type for its meaning. Whether an identifier names a
    # type constructor, whether a form is written at the right arity, and what
    # the type denotes are all later questions. This pass settles only where the
    # binders are, which is the one thing no later phase can recover once the
    # scopes are missing.
    match ty:
        case SList([Symbol() as head, variable, body]) if head.identifier == "@mu":
            # One scope for the whole form: the variable is the binding position
            # and the body is the region, so the body acquires the scope before
            # the descent and the binder acquires it by union.
            scope: Scope = state.fresh()
            body.add(scope)

            return SList(
                [
                    head,
                    type_variable(variable, scope, set(), state),
                    type_form(body, state),
                ]
            )
        case SList([Symbol() as head, SList(variables), body]) if (
            head.identifier == "forall#"
        ):
            # As above, with `seen` shared across the variable list because
            # those are the binders of one form.
            quantifier: Scope = state.fresh()
            seen: set[str] = set()
            body.add(quantifier)

            return SList(
                [
                    head,
                    SList(
                        [
                            type_variable(variable, quantifier, seen, state)
                            for variable in variables
                        ]
                    ),
                    type_form(body, state),
                ]
            )
        case SList(items):
            # Not a binding form, and not necessarily a well-formed one either:
            # a `@mu` or `forall#` at the wrong arity falls here and is descended
            # into as an ordinary list, leaving the written form to the phase
            # whose business it is.
            return SList([type_form(item, state) for item in items])
        case _:
            # A symbol or a datum, re-emitted by identity rather than rebuilt:
            # the scope passes mutate in place, and an enclosing invocation's
            # flip has to reach the symbols a type operand holds.
            return ty


def expand_type(ty: Expr, state: Expansion) -> Expr:
    # An `ascribe#` form's type operand. The operand is a region of its own and
    # takes one scope for the whole of it, minted here by this occurrence as
    # every other scope is minted by its own form's. Every type variable bound
    # within the operand therefore carries it and no term binder ever does,
    # which is what keeps a type variable and a term variable of one identifier
    # from being the same binder.
    scope: Scope = state.fresh()
    ty.add(scope)

    return type_form(ty, state)


def binder_position(
    parameter: Expr,
    register: Callable[[Expr], Symbol],
    ascribed: Callable[[Expr], Expr],
) -> Expr:
    # A binder position is either a bare symbol or an `ascribe#`-wrapped one.
    # Only the name is a binder of the enclosing form; the type operand is a
    # type, and its own binding positions are the `ascribed` pass's to scope.
    # `ascribe#` is a keyword, so the wrapper cannot be mistaken for a binder a
    # macro built.
    #
    # How a binder is minted and how a type operand is scoped are both the
    # caller's business, passed in, so every binding form reads this one shape
    # in this one place and this function holds no state of its own.
    match parameter:
        case SList([head, *_]) if (declared := ascription(parameter)) is not None:
            name, ty = declared
            # The head is re-emitted exactly as it arrived, scopes and all: a
            # macro-introduced ascription carries the scopes of wherever it
            # was written, and rebuilding it bare would discard them.
            return SList([head, register(name), ascribed(ty)])
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
                    binder_position(
                        parameter,
                        lambda s: binder(s, scope, seen, state),
                        lambda t: expand_type(t, state),
                    )
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
                                name,
                                lambda s: binder(s, scope, seen, state),
                                lambda t: expand_type(t, state),
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


def expand_match(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, scrutinee, *clauses] if clauses:
            pass
        case _:
            # A `match` with no clause included: it would be an expression with
            # no value for every scrutinee, which is not something to expand.
            raise ExpandError(f"malformed match: {expr}")

    # The scrutinee is an expression of the enclosing region and binds nothing,
    # so it takes no scope of its own. Every clause does.
    return SList(
        [
            head,
            expand_expr(scrutinee, state),
            *[expand_clause(clause, state) for clause in clauses],
        ]
    )


def expand_clause(clause: Expr, state: Expansion) -> Expr:
    match clause:
        case SList([pattern, body]):
            pass
        case _:
            raise ExpandError(f"malformed match clause: {clause}")

    # One scope per clause, where every other binding form mints one for the
    # whole of itself. A clause's pattern is its binding position and its body
    # is the region those binders apply to, and two clauses are two regions:
    # both may bind the one spelling, which `checked` refuses within a single
    # `seen` set. So the set is per clause as well.
    scope: Scope = state.fresh()
    seen: set[str] = set()
    body.add(scope)

    return SList(
        [
            expand_pattern(pattern, scope, seen, state),
            expand_expr(body, state),
        ]
    )


def expand_pattern(
    pattern: Expr, scope: Scope, seen: set[str], state: Expansion
) -> Expr:
    # A pattern, descended into for its binding positions alone — the role
    # `type_form` plays for a written type. Whether the head of a constructor
    # pattern names a constructor, and whether the operands are written at the
    # arity its payload has, are later questions; this pass settles only where
    # the binders are.
    match pattern:
        case SList([]):
            # No constructor is spelled by nothing, and the empty combination
            # is refused at expression position too.
            raise ExpandError(f"not a pattern: {pattern}")
        case SList([constructor, *operands]) if not heads_ascribe(pattern):
            # The head is re-emitted exactly as it arrived, scopes and all, for
            # the reason `binder_position` re-emits an `ascribe#` head: a
            # macro-introduced constructor carries the scopes of wherever it was
            # written, and resolution is what decides whether it names a
            # constructor or a binding that merely shares the spelling.
            return SList(
                [
                    constructor,
                    *[
                        expand_pattern(operand, scope, seen, state)
                        for operand in operands
                    ],
                ]
            )
        case Integer() | Char():
            # A literal matches by value and binds nothing.
            return pattern
        case _:
            # A binder position: a bare symbol, or an `ascribe#` wrapping one.
            # Scoped exactly as a lambda parameter is, which is what lets a
            # pattern binder carry a written type.
            return binder_position(
                pattern,
                lambda s: binder(s, scope, seen, state),
                lambda t: expand_type(t, state),
            )


def expand_forall(expr: SList, state: Expansion) -> Expr:
    match expr.items:
        case [Symbol() as head, operand]:
            pass
        case _:
            raise ExpandError(f"malformed forall: {expr}")

    # Not a region-opening form, and not a region at all: the form is the
    # binding position and the only occurrence of what it binds, so there is no
    # body for a scope to apply to. Its binder is therefore free in the whole
    # program, which is what `opaque_binder` mints.
    if not heads_ascribe(operand):
        # Refused here rather than at the binder position below, so a bare
        # symbol reads as the missing ascription it is: a free variable has
        # nowhere else to carry its type, because no enclosing form binds it.
        raise ExpandError(f"forall takes an ascribed binder: {expr}")

    return SList(
        [
            head,
            binder_position(
                operand,
                lambda s: opaque_binder(s, set(), state),
                lambda t: expand_type(t, state),
            ),
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
    bound: Expr = binder_position(
        name,
        lambda s: toplevel_binder(s, seen, state),
        lambda t: expand_type(t, state),
    )

    return SList([head, bound, expand_expr(init, state)])
