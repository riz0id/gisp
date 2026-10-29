from dataclasses import dataclass, field
from typing import Final

from Resolve import BindingTable
from Syntax import Char, Expr, Integer, SList, Symbol, ascription, binder_name
from Syntax.Type import (
    BoolType,
    ConstructorArity,
    Context,
    FunType,
    ListType,
    MuType,
    ProdType,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    constructor_arity,
    is_constructor,
    meaning,
    nullary_type_of,
    unfold,
)


class TypeCheckError(Exception):
    pass


# The list selectors, which `selection` checks instead of reading an
# ascription for. These are bare spellings, so a binder that shadows `head` or
# `tail` resolves to a scoped symbol that is not one of them, and reads through
# the store like any other reference.
SELECTORS: Final[frozenset[Symbol]] = frozenset({Symbol("head"), Symbol("tail")})


@dataclass(frozen=True, slots=True)
class TypeChecker:
    # The state a check-only judgment carries: the table every reference
    # resolves against, and the store that answers what each binder is
    # ascribed. The store accumulates — `abstraction` registers a lambda's
    # parameters into it — so a checker is built per judgment, and not shared
    # across two checks of one lambda at two different function types, which
    # `Context.register` refuses.
    table: BindingTable = field(default_factory=BindingTable.default)
    store: Context = field(default_factory=Context.default)

    def check(self, expr: Expr, ty: Type) -> None:
        # Check `expr` against `ty` under the ascriptions in the store, raising
        # `TypeCheckError` when the expression does not inhabit the type. The
        # judgment is check-only: nothing is synthesized, so every variable,
        # letrec binder, and application head must already be ascribed — save
        # for the selectors in `SELECTORS`, whose polymorphic types the checked
        # type instantiates.
        #
        # `quote` cannot arrive here, so a `quote` head is refused below like
        # any other unascribed application head.
        match expr:
            case Integer() | Char():
                # A literal is its own constructor occurrence.
                if not is_constructor(ty, expr, self.store, self.table):
                    raise TypeCheckError(f"{expr} does not construct {ty}.")
            case Symbol() as symbol:
                # A reference inhabits exactly its ascription. This subsumes
                # `#t` and `#f` against `@bool`, which `Context.default()`
                # ascribes.
                binder: Symbol = meaning(symbol, self.table)

                if binder not in self.store:
                    raise TypeCheckError(f"{symbol} has no ascription.")

                if self.store[binder] != ty:
                    raise TypeCheckError(
                        f"{symbol} is ascribed {self.store[binder]}, not {ty}."
                    )
            case SList([Symbol() as head, SList(parameters), body]) if (
                head.identifier == "lambda"
            ):
                self.abstraction(parameters, body, ty)
            case SList([Symbol() as head, test, consequent, alternative]) if (
                head.identifier == "if"
            ):
                # Only the test is constrained on its own; both branches must
                # inhabit the checked type.
                self.check(test, BoolType())
                self.check(consequent, ty)
                self.check(alternative, ty)
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                for pair in pairs:
                    self.binding(pair)
                self.check(body, ty)
            case SList([Symbol() as head, *_]) if head.identifier == "define":
                # A declaration, not an expression: only `declaration` gives
                # it a meaning. Refused here so it cannot fall into the
                # application arm below and be misread as a call to an
                # unascribed `define`.
                raise TypeCheckError("define: only allowed at the top level")
            case SList([Symbol() as head, *operands]):
                if is_constructor(ty, head, self.store, self.table):
                    self.construction(head, operands, ty)
                elif meaning(head, self.table) in SELECTORS:
                    self.selection(head, operands, ty)
                else:
                    self.application(head, operands, ty)
            case _:
                # Everything else: empty combinations, applications of
                # non-symbol heads (check-only cannot invent a function type
                # for them), and the programmatic `Boolean`/`Nil` nodes no
                # source text reaches.
                raise TypeCheckError(f"not a checkable expression: {expr}")

    def abstraction(self, parameters: list[Expr], body: Expr, ty: Type) -> None:
        # A lambda checks only against a function type over a product: the
        # lambda form fixes its arity, so no abstraction inhabits a variadic
        # run.
        match ty:
            case MuType() as recursive:
                # A recursive type is taken apart as its unfolding, which is
                # where a `@fun` written through a `@mu` becomes visible.
                self.abstraction(parameters, body, unfold(recursive))
            case FunType(return_type=return_type, argument_type=ProdType() as product):
                field_types: tuple[Type, ...] = product.types()

                if len(parameters) != len(field_types):
                    raise TypeCheckError(
                        f"a lambda of {len(parameters)} parameters does not"
                        f" inhabit a product of {len(field_types)} fields."
                    )
                for parameter, field_type in zip(parameters, field_types, strict=True):
                    match parameter:
                        case _ if (declared := ascription(parameter)) is not None:
                            binder, ty_symbol = declared
                            # The function type already fixes this position's
                            # type, so the ascription is checked against it
                            # rather than read for it.
                            try:
                                ascribed: Type = nullary_type_of(ty_symbol, self.store)
                            except ValueError as error:
                                raise TypeCheckError(str(error)) from error
                            if ascribed != field_type:
                                raise TypeCheckError(
                                    f"{binder} is ascribed {ascribed},"
                                    f" not {field_type}."
                                )
                            self.store.register(binder, field_type)
                        case Symbol() as binder:
                            # Post-parse binders are globally distinct symbols,
                            # so registering into the one mutable store is
                            # sound.
                            self.store.register(binder, field_type)
                        case _:
                            # `parse` guarantees that binders are symbols.
                            raise TypeCheckError(f"not a parameter: {parameter}")
                self.check(body, return_type)
            case FunType():
                raise TypeCheckError(
                    "a lambda has fixed arity, and so does not inhabit a function"
                    " type over a variadic run."
                )
            case _:
                raise TypeCheckError(f"a lambda does not inhabit {ty}.")

    def declaration(self, form: Expr) -> None:
        # A top-level `(define x e)` or `(define (ascribe# x t) e)`, which is
        # to a declaration what `binding` is to a letrec pair: an ascribed
        # binder carries the type it is checked at, and a bare one must have
        # been ascribed by the caller already.
        match form:
            case SList([Symbol() as head, name, initializer]) if (
                head.identifier == "define"
            ):
                pass
            case _:
                raise TypeCheckError(f"not a declaration: {form}")

        bound: Symbol | None = binder_name(name)

        if bound is None:
            raise TypeCheckError(f"not a define binder: {name}")

        declared: tuple[Symbol, Symbol] | None = ascription(name)

        if declared is None:
            if bound not in self.store:
                raise TypeCheckError(f"define binder {bound} has no ascription.")
        else:
            try:
                # As in `binding`: `Context.register` refuses a binder already
                # ascribed something else, so one call both registers a new
                # ascription and refuses a contradicting one.
                self.store.register(bound, nullary_type_of(declared[1], self.store))
            except ValueError as error:
                raise TypeCheckError(str(error)) from error

        self.check(initializer, self.store[bound])

    def binding(self, pair: Expr) -> None:
        # A letrec binding `(x e)` or `((ascribe# x t) e)`. Check-only has
        # nowhere to synthesize a binder's type from, so a bare binder must
        # have been ascribed by the caller already; an ascribed one carries the
        # type it is checked at, and is the one place a type written in source
        # reaches the store.
        match pair:
            case SList([name, initializer]) if (
                declared := ascription(name)
            ) is not None:
                binder, ty_symbol = declared
                try:
                    # `Context.register` refuses a binder already ascribed
                    # something else, so this one call both registers a new
                    # ascription and refuses one contradicting an ascription
                    # the binder already carries.
                    self.store.register(binder, nullary_type_of(ty_symbol, self.store))
                except ValueError as error:
                    raise TypeCheckError(str(error)) from error
                self.check(initializer, self.store[binder])
            case SList([Symbol() as binder, initializer]):
                if binder not in self.store:
                    raise TypeCheckError(f"letrec binder {binder} has no ascription.")
                self.check(initializer, self.store[binder])
            case _:
                # `parse` refuses malformed letrec bindings before they get
                # here.
                raise TypeCheckError(f"not a letrec binding: {pair}")

    def construction(self, head: Symbol, operands: list[Expr], ty: Type) -> None:
        # `head` is a constructor occurrence of `ty`, and the operands are the
        # constructor's arguments.
        arity: ConstructorArity = constructor_arity(ty, self.store)
        if arity is not None and len(operands) != arity:
            raise TypeCheckError(
                f"{head} constructs {ty} from {arity} arguments, not {len(operands)}."
            )
        match ty:
            case MuType() as recursive:
                # A recursive type is taken apart as its unfolding, whose
                # recursive positions are the folded type again — so the
                # operands below are checked against types in the form they
                # were written, and no comparison ever meets an unfolding.
                self.construction(head, operands, unfold(recursive))
            case ListType(element=element):
                for operand in operands:
                    self.check(operand, element)
            case ProdType() as product:
                for operand, field_type in zip(operands, product.types(), strict=True):
                    self.check(operand, field_type)
            case SumType() as total:
                [payload] = operands
                self.check(payload, total.cons[meaning(head, self.table)])
            case _:
                # The lambda arm of `check` consumes every well-formed
                # `FunType` construction, and no nullary constructor heads an
                # application.
                raise TypeCheckError(f"{head} does not construct {ty} applied.")

    def selection(self, head: Symbol, operands: list[Expr], ty: Type) -> None:
        # `head` and `tail` select from a list, and both are polymorphic: the
        # store ascribes one type to a binder, and no one function type covers
        # every element type. Check-only never needs that ascription, because
        # the checked type is what a synthesizing judgment would have to
        # recover — it names the instance, and the operand follows from it.
        if len(operands) != 1:
            raise TypeCheckError(f"{head} takes one argument, not {len(operands)}.")

        [operand] = operands

        if meaning(head, self.table) == Symbol("head"):
            # The head of a `(@list t)` is a `t`, whichever `t` is checked, so
            # every type instantiates `head`.
            self.check(operand, ListType(ty))
        else:
            # The tail of a list is a list of the same elements, so `tail`
            # inhabits no other type.
            match ty:
                case ListType():
                    self.check(operand, ty)
                case _:
                    raise TypeCheckError(f"{head} returns a list, not {ty}.")

    def application(self, head: Symbol, operands: list[Expr], ty: Type) -> None:
        # An ordinary application: the head must be ascribed a function type
        # whose return is the checked type, and the operands check against the
        # function type's arguments.
        binder: Symbol = meaning(head, self.table)

        if binder not in self.store:
            raise TypeCheckError(f"application head {head} has no ascription.")

        match self.store[binder]:
            case FunType(return_type=return_type, argument_type=argument_type):
                if return_type != ty:
                    raise TypeCheckError(f"{head} returns {return_type}, not {ty}.")

                match argument_type:
                    case ProdType() as product:
                        field_types: tuple[Type, ...] = product.types()

                        if len(operands) != len(field_types):
                            raise TypeCheckError(
                                f"{head} takes {len(field_types)} arguments,"
                                f" not {len(operands)}."
                            )
                        for operand, field_type in zip(
                            operands, field_types, strict=True
                        ):
                            self.check(operand, field_type)
                    case VariadicType(argument=argument):
                        # Zero or more arguments of the one type.
                        for operand in operands:
                            self.check(operand, argument)
                    case VariadicStarType(argument=argument):
                        # One or more arguments of the one type.
                        if len(operands) < 1:
                            raise TypeCheckError(
                                f"{head} takes one or more arguments, not zero."
                            )
                        for operand in operands:
                            self.check(operand, argument)
            case _:
                raise TypeCheckError(f"{head} is not ascribed a function type.")
