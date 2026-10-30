from dataclasses import dataclass, field, replace

from Resolve import BindingTable
from Syntax import Char, Expr, Integer, SList, Symbol, ascription, binder_name
from Syntax.Type import (
    BoolType,
    CharType,
    ConstructorArity,
    Context,
    FunType,
    IntType,
    ListType,
    MuType,
    ProdType,
    Scheme,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    constructor_arity,
    is_constructor,
    meaning,
    solve,
    substitute,
    unfold,
)
from TypeCheck.Elaborate import ElaborateError, elaborate

__all__: list[str] = [
    "ElaborateError",
    "TypeCheckError",
    "TypeChecker",
    "check_toplevel",
    "elaborate",
]


class TypeCheckError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class TypeChecker:
    # The state a check-only judgment carries: the table every reference
    # resolves against, and the store that answers what each binder is
    # ascribed. The store accumulates — `abstraction` registers a lambda's
    # parameters into it — so a checker is built per judgment, and not shared
    # across two checks of one lambda at two different function types, which
    # `Context.register` refuses.
    #
    # The store may nonetheless outlive one checker, and a driver over a
    # sequence of top-level forms passes the same one in: a declaration's
    # ascription has to be there when a later form applies the name. The table
    # does not, because it is rebuilt from each parsed form.
    table: BindingTable = field(default_factory=BindingTable.default)
    store: Context = field(default_factory=Context.default)

    def check(self, expr: Expr, ty: Type) -> None:
        # Check `expr` against `ty` under the ascriptions in the store, raising
        # `TypeCheckError` when the expression does not inhabit the type. The
        # judgment is check-only: nothing is synthesized, so every variable,
        # letrec binder, and application head must already be ascribed.
        #
        # A polymorphic head is no exception. `ty` is what instantiates its
        # scheme, so the judgment reads an instance without ever synthesizing
        # one — which is also the limit of it: a scheme writing a quantified
        # variable only among its arguments has nothing in `ty` to be read from,
        # and `instantiated` refuses it.
        #
        # A `quote` does arrive here — `parse` emits one — and is taken by the
        # arm that reads its datum, below.
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

                if binder in self.store.schemes:
                    # A polymorphic binder is instantiated where it is applied,
                    # so a reference to one is refused rather than read as any
                    # instance. Said here because the message below would claim
                    # the binder has no ascription, and it has a scheme.
                    raise TypeCheckError(f"{symbol} is polymorphic, and not applied.")

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
            case SList([Symbol() as head, datum]) if head.identifier == "quote":
                self.quotation(datum, ty)
            case SList([Symbol() as head, *_]) if head.identifier == "define":
                # A declaration, not an expression: only `declaration` gives
                # it a meaning. Refused here so it cannot fall into the
                # application arm below and be misread as a call to an
                # unascribed `define`.
                raise TypeCheckError("define: only allowed at the top level")
            case SList([Symbol() as head, *operands]):
                if is_constructor(ty, head, self.store, self.table):
                    self.construction(head, operands, ty)
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
                            binder, written = declared
                            # The function type already fixes this position's
                            # type, so the ascription is checked against it
                            # rather than read for it.
                            ascribed: Scheme = self.scheme_of(written)

                            if not ascribed.monomorphic():
                                # A parameter is used at the one type its
                                # position fixes, so a quantifier over it would
                                # bind variables no use could instantiate. The
                                # rank-1 restriction read at a binder.
                                raise TypeCheckError(
                                    f"{binder} is a parameter, and cannot be"
                                    f" ascribed the polymorphic {written}."
                                )
                            if ascribed.body != field_type:
                                raise TypeCheckError(
                                    f"{binder} is ascribed {ascribed.body},"
                                    f" not {field_type}."
                                )
                            self.ascribe(binder, Scheme(variables=(), body=field_type))
                        case Symbol() as binder:
                            # Post-parse binders are globally distinct symbols,
                            # so registering into the one mutable store is
                            # sound.
                            #
                            # Registered through `ascribe` rather than against
                            # the store directly, so a conflict is reported as
                            # the `TypeCheckError` it is: `Context.register`
                            # raises `ValueError`, and a caller catching this
                            # module's errors would not catch that.
                            self.ascribe(binder, Scheme(variables=(), body=field_type))
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

    def quotation(self, datum: Expr, ty: Type) -> None:
        # A quoted datum, checked against the type it is written at. Quotation
        # is the one runtime constructor of list and symbol data, so this is the
        # only place a type is read against a datum rather than an expression:
        # the datum *is* the value, and nothing within it is a reference that
        # could carry an ascription.
        #
        # Where this overlaps `is_constructor` the two agree — an `Integer`
        # constructs `@int` and a `Char` constructs `@char` — and it extends
        # that to the one shape no expression has, a list written as data rather
        # than applied to `list`.
        match ty:
            case MuType() as recursive:
                # Taken apart as its unfolding, as in `construction`: the
                # recursive positions come back as the folded type, so a datum
                # nested within one is read at the type it was written at.
                self.quotation(datum, unfold(recursive))
            case ListType(element=element):
                match datum:
                    case SList(items):
                        # The empty list inhabits every list type: it is the
                        # one datum that constructs a list without naming its
                        # element type.
                        for item in items:
                            self.quotation(item, element)
                    case _:
                        raise TypeCheckError(f"quoted {datum} does not construct {ty}.")
            case IntType() if isinstance(datum, Integer):
                pass
            case CharType() if isinstance(datum, Char):
                pass
            case _:
                # A quoted symbol lands here at every type, which is the whole
                # of what symbol data means to the checker: the language has no
                # type that symbols inhabit, so one written as data is refused
                # rather than classified.
                raise TypeCheckError(f"quoted {datum} does not construct {ty}.")

    def quoted(self, datum: Expr) -> Type:
        # The type a quoted datum already has. A list takes the type of its
        # first element, and `quotation` checks the rest against it — as an
        # `if` takes its consequent's type and checks its alternative there.
        match datum:
            case Integer():
                return IntType()
            case Char():
                return CharType()
            case SList([first, *_]):
                return ListType(self.quoted(first))
            case _:
                # The empty list and a symbol, neither of which names a type:
                # the empty list is written at every list type and determines
                # none of them, and no type classifies symbol data.
                raise TypeCheckError(f"quoted {datum} determines no type.")

    def type_of(self, expr: Expr) -> Type:
        # The type an expression already has. Every binder of a parsed program
        # carries an ascription, so a form's type is read off the form and the
        # store rather than solved for as an unknown.
        #
        # This exists for the two positions where nothing hands a type over: a
        # top-level expression, which no binder encloses, and an operand that
        # determines a scheme's instance. `check` is untouched and still
        # check-only — this reads a type where none is supplied, and never
        # replaces one that is.
        match expr:
            case Integer():
                return IntType()
            case Char():
                return CharType()
            case Symbol() as symbol:
                binder: Symbol = meaning(symbol, self.table)

                if binder in self.store.schemes:
                    # As in `check`: a polymorphic binder is instantiated where
                    # it is applied, and a reference to one names no instance.
                    raise TypeCheckError(f"{symbol} is polymorphic, and not applied.")

                if binder not in self.store:
                    raise TypeCheckError(f"{symbol} has no ascription.")

                return self.store[binder]
            case SList([Symbol() as head, datum]) if head.identifier == "quote":
                return self.quoted(datum)
            case SList([Symbol() as head, test, consequent, _]) if (
                head.identifier == "if"
            ):
                self.check(test, BoolType())

                return self.type_of(consequent)
            case SList([Symbol() as head, SList(pairs), body]) if (
                head.identifier == "letrec"
            ):
                # The body's type, which can only be read once the pairs are in
                # the store. `Context.register` is idempotent at an equal type,
                # so the `check` that follows registers them again harmlessly.
                for pair in pairs:
                    self.binding(pair)

                return self.type_of(body)
            case SList([Symbol() as head, *_]) if head.identifier == "lambda":
                # A lambda's type is what a binder's ascription supplies: the
                # parameters carry their own types but the return type is
                # written nowhere, so there is nothing here to read.
                raise TypeCheckError(f"a lambda's type is read from a binder: {expr}")
            case SList([Symbol() as head, *_]) if head.identifier == "define":
                # A declaration is not an expression and has no type, which is
                # what `check_toplevel` dispatches on before reaching here.
                raise TypeCheckError("define: only allowed at the top level")
            case SList([Symbol() as head, *operands]):
                binder = meaning(head, self.table)

                if binder in self.store.schemes:
                    return self.instance(head, operands).return_type

                if binder not in self.store:
                    raise TypeCheckError(f"application head {head} has no ascription.")

                match self.store[binder]:
                    case FunType(return_type=return_type):
                        return return_type
                    case _:
                        raise TypeCheckError(f"{head} is not ascribed a function type.")
            case _:
                raise TypeCheckError(f"not an expression with a type: {expr}")

    def scheme_of(self, ty: Expr) -> Scheme:
        # The scheme a binder is ascribed, read from the type the parse phase
        # emitted. There is no `ParseError` to convert: that phase has already
        # refused a type that is not well formed, and resolved every variable
        # within one, so `elaborate` reads what a type denotes and nothing else.
        #
        # A type written without `forall#` reads as the scheme quantifying
        # nothing, so every caller takes one path and only the ones that care
        # about quantification ask whether it is `monomorphic`.
        return elaborate(ty)

    def ascribe(self, binder: Symbol, scheme: Scheme) -> None:
        try:
            if scheme.monomorphic():
                self.store.register(binder, scheme.body)
            else:
                self.store.register_scheme(binder, scheme)
        except ValueError as error:
            raise TypeCheckError(str(error)) from error

    def initializer(self, binder: Symbol, expr: Expr) -> None:
        # An ascribed binder's initializer, checked at what the binder carries.
        #
        # A polymorphic binder is checked once, against its scheme's body with
        # the quantified variables made abstract — not once per instance. That
        # is what makes one check stand for every use: the body may assume
        # nothing about the variables beyond their being types, so whatever it
        # does holds at each instance a use reads off. It is also why
        # `Context.register` never has to admit a parameter twice.
        if binder in self.store.schemes:
            scheme: Scheme = self.store.scheme(binder)

            replace(self, store=self.store.abstract(scheme.variables)).check(
                expr, scheme.body
            )
        else:
            self.check(expr, self.store[binder])

    def declaration(self, form: Expr) -> None:
        # A top-level `(define x e)` or `(define (ascribe# x t) e)`, which is
        # to a declaration what `binding` is to a letrec pair: an ascribed
        # binder carries the type it is checked at, and a bare one must have
        # been ascribed by the caller already.
        match form:
            case SList([Symbol() as head, name, init]) if head.identifier == "define":
                pass
            case _:
                raise TypeCheckError(f"not a declaration: {form}")

        bound: Symbol | None = binder_name(name)

        if bound is None:
            raise TypeCheckError(f"not a define binder: {name}")

        declared: tuple[Symbol, Expr] | None = ascription(name)

        if declared is None:
            if bound not in self.store and bound not in self.store.schemes:
                raise TypeCheckError(f"define binder {bound} has no ascription.")
        else:
            self.ascribe(bound, self.scheme_of(declared[1]))

        self.initializer(bound, init)

    def binding(self, pair: Expr) -> None:
        # A letrec binding `(x e)` or `((ascribe# x t) e)`. A bare binder must
        # have been ascribed by the caller already — there is nothing to read a
        # binder's own type off of — while an ascribed one carries the type it
        # is checked at, and is the one place a type written in source reaches
        # the store.
        match pair:
            case SList([name, init]) if (declared := ascription(name)) is not None:
                binder, written = declared
                self.ascribe(binder, self.scheme_of(written))
                self.initializer(binder, init)
            case SList([Symbol() as binder, init]):
                if binder not in self.store and binder not in self.store.schemes:
                    raise TypeCheckError(f"letrec binder {binder} has no ascription.")
                self.initializer(binder, init)
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

    def instantiated(
        self, head: Symbol, scheme: Scheme, solution: dict[Symbol, Type]
    ) -> Type:
        unsolved: list[Symbol] = [
            variable for variable in scheme.variables if variable not in solution
        ]

        if unsolved:
            raise TypeCheckError(
                f"{head} is polymorphic in {', '.join(map(str, unsolved))}, which"
                f" the checked type does not determine."
            )

        instantiated: Type = scheme.body

        for variable in scheme.variables:
            instantiated = substitute(instantiated, variable, solution[variable])

        return instantiated

    def application(self, head: Symbol, operands: list[Expr], ty: Type) -> None:
        # An ordinary application: the head must be ascribed a function type
        # whose return is the checked type, and the operands check against the
        # function type's arguments. A polymorphic head is instantiated first,
        # and then is one of those.
        binder: Symbol = meaning(head, self.table)

        if binder in self.store.schemes:
            self.applied(head, operands, self.polymorphic(head, operands, ty), ty)
            return

        if binder not in self.store:
            raise TypeCheckError(f"application head {head} has no ascription.")

        match self.store[binder]:
            case FunType() as function:
                self.applied(head, operands, function, ty)
            case _:
                raise TypeCheckError(f"{head} is not ascribed a function type.")

    def solved(
        self, scheme: Scheme, operands: list[Expr], solution: dict[Symbol, Type]
    ) -> None:
        # The scheme's variables that remain, determined from the operands' own
        # types. This is the second source the instance may be read from, and
        # the only one for a scheme writing a variable solely among its
        # arguments: `eq` returns `@bool` and `len` returns `@int`, so neither
        # names its instance in a return type and both name it in an operand.
        #
        # Reading an operand's type is as much a reading as `solve` is — the
        # operand is a reference or an application whose head is ascribed, so
        # its type is already fixed. An operand whose type cannot be read is
        # skipped rather than refused: another may determine the variable, and
        # one still undetermined after all of them is `instantiated`'s refusal.
        #
        # A `solve` that fails is likewise passed over. It means this operand
        # disagrees with an earlier one — `(eq 1 #t)`, where `a` is already
        # `@int` — and `applied` refuses the operand at the instance, which says
        # which operand and at what type.
        match scheme.body:
            case FunType(argument_type=ProdType() as product):
                declared: tuple[Type, ...] = product.types()
            case FunType(
                argument_type=VariadicType(argument=argument)
                | VariadicStarType(argument=argument)
            ):
                # Every argument of a run has the one type, so each operand
                # reads against it.
                declared = tuple(argument for _ in operands)
            case _:
                return

        variables: frozenset[Symbol] = frozenset(scheme.variables)

        for operand, declared_type in zip(operands, declared, strict=False):
            if all(variable in solution for variable in scheme.variables):
                return

            try:
                read: Type = self.type_of(operand)
            except TypeCheckError:
                continue

            solve(declared_type, read, variables, solution)

    def instance(self, head: Symbol, operands: list[Expr]) -> FunType:
        # A polymorphic head's instance read from the operands alone, for the
        # one caller that has no checked type to read it from: `type_of` is
        # answering what the application's type *is*.
        scheme: Scheme = self.store.scheme(meaning(head, self.table))
        solution: dict[Symbol, Type] = {}

        self.solved(scheme, operands, solution)

        match self.instantiated(head, scheme, solution):
            case FunType() as function:
                return function
            case instantiated:
                raise TypeCheckError(f"{head} is not a function: {instantiated}.")

    def polymorphic(self, head: Symbol, operands: list[Expr], ty: Type) -> FunType:
        scheme: Scheme = self.store.scheme(meaning(head, self.table))
        solution: dict[Symbol, Type] = {}

        match scheme.body:
            case FunType(return_type=return_type):
                pass
            case _:
                raise TypeCheckError(f"{head} is not ascribed a function type.")

        if not solve(return_type, ty, frozenset(scheme.variables), solution):
            raise TypeCheckError(f"{head} returns {return_type}, not {ty}.")

        # The checked type first, because it is the instance the judgment was
        # handed; the operands only for what it leaves undetermined.
        self.solved(scheme, operands, solution)

        match self.instantiated(head, scheme, solution):
            case FunType() as function:
                return function
            case instantiated:
                raise TypeCheckError(f"{head} is not a function: {instantiated}.")

    def applied(
        self, head: Symbol, operands: list[Expr], function: FunType, ty: Type
    ) -> None:
        if function.return_type != ty:
            raise TypeCheckError(f"{head} returns {function.return_type}, not {ty}.")

        match function.argument_type:
            case ProdType() as product:
                field_types: tuple[Type, ...] = product.types()

                if len(operands) != len(field_types):
                    raise TypeCheckError(
                        f"{head} takes {len(field_types)} arguments,"
                        f" not {len(operands)}."
                    )
                for operand, field_type in zip(operands, field_types, strict=True):
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


def check_toplevel(form: Expr, checker: TypeChecker) -> None:
    # A top-level form, checked. `define` is a declaration and means something
    # only here — `check` refuses one at expression position — and everything
    # else is an expression, checked at the type it already has. The split
    # mirrors `evaluate_toplevel` and `expand_toplevel`, and the sequence of
    # forms lives in the caller for the same reason it does there.
    #
    # The checker is the caller's, because a declaration's ascription has to
    # outlive the form that wrote it: the store is what makes a later form's
    # reference to a defined name resolvable.
    match form:
        case SList([Symbol() as head, *_]) if head.identifier == "define":
            checker.declaration(form)
        case _:
            checker.check(form, checker.type_of(form))
