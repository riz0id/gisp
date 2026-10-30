from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final

from Resolve import BindingTable, meaning
from Syntax import Char, Expr, Integer, Symbol
from Syntax.Type.CharType import CharType
from Syntax.Type.FunType import FunType
from Syntax.Type.IntType import IntType
from Syntax.Type.MuType import MuType
from Syntax.Type.ProdType import ProdType
from Syntax.Type.Scheme import Scheme
from Syntax.Type.SumType import SumType
from Syntax.Type.Type import ConstructorArity, Type
from Syntax.Type.VariadicType import VariadicStarType, VariadicType
from Syntax.Type.VarType import VarType

__all__: list[str] = [
    "CharType",
    "ABSTRACTION",
    "BOOL",
    "CONS",
    "CONSTRUCTORS",
    "FALSE",
    "NIL",
    "PRODUCT",
    "TRUE",
    "ConstructorArity",
    "Context",
    "FunType",
    "IntType",
    "MuType",
    "ProdType",
    "Scheme",
    "SumType",
    "Type",
    "VarType",
    "VariadicStarType",
    "VariadicType",
    "constructor_arity",
    "constructors",
    "guarded",
    "is_constructor",
    "is_nullary_type",
    "list_of",
    "meaning",
    "solve",
    "substitute",
    "substitute_arguments",
    "type_constructor_arity",
    "type_constructor_symbol",
    "unfold",
]


def type_constructor_arity(ty: Type, context: Context) -> int:
    # How many type arguments `ty`'s own constructor is written with, which is
    # not the same as how many arguments construct a term of it.
    match ty:
        case IntType() | CharType():
            # A nullary constructor takes no type arguments, and thus has
            # arity zero. `@bool` is not among them: it is written as a bare
            # symbol but denotes a sum, and this answers about the type and not
            # the spelling it was reached through.
            return 0
        case VarType(binder=binder) if binder in context.rigid:
            # An abstract variable denotes no one type, so there is no arity to
            # answer with. Raised rather than answered, as for an unbound one,
            # but diagnosed apart from it: this is a type checked inside the
            # body of the scheme that quantifies it, and not a bug.
            raise ValueError(f"abstract type variable has no arity: {binder}")
        case VarType(binder=binder):
            # A variable is not a constructor and has no arity of its own: it
            # stands for the type bound to it, and answers whatever that type
            # answers. Zero here would be the arity of the name `a`, not of
            # what `a` denotes.
            return type_constructor_arity(context.variable(binder), context)
        case VariadicType() | VariadicStarType():
            # `(@variadic t)` and `(@variadic* t)` take exactly one type
            # argument: the argument type.
            return 1
        case SumType(cons=cons):
            # `(@sum t_0 t_1 ... t_n)` takes one or more type-arguments.
            return len(cons)
        case ProdType(fields=fields):
            # Only finite products can be written, and so the type
            # constructor's arguments are finite.
            return len(fields)
        case FunType(argument_type=argument_type):
            # The function type constructor is finite, and so the length can be
            # computed.
            return 1 + type_constructor_arity(argument_type, context)
        case MuType():
            # `(@mu a t)` is written with two arguments, but only `t` is a type:
            # `a` is a binder. A fact about the written form, so the body is
            # neither bound nor descended into.
            return 1
        case _:
            raise ValueError(f"not a type: {ty}")


def type_constructor_symbol(ty: Type, context: Context) -> Symbol:
    # How `ty` is spelled at the type-level.
    match ty:
        case IntType():
            return Symbol(identifier="@int")
        case CharType():
            return Symbol(identifier="@char")
        case SumType():
            return Symbol(identifier="@sum")
        case ProdType():
            return Symbol(identifier="@prod")
        case VariadicType():
            return Symbol(identifier="@variadic")
        case VariadicStarType():
            return Symbol(identifier="@variadic*")
        case FunType():
            return Symbol(identifier="@fun")
        case MuType():
            # As in `type_constructor_arity`: a fact about the written form, so
            # the body is neither bound nor descended into.
            return Symbol(identifier="@mu")
        case VarType(binder=binder) if binder in context.rigid:
            # As in `type_constructor_arity`: an abstract variable denotes no
            # one type, so it has no constructor to be spelled. Its own binder
            # is the spelling, and that names a variable rather than a type.
            raise ValueError(f"abstract type variable has no constructor: {binder}")
        case VarType(binder=binder):
            # A variable has no spelling of its own: it is written as its
            # binder, but it *is* the type bound to it, and that is what is
            # being asked for.
            return type_constructor_symbol(context.variable(binder), context)
        case _:
            raise ValueError(f"not a type: {ty}")


def constructor_arity(ty: Type, context: Context) -> ConstructorArity:
    # How many arguments construct a term of `ty`, or `None` where the
    # constructor is written with an ellipsis and so has no bounded arity.
    match ty:
        case IntType() | CharType():
            # Literals have zero-arity, meaning the constructor (some integer
            # or character literal) has no arguments. A boolean is not among
            # them: `@bool` is a sum, and a sum constructor is written applied
            # to its payload.
            return 0
        case VarType(binder=binder) if binder in context.rigid:
            # Nothing constructs an abstract type: a term of it would have to
            # inhabit every instance the variable can be taken at. Unreachable
            # from `check`, whose dispatch consults `is_constructor` first and
            # is answered `False` there, and raised rather than answered so a
            # caller that did reach it is not handed an arity it could use.
            raise ValueError(f"abstract type variable is not constructed: {binder}")
        case VarType(binder=binder):
            # A type variable is constructed however the type bound to it is.
            return constructor_arity(context.variable(binder), context)
        case VariadicType() | VariadicStarType():
            # A variadic run has no bounded arity.
            return None
        case SumType():
            return 1
        case ProdType(fields=fields):
            # Only finite products have a type, and so only finite products can
            # be constructed.
            return len(fields)
        case FunType():
            # The "lambda" form, that is `(lambda (x ...) b)`, has two
            # arguments: the first `(x ...)`, and the second `b`.
            return 2
        case MuType() as recursive:
            # `(@mu a t)` binds `a` to the whole of itself within `t`, so the
            # body is read under that binding rather than through an unfolding.
            # The two agree — they differ only at free occurrences of `a`, which
            # no arm below reads — and nothing is rebuilt. `guarded` is what
            # makes this terminate, the same role it plays for `unfold`.
            return constructor_arity(
                ty=recursive.body,
                context=context.bind(variable=recursive.variable, ty=recursive),
            )
        case _:
            raise ValueError(f"not a type: {ty}")


# The binders that construct a term of a sort that is not written as a literal.
# Named once each, because `is_constructor` compares a resolved head against the
# one its sort expects while a reader without a type to check against can only
# ask whether a head is any of them — and those two readings must not drift.
PRODUCT: Final[Symbol] = Symbol("prod#")
ABSTRACTION: Final[Symbol] = Symbol("lambda")

# The two constructors of `@bool`, each carrying the nullary product. They are
# ordinary sum constructors and are read as such: `is_constructor` resolves a
# head against the keys of the sum, which is what makes these a closed set
# rather than "any symbol ascribed `@bool`".
TRUE: Final[Symbol] = Symbol("#t")
FALSE: Final[Symbol] = Symbol("#f")

# What `@bool` denotes. The spelling is the built-in — the parse phase reads
# `@bool` by spelling and re-emits it unresolved, as it reads a core form's
# identifier, so no program can shadow or rebind it — and this is the type that
# spelling means. `@bool` is not a sort of its own, and no query recognises this
# value: every reading of a boolean goes through the `SumType` arms, so a
# hand-written `(@sum (#t (@prod)) (#f (@prod)))` *is* `@bool` and not merely
# equivalent to it.
#
# Named here rather than built at each use, so `elaborate_type` and
# `Analysis.GROUNDS` cannot drift apart.
BOOL: Final[SumType] = SumType({TRUE: ProdType(), FALSE: ProdType()})

# The two constructors of `(@list t)`, and the fields a cons cell's payload is
# written with. A record and not a tuple, so the two fields a
# `(cons (prod# x xs))` pattern names are named where the type is written too.
NIL: Final[Symbol] = Symbol("nil")
CONS: Final[Symbol] = Symbol("cons")
ELEMENT: Final[Symbol] = Symbol("x")
REST: Final[Symbol] = Symbol("xs")

# The variable the recursion is written over. `Symbol("")` is the one symbol no
# source text can produce — no reader production yields an empty identifier,
# and `Symbol.__post_init__` holds its scope set empty — so no `@mu` or
# `forall#` a program writes can bind it.
#
# That is what makes `list_of` safe. `substitute` renames nothing, and the
# element type is routinely written over a variable a program chose: the
# builtin schemes write `(@list a)`, and were the binder below named `a` too,
# instantiating one of them would replace the element and leave the recursion
# pointing at `@int`. An unbindable binder cannot capture anything.
#
# `Patterns.ANY` is this symbol used for the same reason one level up.
SELF: Final[Symbol] = Symbol.toplevel()


def list_of(element: Type) -> MuType:
    # What `(@list t)` denotes. The written form is the built-in — the parse
    # phase reads `@list` by identifier and re-emits it unresolved, as it reads
    # a core form's identifier, so no program can shadow or rebind it — and
    # this is the type it means. `(@list t)` is not a sort of its own: every
    # one of the five type queries reads a list through the `MuType` and
    # `SumType` arms, and none of them compares against this.
    #
    # One reader does recognise the result, and has to: `TypeChecker.quotation`
    # reads an `SList` datum against the sum, because `SList` is the runtime
    # representation of list data and nothing else says which type that
    # representation inhabits. `Analysis.GROUNDS` names `BOOL` for the same
    # reason.
    #
    # A function rather than a constant, as `BOOL` is a constant, because this
    # one carries a type argument. Named here rather than built at each use, so
    # `elaborate_type` and that quoted-datum judgment cannot drift apart.
    return MuType(
        SELF,
        SumType(
            {
                CONS: ProdType({ELEMENT: element, REST: VarType(SELF)}),
                NIL: ProdType(),
            }
        ),
    )


CONSTRUCTORS: Final[frozenset[Symbol]] = frozenset({NIL, CONS, PRODUCT, ABSTRACTION})


def is_constructor(ty: Type, expr: Expr, context: Context, table: BindingTable) -> bool:
    # Whether `expr` is a constructor occurrence of `ty`: a literal of the sort
    # it classifies, or a reference resolving to the binder that constructs it.
    match ty:
        case IntType():
            return isinstance(expr, Integer)
        case CharType():
            return isinstance(expr, Char)
        case VarType(binder=binder) if binder in context.rigid:
            # No expression is a constructor occurrence of an abstract type,
            # which is what makes the body of a polymorphic function checkable:
            # a literal checked against the variable it quantifies is refused
            # by the caller as the non-inhabitant it is, rather than resolving
            # the variable to a type the instance need not be.
            return False
        case VarType(binder=binder):
            # A type variable is constructed however the type bound to it is.
            return is_constructor(context.variable(binder), expr, context, table)
        case SumType(cons=cons):
            # `cons` is keyed by binders, so a reference names a constructor
            # exactly when it resolves to one of the keys.
            match expr:
                case Symbol() as symbol:
                    return meaning(symbol, table) in cons
                case _:
                    return False
        case ProdType():
            match expr:
                case Symbol() as symbol:
                    return meaning(symbol, table) == PRODUCT
                case _:
                    return False
        case FunType():
            match expr:
                case Symbol() as symbol:
                    return meaning(symbol, table) == ABSTRACTION
                case _:
                    return False
        case VariadicType() | VariadicStarType():
            # No term constructs a variadic run on its own; the run exists only
            # as the arguments of an application.
            return False
        case MuType() as recursive:
            # As in `constructor_arity`: the body is read under the binding the
            # `@mu` makes, which agrees with reading its unfolding.
            return is_constructor(
                recursive.body,
                expr,
                context.bind(recursive.variable, recursive),
                table,
            )
        case _:
            raise ValueError(f"not a type: {ty}")


def constructors(ty: Type, context: Context) -> dict[Symbol, Type] | None:
    # Every way a term of `ty` is built, as ⟨constructor, payload type⟩, or
    # `None` where the sort has no closed set of them. This is what makes a
    # `match` exhaustive: a sort answering a mapping can be covered by naming
    # each key, and one answering `None` can only be covered by a pattern that
    # matches everything.
    #
    # The fifth query over a type, and the one that reads a sort as a sum. The
    # other four answer about one constructor at a time — whether a head is one
    # of them, how many arguments it takes — and none of them can enumerate.
    #
    # `None` is not "no constructors". `@int` and `@char` have one per value,
    # which is why they cannot be enumerated; a `@fun` is built by `lambda` but
    # nothing takes a procedure apart, so a `match` over one has nothing to
    # read; and the variadic types classify no values at all. All four are the
    # same answer to the one question asked here.
    match ty:
        case SumType(cons=cons):
            # The one sort that already is a mapping. Copied, so a caller
            # cannot reach into the type through it.
            return dict(cons)
        case ProdType() as product:
            # One constructor, so a pattern over a product refutes nothing: it
            # names the fields and always matches.
            return {PRODUCT: product}
        case IntType() | CharType():
            return None
        case FunType() | VariadicType() | VariadicStarType():
            return None
        case VarType(binder=binder) if binder in context.rigid:
            # An abstract variable stands for any type at all, so no set of
            # constructors is the one its instance has. Answered rather than
            # raised, as `is_nullary_type` answers: this is a question a
            # `match` in a polymorphic body legitimately asks, and the answer
            # is that it may only match everything.
            return None
        case VarType(binder=binder):
            return constructors(context.variable(binder), context)
        case MuType() as recursive:
            # Taken apart as its unfolding and not under a binding, unlike
            # `constructor_arity` and `is_constructor`: those read the body
            # without rebuilding it, while the payload types here cross back to
            # the caller and a `VarType` among them would arrive somewhere the
            # binding does not reach. The unfolding leaves every recursive
            # position folded, so a payload is a type the program wrote.
            return constructors(unfold(recursive), context)
        case _:
            raise ValueError(f"not a type: {ty}")


def is_nullary_type(ty: Type, context: Context) -> bool:
    # Whether `ty` is a ground type: one written as a bare symbol, classifying
    # values that have a representation of their own. Read by resolution and
    # not by equality against a set of instances, so a variable bound to a
    # ground type is one: a variable bound to `@int` stands for `@int` however
    # it is spelled.
    match ty:
        case IntType() | CharType():
            return True
        case VarType(binder=binder) if binder in context.rigid:
            # An abstract variable is written as a bare name, but it is not a
            # ground type: it classifies whatever its instance classifies, and
            # a scheme may be instantiated at a compound type as readily as a
            # ground one. Answered rather than raised, because this is a
            # question about a spelling and an abstract variable has one.
            return False
        case VarType(binder=binder):
            # As in the four queries: a variable answers for the type bound to
            # it, and `context.variable` refuses an unbound one rather than
            # letting it read as a type it is not.
            return is_nullary_type(context.variable(binder), context)
        case (
            SumType()
            | ProdType()
            | VariadicType()
            | VariadicStarType()
            | FunType()
            | MuType()
        ):
            # Each is written as a list at the type-level, whatever it is
            # written over. The nullary product is the clearest case: `(@prod)`
            # takes no type arguments, and so has type-constructor arity zero,
            # but it is still written as a list and is not a ground type.
            return False
        case _:
            raise ValueError(f"not a type: {ty}")


def unfold(recursive: MuType) -> Type:
    # The body with the variable standing for the whole type, which is what a
    # recursive type is taken apart as. Only the top node changes form: every
    # recursive position in the result is that type again, folded, exactly as
    # it was written.
    return substitute(recursive.body, recursive.variable, recursive)


def guarded(variable: Symbol, body: Type) -> bool:
    # Whether no occurrence of `variable` in `body` sits outside a type
    # constructor. `@mu` is not one, so the reading descends through nested
    # `@mu` bodies and stops at every other sort, each of which guards
    # whatever is beneath it.
    #
    # This is what makes `unfold` progress. `(@mu a a)` unfolds to itself, so
    # anything that reads a recursive type through its unfolding would recur
    # forever; with the variable guarded, each unfolding strictly shortens the
    # run of leading `@mu`s.
    match body:
        case VarType(binder=binder):
            return binder != variable
        case MuType(variable=inner, body=inner_body):
            # An inner `@mu` of the same variable shadows this one, so no
            # occurrence beneath it is an occurrence of this variable.
            return inner == variable or guarded(variable, inner_body)
        case _:
            return True


def substitute(ty: Type, variable: Symbol, replacement: Type) -> Type:
    # `ty` with `replacement` standing in for every free occurrence of
    # `variable`. Nothing is renamed, and nothing needs to be: `unfold` is the
    # only caller, and the type it substitutes is the one being taken apart,
    # which has no free variable to capture.
    match ty:
        case IntType() | CharType():
            return ty
        case VarType(binder=binder):
            return replacement if binder == variable else ty
        case SumType() as total:
            return SumType(
                {
                    constructor: substitute(payload, variable, replacement)
                    for constructor, payload in total.cons.items()
                }
            )
        case ProdType() | VariadicType() | VariadicStarType():
            return substitute_arguments(ty, variable, replacement)
        case FunType(return_type=return_type, argument_type=argument_type):
            return FunType(
                return_type=substitute(return_type, variable, replacement),
                argument_type=substitute_arguments(
                    argument_type, variable, replacement
                ),
            )
        case MuType(variable=bound, body=body) if bound != variable:
            return MuType(bound, substitute(body, variable, replacement))
        case MuType():
            # The body rebinds the variable, so every occurrence beneath it is
            # an occurrence of that binder and not of the one being replaced.
            return ty
        case _:
            raise ValueError(f"cannot substitute into {ty}")


def substitute_arguments(
    argument_type: ProdType | VariadicType | VariadicStarType,
    variable: Symbol,
    replacement: Type,
) -> ProdType | VariadicType | VariadicStarType:
    # Substitution over the three sorts a function type's arguments can be,
    # which is what lets a `@fun` be rebuilt without widening that field.
    match argument_type:
        case ProdType() as product:
            return product.with_types(
                types=[
                    substitute(
                        ty=field_type, variable=variable, replacement=replacement
                    )
                    for field_type in product.types()
                ]
            )
        case VariadicType(argument=argument):
            return VariadicType(
                argument=substitute(
                    ty=argument, variable=variable, replacement=replacement
                )
            )
        case VariadicStarType(argument=argument):
            return VariadicStarType(
                argument=substitute(
                    ty=argument, variable=variable, replacement=replacement
                )
            )


def solve(
    pattern: Type,
    target: Type,
    variables: frozenset[Symbol],
    solution: dict[Symbol, Type],
) -> bool:
    # Whether `target` is `pattern` with each of `variables` taken at some type,
    # recording those types in `solution`. This is how a scheme is instantiated:
    # the instance is read off a type the judgment already has, rather than
    # solved for as an unknown.
    #
    # One-way matching and not unification. Only `pattern` may be written over
    # `variables`; `target` is a type the checker was handed, so it is ground or
    # written over the *rigid* variables of an enclosing scheme, which match as
    # the opaque types they are. Nothing is ever solved to a type still
    # containing an unsolved variable, so there is no occurs check to make.
    match pattern:
        case VarType(binder=binder) if binder in variables:
            if binder in solution:
                # A variable occurring twice must be taken at one type, which is
                # what makes `(eq 1 #t)` a refusal rather than two instances.
                return solution[binder] == target
            else:
                solution[binder] = target
                return True
        case IntType() | CharType() | VarType():
            # A ground type, or a variable this scheme does not quantify: there
            # is nothing to solve, so matching is equality.
            return pattern == target
        case VariadicType(argument=argument):
            match target:
                case VariadicType(argument=other):
                    return solve(argument, other, variables, solution)
                case _:
                    return False
        case VariadicStarType(argument=argument):
            match target:
                case VariadicStarType(argument=other):
                    return solve(argument, other, variables, solution)
                case _:
                    return False
        case ProdType() as product:
            match target:
                case ProdType() as other if product.names() == other.names() and len(
                    product.types()
                ) == len(other.types()):
                    # Shapes must agree, names and all. A tuple and a record of
                    # the same field types are distinct under `==` today, and
                    # matching inherits that rather than deciding it afresh.
                    return all(
                        solve(field_type, other_type, variables, solution)
                        for field_type, other_type in zip(
                            product.types(), other.types(), strict=True
                        )
                    )
                case _:
                    return False
        case SumType() as total:
            match target:
                case SumType() as other if total.cons.keys() == other.cons.keys():
                    return all(
                        solve(payload, other.cons[constructor], variables, solution)
                        for constructor, payload in total.cons.items()
                    )
                case _:
                    return False
        case FunType(return_type=return_type, argument_type=argument_type):
            match target:
                case FunType(return_type=other_return, argument_type=other_argument):
                    return solve(
                        return_type, other_return, variables, solution
                    ) and solve(argument_type, other_argument, variables, solution)
                case _:
                    return False
        case MuType(variable=bound, body=body):
            match target:
                case MuType(variable=other_bound, body=other_body) if (
                    bound == other_bound
                ):
                    # The variable is bound by the `@mu` and not quantified by
                    # the scheme, so it is removed from what may be solved.
                    return solve(body, other_body, variables - {bound}, solution)
                case _:
                    # Two recursive types written over different variables are
                    # compared whole. Matching equirecursive types up to the
                    # renaming of their binders is not attempted, so a scheme
                    # instantiated at such a type is refused rather than
                    # silently taken at the wrong one.
                    return pattern == target
        case _:
            raise ValueError(f"not a type: {pattern}")


@dataclass(frozen=True, slots=True)
class Context:
    # Term binders and the types they are ascribed. These accumulate
    # monotonically across one judgment, which is why `register` refuses to
    # rebind one.
    ascriptions: dict[Symbol, Type]

    # Type variables and the types they stand for. Scoped rather than
    # accumulated: a variable means something only inside the body that binds
    # it, so `bind` returns a new context instead of mutating this one.
    variables: Mapping[Symbol, Type] = field(default_factory=dict)

    # Binders ascribed a polymorphic type. Kept in a field of their own rather
    # than widening `ascriptions` to hold either sort, because `__contains__`
    # and `__getitem__` answer for `ascriptions` alone and every reader outside
    # this module reads a type through them: `Analysis.ascription` asks what a
    # binder is ascribed and expects a `Type`, and a polymorphic binder is
    # simply absent from what it reads, which is the answer it already gives
    # one. Instantiation is the type checker's to do, and only it looks here.
    schemes: dict[Symbol, Scheme] = field(default_factory=dict)

    # Type variables that are abstract: quantified by a scheme whose body is
    # being checked, and so standing for nothing yet.
    #
    # The distinction `variables` cannot make. A variable bound there stands for
    # a known type and every query answers for that type; one named here stands
    # for any type at all, so a query must answer for none of them rather than
    # resolve it. `variable` raising on an unbound name is what keeps the two
    # from being confused — an abstract variable is not an unbound one, and the
    # `VarType` arms read this set before they resolve.
    rigid: frozenset[Symbol] = frozenset()

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.ascriptions)

    def __contains__(self, binder: object) -> bool:
        return binder in self.ascriptions

    def __len__(self) -> int:
        return len(self.ascriptions)

    def __getitem__(self, binder: Symbol) -> Type:
        return self.ascriptions[binder]

    def register(self, binder: Symbol, ty: Type) -> None:
        if binder in self.ascriptions and self.ascriptions[binder] != ty:
            raise ValueError(
                f"Attempt to bind {binder}={ty}, but {binder}={self.ascriptions[binder]} already bound."
            )
        else:
            self.ascriptions[binder] = ty

    def register_scheme(self, binder: Symbol, scheme: Scheme) -> None:
        if binder in self.ascriptions:
            raise ValueError(
                f"Attempt to bind {binder}={scheme}, but"
                f" {binder}={self.ascriptions[binder]} already bound."
            )
        elif binder in self.schemes and self.schemes[binder] != scheme:
            raise ValueError(
                f"Attempt to bind {binder}={scheme}, but"
                f" {binder}={self.schemes[binder]} already bound."
            )
        else:
            self.schemes[binder] = scheme

    def bind(self, variable: Symbol, ty: Type) -> Context:
        # An inner `@mu` of the same variable shadows the outer one rather than
        # conflicting with it, which is the rule `guarded` and `substitute`
        # already follow — so this extends functionally where `register`
        # refuses.
        #
        # A binding must name the `@mu` that binds the variable, or a type with
        # no free occurrence of it. One naming another variable that cycles
        # would make the `VarType` arms recur forever, which is what `guarded`
        # rules out — assumed of every type the language builds and, as with
        # `substitute`'s no-capture precondition, documented and not enforced.
        #
        # A variable given a meaning here is no longer abstract, so it leaves
        # `rigid`: a scheme's body may well write `(@mu a ...)` over one of its
        # own quantified spellings, and within that `@mu` the name stands for
        # the recursive type and not for the instance.
        return Context(
            self.ascriptions,
            {**self.variables, variable: ty},
            self.schemes,
            self.rigid - {variable},
        )

    def abstract(self, variables: tuple[Symbol, ...]) -> Context:
        # This context with `variables` made abstract, which is what a scheme's
        # body is checked under. Functional like `bind` and for the same reason:
        # a quantified variable means something only inside the body that binds
        # it. The two mutable namespaces are shared by reference, so a parameter
        # registered while checking that body lands in the one store.
        return Context(
            self.ascriptions,
            self.variables,
            self.schemes,
            self.rigid | set(variables),
        )

    def scheme(self, binder: Symbol) -> Scheme:
        if binder in self.schemes:
            return self.schemes[binder]
        else:
            raise ValueError(f"no scheme ascribed: {binder}")

    def variable(self, binder: Symbol) -> Type:
        # The counterpart to `__getitem__` for the other namespace, raising
        # rather than answering so an unresolved variable cannot be read as a
        # nullary type.
        if binder in self.variables:
            return self.variables[binder]
        else:
            raise ValueError(f"unbound type variable: {binder}")

    @classmethod
    def default(cls) -> Context:
        binop: ProdType = ProdType({Symbol("a"): IntType(), Symbol("b"): IntType()})

        # The one variable every builtin scheme is written over. Schemes are
        # instantiated by substitution and never nest, so one spelling serves
        # them all: no two of these is ever read under the other.
        a: Symbol = Symbol("a")
        element: VarType = VarType(a)
        listed: MuType = list_of(element)

        return cls(
            {
                # `#t` and `#f` are absent, and deliberately: an ascription
                # here is what would let a bare `#t` check through `check`'s
                # reference arm, and a boolean is written as the sum
                # constructor applied to its payload — `(#t (prod#))`.
                #
                # `prod#` stays. It is not what checks a `(prod# ...)`, which
                # `is_constructor` reads by resolution, but it is what lets
                # `type_of` read `(prod#)`'s type — and that payload is read
                # nowhere else.
                Symbol("prod#"): FunType(return_type=ProdType()),
                Symbol("+"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("-"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("*"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("mod"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("lt"): FunType(return_type=BOOL, argument_type=binop),
                Symbol("le"): FunType(return_type=BOOL, argument_type=binop),
            },
            schemes={
                # The selectors. Each writes its variable into its result, so
                # the checked type names the instance on its own and no operand
                # has to be synthesized for them.
                #
                # `.` is absent, and deliberately: it was the cell constructor
                # standing in for one `(@list t)` had no name for, and the sum
                # names it `cons`. A list is built as its constructors applied
                # to their payloads — `(nil (prod#))` and
                # `(cons (prod# x xs))` — which `is_constructor` reads by
                # resolution, so no ascription here would be what checks one.
                Symbol("head"): Scheme(
                    variables=(a,),
                    body=FunType(
                        return_type=element,
                        argument_type=ProdType([listed]),
                    ),
                ),
                Symbol("tail"): Scheme(
                    variables=(a,),
                    body=FunType(
                        return_type=listed,
                        argument_type=ProdType([listed]),
                    ),
                ),
                # The two that write their variable only among their arguments,
                # and so are instantiated from an operand rather than from the
                # checked type. `solve` already anticipates one of them: it is
                # the repeated variable that makes `(eq 1 #t)` a refusal.
                Symbol("eq"): Scheme(
                    variables=(a,),
                    body=FunType(
                        return_type=BOOL,
                        argument_type=ProdType([element, element]),
                    ),
                ),
                Symbol("len"): Scheme(
                    variables=(a,),
                    body=FunType(
                        return_type=IntType(),
                        argument_type=ProdType([listed]),
                    ),
                ),
            },
        )
