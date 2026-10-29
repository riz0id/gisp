from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from Resolve import BindingTable
from Syntax import Char, Expr, Integer, Symbol
from Syntax.Type.BoolType import BoolType
from Syntax.Type.CharType import CharType
from Syntax.Type.FunType import FunType
from Syntax.Type.IntType import IntType
from Syntax.Type.ListType import ListType
from Syntax.Type.MuType import MuType
from Syntax.Type.ProdType import ProdType
from Syntax.Type.SumType import SumType
from Syntax.Type.Type import ConstructorArity, Type, meaning
from Syntax.Type.VariadicType import VariadicStarType, VariadicType
from Syntax.Type.VarType import VarType

__all__: list[str] = [
    "BoolType",
    "CharType",
    "ConstructorArity",
    "Context",
    "FunType",
    "IntType",
    "ListType",
    "MuType",
    "ProdType",
    "SumType",
    "Type",
    "VarType",
    "VariadicStarType",
    "VariadicType",
    "constructor_arity",
    "guarded",
    "is_constructor",
    "is_nullary_type",
    "meaning",
    "nullary_type_of",
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
        case IntType() | CharType() | BoolType():
            # A nullary constructor takes no type arguments, and thus has
            # arity zero.
            return 0
        case VarType(binder=binder):
            # A variable is not a constructor and has no arity of its own: it
            # stands for the type bound to it, and answers whatever that type
            # answers. Zero here would be the arity of the name `a`, not of
            # what `a` denotes.
            return type_constructor_arity(context.variable(binder), context)
        case ListType():
            # `(@list t)` takes exactly one type argument: the element type.
            return 1
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
        case BoolType():
            return Symbol(identifier="@bool")
        case ListType():
            return Symbol(identifier="@list")
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
        case IntType() | CharType() | BoolType():
            # Literals have zero-arity, meaning the constructor (some integer,
            # character, or boolean literal) has no arguments.
            return 0
        case VarType(binder=binder):
            # A type variable is constructed however the type bound to it is.
            return constructor_arity(context.variable(binder), context)
        case ListType():
            # The `(list#x ...)` constructor has no bounded arity (because of
            # the ellipsis).
            return None
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


def is_constructor(ty: Type, expr: Expr, context: Context, table: BindingTable) -> bool:
    # Whether `expr` is a constructor occurrence of `ty`: a literal of the sort
    # it classifies, or a reference resolving to the binder that constructs it.
    match ty:
        case IntType():
            return isinstance(expr, Integer)
        case CharType():
            return isinstance(expr, Char)
        case BoolType():
            match expr:
                case Symbol() as symbol:
                    binder: Symbol = meaning(symbol, table)

                    return binder in context and context[binder] == BoolType()
                case _:
                    return False
        case VarType(binder=binder):
            # A type variable is constructed however the type bound to it is.
            return is_constructor(context.variable(binder), expr, context, table)
        case ListType():
            match expr:
                case Symbol() as symbol:
                    return meaning(symbol, table) == Symbol("list")
                case _:
                    return False
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
                    return meaning(symbol, table) == Symbol("prod#")
                case _:
                    return False
        case FunType():
            match expr:
                case Symbol() as symbol:
                    return meaning(symbol, table) == Symbol("lambda")
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


def is_nullary_type(ty: Type, context: Context) -> bool:
    # Whether `ty` is a ground type: one written as a bare symbol, classifying
    # values that have a representation of their own. Read by resolution and
    # not by equality against a set of instances, so a variable bound to a
    # ground type is one: a variable bound to `@int` stands for `@int` however
    # it is spelled.
    match ty:
        case IntType() | CharType() | BoolType():
            return True
        case VarType(binder=binder):
            # As in the four queries: a variable answers for the type bound to
            # it, and `context.variable` refuses an unbound one rather than
            # letting it read as a type it is not.
            return is_nullary_type(context.variable(binder), context)
        case (
            ListType()
            | SumType()
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


def nullary_type_of(symbol: Symbol, context: Context) -> Type:
    # The reverse of each nullary type's own `type_constructor_symbol`, read by
    # identifier only: the symbol reaching here is `parse()`'s scope-stripped
    # output, so comparing full `Symbol` equality would be an accident to rely
    # on. Only the nullary types are listed — a compound type is written as a
    # list, and no spelling of one arrives here as a bare symbol.
    for ty in (IntType(), CharType(), BoolType()):
        if type_constructor_symbol(ty, context).identifier == symbol.identifier:
            return ty

    raise ValueError(f"not a nullary type: {symbol}")


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
        case IntType() | CharType() | BoolType():
            return ty
        case VarType(binder=binder):
            return replacement if binder == variable else ty
        case ListType(element=element):
            return ListType(substitute(element, variable, replacement))
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
        return Context(self.ascriptions, {**self.variables, variable: ty})

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

        return cls(
            {
                Symbol("#t"): BoolType(),
                Symbol("#f"): BoolType(),
                Symbol("prod#"): FunType(return_type=ProdType()),
                Symbol("+"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("-"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("*"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("mod"): FunType(return_type=IntType(), argument_type=binop),
                Symbol("lt"): FunType(return_type=BoolType(), argument_type=binop),
                Symbol("le"): FunType(return_type=BoolType(), argument_type=binop),
            }
        )
