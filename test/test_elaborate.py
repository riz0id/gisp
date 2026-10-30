from collections.abc import Callable
from itertools import count

import pytest

from Expander import expand, parse
from Reader import read
from Resolve import bindings
from Syntax import Expr, Scope, SList, Symbol
from Syntax.Type import (
    BOOL,
    CharType,
    FunType,
    IntType,
    MuType,
    ProdType,
    Scheme,
    SumType,
    Type,
    VariadicStarType,
    VariadicType,
    VarType,
    list_of,
    unfold,
)
from TypeCheck.Elaborate import ElaborateError, elaborate, elaborate_type

# Elaboration reads a type the parse phase has already checked and resolved, and
# builds the `Type` it denotes. Two rules are its own, because each is about
# meaning rather than written form: the rank-1 restriction, and that a `@fun`'s
# first operand must denote an argument run.
#
# Fixtures inject deterministic counters from the 1700s so no expectation
# depends on the global fresh-uid counter and no uid collides with another test
# file's hand-written scopes.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def type_operand(source: str, expand_at: int = 1700) -> Expr:
    # The parsed type operand of `(lambda ((ascribe# x <source>)) x)` — what
    # elaboration is handed, and the only thing it is ever handed.
    program = f"(lambda ((ascribe# x {source})) x)"
    expanded = expand(read(program), fresh=counter(expand_at))

    match parse(expanded, bindings(expanded)):
        case SList([_, SList([SList([_, _, ty])]), _]):
            return ty
        case parsed:
            raise AssertionError(f"not an ascribed lambda: {parsed}")


def type_of(source: str) -> Type:
    return elaborate_type(type_operand(source))


def scheme_of(source: str) -> Scheme:
    return elaborate(type_operand(source))


# -- ground and compound types ---------------------------------------------------


def test_every_ground_type_is_written_as_a_bare_identifier():
    assert type_of("@int") == IntType()
    assert type_of("@char") == CharType()
    assert type_of("@bool") == BOOL


def test_bool_denotes_a_sum_of_two_nullary_constructors():
    # The spelling is the built-in and this is the type it means. `@bool` is
    # not a sort of its own: it denotes a sum like any other, which is what
    # lets `constructors` enumerate it and a `match` take a boolean apart.
    assert BOOL == SumType({Symbol("#t"): ProdType(), Symbol("#f"): ProdType()})
    assert type_of("@bool") == BOOL


def test_bool_and_the_sum_written_out_are_one_type():
    # Not merely equivalent — the same value, so every judgment over one is the
    # judgment over the other.
    assert type_of("(@sum (#t (@prod)) (#f (@prod)))") == type_of("@bool")


def test_a_list_type_is_written_over_its_element():
    assert type_of("(@list @int)") == list_of(IntType())
    assert type_of("(@list (@list @char))") == list_of(list_of(CharType()))


def test_a_list_denotes_a_recursive_sum_of_two_constructors():
    # The point of the change. `(@list t)` is not a sort of its own: it is the
    # `@mu` over a sum whose `cons` carries a record of the head and the rest,
    # and whose `nil` carries the nullary product. Every reading of a list goes
    # through the `MuType` and `SumType` arms because of this.
    elaborated = type_of("(@list @int)")

    match elaborated:
        case MuType(variable=bound, body=SumType(cons=cons)):
            pass
        case _:
            raise AssertionError(f"not a recursive sum: {elaborated}")

    assert set(cons) == {Symbol("cons"), Symbol("nil")}
    assert cons[Symbol("nil")] == ProdType()
    assert cons[Symbol("cons")] == ProdType(
        {Symbol("x"): IntType(), Symbol("xs"): VarType(bound)}
    )


def test_unfolding_a_list_leaves_its_tail_folded():
    # What makes a `(cons x xs)` pattern bind `xs` at the list type as written:
    # only the top node changes form, so the recursive position comes back as
    # the folded type and no comparison ever meets an unfolding.
    elaborated = type_of("(@list @int)")
    unfolded = unfold(elaborated)

    assert isinstance(unfolded, SumType)
    assert unfolded.cons[Symbol("cons")] == ProdType(
        {Symbol("x"): IntType(), Symbol("xs"): elaborated}
    )


def test_the_recursion_variable_is_one_no_program_can_bind():
    # The binder has to be unwritable, because `substitute` renames nothing and
    # the element type is routinely written over a variable a program chose:
    # `(@list a)` inside a quantifier is exactly that. A binder a program could
    # also write would capture the element on instantiation.
    match type_of("(@list @int)"):
        case MuType(variable=bound):
            pass
        case elaborated:
            raise AssertionError(f"not a recursive type: {elaborated}")

    assert bound == Symbol("")
    assert not bound.scopes


def test_a_hand_written_list_sum_is_a_type_of_its_own():
    # The one place `(@list t)` differs from `@bool`, where the sum written out
    # *is* the type. The recursion variable cannot be written, so a program's
    # own list sum is a distinct type — read identically by every pattern, and
    # not interchangeable with `(@list @int)`.
    written = type_of("(@mu r (@sum (cons (@prod (x @int) (xs r))) (nil (@prod))))")

    assert isinstance(written, MuType)
    assert written != list_of(IntType())


def test_the_nullary_product_is_written_with_no_fields():
    assert type_of("(@prod)") == ProdType()


def test_a_positional_product_keeps_its_fields_in_order():
    assert type_of("(@prod @int @bool)") == ProdType([IntType(), BOOL])


def test_a_named_product_is_a_record_of_bare_names():
    record = type_of("(@prod (head @int) (tail @bool))")

    assert record == ProdType({Symbol("head"): IntType(), Symbol("tail"): BOOL})
    assert isinstance(record, ProdType)
    assert record.names() == (Symbol("head"), Symbol("tail"))


def test_a_field_written_as_a_type_form_stays_positional():
    # The disambiguation: `(@list @int)` is a type and not a field named
    # `@list`, which the leading `@` tells apart.
    assert type_of("(@prod (@list @int))") == ProdType([list_of(IntType())])


def test_a_sum_is_written_as_its_constructors():
    assert type_of("(@sum (zero (@prod)) (succ @int))") == SumType(
        {Symbol("zero"): ProdType(), Symbol("succ"): IntType()}
    )
    assert type_of("(@sum (only @int))") == SumType({Symbol("only"): IntType()})


def test_the_variadic_runs_are_written_over_one_type():
    assert type_of("(@variadic @int)") == VariadicType(IntType())
    assert type_of("(@variadic* @int)") == VariadicStarType(IntType())


def test_a_function_type_is_written_run_then_return():
    assert type_of("(@fun (@prod @int @int) @bool)") == FunType(
        return_type=BOOL, argument_type=ProdType([IntType(), IntType()])
    )
    assert type_of("(@fun (@prod) @bool)") == FunType(return_type=BOOL)
    assert type_of("(@fun (@prod (n @int)) @int)") == FunType(
        return_type=IntType(), argument_type=ProdType({Symbol("n"): IntType()})
    )


def test_a_function_type_is_written_over_a_variadic_run():
    assert type_of("(@fun (@variadic @int) @int)") == FunType(
        return_type=IntType(), argument_type=VariadicType(IntType())
    )
    assert type_of("(@fun (@variadic* @int) @int)") == FunType(
        return_type=IntType(), argument_type=VariadicStarType(IntType())
    )


def test_a_function_type_over_a_type_that_is_no_run_is_refused():
    # `@int` classifies values, and a function is applied to an argument list.
    # Written form says nothing about this, so the parse phase admits it and the
    # refusal is here.
    assert type_operand("(@fun @int @int)") is not None

    with pytest.raises(ElaborateError):
        type_of("(@fun @int @int)")


# -- recursive types -------------------------------------------------------------


def test_a_recursive_type_binds_its_variable_in_its_body():
    # The `nat()` fixture of `test_recursive_types.py`. The binder is whatever
    # `@mu` minted, so the test states the binding relation rather than a uid.
    match type_of("(@mu a (@sum (zero (@prod)) (succ a)))"):
        case MuType(variable=bound, body=SumType() as total):
            pass
        case ty:
            raise AssertionError(f"not a recursive sum: {ty}")

    assert total.cons[Symbol("zero")] == ProdType()
    assert total.cons[Symbol("succ")] == VarType(bound)


def test_a_recursive_record_is_written_through_its_payloads():
    # The `ints()` fixture, whose recursive position sits two sorts down.
    source = "(@mu a (@sum (nil (@prod)) (cons (@prod (head @int) (tail a)))))"

    match type_of(source):
        case MuType(variable=bound, body=SumType() as total):
            pass
        case ty:
            raise AssertionError(f"not a recursive sum: {ty}")

    assert total.cons[Symbol("nil")] == ProdType()
    assert total.cons[Symbol("cons")] == ProdType(
        {Symbol("head"): IntType(), Symbol("tail"): VarType(bound)}
    )


def test_an_elaborated_recursive_type_unfolds():
    # What says the symbols the parse phase emits are the ones substitution
    # needs: `Symbol` equality covers scopes, so a binder and its occurrences
    # have to be one value for `unfold` to replace anything at all. Resolution
    # now guarantees that, where a set of identifier strings approximated it.
    elaborated = type_of("(@mu a (@list a))")

    assert isinstance(elaborated, MuType)
    assert unfold(elaborated) == list_of(elaborated)


def test_a_nested_recursive_type_binds_a_distinct_variable():
    match type_of("(@mu a (@prod (@mu a (@prod a))))"):
        case MuType(variable=outer, body=ProdType() as fields):
            pass
        case ty:
            raise AssertionError(f"not a recursive product: {ty}")

    match fields.types():
        case (MuType(variable=inner, body=ProdType() as inner_fields),):
            pass
        case types:
            raise AssertionError(f"not a nested @mu: {types}")

    assert outer != inner
    assert inner_fields.types() == (VarType(inner),)


def test_an_outer_binding_reaches_inside_an_inner_one():
    match type_of("(@mu a (@prod (@mu b (@prod a))))"):
        case MuType(variable=outer, body=ProdType() as fields):
            pass
        case ty:
            raise AssertionError(f"not a recursive product: {ty}")

    match fields.types():
        case (MuType(body=ProdType() as inner_fields),):
            pass
        case types:
            raise AssertionError(f"not a nested @mu: {types}")

    assert inner_fields.types() == (VarType(outer),)


# -- schemes ---------------------------------------------------------------------


def test_a_type_written_without_a_quantifier_is_a_trivial_scheme():
    # Every ascription reads as a scheme, so the monomorphic ones have to read
    # as the scheme that quantifies nothing rather than being a separate shape.
    assert scheme_of("@int") == Scheme(variables=(), body=IntType())
    assert scheme_of("(@list @int)") == Scheme(variables=(), body=list_of(IntType()))
    assert scheme_of("@int").monomorphic()


def test_a_quantifier_binds_its_variable_in_the_body():
    scheme = scheme_of("(forall# (a) (@fun (@prod a) a))")

    match scheme.variables:
        case (bound,):
            pass
        case variables:
            raise AssertionError(f"not one quantified variable: {variables}")

    assert scheme.body == FunType(
        return_type=VarType(bound), argument_type=ProdType([VarType(bound)])
    )
    assert not scheme.monomorphic()


def test_a_quantifier_binds_several_variables_in_order():
    scheme = scheme_of("(forall# (a b) (@fun (@prod a b) b))")

    match scheme.variables:
        case (first, second):
            pass
        case variables:
            raise AssertionError(f"not two quantified variables: {variables}")

    assert first.identifier == "a"
    assert second.identifier == "b"
    assert scheme.body == FunType(
        return_type=VarType(second),
        argument_type=ProdType([VarType(first), VarType(second)]),
    )


def test_a_quantified_variable_is_the_binder_the_quantifier_minted():
    # The counterpart of the old expectation that a quantified variable was
    # emitted bare. It is not bare any more: it is the binder `forall#` minted,
    # and `solve` matches it by that value rather than by its identifier.
    scheme = scheme_of("(forall# (a) a)")

    match scheme.variables:
        case (bound,):
            pass
        case variables:
            raise AssertionError(f"not one quantified variable: {variables}")

    assert bound.scopes != set()
    assert scheme.body == VarType(bound)


def test_a_mu_within_a_quantifier_binds_its_own_variable():
    # The inner `@mu` is the binding its body is read under, which is the rule
    # a `@mu` within a `@mu` already follows.
    scheme = scheme_of("(forall# (a) (@mu a (@list a)))")

    match scheme.variables:
        case (quantified,):
            pass
        case variables:
            raise AssertionError(f"not one quantified variable: {variables}")

    match scheme.body:
        case MuType(variable=bound, body=MuType()):
            # A `@mu` whose body is another: `(@list a)` is itself the `@mu`
            # over a sum, so a `@mu` written over a list type nests.
            pass
        case body:
            raise AssertionError(f"not a recursive list: {body}")

    assert quantified != bound
    # The element is the inner `@mu`'s own binder and not the quantifier's,
    # which is the whole of what this test is about.
    assert scheme.body == MuType(bound, list_of(VarType(bound)))


# -- rank-1 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        # A quantifier is written outermost and nowhere within a type, which is
        # the whole of the rank-1 restriction. Each of these parses: the rule is
        # about what the type means, so it is read here.
        "(@list (forall# (a) a))",
        "(forall# (a) (forall# (b) (@fun (@prod a) b)))",
        "(@fun (@prod (forall# (a) a)) @int)",
    ],
)
def test_a_misplaced_quantifier_is_refused(source: str):
    with pytest.raises(ElaborateError):
        scheme_of(source)


def test_a_quantifier_over_nothing_is_refused():
    # Written where a reader expects a variable to be bound, so an empty binder
    # list is far more likely a mistake than an intent. The written form is
    # well formed, so the parse phase admits it.
    assert type_operand("(forall# () @int)") is not None

    with pytest.raises(ElaborateError):
        scheme_of("(forall# () @int)")
