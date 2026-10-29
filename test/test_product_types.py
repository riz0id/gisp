import pytest

from Resolve import BindingTable
from Syntax import Symbol
from Syntax.Type import (
    BoolType,
    CharType,
    Context,
    IntType,
    ProdType,
    Type,
    constructor_arity,
    is_constructor,
    is_nullary_type,
    type_constructor_arity,
    type_constructor_symbol,
)

# -- shape ------------------------------------------------------------------------


def test_a_tuple_reports_its_types_in_order_and_has_no_names():
    ty = ProdType([IntType(), BoolType()])

    assert not ty.is_record()
    assert ty.names() is None
    assert ty.types() == (IntType(), BoolType())


def test_a_record_reports_both_its_names_and_its_types_in_order():
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BoolType()})

    assert ty.is_record()
    assert ty.names() == (Symbol("a"), Symbol("b"))
    assert ty.types() == (IntType(), BoolType())


def test_a_tuple_and_a_record_over_the_same_types_are_different_types():
    assert ProdType([IntType()]) != ProdType({Symbol("a"): IntType()})


# -- equality and hashing ---------------------------------------------------------


def test_a_tuples_equality_is_positional():
    assert ProdType([IntType(), BoolType()]) != ProdType([BoolType(), IntType()])
    assert ProdType([IntType(), BoolType()]) == ProdType([IntType(), BoolType()])


def test_equal_products_of_either_shape_hash_equal():
    assert hash(ProdType([IntType()])) == hash(ProdType([IntType()]))

    record: dict[Symbol, Type] = {Symbol("a"): IntType()}
    assert hash(ProdType(record)) == hash(ProdType(dict(record)))

    assert len({ProdType([IntType()]), ProdType([IntType()])}) == 1


# -- the nullary product ----------------------------------------------------------


def test_the_nullary_product_is_the_empty_tuple_however_it_is_written():
    assert ProdType() == ProdType([]) == ProdType({})

    assert not ProdType().is_record()
    assert ProdType().names() is None
    assert ProdType().types() == ()


def test_the_nullary_product_is_not_a_nullary_type():
    # Type-constructor arity zero without being a ground type: `(@prod)` takes
    # no type arguments, and is still written as a list.
    assert type_constructor_arity(ProdType(), Context.default()) == 0
    assert not is_nullary_type(ProdType(), Context.default())


# -- arity ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ty",
    [
        ProdType([IntType(), BoolType()]),
        ProdType({Symbol("a"): IntType(), Symbol("b"): BoolType()}),
    ],
)
def test_both_arities_count_the_fields_of_either_shape(ty: ProdType):
    assert type_constructor_arity(ty, Context.default()) == 2
    assert constructor_arity(ty, Context.default()) == 2
    assert type_constructor_symbol(ty, Context.default()) == Symbol("@prod")


# -- rebuilding -------------------------------------------------------------------


def test_rebuilding_a_tuple_keeps_it_a_tuple():
    rebuilt = ProdType([IntType(), BoolType()]).with_types([CharType(), IntType()])

    assert rebuilt == ProdType([CharType(), IntType()])


def test_rebuilding_a_record_keeps_its_names_in_order():
    ty = ProdType({Symbol("a"): IntType(), Symbol("b"): BoolType()})
    rebuilt = ty.with_types([CharType(), IntType()])

    assert rebuilt == ProdType({Symbol("a"): CharType(), Symbol("b"): IntType()})


@pytest.mark.parametrize(
    "ty",
    [ProdType([IntType()]), ProdType({Symbol("a"): IntType()})],
)
def test_rebuilding_from_the_wrong_number_of_types_is_an_error(ty: ProdType):
    with pytest.raises(ValueError):
        ty.with_types([IntType(), BoolType()])


# -- the caller's collection ------------------------------------------------------


def test_a_product_does_not_change_when_the_caller_mutates_what_built_it():
    names: dict[Symbol, Type] = {Symbol("a"): IntType()}
    record = ProdType(names)
    names[Symbol("b")] = BoolType()

    assert record == ProdType({Symbol("a"): IntType()})

    types: list[Type] = [IntType()]
    anonymous = ProdType(types)
    types.append(BoolType())

    assert anonymous == ProdType([IntType()])


# -- construction -----------------------------------------------------------------


@pytest.mark.parametrize(
    "ty",
    [ProdType([IntType()]), ProdType({Symbol("a"): IntType()})],
)
def test_prod_constructs_either_shape(ty: ProdType):
    assert is_constructor(ty, Symbol("prod#"), Context.default(), BindingTable())
