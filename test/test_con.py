import pytest
from hypothesis import given
from Strategies import expressions

from Syntax import (
    Char,
    Integer,
    RecordConExpr,
    SList,
    SumConExpr,
    Symbol,
    as_data,
    as_syntax,
    boolean,
    elements,
    listed,
    product,
    spread,
    unit,
)
from Syntax.Type import IntType, ProdType
from TypeCheck.Patterns import product_fields

X = Symbol("x")
XS = Symbol("xs")


# -- the two product shapes ------------------------------------------------------


def test_a_tuple_has_no_field_names_and_a_record_does():
    assert product([Integer(1)]).names() is None
    assert RecordConExpr({X: Integer(1)}).names() == (X,)


def test_both_shapes_answer_with_their_fields_in_order():
    fields = [Integer(1), Integer(2)]

    assert product(fields).values() == (Integer(1), Integer(2))
    assert RecordConExpr({X: Integer(1), XS: Integer(2)}).values() == (
        Integer(1),
        Integer(2),
    )


def test_a_record_is_never_equal_to_a_tuple_of_the_same_fields():
    # The value side mirrors the type side: `ProdType({x: t})` is not
    # `ProdType([t])`, so neither are their values.
    assert RecordConExpr({X: Integer(1)}) != product([Integer(1)])
    assert ProdType({X: IntType()}) != ProdType([IntType()])


def test_the_nullary_product_is_the_empty_tuple():
    # A record with no field to name is the empty tuple however it was
    # written, which is the reading `ProdType.__post_init__` already makes.
    assert unit() == product([])
    assert ProdType({}) == ProdType([])

    with pytest.raises(ValueError):
        RecordConExpr({})


# -- reading a product -----------------------------------------------------------


def test_spread_reads_a_product_as_product_fields_reads_its_type():
    # The two are one rule over the two sides: a product is read as its
    # fields, and anything else as the one value it is. What a constructor
    # pattern is written with is `operand_types`, and not this.
    products = [
        (unit(), ProdType()),
        (product([Integer(1), Integer(2)]), ProdType([IntType(), IntType()])),
        (
            RecordConExpr({X: Integer(1), XS: Integer(2)}),
            ProdType({X: IntType(), XS: IntType()}),
        ),
        (Integer(1), IntType()),
    ]

    for value, ty in products:
        assert len(spread(value)) == len(product_fields(ty))


# -- lists -----------------------------------------------------------------------


def test_a_list_is_a_nest_of_cells_ending_in_nil():
    assert listed([]) == SumConExpr(Symbol("nil"), unit())
    assert listed([Integer(1)]) == SumConExpr(
        Symbol("cons"), product([Integer(1), listed([])])
    )


def test_elements_reads_a_list_back_and_declines_anything_else():
    assert elements(listed([])) == []
    assert elements(listed([Integer(1), Integer(2)])) == [Integer(1), Integer(2)]
    assert elements(boolean(True)) is None
    assert elements(unit()) is None
    assert elements(Integer(1)) is None


def test_a_cell_whose_payload_is_a_record_is_still_a_list():
    # `elements` reads a cell through `spread`, so it does not care which
    # product shape the payload is — which is what leaves a record-payloaded
    # cell readable if one is ever built.
    cell = SumConExpr(Symbol("cons"), RecordConExpr({X: Integer(1), XS: listed([])}))

    assert elements(cell) == [Integer(1)]


# -- crossing between syntax and data --------------------------------------------


@given(expressions)
def test_a_datum_and_the_value_it_denotes_print_alike(datum):
    # The invariant the printer is written for: a quoted list and the list it
    # evaluates to are the same text, so list and string notation survive a
    # value having no `SList` in it.
    assert str(as_data(datum)) == str(datum)


@given(expressions)
def test_as_syntax_undoes_as_data(datum):
    assert as_syntax(as_data(datum)) == datum


def test_as_data_leaves_an_atom_alone():
    for datum in [Integer(1), Char(ord("a")), Symbol("x")]:
        assert as_data(datum) == datum


def test_as_syntax_leaves_a_value_that_is_not_a_list_alone():
    assert as_syntax(boolean(True)) == boolean(True)
    assert as_syntax(unit()) == unit()


# -- printing --------------------------------------------------------------------


def test_a_construction_prints_as_itself_where_no_notation_reads_it_back():
    assert str(unit()) == "(prod#)"
    assert str(product([Integer(1), Integer(2)])) == "(prod# 1 2)"
    assert str(boolean(True)) == "(#t (prod#))"
    assert str(boolean(False)) == "(#f (prod#))"
    assert (
        str(RecordConExpr({X: Integer(1), XS: unit()})) == "(prod# (x 1) (xs (prod#)))"
    )


def test_a_list_prints_in_the_notation_the_reader_has_for_one():
    assert str(listed([])) == "()"
    assert str(listed([Integer(1), Integer(2)])) == "(1 2)"
    assert str(listed([Char(ord("h")), Char(ord("i"))])) == '"hi"'
    assert str(listed([listed([Integer(1)]), Integer(2)])) == "((1) 2)"


def test_a_sum_constructor_over_a_bare_payload_prints_as_a_construction():
    assert str(SumConExpr(Symbol("succ"), Integer(1))) == "(succ 1)"


# -- a value is never syntax -----------------------------------------------------


def test_no_constructed_value_is_an_slist():
    # The whole of what this change separates: `SList` is what the reader
    # builds, and nothing a program evaluates to.
    for value in [unit(), product([Integer(1)]), boolean(True), listed([Integer(1)])]:
        assert not isinstance(value, SList)
