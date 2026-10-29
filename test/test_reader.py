import pytest
from hypothesis import given
from hypothesis import strategies as st
from Strategies import expressions

from Reader import ReadError, read, read_all
from Syntax import Char, Integer, Keyword, SList, Symbol

blank_sources = st.sampled_from(["", "   ", "\n\t ", "; comment", "; a\n; b\n"])


@given(st.lists(expressions, max_size=6))
def test_read_all_recovers_every_form(exprs):
    assert read_all(" ".join(str(expr) for expr in exprs)) == exprs


@given(blank_sources)
def test_read_all_of_formless_source_is_empty(source):
    assert read_all(source) == []


@given(blank_sources)
def test_read_of_formless_source_is_an_error(source):
    with pytest.raises(ReadError):
        read(source)


@given(expressions)
def test_quote_sugar_expands_to_a_quote_form(expr):
    printed = str(expr)
    expected = SList([Symbol("quote"), expr])
    assert read("'" + printed) == expected
    assert read("(quote " + printed + ")") == expected


@given(blank_sources)
def test_quote_without_a_following_form_is_an_error(source):
    with pytest.raises(ReadError):
        read("'" + source)


# Pinned directly because the symbol strategies exclude `.` from their
# alphabet: the atom terminal has no reserved dot, and no float literal
# competes for it, so a bare dot is an ordinary symbol — the `.` builtin
# needs no reader support.
def test_a_bare_dot_reads_as_a_symbol():
    assert read(".") == Symbol(".")


@given(expressions)
def test_comments_and_whitespace_are_transparent(expr):
    assert read("  " + str(expr) + " ; comment\n") == expr


@given(expressions)
def test_unbalanced_sources_are_errors(expr):
    printed = str(expr)
    with pytest.raises(ReadError):
        read(printed + ")")
    with pytest.raises(ReadError):
        read("(" + printed)


@given(expressions, expressions)
def test_trailing_forms_are_rejected_by_read_but_not_read_all(first, second):
    source = str(first) + " " + str(second)
    with pytest.raises(ReadError):
        read(source)
    assert read_all(source) == [first, second]


# A keyword is the datum this reader was changed to produce: before the
# `KEYWORD` terminal, `ATOM` swallowed `#:x` and `atom` read it as an
# identifier. Pinned rather than left to the round-trip property, since the
# property is satisfied by any sort that prints and reads back consistently —
# including the `Symbol` this must no longer be.
def test_a_keyword_reads_as_a_keyword_and_not_a_symbol():
    assert read("#:x") == Keyword("x")
    assert read("#:x") != Symbol("#:x")


def test_a_keyword_is_one_form_in_operand_position():
    assert read("(f #:x 1)") == SList([Symbol("f"), Keyword("x"), Integer(1)])


# Keyword names are not identifiers, so the integer check in `atom` never
# reaches them.
def test_a_numeric_keyword_name_is_not_an_integer():
    assert read("#:1") == Keyword("1")


# `CHAR` and `KEYWORD` share a priority and a leading `#`, so the colon
# character is the overlap worth pinning.
def test_the_colon_character_still_reads_as_a_character():
    assert read("#\\:") == Char(ord(":"))


# The terminal matches the bare marker on purpose: claiming `#:` here is what
# keeps it from falling through to `ATOM` and reading as the symbol `#:`, both
# at end of input and before a delimiter.
@pytest.mark.parametrize("source", ["#:", "#: ", "#:(a)", "(f #:)"])
def test_a_keyword_without_a_name_is_an_error(source):
    with pytest.raises(ReadError):
        read(source)


def test_a_quoted_keyword_is_a_quote_form():
    assert read("'#:x") == SList([Symbol("quote"), Keyword("x")])
