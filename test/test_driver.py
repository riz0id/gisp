import io
from pathlib import Path

import pytest
from gisp.__main__ import main

LIB: Path = Path(__file__).parent.parent / "lib"

SORTED: str = "(1 1 2 3 4 5 6 9)\n"


# -- the libraries ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["bubblesort.gisp", "quicksort.gisp"])
def test_a_fully_ascribed_library_checks_and_runs(capsys, name):
    # What the pass is for: every binder in these files carries a type, and the
    # driver reads them before anything runs.
    assert main([str(LIB / name)]) == 0
    assert capsys.readouterr().out == SORTED


# -- what the pass refuses -------------------------------------------------------


def test_a_declaration_its_initializer_does_not_inhabit_is_refused(capsys):
    assert main(["-e", "(define (ascribe# n @int) (#t (prod#)))"]) == 1
    assert "gisp:" in capsys.readouterr().err


def test_an_unascribed_declaration_is_refused(capsys):
    # Check-only has nowhere to read a declaration's type from.
    assert main(["-e", "(define n 5)"]) == 1
    assert "gisp:" in capsys.readouterr().err


def test_a_misapplied_builtin_is_refused(capsys):
    assert main(["-e", "(le (#t (prod#)) 1)"]) == 1
    assert "gisp:" in capsys.readouterr().err


def test_an_un_expanded_core_form_is_refused_by_the_parse_phase(capsys):
    # The phases ahead of the checker report through the same handler.
    assert main(["-e", "(lambda)"]) == 1
    assert "gisp:" in capsys.readouterr().err


def test_a_malformed_source_is_refused_by_the_reader(capsys):
    assert main(["-e", "(+ 1"]) == 1
    assert "gisp:" in capsys.readouterr().err


# -- the pass precedes evaluation ------------------------------------------------


def test_nothing_is_evaluated_when_a_later_form_is_refused(capsys):
    # The first form would fail at runtime. The second does not check, so the
    # reported error is the type error and not `head`'s — which is only true if
    # the whole source is checked before any of it runs.
    source = "(head '()) (define n 5)"

    assert main(["-e", source]) == 1

    captured = capsys.readouterr()

    assert captured.out == ""
    assert "empty" not in captured.err


def test_a_program_that_checks_still_reports_its_runtime_error(capsys):
    # The converse: checking does not make `head` total.
    assert main(["-e", "(head '())"]) == 1
    assert "gisp:" in capsys.readouterr().err


# -- declarations across forms ---------------------------------------------------


def test_a_later_form_applies_an_earlier_declaration(capsys):
    source = (
        "(define (ascribe# double (@fun (@prod @int) @int)) (lambda (x) (+ x x)))"
        " (double 21)"
    )

    assert main(["-e", source]) == 0
    assert capsys.readouterr().out == "42\n"


def test_a_declaration_is_refused_a_second_time_at_another_type(capsys):
    source = "(define (ascribe# n @int) 5) (define (ascribe# n @bool) (#t (prod#)))"

    assert main(["-e", source]) == 1
    assert "gisp:" in capsys.readouterr().err


# -- the repl --------------------------------------------------------------------


def repl(monkeypatch, lines: str) -> int:
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    return main([])


def test_the_repl_checks_each_line(monkeypatch, capsys):
    assert repl(monkeypatch, "(+ 1 2)\n") == 0
    assert capsys.readouterr().out == "3\n"


def test_the_repl_carries_a_declaration_across_lines(monkeypatch, capsys):
    # The store crosses lines for the reason the environment does.
    lines = (
        "(define (ascribe# double (@fun (@prod @int) @int)) (lambda (x) (+ x x)))\n"
        "(double 21)\n"
    )

    assert repl(monkeypatch, lines) == 0
    assert capsys.readouterr().out == "42\n"


def test_the_repl_continues_past_a_line_that_does_not_check(monkeypatch, capsys):
    lines = "(le (#t (prod#)) 1)\n(+ 1 2)\n"

    assert repl(monkeypatch, lines) == 0

    captured = capsys.readouterr()

    assert captured.out == "3\n"
    assert "gisp:" in captured.err
