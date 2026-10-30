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


# -- the walk report -------------------------------------------------------------

# Bubblesort over three elements, which is small enough for the report to show
# every walk of it.
BUBBLE: str = """
(define (ascribe# pass (@fun (@prod (@list @int)) (@list @int)))
  (lambda ((ascribe# xs (@list @int)))
    (match xs
      ((nil (prod#)) xs)
      ((cons (prod# x (nil (prod#)))) xs)
      ((cons (prod# x (cons (prod# y ys))))
       (match (le x y)
         ((#t (prod#)) (cons (prod# x (pass (cons (prod# y ys))))))
         ((#f (prod#)) (cons (prod# y (pass (cons (prod# x ys)))))))))))
(define (ascribe# sort (@fun (@prod (@list @int)) (@list @int)))
  (lambda ((ascribe# xs (@list @int)))
    (letrec (((ascribe# ys (@list @int)) (pass xs)))
      (match (eq xs ys)
        ((#t (prod#)) xs)
        ((#f (prod#)) (sort ys))))))
(sort '(3 1 2))
"""


def test_the_report_gives_a_summary_per_procedure(capsys):
    # The walks are the procedure's, so they are said once each and not per
    # call. `pass` splits its argument three ways and `sort` two, which is the
    # whole of either answer — there is no count to cap and nothing truncated.
    assert main(["--analyse", "-e", BUBBLE]) == 0

    out: str = capsys.readouterr().out

    assert "procedure pass (xs)" in out
    assert "procedure sort (xs)" in out
    assert out.count("  class ") == 5

    for gone in ["truncated", "dropped", "more walks not shown", "walks:"]:
        assert gone not in out, gone


def test_a_comparison_within_a_class_is_weighed_exactly(capsys):
    # `pass`'s widest class binds two elements and compares them, so the two
    # orderings of those elements go one to each branch.
    assert main(["--analyse", "-e", BUBBLE]) == 0

    out: str = capsys.readouterr().out

    assert "  class 0: (cons (prod# _ (cons _)))" in out
    assert out.count("p 0.5000") == 2
    assert "p 0.5000: x0<=x1 -> (cons (prod# x0 <pass@0,1>))" in out
    assert "p 0.5000: x1<x0 -> (cons (prod# x1 <pass@0,1>))" in out


def test_a_recursive_call_is_printed_as_a_reference(capsys):
    # `<pass@0,1>` is the whole of how recursion appears: a reference to the
    # table above at the classes the argument can be. Nothing was followed to
    # print it.
    assert main(["--analyse", "-e", BUBBLE]) == 0

    out: str = capsys.readouterr().out

    assert "<pass@0,1>" in out
    assert "<sort@0,1>" in out


def test_a_reference_names_classes_the_table_is_headed_by(capsys):
    # The indices in `<pass@0,1>` are only meaningful if the lines they address
    # carry them, so the class heads are written with the index a reference
    # would name. `@0,1` says the recursive argument is a list of two-or-more
    # or of one, never the empty list that class 2 heads.
    assert main(["--analyse", "-e", BUBBLE]) == 0

    out: str = capsys.readouterr().out

    assert "  class 0: (cons (prod# _ (cons _)))" in out
    assert "  class 1: (cons (prod# _ (nil _)))" in out
    assert "  class 2: (nil _)" in out


def test_a_reference_to_a_two_parameter_procedure_names_one_class_each(capsys):
    # `append`'s heads are keyed by a class per parameter, written in the order
    # `<append@0,1 0>` writes them: that reference addresses the lines headed
    # `class 0 0` and `class 1 0`, which is readable only because the heads say
    # so.
    assert main(["--analyse", str(LIB / "quicksort.gisp")]) == 0

    out: str = capsys.readouterr().out

    assert "  class 0 0: (cons _) (cons _)" in out
    assert "  class 1 0: (nil _) (cons _)" in out
    assert "<append@0,1 0>" in out


def test_a_branch_decided_by_a_calls_result_prints_no_probability(capsys):
    # `sort` branches on `(eq xs ys)` where `ys` is what `pass` returned, and a
    # call is read by reference — so there is no probability to print, and `?`
    # is printed rather than a number that would be wrong.
    assert main(["--analyse", "-e", BUBBLE]) == 0

    out: str = capsys.readouterr().out

    assert "p ?" in out


def test_a_form_is_answered_as_the_class_its_argument_falls_in(capsys):
    assert main(["--analyse", "-e", BUBBLE]) == 0

    assert "applies sort with xs in (cons _)" in capsys.readouterr().out


def test_the_literals_length_is_not_read(capsys):
    # The statement that nothing unrolls, read off the report: a one-element
    # list and an eight-element one are both a `cons`, and the summary above
    # them is the same table either way.
    assert main(["--analyse", "-e", BUBBLE.replace("'(3 1 2)", "'(1)")]) == 0
    one: str = capsys.readouterr().out

    assert (
        main(["--analyse", "-e", BUBBLE.replace("'(3 1 2)", "'(3 1 4 1 5 9 2 6)")]) == 0
    )
    many: str = capsys.readouterr().out

    assert one.replace("(3 1 2)", "") == many.replace("(3 1 2)", "").replace(
        "(quote (3 1 4 1 5 9 2 6))", "(quote (1))"
    )


def test_the_library_is_summarised_without_a_bound(capsys):
    # Eight elements and no budget: the same finite table, because the answer
    # is about any non-empty list and not about eight of anything.
    assert main(["--analyse", str(LIB / "bubblesort.gisp")]) == 0

    out: str = capsys.readouterr().out

    assert "procedure pass (xs)" in out
    assert "applies sort with xs in (cons _)" in out

    for gone in ["truncated", "dropped", "more walks not shown"]:
        assert gone not in out, gone
