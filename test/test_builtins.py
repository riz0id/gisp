import pytest

from Evaluate import BUILTINS, Environment, EvalError, Value, evaluate
from Reader import read
from Syntax import (
    Integer,
    SList,
    Symbol,
    as_syntax,
    boolean,
    listed,
    product,
    unit,
)


def run(source: str) -> Value:
    return evaluate(read(source), Environment(BUILTINS))


def test_head_returns_the_first_element():
    assert run("(head '(a b))") == Symbol("a")


def test_tail_returns_the_rest_of_the_list():
    assert run("(tail '(a b))") == listed([Symbol("b")])


def test_tail_of_a_singleton_is_the_empty_list():
    assert run("(tail '(a))") == listed([])


def test_a_product_is_one_sort_at_every_width():
    # The nullary product is the empty tuple and not a sort of its own, which
    # is what `(#t (prod#))` carries and a `(prod#)` pattern matches.
    assert run("(prod#)") == unit()
    assert run("(prod#)") == product([])
    assert run("(prod# 1 2)") == product([Integer(1), Integer(2)])
    assert run("(prod# 1)") == product([Integer(1)])


def test_nil_is_the_empty_list():
    assert run("(nil (prod#))") == listed([])


def test_cons_conses_onto_the_head():
    assert run("(cons (prod# 1 '(2 3)))") == listed(
        [Integer(1), Integer(2), Integer(3)]
    )


def test_cons_onto_the_empty_list_makes_a_singleton():
    assert run("(cons (prod# 1 (nil (prod#))))") == listed([Integer(1)])


def test_head_undoes_cons():
    assert run("(head (cons (prod# 1 '(2))))") == Integer(1)


def test_cons_leaves_its_argument_intact():
    source = "(letrec ((xs '(2))) ((lambda (ys) xs) (cons (prod# 1 xs))))"
    assert run(source) == listed([Integer(2)])


def test_quote_reads_a_datum_as_the_value_it_denotes():
    # An atom is already the value it denotes; a list is read as the `cons`
    # nest that is one, because no value is an `SList`.
    assert run("'x") == Symbol("x")
    assert run("'()") == listed([])
    assert run("''x") == listed([Symbol("quote"), Symbol("x")])


def test_a_lambda_binder_shadows_a_builtin():
    assert run("((lambda (head) head) 5)") == Integer(5)


def test_le_is_less_than_or_equal():
    assert run("(le 1 2)") == boolean(True)
    assert run("(le 2 2)") == boolean(True)
    assert run("(le 3 2)") == boolean(False)


def test_gt_lt_ge_order_integers():
    assert run("(lt 1 2)") == boolean(True)
    assert run("(lt 2 2)") == boolean(False)


def test_arithmetic_operates_on_integers():
    assert run("(+ 2 3)") == Integer(5)
    assert run("(- 2 3)") == Integer(-1)
    assert run("(* 2 3)") == Integer(6)


def test_mod_takes_the_divisors_sign():
    assert run("(mod 7 2)") == Integer(1)
    assert run("(mod (- 0 7) 2)") == Integer(1)
    assert run("(mod 7 (- 0 2))") == Integer(-1)


def test_eq_compares_integers():
    assert run("(eq 1 1)") == boolean(True)
    assert run("(eq 1 2)") == boolean(False)


def test_eq_compares_symbols_by_name():
    assert run("(eq 'a 'a)") == boolean(True)
    assert run("(eq 'a 'b)") == boolean(False)


def test_eq_compares_lists_structurally():
    assert run("(eq '(1 2) '(1 2))") == boolean(True)
    assert run("(eq '() '())") == boolean(True)


def test_eq_across_types_is_false():
    assert run("(eq 1 'a)") == boolean(False)


def test_len_counts_top_level_items():
    assert run("(len '())") == Integer(0)
    assert run("(len '(a b c))") == Integer(3)
    assert run("(len '((a b) c))") == Integer(2)


def test_predicates_drive_a_match():
    source = "(match (le (head '(1 2)) (len '(a b))) ((#t) 'yes) ((#f) 'no))"

    assert run(source) == Symbol("yes")


def test_expand_scopes_quoted_syntax():
    # The builtin draws scopes from the global counter, so the expectation
    # is structural: the binder and its occurrence acquire the same fresh
    # scopes, and the keyword stays in the empty top-level region.
    #
    # Read back through `as_syntax`, because the builtin answers with the list
    # data denoting the program and `expand` is written over syntax.
    result = as_syntax(run("(expand '(lambda (x) x))"))

    match result:
        case SList([Symbol() as head, SList([Symbol() as binder]), Symbol() as body]):
            assert head.identifier == "lambda"
            assert not head.scopes
            assert binder.scopes
            assert binder == body
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_resolves_expanded_syntax():
    result = as_syntax(run("(parse (expand '((lambda (x) x) 5)))"))

    match result:
        case SList(
            [
                SList(
                    [
                        Symbol() as head,
                        SList([Symbol() as parameter]),
                        Symbol() as body,
                    ]
                ),
                Integer(5),
            ]
        ):
            assert head == Symbol("lambda")
            assert body == parameter
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_expand_and_parse_compose_over_macros():
    # Hygiene reaches the language level: the macro-introduced binder is
    # spelled like the user's, and the substituted reference still comes out
    # as the user's binder symbol in the parsed result.
    result = as_syntax(
        run(
            "(parse (expand"
            " '(lambda (x)"
            "   (letrec-syntax ((m (syntax-rules () ((_ e) (lambda (x) e)))))"
            "    (m x)))))"
        )
    )

    match result:
        case SList(
            [
                _,
                SList([Symbol() as outer]),
                SList([_, SList([Symbol() as inner]), Symbol() as body]),
            ]
        ):
            assert body == outer
            assert body != inner
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_strips_quoted_data():
    result = as_syntax(run("(parse (expand '(lambda (x) 'x)))"))

    match result:
        case SList([_, _, SList([Symbol() as head, Symbol() as datum])]):
            assert head == Symbol("quote")
            assert datum == Symbol("x")
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_output_is_syntax_to_the_other_builtins():
    # The parsed tree is list data like any other, so the list builtins consume
    # it directly — which is what the conversion at the builtin's boundary is
    # for.
    assert run("(head (parse (expand '(f x))))") == Symbol("f")


def test_parse_reads_an_ascribed_programs_type():
    # An ascription survives into the parsed tree with its type read as a type:
    # resolved where it binds, and emitted bare where it names a constructor.
    result = as_syntax(run("(parse (expand '(lambda ((ascribe# x @int)) x)))"))

    match result:
        case SList([_, SList([SList([keyword, _, ty])]), _]):
            assert keyword == Symbol("ascribe#")
            assert ty == Symbol("@int")
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_refuses_an_ascribed_program_whose_type_is_malformed():
    # The failure mode the type operand gains by being read at all: this once
    # parsed, and was refused only when the type checker came to read the type.
    with pytest.raises(EvalError):
        run("(parse (expand '(lambda ((ascribe# x (@list @int @int))) x)))")


@pytest.mark.parametrize(
    "source",
    [
        "(head '())",
        "(tail '())",
        "(head 5)",
        "(tail 5)",
        "(cons (prod# 1 2))",
        "(cons (prod# (lambda (x) x) '()))",
        "(cons 1)",
        "(cons (prod# 1))",
        "(nil 1)",
        "(nil (prod# 1 2))",
        "(prod# (lambda (x) x))",
        "(head '(a) '(b))",
        "(tail)",
        "(le 1 'a)",
        '(le "x" 2)',
        "(lt 'a 1)",
        "(+ 1 'a)",
        "(- 'a 1)",
        "(* 1 '())",
        "(mod 1 'a)",
        "(mod 7 0)",
        "(+ 1 2 3)",
        "(eq head 1)",
        "(eq 1 (lambda (x) x))",
        "(len 5)",
        '(len "abc")',
        "(le 1)",
        "(eq 1)",
        "(len '(a) '(b))",
        "(expand head)",
        "(expand '(lambda (x)))",
        "(parse '(letrec-syntax ((m (syntax-rules () ((_) x)))) y))",
        "(parse '())",
        "(parse head)",
        "(expand '(a) '(b))",
        "(expand '((lambda () (define x 5))))",
    ],
)
def test_malformed_builtin_applications_are_refused(source: str):
    with pytest.raises(EvalError):
        run(source)
