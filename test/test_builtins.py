import pytest

from Evaluate import BUILTINS, Environment, EvalError, Value, evaluate
from Reader import read
from Syntax import Boolean, Integer, SList, Symbol


def run(source: str) -> Value:
    return evaluate(read(source), Environment(BUILTINS))


def test_head_returns_the_first_element():
    assert run("(head '(a b))") == Symbol("a")


def test_tail_returns_the_rest_of_the_list():
    assert run("(tail '(a b))") == SList([Symbol("b")])


def test_tail_of_a_singleton_is_the_empty_list():
    assert run("(tail '(a))") == SList([])


def test_dot_conses_onto_the_head():
    assert run("(. 1 '(2 3))") == SList([Integer(1), Integer(2), Integer(3)])


def test_dot_onto_the_empty_list_makes_a_singleton():
    assert run("(. 1 '())") == SList([Integer(1)])


def test_head_undoes_dot():
    assert run("(head (. 1 '(2)))") == Integer(1)


def test_dot_leaves_its_argument_intact():
    source = "(letrec ((xs '(2))) ((lambda (ys) xs) (. 1 xs)))"
    assert run(source) == SList([Integer(2)])


def test_quote_returns_the_datum_verbatim():
    assert run("'x") == Symbol("x")
    assert run("'()") == SList([])
    assert run("''x") == SList([Symbol("quote"), Symbol("x")])


def test_a_lambda_binder_shadows_a_builtin():
    assert run("((lambda (head) head) 5)") == Integer(5)


def test_le_is_less_than_or_equal():
    assert run("(le 1 2)") == Boolean(True)
    assert run("(le 2 2)") == Boolean(True)
    assert run("(le 3 2)") == Boolean(False)


def test_gt_lt_ge_order_integers():
    assert run("(gt 2 1)") == Boolean(True)
    assert run("(gt 1 1)") == Boolean(False)
    assert run("(lt 1 2)") == Boolean(True)
    assert run("(lt 2 2)") == Boolean(False)


def test_arithmetic_operates_on_integers():
    assert run("(+ 2 3)") == Integer(5)
    assert run("(- 2 3)") == Integer(-1)
    assert run("(* 2 3)") == Integer(6)


def test_mod_takes_the_divisors_sign():
    assert run("(mod 7 2)") == Integer(1)
    assert run("(mod (- 0 7) 2)") == Integer(1)
    assert run("(mod 7 (- 0 2))") == Integer(-1)


def test_the_motivating_recursion_returns_the_singleton():
    source = (
        "(letrec ((go (lambda (xs)"
        " (letrec ((n (len xs)))"
        "  (if (gt n 0)"
        "   (letrec ((x (head xs)))"
        "    (if (eq (mod (* n x) 2) 0) (go (. x xs)) xs))"
        "   (go (. 1 xs)))))))"
        " (go '()))"
    )
    assert run(source) == SList([Integer(1)])


def test_eq_compares_integers():
    assert run("(eq 1 1)") == Boolean(True)
    assert run("(eq 1 2)") == Boolean(False)


def test_eq_compares_symbols_by_name():
    assert run("(eq 'a 'a)") == Boolean(True)
    assert run("(eq 'a 'b)") == Boolean(False)


def test_eq_compares_lists_structurally():
    assert run("(eq '(1 2) '(1 2))") == Boolean(True)
    assert run("(eq '() '())") == Boolean(True)


def test_eq_across_types_is_false():
    assert run("(eq 1 'a)") == Boolean(False)


def test_len_counts_top_level_items():
    assert run("(len '())") == Integer(0)
    assert run("(len '(a b c))") == Integer(3)
    assert run("(len '((a b) c))") == Integer(2)


def test_predicates_drive_if():
    assert run("(if (le (head '(1 2)) (len '(a b))) 'yes 'no)") == Symbol("yes")


def test_expand_scopes_quoted_syntax():
    # The builtin draws scopes from the global counter, so the expectation
    # is structural: the binder and its occurrence acquire the same fresh
    # scopes, and the keyword stays in the empty top-level region.
    result = run("(expand '(lambda (x) x))")

    match result:
        case SList([Symbol() as head, SList([Symbol() as binder]), Symbol() as body]):
            assert head.identifier == "lambda"
            assert not head.scopes
            assert binder.scopes
            assert binder == body
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_resolves_expanded_syntax():
    result = run("(parse (expand '((lambda (x) x) 5)))")

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
    result = run(
        "(parse (expand"
        " '(lambda (x)"
        "   (letrec-syntax ((m (syntax-rules () ((_ e) (lambda (x) e)))))"
        "    (m x)))))"
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
    result = run("(parse (expand '(lambda (x) 'x)))")

    match result:
        case SList([_, _, SList([Symbol() as head, Symbol() as datum])]):
            assert head == Symbol("quote")
            assert datum == Symbol("x")
        case _:
            pytest.fail(f"unexpected shape: {result}")


def test_parse_output_is_syntax_to_the_other_builtins():
    # The parsed tree is the same S-expression sort as any other datum, so
    # the list builtins consume it directly.
    assert run("(head (parse (expand '(f x))))") == Symbol("f")


@pytest.mark.parametrize(
    "source",
    [
        "(head '())",
        "(tail '())",
        "(head 5)",
        "(tail 5)",
        "(. 1 2)",
        "(. (lambda (x) x) '())",
        "(head '(a) '(b))",
        "(tail)",
        "(. 1)",
        "(le 1 'a)",
        '(le "x" 2)',
        "(gt 1 'a)",
        "(lt 'a 1)",
        "(+ 1 'a)",
        "(- 'a 1)",
        "(* 1 '())",
        "(mod 1 'a)",
        "(mod 7 0)",
        "(gt 1)",
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
