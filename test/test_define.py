import pytest

from Evaluate import BUILTINS, Environment, EvalError, Value, evaluate_toplevel
from Reader import read_all
from Syntax import Integer, Symbol, listed


def run(source: str) -> Value | None:
    environment = Environment(BUILTINS)
    result: Value | None = None
    for form in read_all(source):
        result = evaluate_toplevel(form, environment)
    return result


def test_a_defined_name_is_visible_to_later_forms():
    assert run("(define x 5) x") == Integer(5)


def test_the_init_is_evaluated():
    assert run("(define x (head '(1 2))) x") == Integer(1)


def test_a_defined_procedure_is_self_recursive():
    source = """
    (define last (lambda (xs)
                   (match xs
                     ((cons x (nil)) x)
                     ((cons x rest) (last rest)))))
    (last '(1 2 3))
    """
    assert run(source) == Integer(3)


def test_a_forward_reference_resolves_at_call_time():
    assert run("(define f (lambda () g)) (define g 5) (f)") == Integer(5)


def test_redefinition_overwrites():
    assert run("(define x 1) (define x 2) x") == Integer(2)


def test_a_define_form_yields_no_value():
    assert run("(define x 5)") is None


def test_a_quoted_define_is_data():
    assert run("'(define x 5)") == listed(
        [Symbol("define"), Symbol("x"), Integer(5)]
    )


@pytest.mark.parametrize(
    "source",
    [
        "((lambda () (define x 5)))",
        "(letrec ((y (define x 5))) y)",
    ],
)
def test_a_nested_define_is_refused(source: str):
    with pytest.raises(EvalError):
        run(source)


@pytest.mark.parametrize(
    "source",
    [
        "(define)",
        "(define x)",
        "(define x 1 2)",
        "(define 5 1)",
        "(define (f x) 1)",
    ],
)
def test_a_malformed_define_is_refused(source: str):
    with pytest.raises(EvalError):
        run(source)


# -- type ascription -------------------------------------------------------------


def test_an_ascribed_define_binds_its_name():
    assert run("(define (ascribe# x @int) 5) x") == Integer(5)


def test_an_ascribed_define_is_visible_to_later_forms():
    assert run("(define (ascribe# x @int) 5) (define y x) y") == Integer(5)


def test_an_ascribed_defined_procedure_is_self_recursive():
    # Ascription is erased at runtime, so an ascribed define keeps the
    # self-reference a bare one has.
    source = """
    (define (ascribe# fact @int)
      (lambda ((ascribe# n @int))
        (match n
          (0 1)
          (m (* m (fact (- m 1)))))))
    (fact 5)
    """
    assert run(source) == Integer(120)


@pytest.mark.parametrize(
    "source",
    ["(define (ascribe# x) 5)", "(define (ascribe# x @int @int) 5)"],
)
def test_a_malformed_ascribed_define_is_refused(source: str):
    with pytest.raises(EvalError):
        run(source)
