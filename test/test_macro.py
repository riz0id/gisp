from collections.abc import Callable
from itertools import count

import pytest

from Expander import ExpandError, expand
from Reader import read
from Resolve import bindings, resolve
from Syntax import Scope, SList, Symbol

# Fixtures inject deterministic counters from the 1100s band, unclaimed by
# every other test file, so no expectation depends on the global fresh-uid
# counter. Each invocation mints its use-site scope before its introduction
# scope; the trace comments spell both out so a deviation in minting order
# or flip semantics fails with exact values.


def counter(start: int) -> Callable[[], Scope]:
    uids = count(start)
    return lambda: Scope(next(uids))


def test_macro_introduced_references_resolve_at_the_definition_site():
    # The template's x carries the definition context plus the introduction
    # scope but never the inner lambda's scope, so it resolves to the outer
    # binder despite sitting textually under the shadow.
    #
    # 1110 outer lambda; 1111 letrec-syntax; 1112 inner lambda;
    # 1113 use-site; 1114 introduction.
    expanded = expand(
        read(
            "(lambda (x)"
            " (letrec-syntax ((m (syntax-rules () ((_) x))))"
            "  ((lambda (x) (m)) y)))"
        ),
        fresh=counter(1110),
    )

    outer = Symbol("x", {Scope(1110)})
    inner = Symbol("x", {Scope(1110), Scope(1111), Scope(1112)})
    introduced = Symbol("x", {Scope(1110), Scope(1111), Scope(1114)})
    assert expanded == SList(
        [
            Symbol("lambda"),
            SList([outer]),
            SList(
                [
                    SList(
                        [
                            Symbol("lambda", {Scope(1110), Scope(1111)}),
                            SList([inner]),
                            introduced,
                        ]
                    ),
                    Symbol("y", {Scope(1110), Scope(1111)}),
                ]
            ),
        ]
    )

    assert resolve(introduced, bindings(expanded)).binder == outer


def test_a_use_site_identifier_as_binder_does_not_capture_the_template():
    # The identity/misc-id shape: the macro puts a use-site identifier into
    # binding position around a template reference. The matched x keeps the
    # use-site scope and loses the introduction scope, so the manufactured
    # binder can never be a candidate for the template's x — which resolves
    # to the template lambda, unambiguously.
    #
    # 1100 letrec-syntax; 1101 use-site; 1102 introduction; 1103 template
    # lambda; 1104 manufactured lambda.
    expanded = expand(
        read(
            "(letrec-syntax ((identity (syntax-rules ()"
            "                  ((_ misc-id)"
            "                   (lambda (x) ((lambda (misc-id) x) other))))))"
            " (identity x))"
        ),
        fresh=counter(1100),
    )

    template_binder = Symbol("x", {Scope(1100), Scope(1102), Scope(1103)})
    manufactured = Symbol("x", {Scope(1100), Scope(1101), Scope(1103), Scope(1104)})
    occurrence = Symbol("x", {Scope(1100), Scope(1102), Scope(1103), Scope(1104)})
    assert expanded == SList(
        [
            Symbol("lambda", {Scope(1100), Scope(1102)}),
            SList([template_binder]),
            SList(
                [
                    SList(
                        [
                            Symbol("lambda", {Scope(1100), Scope(1102), Scope(1103)}),
                            SList([manufactured]),
                            occurrence,
                        ]
                    ),
                    Symbol("other", {Scope(1100), Scope(1102), Scope(1103)}),
                ]
            ),
        ]
    )

    table = bindings(expanded)
    assert resolve(occurrence, table).binder == template_binder
    assert resolve(occurrence, table).binder != manufactured


def test_a_template_binder_does_not_capture_a_use_site_reference():
    # The symmetric capture: the template binds its own x around the matched
    # use-site x. The use-site x lacks the introduction scope the template
    # binder carries, so it still resolves to the enclosing lambda.
    #
    # 1120 outer lambda; 1121 letrec-syntax; 1122 use-site;
    # 1123 introduction; 1124 template lambda.
    expanded = expand(
        read(
            "(lambda (x)"
            " (letrec-syntax ((m (syntax-rules () ((_ misc-id) (lambda (misc-id) x)))))"
            "  (m x)))"
        ),
        fresh=counter(1120),
    )

    outer = Symbol("x", {Scope(1120)})
    occurrence = Symbol("x", {Scope(1120), Scope(1121), Scope(1123), Scope(1124)})
    assert resolve(occurrence, bindings(expanded)).binder == outer


def test_nested_uses_expand_inside_out_through_reexpansion():
    # The outer invocation matches the whole inner use unexpanded and splices
    # it into the template; re-expansion then fires the inner use, so each
    # invocation mints its own use/introduction pair.
    #
    # 1130 lambda; 1131 letrec-syntax; 1132/1133 outer use; 1134/1135 inner.
    expanded = expand(
        read(
            "(lambda (x) (letrec-syntax ((m (syntax-rules () ((_ v) v)))) (m (m x))))"
        ),
        fresh=counter(1130),
    )

    binder = Symbol("x", {Scope(1130)})
    occurrence = Symbol("x", {Scope(1130), Scope(1131), Scope(1132), Scope(1134)})
    assert expanded == SList([Symbol("lambda"), SList([binder]), occurrence])
    assert resolve(occurrence, bindings(expanded)).binder == binder


def test_sibling_macros_resolve_through_the_shared_form_scope():
    # letrec-style scoping: both rules forms carry the form's one scope, so
    # m's template reference to n resolves at transcription time.
    #
    # 1140 letrec-syntax; 1141/1142 m's use; 1143/1144 n's use.
    expanded = expand(
        read(
            "(letrec-syntax ((m (syntax-rules () ((_ v) (n v))))"
            "                (n (syntax-rules () ((_ w) w))))"
            " (m y))"
        ),
        fresh=counter(1140),
    )

    assert expanded == Symbol("y", {Scope(1140), Scope(1141), Scope(1143)})


def test_a_lambda_binder_shadows_a_macro_name():
    # The head resolves to the lambda's binding, which carries no
    # transformer, so the form is a plain application and the macro never
    # fires.
    expanded = expand(
        read("(letrec-syntax ((m (syntax-rules () ((_) x)))) (lambda (m) (m y)))"),
        fresh=counter(1150),
    )

    binder = Symbol("m", {Scope(1150), Scope(1151)})
    assert expanded == SList(
        [
            Symbol("lambda", {Scope(1150)}),
            SList([binder]),
            SList([binder, Symbol("y", {Scope(1150), Scope(1151)})]),
        ]
    )


def test_a_macro_name_outside_operator_position_is_refused():
    with pytest.raises(ExpandError):
        expand(read("(letrec-syntax ((m (syntax-rules () ((_) x)))) (f m))"))


def test_quoted_template_data_is_immune_to_substitution():
    # Quoted forms never acquire scopes, so a quoted spelling can never equal a
    # pattern variable: substitution cannot reach into data.
    #
    # 1160 letrec-syntax; 1161 use-site; 1162 introduction.
    expanded = expand(
        read("(letrec-syntax ((m (syntax-rules () ((_ v) (f v 'v))))) (m y))"),
        fresh=counter(1160),
    )

    assert expanded == SList(
        [
            Symbol("f", {Scope(1160), Scope(1162)}),
            Symbol("y", {Scope(1160), Scope(1161)}),
            SList([Symbol("quote"), Symbol("v")]),
        ]
    )


@pytest.mark.parametrize(
    "source",
    [
        "(letrec-syntax ((m (syntax-rules (lit) ((_) x)))) y)",
        "(letrec-syntax ((m (syntax-rules ()))) y)",
        "(letrec-syntax ((m (syntax-rules () ((_ a ...) x)))) y)",
        "(letrec-syntax ((m (syntax-rules () ((_ a) (a ...))))) y)",
        "(letrec-syntax ((m (syntax-rules () ((_ a a) x)))) y)",
        "(letrec-syntax ((m (syntax-rules () (p x)))) y)",
        "(letrec-syntax ((m (lambda (x) x))) y)",
        "(letrec-syntax ((m (syntax-rules () ((_ a) a)))) (m))",
        "(letrec-syntax ((m (syntax-rules () ((_) x)))"
        "                (m (syntax-rules () ((_) y)))) z)",
        "(letrec-syntax ((lambda (syntax-rules () ((_) x)))) y)",
        "(letrec-syntax () y)",
        "(letrec-syntax ((m (syntax-rules () ((_) x)))))",
        "(lambda (x) (syntax-rules () ((_) x)))",
    ],
)
def test_malformed_macro_forms_are_refused(source: str):
    with pytest.raises(ExpandError):
        expand(read(source))
