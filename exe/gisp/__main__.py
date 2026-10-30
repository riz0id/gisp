import argparse
import sys

from Analysis import (
    Guard,
    OccurrenceError,
    arms,
    coverages,
    opaque,
    path_conditions,
    refinements,
    unreachable,
)
from Analysis.CExpr import CExprError
from Analysis.Path import Path
from Analysis.PatternCoverage import Coverage, CoverageError
from Analysis.Support import Support
from Evaluate import BUILTINS, Environment, EvalError, evaluate_toplevel
from Expander import ExpandError, ParseError, expand, parse
from Reader import ReadError, read_all
from Resolve import Ambiguous, BindingTable, bindings
from Syntax import Expr, SList, Symbol, Value, binder_name
from Syntax.Type import Context
from TypeCheck import ElaborateError, TypeChecker, TypeCheckError, check_toplevel

# Every error the phases raise, so one handler reports whichever phase refused
# the program. `Ambiguous` is here because `resolve`'s callers catch `Unbound`
# alone: an identifier with two candidate binders is refused by `resolve`
# itself, and reaches this far from both the parse phase and the type checker.
#
# The last three are the analysis phase's, which only `--analyse` runs: a report
# is worth no more than an evaluation is when a phase refuses the program.
# `CoverageError` cannot be raised from here, because the type checker refuses
# an incomplete `match` first — it is the analysis stating its own precondition,
# and it is listed so that a caller who reaches the analysis another way is
# reported about in the same words.
INTERPRETER_ERRORS = (
    ReadError,
    ExpandError,
    ParseError,
    Ambiguous,
    TypeCheckError,
    ElaborateError,
    EvalError,
    OccurrenceError,
    CExprError,
    CoverageError,
)


def checked(form: Expr, store: Context) -> Expr:
    # One top-level form taken through the phases that precede evaluation, and
    # returned as the parsed program evaluation runs.
    #
    # The table is rebuilt per form rather than accumulated across them. A
    # `define` mints no scope, so its binder and a later form's reference to it
    # are the same symbol, and it is `store` — carried by the caller — that
    # makes the later form see the ascription.
    expanded: Expr = expand(form)
    program: Expr = parse(expanded, bindings(expanded))

    check_toplevel(program, TypeChecker(table=bindings(program), store=store))

    return program


def evaluate_source(
    source: str, environment: Environment, store: Context
) -> Value | None:
    # The whole source is checked before any of it is evaluated, so a type error
    # in a later form means no earlier one runs.
    programs: list[Expr] = [checked(form, store) for form in read_all(source)]
    result: Value | None = None

    for program in programs:
        result = evaluate_toplevel(program, environment)

    return result


def analysed(source: str, store: Context) -> list[tuple[Expr, BindingTable]]:
    # The whole source taken through the phases that precede evaluation, with
    # each form's table kept beside it. `checked` lets its table go because
    # evaluation is scope-blind; the analysis is not — it resolves every
    # reference it reads against the table the form was parsed with.
    forms: list[tuple[Expr, BindingTable]] = []

    for form in read_all(source):
        expanded: Expr = expand(form)
        program: Expr = parse(expanded, bindings(expanded))
        table: BindingTable = bindings(program)

        check_toplevel(program, TypeChecker(table=table, store=store))
        forms.append((program, table))

    return forms


def described(form: Expr) -> str:
    # The line naming the form a report is about: a declaration by the name it
    # declares, and anything else by the way it is written. A `define`'s
    # initializer is the body of a procedure and too long to read back.
    match form:
        case SList([Symbol() as head, name, _]) if (head.identifier == "define") and (
            bound := binder_name(name)
        ) is not None:
            return f"(define {bound} ...)"
        case _:
            return str(form)


def report_arm(
    path: Path,
    program: Expr,
    table: BindingTable,
    store: Context,
    free: list[Symbol],
) -> None:
    # One clause of one `match`: whether it is reached, the guards that decide
    # that, and what each free variable is narrowed to along the way.
    guards: list[Guard] = path_conditions(path, program, table)
    dead: bool = unreachable(path, program, table, store)

    print(f"  {path}: {'unreachable' if dead else 'reachable'}")

    for guard in guards:
        # The scrutinee as written, which is what `to_expr` reconstructs, and
        # the pattern that reaching this clause commits it to — or commits it
        # to having failed, for every clause tried before this one.
        taken: str = "matches" if guard.consequent else "fails"
        print(f"      {guard.test.to_expr()} {taken} {guard.pattern}")

    for variable in free:
        support: Support = refinements(variable, path, program, table, store)[-1]
        narrowed: str = "no value" if support.is_empty() else str(support)
        print(f"      {variable} takes {narrowed}")


def report_coverage(path: Path, verdict: Coverage | None) -> None:
    # One `match`: whether its clauses leave a value unmatched, and the pattern
    # that one would be written as.
    #
    # Said of the form and not of a clause, which is what distinguishes it from
    # the arm lines below: those answer whether a clause is reached, and this
    # answers whether the clauses between them cover the type.
    #
    # The type is not named here. It would be the whole of a `(@list @int)`
    # written out, which is a line long, and the witness names the constructor
    # it is about already.
    if verdict is None:
        # Coverage declined the question: the scrutinee's type cannot be read,
        # or a clause's pattern is not one for it.
        print(f"  match at {path}: coverage unknown")
    elif verdict.complete():
        print(f"  match at {path}: complete")
    else:
        print(f"  match at {path}: leaves {verdict.witness} unmatched")


def report_analysis(source: str, store: Context) -> None:
    for index, (program, table) in enumerate(analysed(source, store)):
        free: list[Symbol] = opaque(program, table)
        branches: list[Path] = arms(program)

        print(f"form {index}: {described(program)}")

        if free:
            print(
                "  free variables: "
                + ", ".join(f"{variable} : {store[variable]}" for variable in free)
            )

        for path, verdict in coverages(program, table, store):
            report_coverage(path, verdict)

        if not branches:
            # Said rather than left blank: a form with no `match` in it is one
            # the analysis has nothing to answer about, which is not the same
            # as a form whose every clause is reachable.
            print("  no clauses")

        for path in branches:
            report_arm(path, program, table, store, free)

        print()


def report(error: object) -> None:
    print(f"gisp: {error}", file=sys.stderr)


def run_source(source: str, environment: Environment, store: Context) -> int:
    try:
        result = evaluate_source(source, environment, store)
    except INTERPRETER_ERRORS as error:
        report(error)
        return 1
    if result is not None:
        print(str(result))
    return 0


def run_report(source: str, store: Context) -> int:
    # The analysis phase in place of evaluation, and not after it: a report is
    # about the branches a program has, and running the program would take one
    # of them.
    try:
        report_analysis(source, store)
    except INTERPRETER_ERRORS as error:
        report(error)
        return 1
    return 0


def run_repl(environment: Environment, store: Context) -> int:
    # A line at a time, because a line does not exist until the one before it
    # has run — so this is the one driver that cannot check a whole program
    # before evaluating any of it. The store crosses lines for the reason the
    # environment does: a line refers to what an earlier line declared.
    interactive = sys.stdin.isatty()
    while True:
        if interactive:
            sys.stdout.write("gisp> ")
            sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            if interactive:
                sys.stdout.write("\n")
            return 0
        if not line.strip():
            continue
        try:
            result = evaluate_source(line, environment, store)
        except INTERPRETER_ERRORS as error:
            report(error)
            continue
        if result is not None:
            print(str(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gisp")
    parser.add_argument("file", nargs="?")
    parser.add_argument("-e", "--eval", dest="expression")
    parser.add_argument("--analyse", action="store_true")
    arguments = parser.parse_args(argv)
    environment = Environment(BUILTINS)
    store = Context.default()
    if arguments.expression is not None:
        if arguments.analyse:
            return run_report(arguments.expression, store)
        return run_source(arguments.expression, environment, store)
    if arguments.file is not None:
        try:
            with open(arguments.file, encoding="utf-8") as handle:
                source = handle.read()
        except OSError as error:
            report(error)
            return 1
        if arguments.analyse:
            return run_report(source, store)
        return run_source(source, environment, store)
    if arguments.analyse:
        # The repl has no program to report on: a line is checked and run
        # before the next one exists, and the branches of one line are the
        # whole of what a report could cover.
        report("--analyse needs a file or an expression")
        return 1
    return run_repl(environment, store)


if __name__ == "__main__":
    sys.exit(main())
