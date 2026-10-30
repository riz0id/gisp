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
from Analysis.Measure import probability
from Analysis.Path import Path
from Analysis.PatternCoverage import Coverage, CoverageError
from Analysis.Support import Support
from Analysis.Walk import (
    Branch,
    Procedure,
    Summary,
    applied,
    elements,
    procedures,
    summaries,
    weight,
)
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


def ordering(branch: Branch) -> str:
    # What a branch committed its class's ground binders to, as the
    # comparisons that said so.
    #
    # The ordering and not the guards. The ordering is the whole of what
    # decides which value of a class takes the branch, and is short enough to
    # read — a class binds as many elements as its clauses named, which is two
    # or three, not as many as an input has.
    #
    # One line per pair, and the strict reading where a pair was compared both
    # ways.
    strictest: dict[tuple[Symbol, Symbol], bool] = {}

    for edge in branch.constraints.order:
        pair: tuple[Symbol, Symbol] = (edge.lesser, edge.greater)
        strictest[pair] = strictest.get(pair, False) or edge.strict

    return ", ".join(
        f"{lesser}{'<' if strict else '<='}{greater}"
        for (lesser, greater), strict in strictest.items()
    )


def elided(expr: object, width: int = 56) -> str:
    # `expr` written out, cut to a width it can be read at. The cut is marked,
    # so a long answer can be told from a complete one.
    written: str = str(expr)

    return written if len(written) <= width else f"{written[: width - 1]}…"


def report_branch(branch: Branch, over: tuple[Symbol, ...]) -> None:
    share: float | None = probability(weight(branch, over), over)
    said: list[str] = ["p ?" if share is None else f"p {share:.4f}"]
    taken: str = ordering(branch)

    if taken:
        said.append(elided(taken))

    print(f"      {': '.join(said)} -> {elided(branch.result.to_value())}")


def report_summary(summary: Summary) -> None:
    # Every way one procedure's data can move, by the class its argument is in.
    #
    # Said of the procedure and not of a call, which is what makes this finite:
    # a recursive call appears in a result as `<name@classes>`, a reference to
    # this very table, rather than as a walk followed into. So there is no
    # count of walks to cap here and nothing to report as truncated.
    written: tuple[str, ...] = tuple(
        parameter.identifier for parameter in summary.parameters
    )

    print(f"procedure {summary.procedure} ({', '.join(written)})")

    for key, branches in summary.branches.items():
        named: str = " ".join(
            str(summary.classes[position][index]) for position, index in enumerate(key)
        )
        # The indices this line is addressed by, written as a reference writes
        # them: one per parameter, in order. `<append@0,1 1>` in a result stands
        # for the two lines headed `class 0 1` and `class 1 1`, and without the
        # index here there is nothing those digits could be read against.
        at: str = " ".join(str(index) for index in key)
        over: tuple[Symbol, ...] = elements(branches)

        print(f"  class {at}: {named}")

        for branch in branches:
            report_branch(branch, over)


def report_application(
    form: Expr,
    table: BindingTable,
    store: Context,
    known: dict[Symbol, Procedure],
    summarised: dict[Symbol, Summary | None],
) -> None:
    # Which of a procedure's classes a form's arguments fall in.
    #
    # All a report says about an expression, and all there is to say: the walks
    # are the procedure's and were given above, so a call adds only which of
    # them it reaches. `'(1)` and `'(3 1 4 1 5 9 2 6)` are both a `cons`, which
    # is why this says the same of either.
    found: tuple[Symbol, tuple[tuple[int, ...], ...]] | None = applied(
        form, table, store, known
    )

    if found is None:
        return

    binder, indices = found
    summary: Summary | None = summarised.get(binder)

    if summary is None:
        return

    for position, reached in enumerate(indices):
        parameter: Symbol = summary.parameters[position]
        named: str = " or ".join(
            str(summary.classes[position][index]) for index in reached
        )

        print(f"  applies {binder} with {parameter} in {named or 'no class'}")


def report_analysis(source: str, store: Context) -> None:
    forms: list[tuple[Expr, BindingTable]] = analysed(source, store)
    known: dict[Symbol, Procedure] = procedures(forms, store)
    summarised: dict[Symbol, Summary | None] = summaries(forms, store)

    # The procedures first and once each, because that is where the walks are.
    # A form that applies one adds only which of its classes it reaches.
    for binder in known:
        summary: Summary | None = summarised[binder]

        if summary is not None:
            report_summary(summary)
            print()

    for index, (program, table) in enumerate(forms):
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

        report_application(program, table, store, known, summarised)

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
