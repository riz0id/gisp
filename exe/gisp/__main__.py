import argparse
import sys

from Evaluate import BUILTINS, Environment, EvalError, evaluate_toplevel
from Expander import ExpandError, ParseError, expand, parse
from Reader import ReadError, read_all
from Resolve import Ambiguous, bindings
from Syntax import Expr, Value
from Syntax.Type import Context
from TypeCheck import ElaborateError, TypeChecker, TypeCheckError, check_toplevel

# Every error the phases raise, so one handler reports whichever phase refused
# the program. `Ambiguous` is here because `resolve`'s callers catch `Unbound`
# alone: an identifier with two candidate binders is refused by `resolve`
# itself, and reaches this far from both the parse phase and the type checker.
INTERPRETER_ERRORS = (
    ReadError,
    ExpandError,
    ParseError,
    Ambiguous,
    TypeCheckError,
    ElaborateError,
    EvalError,
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


def evaluate_source(source: str, environment: Environment, store: Context) -> Value | None:
    # The whole source is checked before any of it is evaluated, so a type error
    # in a later form means no earlier one runs.
    programs: list[Expr] = [checked(form, store) for form in read_all(source)]
    result: Value | None = None

    for program in programs:
        result = evaluate_toplevel(program, environment)

    return result


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
    arguments = parser.parse_args(argv)
    environment = Environment(BUILTINS)
    store = Context.default()
    if arguments.expression is not None:
        return run_source(arguments.expression, environment, store)
    if arguments.file is not None:
        try:
            with open(arguments.file, encoding="utf-8") as handle:
                source = handle.read()
        except OSError as error:
            report(error)
            return 1
        return run_source(source, environment, store)
    return run_repl(environment, store)


if __name__ == "__main__":
    sys.exit(main())
