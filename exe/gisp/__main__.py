import argparse
import sys

from Evaluate import BUILTINS, Environment, EvalError, evaluate_toplevel
from Reader import ReadError, read_all
from Syntax import Value

INTERPRETER_ERRORS = (ReadError, EvalError)


def evaluate_source(source: str, environment: Environment) -> Value | None:
    result: Value | None = None
    for form in read_all(source):
        result = evaluate_toplevel(form, environment)
    return result


def report(error: object) -> None:
    print(f"gisp: {error}", file=sys.stderr)


def run_source(source: str, environment: Environment) -> int:
    try:
        result = evaluate_source(source, environment)
    except INTERPRETER_ERRORS as error:
        report(error)
        return 1
    if result is not None:
        print(str(result))
    return 0


def run_repl(environment: Environment) -> int:
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
            result = evaluate_source(line, environment)
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
    if arguments.expression is not None:
        return run_source(arguments.expression, environment)
    if arguments.file is not None:
        try:
            with open(arguments.file, encoding="utf-8") as handle:
                source = handle.read()
        except OSError as error:
            report(error)
            return 1
        return run_source(source, environment)
    return run_repl(environment)


if __name__ == "__main__":
    sys.exit(main())
