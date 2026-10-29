from typing import Final

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import UnexpectedInput, VisitError

from Reader.Error import ReadError
from Syntax import Char, Expr, Integer, Keyword, SList, Symbol

__all__ = [
    "ReadError",
    "read",
    "read_all",
]

GRAMMAR: Final[str] = r"""
start: form*

?form: slist | quote | string | char | keyword | atom

slist: "(" form* ")"
quote: "'" form
string: STRING
char: CHAR
keyword: KEYWORD
atom: ATOM

STRING: /"(\\[nrt"\\]|[^"\\])*"/
CHAR.2: /#\\(x[0-9a-fA-F]+|[^\s])/
KEYWORD.2: /#:[^\s()';"]*/
ATOM: /[^\s()';"]+/

%ignore /\s+/
%ignore /;[^\n]*/
"""

PARSER: Final = Lark(GRAMMAR, parser="lalr")

ESCAPED_CHARACTERS: dict[str, str] = {
    "\\": "\\",
    '"': '"',
    "n": "\n",
    "t": "\t",
    "r": "\r",
}


def unescape(text: str) -> str:
    body: str = text[1:-1]
    characters: list[str] = []
    index: int = 0

    while index < len(body):
        if body[index] == "\\":
            characters.append(ESCAPED_CHARACTERS[body[index + 1]])
            index += 2
        else:
            characters.append(body[index])
            index += 1

    return "".join(characters)


class ToExpr(Transformer[Token, list[Expr]]):
    def start(self, children: list[Expr]) -> list[Expr]:
        return children

    def slist(self, children: list[Expr]) -> Expr:
        return SList(children)

    def quote(self, children: list[Expr]) -> Expr:
        return SList([Symbol("quote"), children[0]])

    def string(self, children: list[Token]) -> Expr:
        return SList([Char(ord(ch)) for ch in unescape(str(children[0]))])

    def char(self, children: list[Token]) -> Expr:
        # `#\x` with no digits is the character x: the hex branch of the
        # terminal requires at least one digit, so the lexer agrees.
        body: str = str(children[0])[2:]

        if body[0] == "x" and len(body) > 1:
            return Char(int(body[1:], 16))
        else:
            return Char(ord(body))

    def keyword(self, children: list[Token]) -> Expr:
        # The terminal takes the marker with an empty name so that `#:` is
        # claimed here and refused, rather than falling through to `ATOM` and
        # reading as the symbol `#:`. `Keyword.__post_init__` cannot carry
        # this: its `assert` is stripped under `-O`, and a malformed source
        # deserves a `ReadError` rather than an `AssertionError`.
        name: str = str(children[0])[2:]

        if not name:
            raise ReadError("expected a name after `#:`")

        return Keyword(name)

    def atom(self, children: list[Token]) -> Expr:
        text = str(children[0])
        digits = text[1:] if text[0] in "+-" else text
        if digits.isascii() and digits.isdigit():
            return Integer(int(text))
        return Symbol(text)


TRANSFORMER: Final = ToExpr()


def read_all(source: str) -> list[Expr]:
    try:
        tree: Tree[Token] = PARSER.parse(source)
    except UnexpectedInput as error:
        raise ReadError(f"syntax error at offset {error.pos_in_stream}") from error

    try:
        return TRANSFORMER.transform(tree)
    except VisitError as error:
        # Lark wraps every exception a callback raises. A `ReadError` from a
        # callback is the reader's own diagnosis and must reach the caller as
        # itself; anything else is a bug and propagates wrapped.
        if isinstance(error.orig_exc, ReadError):
            raise error.orig_exc from error
        raise


def read(source: str) -> Expr:
    match read_all(source):
        case []:
            raise ReadError("expected a form but found none")
        case [form]:
            return form
        case forms:
            raise ReadError(f"unexpected trailing form ({len(forms)} forms in source)")
