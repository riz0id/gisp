import string

from hypothesis import strategies as st

from Syntax import Char, Integer, Keyword, SList, Symbol

SYMBOL_PUNCTUATION = "+-*/<>=!?_"
SYMBOL_FIRST_CHARACTERS = string.ascii_letters + SYMBOL_PUNCTUATION
SYMBOL_REST_CHARACTERS = string.ascii_letters + string.digits + SYMBOL_PUNCTUATION
NON_SYMBOL_NAMES = frozenset({"#t", "#f"})


def is_symbol_name(name: str) -> bool:
    if name in NON_SYMBOL_NAMES:
        return False
    try:
        int(name)
    except ValueError:
        return True
    return False


symbol_names = st.builds(
    lambda first, rest: first + rest,
    st.sampled_from(SYMBOL_FIRST_CHARACTERS),
    st.text(alphabet=SYMBOL_REST_CHARACTERS, max_size=7),
).filter(is_symbol_name)

symbols = symbol_names.map(Symbol)

# A keyword name has no reserved spellings and is not an identifier, so no
# filter stands here: `#:1` is a keyword like any other. The alphabet is a
# subset of the `KEYWORD` terminal's body class, so every name round-trips.
keyword_names = st.text(alphabet=SYMBOL_REST_CHARACTERS, min_size=1, max_size=8)

keywords = keyword_names.map(Keyword)

characters = st.characters(exclude_categories=("Cs",)).map(lambda ch: Char(ord(ch)))

strings = st.text(st.characters(exclude_categories=("Cs",))).map(
    lambda text: SList([Char(ord(ch)) for ch in text])
)

atoms = st.one_of(
    symbols,
    st.integers().map(Integer),
    characters,
    strings,
    keywords,
)

expressions = st.recursive(
    atoms,
    lambda children: st.lists(children, max_size=4).map(SList),
    max_leaves=12,
)
