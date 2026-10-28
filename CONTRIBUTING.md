# Contributing

Contributions that make the checks more accurate, the page reader more robust,
or the skill clearer are welcome. Sibyl stays small: it checks claims against
the pages they quote and does not search, rank or call models.

English is the language of maintained documentation, code comments, messages,
metadata and maintainer-authored issues, PRs and releases. README.zh-CN.md is the
sole Chinese documentation file and must track README.md's scope, installation
and limitations. Original-language evidence, names and multilingual fixtures
keep their language.

## Development setup

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[pdf]' ruff==0.17.1 mypy==2.4.0
ruff check sibyl tests scripts
ruff format --check sibyl tests scripts
mypy
python scripts/check_docs.py
python -m unittest discover tests -v
```

Tests run offline with fake pages. A change to quote matching or a check must
come with a test showing both the case it catches and a case it must not flag,
and docs/ledger-format.md must describe it. Distribution changes also need
`python scripts/check_wheel.py dist/*.whl <version>` after
`python -m pip wheel . --no-deps --wheel-dir dist`.

Do not describe a passed check as verification of truth or entailment, and do
not advertise a check before it exists.
