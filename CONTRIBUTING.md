# Contributing

Contributions that improve retrieval quality, provenance, safety, portability,
or documentation are welcome.

## Development setup

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[all]'
python -m unittest discover tests -v
```

Before opening a pull request, also run:

```bash
python scripts/eval_retrieval.py --ranker lexical
python scripts/eval_retrieval_pipeline.py --ranker lexical
python scripts/eval_source_quality.py
```

Network-dependent results are not deterministic. Include the date, dataset,
command, and raw metrics when a change relies on `eval_live_retrieval.py`.
Use its `--output` option so failures and timing remain inspectable.

Keep pull requests focused. Add or update tests for behavior changes, preserve
SourceBundle compatibility, and document any new network destination or data
disclosure in `PRIVACY.md`.

Use GitHub issues for bugs and feature proposals. Report vulnerabilities through
the private process in `SECURITY.md`.
