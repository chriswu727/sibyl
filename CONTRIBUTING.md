# Contributing

Contributions that improve retrieval quality, provenance, safety, portability,
or documentation are welcome.

## Development setup

```bash
git clone https://github.com/chriswu727/sibyl.git
cd sibyl
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[report,finance,extract,pdf,monitor]' ruff==0.17.1
ruff check sibyl tests scripts
python -m unittest discover tests -v
```

Before opening a pull request, also run:

```bash
python scripts/eval_retrieval.py --ranker lexical
python scripts/eval_retrieval_pipeline.py --ranker lexical
python scripts/eval_source_quality.py
python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)"
python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)" --monitor
python scripts/check_docs.py
```

The default development checks do not need a paid model or a downloaded ranking
model. Install `[rerank]` separately when changing FlashRank integration.
Read the [module map and extension guide](docs/MCP_ARCHITECTURE.md) before adding
new tools, providers, exporters, or runtime policies.

Network-dependent results are not deterministic. Include the date, dataset,
command, and raw metrics when a change relies on `eval_live_retrieval.py`.
Use its `--output` option so failures and timing remain inspectable.

Keep pull requests focused. Add or update tests for behavior changes, preserve
SourceBundle compatibility, and document any new network destination or data
disclosure in `PRIVACY.md`.

Documentation follows unreleased `main`; PyPI installation examples must not
advertise development-only APIs as published. Start at the [documentation index](docs/README.md).
Update usage, architecture/operations boundaries, and the changelog with behavior
changes. Label historical measurements with their original scope; never turn
retries, extrapolations, or offline pass rates into a new live quality claim.
Model/subscription-backed evaluations require explicit operator budget approval.

Use GitHub issues for bugs and feature proposals. Report vulnerabilities through
the private process in `SECURITY.md`.
