# Releasing Sibyl

Sibyl publishes `sibyl-research` to PyPI and `io.github.chriswu727/sibyl` to the official MCP Registry from `.github/workflows/release.yml` when a GitHub Release is published. Authentication uses GitHub OIDC; the repository does not store a PyPI or MCP Registry token.

## One-time setup

1. Create a GitHub environment named `pypi` and require manual approval for deployments.
2. On the PyPI `sibyl-research` project, add a GitHub Trusted Publisher with these exact values:
   - Owner: `chriswu727`
   - Repository: `sibyl`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Protect `main` and require the test workflow before merging release changes.

The workflow filename and environment name are part of the OIDC identity. Renaming either one requires updating the Trusted Publisher on PyPI.

## Release checklist

Prerequisite: every gate in [Launch readiness](LAUNCH_READINESS.md#release-gates)
must be satisfied, including live/report-quality evaluation and external beta
feedback. The current 0.4.0 development version is not cleared for release.
`server.json` is the next-version manifest, not a statement that PyPI/Registry
already contains that version. `check_release.py` validates matching version
strings; it does not reject an Unreleased changelog or enforce quality gates.

1. Update `sibyl.__version__`, `server.json`, and its package version together; date the matching `CHANGELOG.md` section and remove its Unreleased label only after the gates pass.
2. Run the complete local verification:

   ```bash
   .venv/bin/python -m unittest discover tests -v
   .venv/bin/python scripts/check_docs.py
   .venv/bin/python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)"
   .venv/bin/python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)" --monitor
   .venv/bin/python scripts/eval_retrieval.py --ranker lexical
   .venv/bin/python scripts/eval_retrieval_pipeline.py --ranker lexical
   .venv/bin/python scripts/eval_source_quality.py
   .venv/bin/python scripts/eval_live_retrieval.py --repeats 3 \
     --output evals/results/live-retrieval-YYYY-MM-DD.json
   ```

   The live gate uses public websites and is intentionally not part of CI. If
   the optional report pipeline changed, run its model-backed evaluation with an
   explicit provider budget and retain the model, configuration, and raw result.

   Install the selected extras with `.[report,finance,extract,pdf,monitor]` first.
   The release workflow currently installs report/finance/extract and smoke-tests
   the core wheel. Extend it to PDF/monitoring extras and the authenticated
   monitoring smoke check before the next release; the broader PR CI matrix
   does not replace clean-wheel verification of those extras.

3. Build and validate fresh distributions, not files left in `dist/` from an older version:

   ```bash
   # Use a new empty output directory for this candidate.
   .venv/bin/python -m pip install build==1.5.1 twine==7.0.0 pip-audit==2.10.1
   .venv/bin/python -m pip_audit --vulnerability-service osv --skip-editable
   .venv/bin/python scripts/check_release.py v0.4.0
   .venv/bin/python -m build --outdir /tmp/sibyl-dist-0.4.0-candidate
   .venv/bin/python -m twine check /tmp/sibyl-dist-0.4.0-candidate/*
   ```

4. Merge the release pull request and verify all required checks on `main`.
5. Create a draft GitHub Release for tag `v<package-version>` targeting the verified `main` commit. Use the matching changelog section as its notes.
6. Publish the GitHub Release. Approve the `pypi` environment only after the build job passes. The workflow attaches the distributions to the GitHub Release, uploads them to PyPI, waits for PyPI indexing, and publishes the matching `server.json` to the MCP Registry.
7. Verify the public package from a new environment:

   ```bash
   python3 -m venv /tmp/sibyl-pypi-smoke
   /tmp/sibyl-pypi-smoke/bin/python -m pip install --no-cache-dir sibyl-research==0.4.0
   /tmp/sibyl-pypi-smoke/bin/python -c "import importlib.metadata, sibyl; assert sibyl.__version__ == importlib.metadata.version('sibyl-research') == '0.4.0'"
   /tmp/sibyl-pypi-smoke/bin/sibyl-mcp --version
   /tmp/sibyl-pypi-smoke/bin/sibyl-mcp --list-tools
   ```

8. Confirm that the GitHub Release has both distribution assets, PyPI exposes
   the new version, and the official MCP Registry entry shows the same title,
   description, package version, and website URL.
9. Verify the README and `llms-install.md` examples against that published wheel.
   Only then replace source-checkout quick starts and unreleased notices with
   the matching PyPI/`uvx` installation instructions.

Never reuse a version already uploaded to PyPI. If publishing fails after PyPI accepts one distribution, diagnose the workflow and release a new patch version if any uploaded file must change.
