# Releasing Sibyl

Sibyl publishes `sibyl-research` to PyPI from `.github/workflows/release.yml` when a GitHub Release is published. Authentication uses PyPI Trusted Publishing; the repository does not store a PyPI API token.

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

1. Update `sibyl.__version__` and add the matching `CHANGELOG.md` section in a pull request.
2. Run the complete local verification:

   ```bash
   .venv/bin/python -m unittest discover tests -v
   .venv/bin/python scripts/eval_retrieval.py --ranker lexical
   .venv/bin/python scripts/eval_retrieval_pipeline.py --ranker lexical
   .venv/bin/python scripts/eval_source_quality.py
   ```

3. Build and validate fresh distributions, not files left in `dist/` from an older version:

   ```bash
   rm -rf dist
   .venv/bin/python -m pip install build==1.5.1 twine==6.2.0
   .venv/bin/python scripts/check_release.py v0.3.0
   .venv/bin/python -m build
   .venv/bin/python -m twine check dist/*
   ```

4. Merge the release pull request and verify all required checks on `main`.
5. Create a draft GitHub Release for tag `v<package-version>` targeting the verified `main` commit. Use the matching changelog section as its notes.
6. Publish the GitHub Release. Approve the `pypi` environment only after the build job passes. The workflow then uploads the wheel and source distribution to PyPI.
7. Verify the public package from a new environment:

   ```bash
   python3 -m venv /tmp/sibyl-pypi-smoke
   /tmp/sibyl-pypi-smoke/bin/python -m pip install --no-cache-dir sibyl-research==0.3.0
   /tmp/sibyl-pypi-smoke/bin/python -c "import importlib.metadata, sibyl; assert sibyl.__version__ == importlib.metadata.version('sibyl-research') == '0.3.0'"
   ```

Never reuse a version already uploaded to PyPI. If publishing fails after PyPI accepts one distribution, diagnose the workflow and release a new patch version if any uploaded file must change.
