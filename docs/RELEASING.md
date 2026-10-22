# Releasing Sibyl

Sibyl publishes `sibyl-research` to PyPI and `io.github.chriswu727/sibyl` to the official MCP Registry from `.github/workflows/release.yml` when a GitHub Release is published. Authentication uses GitHub OIDC; the repository does not store a PyPI or MCP Registry token.

## Publication channels

The release path follows the same three-stage distribution flow as Argus:
GitHub Release, PyPI, then the official MCP Registry. Directory listings and
the product page are separate surfaces, not additional package registries.

| Channel | Sibyl destination | Update and verification |
|---|---|---|
| GitHub Releases | [Release history](https://github.com/chriswu727/sibyl/releases) | Publish the verified tag; confirm wheel and source archive assets |
| PyPI | [`sibyl-research`](https://pypi.org/project/sibyl-research/) | Release workflow with Trusted Publishing; install the exact public version |
| Official MCP Registry | `io.github.chriswu727/sibyl` | Workflow runs after PyPI; confirm the public manifest matches the release |
| Glama directory | [Sibyl listing](https://glama.ai/mcp/servers/chriswu727/sibyl) | Check repository indexing, build status, README, and tool schema separately |
| Product page | [yichenwu.dev/projects/sibyl](https://yichenwu.dev/projects/sibyl) | Update the website repository; verify the deployed version and examples |

Pre-beta channel inventory checked on 2026-10-21: GitHub and PyPI expose 0.3.0.
The official Registry's exact-name latest endpoint returned HTTP 404; the
next publish must be verified as a first visible registration, not assumed to
be an update. Glama already lists Sibyl, but its publicly indexed descriptions
and tool schema lag the current repository. Its listing is not evidence that
the latest commit has built successfully. The product page describes an older
source preview and needs to be reconciled with the release candidate.

The GitHub `v0.4.0b1` prerelease and both distribution assets are available.
Its first PyPI attempt failed before upload: the pinned publisher v1.14.0
rejected Metadata 2.5, although the build job's Twine 7 validation passed.
The corrected workflow uses publisher v1.14.3 and can recover the existing
release assets without rebuilding or replacing them. PyPI and Registry
availability still require verification after the protected publication job.

`glama.json` declares the repository maintainer for directory integration. It
does not publish a package, refresh a listing on demand, or prove a successful
build. If directory refresh requires account access or manual review, record
that as pending rather than claiming all channels are synchronized. No npm,
Docker Hub, or other package publication is configured by this workflow.

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

Prerequisite: satisfy the applicable gates in
[Launch readiness](LAUNCH_READINESS.md#release-gates). The maintainer approved
the host-assisted 0.4.0b1 beta scope on 2026-10-21. Engineering, security, and
clean-install checks remain mandatory; failed live-quality results stay visible.
Stable 0.4.0 additionally requires the retained research-quality gates.
`server.json` is the next-version manifest, not a statement that PyPI/Registry
already contains that version. `check_release.py` validates matching version strings.
Publication also passes `--require-release-note`, rejecting an undated/Unreleased
changelog. This does not enforce the research-quality gates.

1. Update `sibyl.__version__`, `server.json`, and its package version together; date the matching `CHANGELOG.md` section and remove its Unreleased label only after the gates pass.
2. Run the complete local verification:

   ```bash
   .venv/bin/python -m unittest discover tests -v
   .venv/bin/ruff check sibyl tests scripts
   .venv/bin/ruff format --check sibyl tests scripts
   .venv/bin/mypy
   .venv/bin/python scripts/check_docs.py
   .venv/bin/python scripts/check_research_workflow.py --pdf
   .venv/bin/python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)"
   .venv/bin/python scripts/check_mcp_stdio.py "$(command -v sibyl-mcp)" --monitor
   .venv/bin/python scripts/eval_retrieval.py --ranker lexical
   .venv/bin/python scripts/eval_retrieval_pipeline.py --ranker lexical
   .venv/bin/python scripts/eval_source_quality.py
   .venv/bin/python scripts/eval_live_retrieval.py --repeats 3 \
     --output evals/results/live-retrieval-YYYY-MM-DD.json
   ```

   The live gate uses public websites and is intentionally not part of CI.
   For 0.4.0b1, retain the 2026-10-21 run and its failed metrics; do not rerun
   until a favorable result or weaken its thresholds. A stable release also
   needs the model-backed evaluation with an explicitly authorized provider
   budget and retained model, configuration, and raw result. Autonomous mode
   remains experimental in this beta.

   Install the selected extras with `.[report,finance,extract,pdf,monitor]` first.
   The release workflow installs all of those extras, then separately validates
   clean core, PDF/monitoring, and report/finance/extraction wheel installations.
   Checks include installed-module origin, dependency consistency, real MCP
   research replay, authenticated monitoring shutdown, PDF parsing/export, and
   optional-tool availability. Temporary environments are removed sequentially.

3. Build and validate fresh distributions, not files left in `dist/` from an older version:

   ```bash
   # Use a new empty output directory for this candidate.
   .venv/bin/python -m pip install build==1.5.1 twine==7.0.0 pip-audit==2.10.1
   .venv/bin/python -m pip_audit --vulnerability-service osv --skip-editable
   .venv/bin/python scripts/check_release.py v0.4.0b1 --require-release-note
   .venv/bin/python -m build --outdir /tmp/sibyl-dist-0.4.0b1-candidate
   .venv/bin/python -m twine check /tmp/sibyl-dist-0.4.0b1-candidate/*
   .venv/bin/python scripts/check_wheel.py /tmp/sibyl-dist-0.4.0b1-candidate/*.whl 0.4.0b1
   ```

4. Merge the release pull request and verify all required checks on `main`.
   After dating the changelog, run `python scripts/check_release.py v0.4.0b1 --require-release-note`.
5. Create a draft GitHub Release for tag `v<package-version>` targeting the verified `main` commit. Use the matching changelog section as its notes. For `0.4.0b1`, select **prerelease**, not latest stable. Keep the same PEP 440 version in the package and Registry manifest; the Registry permits non-semantic versions.
6. Publish the GitHub Release. Approve the `pypi` environment only after the build job passes. The workflow attaches the distributions to the GitHub Release, uploads them to PyPI, waits for PyPI indexing, and publishes the matching `server.json` to the MCP Registry.
7. Verify the public package from a new environment:

   ```bash
   python3 -m venv /tmp/sibyl-pypi-smoke
   /tmp/sibyl-pypi-smoke/bin/python -m pip install --no-cache-dir sibyl-research==0.4.0b1
   /tmp/sibyl-pypi-smoke/bin/python -c "import importlib.metadata, sibyl; assert sibyl.__version__ == importlib.metadata.version('sibyl-research') == '0.4.0b1'"
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

## Recover a publisher-only failure

Use this only when the release assets are already built and tested, and no
uploaded distribution needs to change. Do not move the tag or overwrite assets.

After merging a publisher fix, dispatch the current `release.yml` on `main`:

```bash
gh workflow run release.yml --ref main -f release_tag=v0.4.0b1
```

The recovery path checks out the requested tag, repeats release validation,
tests, and dependency audits, downloads its published wheel and source archive,
and verifies both SHA-256 digests against the GitHub release metadata. It then
runs Twine and all clean-wheel checks before the existing `pypi` environment
approval. It does not rebuild or replace GitHub assets, disable metadata
validation, skip existing PyPI files, or bypass approval. Registry publication
uses the same release tag after successful PyPI publication.

Prereleases are opt-in. An unpinned `pip install sibyl-research` may continue to
select 0.3.0 as the latest stable version. Publish beta installation examples
with `sibyl-research==0.4.0b1`; do not imply that the stable package moved forward.
