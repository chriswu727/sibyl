# Releasing Sibyl

Sibyl publishes `sibyl-research` to PyPI and `io.github.chriswu727/sibyl` to the official MCP Registry from `.github/workflows/release.yml` when a GitHub Release is published. Authentication uses GitHub OIDC; the repository does not store a PyPI or MCP Registry token.

## Publication channels

This runbook follows unreleased host-only `main`. The current target is a stable
release after all [quality gates](LAUNCH_READINESS.md#release-gates) pass, not
another beta. Version `0.4.0b1` remains in source metadata for development only;
it was already published with the old architecture and must never be reused for
these changed distributions. Pick a fresh version at the release decision.

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

### Historical channel checks — 2026-10-21

The following is a dated publication record, not a fresh check of public services.
Pre-beta channel inventory checked on 2026-10-21: GitHub and PyPI exposed 0.3.0.
The official Registry's exact-name latest endpoint returned HTTP 404; the
next publish must be verified as a first visible registration, not assumed to
be an update. Glama already lists Sibyl, but its publicly indexed descriptions
and tool schema lag the current repository. Its listing is not evidence that
the latest commit has built successfully. The product page describes an older
source preview and needs to be reconciled with the release candidate.

The GitHub `v0.4.0b1` prerelease and both distribution assets are available.
Its first PyPI attempt failed before upload: the pinned publisher v1.14.0
rejected Metadata 2.5, although the build job's Twine 7 validation passed.
The corrected workflow uses publisher v1.14.3 and recovered the existing
release assets without rebuilding or replacing them. PyPI now exposes 0.4.0b1;
both file hashes match the GitHub assets. The next Registry step rejected an
overlong description. The corrected manifest and Registry-only recovery path
avoid re-uploading an already published PyPI version.

An isolated installation from public PyPI passed dependency consistency, version
and console-script checks, the real 12-tool MCP stdio smoke test, and the two-round
host research replay with review, citations, exports, restart, and immutability.
The temporary environment was removed. At that time README and installation
examples pinned the verified public beta. Current examples instead install
source because that beta does not contain the host-only refactor.

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

Prerequisite: satisfy all current stable gates in
[Launch readiness](LAUNCH_READINESS.md#release-gates). The old beta approval is
not authority to ship current source. Engineering, security, clean-install and
research-quality checks remain mandatory; failed live results stay visible.
`server.json` is the development manifest and still carries the old published
version label; it is not ready to publish current source. Update it together
with package metadata at the release decision. `check_release.py` validates matching version strings.
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
     --host-sources evals/host_source_cases.jsonl \
     --output evals/results/host-source-retrieval-YYYY-MM-DD.json
   ```

   The live gate uses public websites and is intentionally not part of pull
   request CI. Follow the [frozen protocol](LAUNCH_READINESS.md#host-source-retrieval-gate-protocol).
   To run it on a GitHub-hosted runner instead of a developer network, dispatch
   the `live retrieval gate` workflow on `main` with `gate=host_sources` and
   commit its uploaded JSON as the dated result. The keyless discovery run
   (`gate=keyless`) stays tracked but does not block release.
   Retain every failed run; do not rerun until a favorable result or weaken
   thresholds. A stable release also
   needs the paired host-agent research evaluation with explicitly authorized
   usage and retained model, configuration, rubric and raw results. The current
   source preview removes internal autonomous execution; the published beta
   still contains the previous architecture.

   Install the selected extras with `.[report,extract,pdf,monitor]` first.
   The release workflow installs all of those extras, then separately validates
   clean core, PDF/monitoring, and report/extraction wheel installations.
   Checks include installed-module origin, dependency consistency, real MCP
   research replay, authenticated monitoring shutdown, PDF parsing/export, and
   optional-tool availability. Temporary environments are removed sequentially.

3. Build and validate fresh distributions, not files left in `dist/` from an older version:

   ```bash
   # Use a new empty output directory for this candidate.
   .venv/bin/python -m pip install build==1.5.1 twine==7.0.0 pip-audit==2.10.1
   .venv/bin/python -m pip_audit --vulnerability-service osv --skip-editable
   # Set only after selecting a fresh version and updating matching metadata.
   export SIBYL_RELEASE_VERSION='<fresh-version>'
   .venv/bin/python scripts/check_release.py "v${SIBYL_RELEASE_VERSION}" --require-release-note
   .venv/bin/python -m build --outdir "/tmp/sibyl-dist-${SIBYL_RELEASE_VERSION}-candidate"
   .venv/bin/python -m twine check /tmp/sibyl-dist-${SIBYL_RELEASE_VERSION}-candidate/*
   .venv/bin/python scripts/check_wheel.py /tmp/sibyl-dist-${SIBYL_RELEASE_VERSION}-candidate/*.whl "$SIBYL_RELEASE_VERSION"
   ```

4. Merge the release pull request and verify all required checks on `main`.
   After dating the changelog, run `python scripts/check_release.py "v${SIBYL_RELEASE_VERSION}" --require-release-note`.
5. Create a draft GitHub Release for tag `v<package-version>` targeting the verified `main` commit. Use the matching changelog section as its notes. The current target is a stable version, not a prerelease. Keep the same PEP 440 version in the package and Registry manifest.
6. Publish the GitHub Release. Approve the `pypi` environment only after the build job passes. The workflow attaches the distributions to the GitHub Release, uploads them to PyPI, waits for PyPI indexing, and publishes the matching `server.json` to the MCP Registry.
7. Verify the public package from a new environment:

   ```bash
   python3 -m venv /tmp/sibyl-pypi-smoke
   /tmp/sibyl-pypi-smoke/bin/python -m pip install --no-cache-dir "sibyl-research==${SIBYL_RELEASE_VERSION}"
   /tmp/sibyl-pypi-smoke/bin/python -c 'import importlib.metadata, os, sibyl; assert sibyl.__version__ == importlib.metadata.version("sibyl-research") == os.environ["SIBYL_RELEASE_VERSION"]'
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
gh workflow run release.yml --ref main -f release_tag="v${SIBYL_RELEASE_VERSION}"
```

The recovery path checks out the requested tag, repeats release validation,
tests, and dependency audits, downloads its published wheel and source archive,
and verifies both SHA-256 digests against the GitHub release metadata. It then
runs Twine and all clean-wheel checks before the existing `pypi` environment
approval. It does not rebuild or replace GitHub assets, disable metadata
validation, skip existing PyPI files, or bypass approval. Registry publication
uses the same release tag after successful PyPI publication.

## Recover a Registry-only failure

If PyPI publication succeeded but Registry validation failed, correct
`server.json` on `main` and run:

```bash
gh workflow run release.yml --ref main -f release_tag="v${SIBYL_RELEASE_VERSION}" -f registry_only=true
```

This mode uses the dispatch commit's corrected manifest, requires the `main`
branch, and verifies that its package version and dated changelog match the
requested tag. It confirms the PyPI version exists before Registry publication.
Build, GitHub-asset upload, and PyPI upload jobs are skipped; no PyPI approval
is needed because no package is uploaded. The original tag and distributions
remain unchanged. This recovery is for the version currently described on
`main`, not arbitrary historical manifests.

The release validator enforces the Registry description's 1–100 character
limit before publication. Regression tests also validate the actual repository
manifest, not only synthetic release fixtures.

Prereleases are opt-in. Before a new stable release, an unpinned
`pip install sibyl-research` may still select the older stable package. Do not
replace current source-install instructions with either historical package.
Verify the exact new public wheel before advertising its installation command.
