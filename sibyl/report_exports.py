"""Confirmed local export artifacts and explicit per-format failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .async_cache import CapacityError
from .privacy import display_url
from .report_models import ResearchReport
from .research_errors import ResearchRequestError
from .storage import ArtifactStore

ExportFormat = Literal["md", "json", "pdf"]


class ExportedArtifact(BaseModel):
    format: ExportFormat
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ExportFailure(BaseModel):
    format: ExportFormat
    code: Literal["export_failed", "capacity_exceeded"]
    message: str


class ExportManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    view: Literal["export"] = "export"
    status: Literal["saved", "partial", "failed"]
    job_id: str | None
    report_status: str
    requested_formats: list[ExportFormat]
    artifacts: list[ExportedArtifact]
    failures: list[ExportFailure]

    def text(self) -> str:
        labels = {"md": "Markdown", "json": "JSON", "pdf": "PDF"}
        saved = {artifact.format: artifact for artifact in self.artifacts}
        return "\n".join(
            f"{labels[format]} saved: {saved[format].path}"
            if format in saved
            else f"{labels[format]} failed: could not export report."
            for format in self.requested_formats
        )


class ReportExportError(ResearchRequestError):
    def __init__(self, manifest: ExportManifest) -> None:
        label = "PDF" if manifest.requested_formats == ["pdf"] else "Report"
        super().__init__(
            f"{label} export failed; no saved artifact is confirmed.",
            code="export_failed",
        )
        self.manifest = manifest


def export_report(
    report: ResearchReport,
    root: Path,
    output_dir: str,
    format: ExportFormat,
    job_id: str | None,
) -> ExportManifest:
    from .reporter import PDFRenderingError, _report_to_markdown, generate_pdf

    store = ArtifactStore(root, output_dir)
    formats: list[ExportFormat] = [format]
    artifacts: list[ExportedArtifact] = []
    failures: list[ExportFailure] = []
    for selected in formats:
        try:
            if selected == "pdf":
                path = Path(generate_pdf(report, str(store.root)))
            elif selected == "md":
                path = store.write(
                    _report_to_markdown(report).encode("utf-8"), "report", ".md"
                )
            else:
                data = report.to_dict()
                for source in data["sources"]:
                    source["url"] = display_url(source["url"])
                path = store.write(
                    json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                    "report",
                    ".json",
                )
            if (
                path.parent != store.root
                or path.is_symlink()
                or not path.is_file()
                or path.suffix != f".{selected}"
                or not 0 < path.stat().st_size <= store.MAX_ARTIFACT_BYTES
            ):
                raise ValueError("Export artifact was not confirmed inside the store.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            artifacts.append(
                ExportedArtifact(
                    format=selected,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    sha256=digest,
                )
            )
        except Exception as exc:
            failures.append(
                ExportFailure(
                    format=selected,
                    code="capacity_exceeded"
                    if isinstance(exc, CapacityError)
                    else "export_failed",
                    message=str(exc)
                    if isinstance(exc, PDFRenderingError)
                    else f"Could not confirm a saved {selected} artifact. Do not infer a path.",
                )
            )
    return ExportManifest(
        status="partial"
        if artifacts and failures
        else "saved"
        if artifacts
        else "failed",
        job_id=job_id,
        report_status=report.status,
        requested_formats=formats,
        artifacts=artifacts,
        failures=failures,
    )
