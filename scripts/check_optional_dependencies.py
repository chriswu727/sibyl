"""Offline installed-extra checks; no models, finance APIs, or browser downloads."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io

from sibyl.pdf_reader import read_pdf


async def check_pdf() -> None:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 700 Td (Installed wheel PDF evidence.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    result = await read_pdf(output.getvalue(), 2000)
    if "error" in result or "Installed wheel PDF evidence." not in result["text"]:
        raise RuntimeError(f"Installed PDF worker failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=["core", "pdf-monitor", "report-finance-extract"]
    )
    profile = parser.parse_args().profile
    if profile == "pdf-monitor":
        importlib.import_module("uvicorn")
        asyncio.run(check_pdf())
    if profile == "report-finance-extract":
        for module in (
            "litellm",
            "fpdf",
            "PIL",
            "yfinance",
            "matplotlib",
            "pandas",
            "pytrends",
            "trafilatura",
        ):
            importlib.import_module(module)
    print(f"Installed {profile} optional dependency checks passed")


if __name__ == "__main__":
    main()
