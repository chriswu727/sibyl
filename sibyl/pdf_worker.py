"""Resource-limited PDF text worker; executed as an isolated subprocess."""

import io
import json
import sys


def extract(data, max_chars):
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"error": "PDF support requires the pdf installation extra."}
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        return {"error": "Encrypted PDF is not supported."}
    warnings = []
    page_count = len(reader.pages)
    if page_count > 40:
        warnings.append(
            "PDF extraction limited to the first 40 pages; this is not the complete document."
        )
    parts = []
    length = 0
    for index in range(min(page_count, 40)):
        text = reader.pages[index].extract_text() or ""
        if not text.strip():
            warnings.append(
                f"PDF page {index + 1} has no readable text; OCR was not performed."
            )
            continue
        part = f"[PDF page {index + 1}]\n{text.strip()}\n"
        remaining = max_chars - length
        parts.append(part[:remaining])
        length += len(parts[-1])
        if len(part) > remaining or (length >= max_chars and index + 1 < page_count):
            warnings.append(
                "PDF text was truncated at the extraction character limit; read other sources for missing evidence."
            )
            break
    if not parts:
        return {"error": "PDF contains no readable text; OCR is not supported."}
    title = str((reader.metadata or {}).get("/Title", ""))[:300]
    return {"title": title, "text": "".join(parts), "warnings": warnings}


def main():
    try:
        data = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024 or not data.startswith(b"%PDF-"):
            result = {"error": "Invalid or oversized PDF."}
        else:
            result = extract(data, max(200, min(int(sys.argv[1]), 60000)))
    except Exception:
        result = {"error": "PDF parsing failed or exceeded a resource limit."}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
