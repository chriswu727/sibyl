"""SourceBundle 1.6 consumer contract tests."""
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from pydantic import TypeAdapter

import sibyl.mcp_server as mcp_server
from sibyl.evidence import SourceBundle


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "docs" / "source_bundle_1_6.example.json"


class TestSourceBundleContract(unittest.TestCase):
    def test_example_round_trips_through_typed_contract(self):
        expected = json.loads(EXAMPLE.read_text(encoding="utf-8"))

        bundle = TypeAdapter(SourceBundle).validate_python(expected)

        self.assertEqual(bundle.to_dict(), expected)
        evidence_chars = 0
        for source in bundle.sources:
            for passage in source.evidence:
                evidence_chars += len(passage.text)
                self.assertEqual(
                    passage.content_hash,
                    hashlib.sha256(passage.text.encode("utf-8")).hexdigest(),
                )
                self.assertEqual(
                    passage.end_char - passage.start_char,
                    len(passage.text),
                )
        self.assertEqual(bundle.diagnostics.evidence_chars, evidence_chars)

    def test_contract_schema_exposes_consumer_enums(self):
        schema = TypeAdapter(SourceBundle).json_schema()

        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.6")
        self.assertEqual(
            set(
                schema["$defs"]["EvidenceSource"]["properties"]
                ["content_origin"]["enum"]
            ),
            {"direct_fetch", "jina_reader", "wikipedia_api", "search_snippet"},
        )
        self.assertEqual(
            set(schema["properties"]["status"]["enum"]),
            {"ok", "insufficient_evidence", "invalid_request", "failed"},
        )
        self.assertIn(
            "meta_article_published_time",
            schema["$defs"]["EvidenceSource"]["properties"]
            ["published_at_method"]["enum"],
        )


class TestMcpSourceBundleContract(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_serialization_matches_contract_example(self):
        expected = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        bundle = TypeAdapter(SourceBundle).validate_python(expected)

        with mock.patch(
            "sibyl.mcp_server.gather_source_bundle",
            new=mock.AsyncMock(return_value=bundle),
        ):
            _, structured = await mcp_server.mcp.call_tool(
                "gather_bundle",
                {"query": expected["query"]},
            )

        self.assertEqual(structured, expected)


if __name__ == "__main__":
    unittest.main()
