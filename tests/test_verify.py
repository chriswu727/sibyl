"""Finding verification — JSON verdicts, lexical fallback, alignment. Mocked.

Run: python -m unittest discover tests
"""
import json
import unittest
from unittest import mock

from sibyl.verifier import verify_findings, _lexical_verdict, _extract_citations, FindingVerification
from sibyl.config import Provider
from sibyl.scraper import WebPage


def _resp(content):
    m = mock.Mock()
    m.choices = [mock.Mock(message=mock.Mock(content=content))]
    return m


def _pages():
    return [WebPage(url="u1", title="Revenue report", text="NVIDIA data center revenue reached 40 billion dollars in Q2."),
            WebPage(url="u2", title="Other", text="Cats are small mammals kept as pets.")]


class TestExtractCitations(unittest.TestCase):
    def test_extracts(self):
        self.assertEqual(_extract_citations("Grew 40% [Source 1] and [Source 3]."), [1, 3])
        self.assertEqual(_extract_citations("no citation here"), [])


class TestLexicalVerdict(unittest.TestCase):
    def test_supported_when_overlap(self):
        v = _lexical_verdict(1, "NVIDIA data center revenue reached 40 billion [Source 1]", _pages())
        self.assertTrue(v.supported)

    def test_unsupported_no_citation(self):
        v = _lexical_verdict(1, "Some claim with no citation", _pages())
        self.assertFalse(v.supported)

    def test_unsupported_low_overlap(self):
        v = _lexical_verdict(1, "The pyramids of Egypt were built by aliens [Source 2]", _pages())
        self.assertFalse(v.supported)


class TestVerifyFindings(unittest.IsolatedAsyncioTestCase):
    async def test_parses_json_verdicts(self):
        payload = json.dumps({"verdicts": [
            {"index": 1, "supported": True, "confidence": "high", "cited": [1], "note": "matches"},
            {"index": 2, "supported": False, "confidence": "low", "cited": [2], "note": "unsupported"},
        ]})

        async def fake(**kwargs):
            return _resp(payload)

        with mock.patch("sibyl.verifier.litellm.acompletion", fake):
            out = await verify_findings(["A [Source 1]", "B [Source 2]"], _pages(),
                                        Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].supported)
        self.assertFalse(out[1].supported)

    async def test_alignment_fills_omissions_with_lexical(self):
        # LLM only returns a verdict for finding 1; finding 2 filled by lexical fallback
        payload = json.dumps({"verdicts": [{"index": 1, "supported": True, "confidence": "high", "cited": [1]}]})

        async def fake(**kwargs):
            return _resp(payload)

        findings = ["NVIDIA data center revenue reached 40 billion [Source 1]",
                    "Unrelated aliens claim [Source 2]"]
        with mock.patch("sibyl.verifier.litellm.acompletion", fake):
            out = await verify_findings(findings, _pages(), Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(len(out), 2)      # always 1:1 with findings
        self.assertFalse(out[1].supported)  # lexical fallback flagged the unrelated one

    async def test_non_json_falls_back_no_crash(self):
        async def fake(**kwargs):
            return _resp("not json at all")

        with mock.patch("sibyl.verifier.litellm.acompletion", fake):
            out = await verify_findings(["A [Source 1]", "B"], _pages(),
                                        Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(len(out), 2)  # lexical fallback for all, no crash

    async def test_empty_findings(self):
        out = await verify_findings([], _pages(), Provider(model="deepseek/deepseek-v4-flash", api_key="k"))
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
