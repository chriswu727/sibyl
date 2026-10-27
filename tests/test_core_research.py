"""Host research must not import optional model or rendering SDKs."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path


class TestCoreResearch(unittest.TestCase):
    def test_runtime_source_has_no_removed_provider_imports_or_modules(self):
        root = Path(__file__).resolve().parents[1] / "sibyl"
        forbidden = {"litellm", "openai", "anthropic", "yfinance", "pytrends"}
        removed = {
            "context",
            "analyzer",
            "async_utils",
            "data",
            "researcher",
            "tools",
            "verifier",
        }
        for path in root.glob("*.py"):
            with self.subTest(module=path.name):
                self.assertNotIn(path.stem, removed | forbidden)
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                        if node.level:
                            self.assertNotIn((node.module or "").split(".")[0], removed)
                    else:
                        continue
                    self.assertFalse(
                        {name.split(".")[0] for name in names} & forbidden,
                        f"Removed execution dependency imported by {path.name}",
                    )

    def test_host_report_roundtrip_with_optional_imports_blocked(self):
        script = """
import asyncio
import importlib.abc
import sys
import tempfile
from pathlib import Path

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'litellm', 'openai', 'anthropic', 'fpdf', 'matplotlib', 'pandas', 'yfinance', 'pytrends', 'flashrank', 'pypdf'}:
            raise ImportError('Optional dependency blocked: ' + fullname)

sys.meta_path.insert(0, BlockOptional())
from sibyl.config import Config
from sibyl.evidence import BundleDiagnostics, EvidencePassage, EvidenceSource, SourceBundle
from sibyl.research_state import ResearchNotebook
from sibyl.runtime import RuntimePolicy, SibylRuntime
from sibyl.mcp_server import create_server

async def main():
    with tempfile.TemporaryDirectory() as directory:
        runtime = SibylRuntime(RuntimePolicy(output_root=Path(directory)), Config())
        server = create_server(runtime)
        try:
            response = (await server.call_tool('start_research', {'question': 'q'})).structuredContent
            job = runtime.research_jobs.store.get(response['job_id'])
            notebook = ResearchNotebook.from_dict(job['notebook'])
            passage = EvidencePassage('P1', 'bundle/S1/P1', 'Actual evidence text.', 'hash')
            source = EvidenceSource('S1', 'https://example.com', 'Example', '2026-10-20', 'hash', 'web', 21, [passage])
            notebook.bundles.append(SourceBundle('1.6', 'bundle', 'q', 'ok', [source],
                BundleDiagnostics(1,1,1,1,0,0,0,1,8,8,4000,4000,1)))
            job['notebook'] = notebook.to_dict()
            runtime.research_jobs.store.save(job)
            reviewed = await server.call_tool('assess_research', {
                'job_id': job['job_id'], 'revision': job['revision'],
                'question_id': 'Q0', 'status': 'supported',
                'conclusion': 'Finding', 'counterevidence_note': 'Further investigation needed.',
                'claims': [{'claim_id': 'Q0.C1', 'kind': 'fact', 'status': 'supported',
                    'text': 'Finding', 'supporting_passage_ids': ['bundle/S1/P1'],
                    'quotes': [{'passage_id': 'bundle/S1/P1', 'text': 'Actual evidence text.'}]}]})
            assert not reviewed.isError, reviewed
            completed = await server.call_tool('complete_research', {
                'job_id': job['job_id'], 'revision': reviewed.structuredContent['revision'],
                'claim_ids': ['Q0.C1'],
                'allow_incomplete': True})
            assert not completed.isError, completed
            assert completed.structuredContent['status'] == 'completed'
            for format in ('md', 'json'):
                exported = await server.call_tool('save_report', {'job_id': job['job_id'], 'format': format})
                assert not exported.isError, exported
                assert exported.structuredContent['status'] == 'saved'
                assert len(list(Path(directory).glob('*.' + format))) == 1
            assert 'litellm' not in sys.modules
            assert 'fpdf' not in sys.modules
        finally:
            await runtime.aclose()

asyncio.run(main())
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
        )
        self.assertEqual(result.returncode, 0, result.stderr)
