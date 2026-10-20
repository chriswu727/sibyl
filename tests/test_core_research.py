"""Host research must not import optional model or rendering SDKs."""

import subprocess
import sys
import unittest


class TestCoreResearch(unittest.TestCase):
    def test_host_report_roundtrip_with_optional_imports_blocked(self):
        script = """
import asyncio
import importlib.abc
import sys
import tempfile
from pathlib import Path

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'litellm', 'fpdf', 'matplotlib', 'pandas', 'yfinance', 'pytrends', 'flashrank'}:
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
            _, response = await server.call_tool('start_research', {'question': 'q'})
            job = runtime.research_jobs.store.get(response['job_id'])
            notebook = ResearchNotebook.from_dict(job['notebook'])
            passage = EvidencePassage('P1', 'bundle/S1/P1', 'Actual evidence text.', 'hash')
            source = EvidenceSource('S1', 'https://example.com', 'Example', '2026-10-20', 'hash', 'web', 21, [passage])
            notebook.bundles.append(SourceBundle('1.6', 'bundle', 'q', 'ok', [source],
                BundleDiagnostics(1,1,1,1,0,0,0,1,8,8,4000,4000,1)))
            job['notebook'] = notebook.to_dict()
            runtime.research_jobs.store.save(job)
            _, completed = await server.call_tool('complete_research', {
                'job_id': job['job_id'], 'revision': job['revision'],
                'summary': 'Summary [Source 1]', 'key_findings': ['Finding [Source 1]'],
                'allow_incomplete': True})
            assert completed['status'] == 'completed'
            for format in ('md', 'json'):
                await server.call_tool('save_report', {'job_id': job['job_id'], 'format': format})
                assert len(list(Path(directory).glob('*.' + format))) == 1
            assert 'litellm' not in sys.modules
            assert 'fpdf' not in sys.modules
        finally:
            await runtime.aclose()

asyncio.run(main())
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
