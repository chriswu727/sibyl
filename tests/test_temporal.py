"""Period routing remains separate from evidence and source authority."""

import unittest
from datetime import datetime, timezone

from sibyl.temporal import period_match


class TestPeriodMatch(unittest.TestCase):
    def test_explicit_period_prefers_title_over_crawl_or_update_year(self):
        self.assertEqual(period_match("2026 chip market", "2025 financial results", "https://ex.com", "2026-10-01"), -1)
        self.assertEqual(period_match("2026 chip market", "2026 financial results", "https://ex.com"), 1)

    def test_unknown_dates_are_neutral_and_body_comparisons_are_not_inputs(self):
        self.assertEqual(period_match("2026 chip market", "Financial results", "https://ex.com"), 0)

    def test_historical_and_comparative_queries_do_not_always_prefer_newest(self):
        self.assertEqual(period_match("2024 earnings", "2024 results", "https://ex.com", "2026-01-01"), 1)
        self.assertEqual(period_match("2024 versus 2026 earnings", "2024 results", "https://ex.com"), 1)
        self.assertEqual(period_match("transistor physics", "Study from 1960", "https://ex.com"), 0)

    def test_fiscal_period_is_not_calendar_year(self):
        self.assertEqual(period_match("2026 earnings", "Second quarter fiscal 2027", "https://ex.com", "2026-08-26"), 1)
        self.assertEqual(period_match("2026 earnings", "Second quarter fiscal 2026", "https://ex.com", "2025-08-26"), -1)
        self.assertEqual(period_match("2026 earnings", "Second quarter fiscal 2026", "https://ex.com"), 0)
        self.assertEqual(period_match("fiscal 2026 earnings", "Second quarter fiscal 2026", "https://ex.com", "2025-08-26"), 1)

    def test_latest_and_chinese_queries_use_current_year_without_fixed_date(self):
        year = datetime.now(timezone.utc).year
        for query in ("latest earnings", "当前芯片市场", f"截至{year}年芯片市场"):
            self.assertEqual(period_match(query, f"Results {year}", "https://ex.com"), 1)
            self.assertEqual(period_match(query, f"Results {year - 1}", "https://ex.com"), -1)

    def test_path_is_a_last_resort_not_query_string_tracking(self):
        self.assertEqual(period_match("2026 earnings", "Results", "https://ex.com/2026/results"), 1)
        self.assertEqual(period_match("2026 earnings", "Results", "https://ex.com/results?year=2025"), 0)
