"""Data collection and analysis — financial data, trends, charts."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DataPoint:
    date: str
    value: float
    label: str = ""


@dataclass
class DataSeries:
    name: str
    symbol: str
    points: List[DataPoint]
    current: float = 0.0
    change_pct: float = 0.0       # period change %
    trend: str = ""                # "up", "down", "flat"
    moving_avg_50: float = 0.0
    moving_avg_200: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    summary: str = ""


async def fetch_stock_data(
    symbol: str,
    period: str = "1y",
) -> Optional[DataSeries]:
    """Fetch stock/ETF/index data from Yahoo Finance."""
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return None

        points = []
        for date, row in hist.iterrows():
            points.append(DataPoint(
                date=date.strftime("%Y-%m-%d"),
                value=round(row["Close"], 2),
            ))

        current = points[-1].value if points else 0
        first = points[0].value if points else 0
        change_pct = round((current - first) / first * 100, 2) if first else 0

        # Moving averages
        closes = hist["Close"]
        ma50 = round(closes.rolling(50).mean().iloc[-1], 2) if len(closes) >= 50 else 0
        ma200 = round(closes.rolling(200).mean().iloc[-1], 2) if len(closes) >= 200 else 0

        # 52-week high/low
        high_52w = round(closes.max(), 2)
        low_52w = round(closes.min(), 2)

        # Trend
        if len(closes) >= 20:
            recent = closes.iloc[-20:].mean()
            earlier = closes.iloc[-40:-20].mean() if len(closes) >= 40 else closes.iloc[:20].mean()
            if recent > earlier * 1.02:
                trend = "up"
            elif recent < earlier * 0.98:
                trend = "down"
            else:
                trend = "flat"
        else:
            trend = "unknown"

        info = ticker.info
        name = info.get("shortName", info.get("longName", symbol))

        summary = (
            f"{name} ({symbol}): ${current:.2f} ({'+' if change_pct > 0 else ''}{change_pct}% over {period}). "
            f"Trend: {trend}. 52w range: ${low_52w}-${high_52w}."
        )
        if ma50:
            summary += f" 50-day MA: ${ma50}."
        if ma200:
            summary += f" 200-day MA: ${ma200}."

        return DataSeries(
            name=name,
            symbol=symbol,
            points=points,
            current=current,
            change_pct=change_pct,
            trend=trend,
            moving_avg_50=ma50,
            moving_avg_200=ma200,
            high_52w=high_52w,
            low_52w=low_52w,
            summary=summary,
        )
    except Exception as e:
        return None


async def fetch_multiple(symbols: List[str], period: str = "1y") -> List[DataSeries]:
    """Fetch data for multiple symbols."""
    import asyncio
    results = []
    for sym in symbols:
        ds = await fetch_stock_data(sym, period)
        if ds:
            results.append(ds)
    return results


def generate_chart(
    series_list: List[DataSeries],
    title: str = "Price History",
    output_path: str = "",
) -> str:
    """Generate a chart PNG from one or more DataSeries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 5))

    for series in series_list:
        dates = [datetime.strptime(p.date, "%Y-%m-%d") for p in series.points]
        values = [p.value for p in series.points]
        label = f"{series.name} ({series.symbol})"
        ax.plot(dates, values, label=label, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()

    if not output_path:
        output_path = f"/tmp/sibyl_chart_{datetime.now().strftime('%H%M%S')}.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_data_summary(series_list: List[DataSeries]) -> str:
    """Format data series into a text summary for LLM context."""
    lines = ["## Market Data"]
    for ds in series_list:
        lines.append(f"\n**{ds.name} ({ds.symbol})**")
        lines.append(f"- Current: ${ds.current:.2f}")
        lines.append(f"- Period change: {'+' if ds.change_pct > 0 else ''}{ds.change_pct}%")
        lines.append(f"- Trend: {ds.trend}")
        lines.append(f"- 52-week range: ${ds.low_52w} - ${ds.high_52w}")
        if ds.moving_avg_50:
            pos = "above" if ds.current > ds.moving_avg_50 else "below"
            lines.append(f"- Price is {pos} 50-day MA (${ds.moving_avg_50})")
        if ds.moving_avg_200:
            pos = "above" if ds.current > ds.moving_avg_200 else "below"
            lines.append(f"- Price is {pos} 200-day MA (${ds.moving_avg_200})")
    return "\n".join(lines)
