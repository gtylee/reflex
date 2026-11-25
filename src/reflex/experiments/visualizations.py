"""Standalone SVG charts for experiment outputs without plotting dependencies."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

from .models.results import ExperimentReport


def write_metric_svg(report: ExperimentReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height, left, right, top, bottom = 900, 470, 90, 35, 75, 100
    plot_height = height - top - bottom
    baseline = height - bottom
    variants = list(report.runs)
    slot = (width - left - right) / max(1, len(variants))
    colors = {"precision": "#3b82f6", "recall": "#14b8a6", "f1": "#8b5cf6"}
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">Held-out event detection · {escape(report.scenario)}</text>',
                f'<text x="{width/2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">seed {report.seed} · comparison ready: {str(report.comparison_ready).lower()}</text>']
    for tick in range(0, 11, 2):
        y = baseline - plot_height * tick / 10
        elements.extend([f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                         f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">{tick/10:.1f}</text>'])
    group_width = min(54, slot * 0.68)
    bar_width = group_width / 3
    for index, run in enumerate(variants):
        center = left + slot * (index + 0.5)
        elements.append(f'<text x="{center:.1f}" y="{baseline+22}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600">{escape(run.variant_id)}</text>')
        if run.metrics is None:
            elements.append(f'<text x="{center:.1f}" y="{baseline-plot_height*0.52:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#777">{escape(run.status.replace("_", " "))}</text>')
            continue
        for j, metric in enumerate(("precision", "recall", "f1")):
            value = float(run.metrics.get(metric, 0.0))
            x = center - group_width / 2 + j * bar_width
            bh = plot_height * max(0.0, min(1.0, value))
            elements.append(f'<rect x="{x:.1f}" y="{baseline-bh:.1f}" width="{bar_width-2:.1f}" height="{bh:.1f}" fill="{colors[metric]}"/>')
            elements.append(f'<text x="{x+bar_width/2-1:.1f}" y="{baseline-bh-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">{value:.2f}</text>')
    legend_y = height - 27
    cursor = width / 2 - 145
    for metric, color in colors.items():
        elements.append(f'<rect x="{cursor}" y="{legend_y-11}" width="11" height="11" fill="{color}"/>')
        elements.append(f'<text x="{cursor+17}" y="{legend_y-2}" font-family="sans-serif" font-size="12">{metric}</text>')
        cursor += 105
    elements.append('</svg>')
    target.write_text("".join(elements))
    return target


def write_news_timeline_svg(report_path: str | Path, path: str | Path) -> Path:
    report = json.loads(Path(report_path).read_text())
    reactions = report["story_reactions"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height, left, right, line_y = 1100, 330, 70, 55, 130
    span = width - left - right
    colors = {"notify_portfolio_manager": "#2563eb", "investigate": "#7c3aed",
              "defer_and_monitor": "#d97706", "duplicate_suppressed": "#9ca3af",
              "record_only": "#64748b", "reopen_and_reassess": "#dc2626"}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="550" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">Evolving story: stateful routing with feedback</text>',
              '<text x="550" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">Synthetic story events in chronological order</text>',
              f'<line x1="{left}" y1="{line_y}" x2="{width-right}" y2="{line_y}" stroke="#cbd5e1" stroke-width="3"/>']
    for index, item in enumerate(reactions):
        x = left + (span * index / max(1, len(reactions) - 1))
        action = item["action"]
        color = colors.get(action, "#475569")
        pieces.append(f'<circle cx="{x:.1f}" cy="{line_y}" r="10" fill="{color}"/>')
        pieces.append(f'<text x="{x:.1f}" y="{line_y+34}" text-anchor="middle" font-family="sans-serif" font-size="12">rev {item["revision"]}</text>')
        label = action.replace("_", " ")
        chunks: list[str] = []
        for word in label.split():
            if chunks and len(chunks[-1]) + len(word) + 1 <= 18:
                chunks[-1] += " " + word
            else:
                chunks.append(word)
        for row, chunk in enumerate(chunks[:3]):
            pieces.append(f'<text x="{x:.1f}" y="{line_y+55+row*15}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#333">{escape(chunk)}</text>')
    pieces.append('</svg>')
    target.write_text("".join(pieces))
    return target


__all__ = ["write_metric_svg", "write_news_timeline_svg"]
