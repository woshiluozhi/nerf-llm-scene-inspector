"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path


def write_project_report(
    path: str | Path,
    *,
    title: str,
    scene_name: str,
    backend: str,
    query_rows: list[dict[str, object]] | None = None,
    metrics: dict[str, object] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write a concise markdown project report."""

    query_rows = query_rows or []
    metrics = metrics or {}
    notes = notes or []
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        "This report summarizes a NeRF-LLM Scene Inspector run. The system is built on",
        "Nerfstudio and LERF and is intended as a reproducible research engineering demo.",
        "",
        "## Scene",
        "",
        f"- Scene name: `{scene_name}`",
        f"- Backend: `{backend}`",
        "",
        "## Query Results",
        "",
        "| Query | Target | Top-k Hit | Best IoU | Confidence | Warnings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if query_rows:
        for row in query_rows:
            lines.append(
                "| {query} | {target} | {hit} | {iou:.3f} | {confidence} | {warnings} |".format(
                    query=row.get("query", ""),
                    target=row.get("target_description", ""),
                    hit=row.get("topk_hit", ""),
                    iou=float(row.get("best_iou_2d") or 0.0),
                    confidence=row.get("confidence", ""),
                    warnings=str(row.get("warnings", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("| Pending | Pending | Pending | Pending | Pending | Pending |")

    lines.extend(["", "## Evaluation Summary", "", "| Metric | Value |", "| --- | --- |"])
    if metrics:
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| Pending | Pending |")

    lines.extend(["", "## Notes", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append(
            "- This project demonstrates open-vocabulary 3D scene querying without claiming new state-of-the-art results."
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
