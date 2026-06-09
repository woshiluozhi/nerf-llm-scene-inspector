"""Generate a static portfolio page from a pipeline run directory."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class PortfolioPage:
    """Portable HTML page assembled from run-scoped evidence artifacts."""

    run_dir: str
    scene_name: str
    backend: str
    dry_run: bool
    evidence_level: str
    evidence_score: str
    audit_status: str
    summary: str
    result_status: str = "unknown"
    capture_manifest_status: str = "unknown"
    capture_manifest_fail_count: int = 0
    query_evidence_status: str = "unknown"
    query_counter_evidence_count: int = 0
    query_risk_flag_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    sharing_readiness: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_timestamp)

    def to_html(self) -> str:
        """Render the page as standalone HTML with relative artifact links."""

        dry_run_note = (
            "This is a dry-run smoke demo, not a real trained-scene result."
            if self.dry_run
            else "This page summarizes a real-mode run; inspect upstream versions and artifacts."
        )
        if self.result_status == "blocked":
            dry_run_note = (
                "This run is blocked for external sharing until failed evidence, "
                "query-risk, or claim-calibration checks are resolved."
            )
        elif self.query_risk_flag_count:
            dry_run_note = (
                "This run has unresolved query-risk flags; review query evidence "
                "before using scene-answer claims externally."
            )
        elif self.capture_manifest_fail_count:
            dry_run_note = (
                "This run has capture-manifest validation failures; resolve capture "
                "metadata, privacy, static-scene, or overlap issues before sharing."
            )
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '  <meta charset="utf-8">',
                '  <meta name="viewport" content="width=device-width, initial-scale=1">',
                f"  <title>{_escape(self.scene_name or 'NeRF-LLM Scene Inspector')}</title>",
                "  <style>",
                _style_block(),
                "  </style>",
                "</head>",
                "<body>",
                '  <main class="page">',
                '    <section class="hero">',
                '      <div>',
                '        <p class="eyebrow">NeRF-LLM Scene Inspector</p>',
                f"        <h1>{_escape(self.scene_name or 'Scene Run')}</h1>",
                f"        <p class=\"summary\">{_escape(self.summary)}</p>",
                f"        <p class=\"notice\">{_escape(dry_run_note)}</p>",
                "      </div>",
                '      <div class="score">',
                f"        <span>{_escape(self.evidence_score)}</span>",
                "        <small>Evidence score</small>",
                "      </div>",
                "    </section>",
                '    <section class="metrics">',
                *_stat_cards(self),
                "    </section>",
                '    <section class="grid">',
                '      <article class="panel">',
                "        <h2>Sharing Readiness</h2>",
                _sharing_readiness_panel(self.sharing_readiness),
                "      </article>",
                '      <article class="panel">',
                "        <h2>Top Recommendations</h2>",
                _recommendation_list(self.recommendations),
                "      </article>",
                '      <article class="panel">',
                "        <h2>Metrics</h2>",
                _metrics_table(self.metrics),
                "      </article>",
                "    </section>",
                '    <section class="panel">',
                "      <h2>Visual Evidence</h2>",
                _image_grid(self.images),
                "    </section>",
                '    <section class="panel">',
                "      <h2>Artifacts</h2>",
                _artifact_list(self.artifacts),
                "    </section>",
                f"    <footer>Generated at {_escape(self.generated_at)} from run directory {_escape(self.run_dir)}.</footer>",
                "  </main>",
                "</body>",
                "</html>",
            ]
        )

    def write_html(self, path: str | Path) -> Path:
        """Write the rendered HTML page."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_html(), encoding="utf-8")
        return output_path


def build_portfolio_page(run_dir: str | Path) -> PortfolioPage:
    """Build a static page payload from a run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scorecard = _read_json(root / "evidence_scorecard.json")
    audit = _read_json(root / "run_audit.json")
    result_card = _read_json(root / "run_result_card.json")
    capture = _read_json(root / "capture_manifest_validation.json")
    query_evidence = _read_json(root / "query_evidence_audit.json")
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")
    scene_name = str(summary.get("scene_name") or scorecard.get("scene_name") or root.name)
    score = scorecard.get("score", "?")
    max_score = scorecard.get("max_score", "?")
    dry_run = bool(summary.get("dry_run", scorecard.get("dry_run", False)))
    sharing_readiness = _sharing_readiness(root, capture=capture, dry_run=dry_run)
    counter_evidence_count, risk_flag_count = _query_evidence_counts(query_evidence, sharing_readiness)
    capture_status = _capture_manifest_status(capture, sharing_readiness)
    capture_fail_count = _capture_manifest_fail_count(capture, sharing_readiness)
    return PortfolioPage(
        run_dir=".",
        scene_name=scene_name,
        backend=str(summary.get("backend") or scorecard.get("backend") or "unknown"),
        dry_run=dry_run,
        evidence_level=str(scorecard.get("evidence_level") or "unknown"),
        evidence_score=f"{score}/{max_score}",
        audit_status=str(audit.get("status") or "unknown"),
        result_status=_page_result_status(
            str(result_card.get("result_status") or "unknown"),
            dry_run=dry_run,
            capture_present=bool(capture),
            capture_status=capture_status,
            capture_fail_count=capture_fail_count,
        ),
        capture_manifest_status=capture_status or "unknown",
        capture_manifest_fail_count=capture_fail_count,
        query_evidence_status=str(sharing_readiness.get("query_evidence_status") or query_evidence.get("status") or "unknown"),
        query_counter_evidence_count=counter_evidence_count,
        query_risk_flag_count=risk_flag_count,
        summary=str(scorecard.get("summary") or _fallback_summary(summary)),
        metrics=_metrics(scorecard, evaluation),
        sharing_readiness=sharing_readiness,
        recommendations=_recommendations(scorecard),
        images=_images(root),
        artifacts=_artifacts(root),
    )


def _metrics(scorecard: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = scorecard.get("metrics") if isinstance(scorecard.get("metrics"), dict) else {}
    merged = dict(metrics)
    for key in (
        "top_k_hit_rate",
        "mean_iou_2d",
        "semantic_success_rate",
        "average_relevancy_score",
        "num_evaluated_queries",
        "num_bbox_annotated_queries",
    ):
        if key in evaluation and key not in merged:
            merged[key] = evaluation[key]
    return merged


def _recommendations(scorecard: dict[str, Any]) -> list[str]:
    raw = scorecard.get("top_recommendations")
    if not isinstance(raw, list):
        return ["Review the generated run artifacts before sharing this page."]
    return [str(item) for item in raw[:5]] or ["No scorecard recommendations were recorded."]


def _sharing_readiness(root: Path, *, capture: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    packet = _read_json(root / "submission_packet" / "submission_packet.json")
    query_evidence = _read_json(root / "query_evidence_audit.json")
    summary = packet.get("readiness_summary")
    if isinstance(summary, dict) and summary:
        return _with_capture_validation(
            _with_query_evidence(dict(summary), packet, query_evidence),
            packet,
            capture,
            dry_run=dry_run,
        )
    if packet:
        return _with_capture_validation(
            _with_query_evidence(
                {
                    "status": "unknown",
                    "readiness_level": packet.get("readiness_level", "unknown"),
                    "failed_check_count": _count_items(packet.get("checklist"), "fail"),
                    "warning_check_count": _count_items(packet.get("checklist"), "warn"),
                    "packet_warning_count": len(packet.get("warnings") or []),
                    "pack_ok": packet.get("pack_ok"),
                    "recommended_next_action": packet.get("share_decision") or "Review the submission checklist.",
                },
                packet,
                query_evidence,
            ),
            packet,
            capture,
            dry_run=dry_run,
        )
    if capture:
        return _with_capture_validation({}, {}, capture, dry_run=dry_run)
    return {}


def _images(root: Path) -> list[dict[str, str]]:
    candidates: list[Path] = [
        root / "demo_assets" / "query_grid.png",
        root / "demo_assets" / "demo_montage.gif",
        root / "evaluation" / "annotation_review_contact_sheet.png",
    ]
    candidates.extend(sorted((root / "demo_assets").rglob("*overlay.png")))
    candidates.extend(sorted((root / "queries").rglob("*overlay.png")))
    candidates.extend(sorted((root / "queries").rglob("*relevancy.png")))
    seen: set[Path] = set()
    images: list[dict[str, str]] = []
    for path in candidates:
        if len(images) >= 12 or not path.exists() or path in seen:
            continue
        seen.add(path)
        images.append({"path": _relative(path, root), "label": _relative(path, root)})
    return images


def _artifacts(root: Path) -> list[dict[str, str]]:
    candidates = {
        "Pipeline summary": "pipeline_summary.json",
        "Capture manifest": "capture_manifest.md",
        "Capture validation": "capture_manifest_validation.md",
        "Preflight report": "preflight_report.md",
        "Failure diagnostics": "failure_diagnostics.md",
        "Evidence scorecard": "evidence_scorecard.md",
        "Quality gate": "quality_gate.md",
        "Run readiness gate": "run_readiness.md",
        "Run result card": "run_result_card.md",
        "Run audit": "run_audit.md",
        "Query evidence audit": "query_evidence_audit.md",
        "Recommendations": "run_recommendations.md",
        "Scene inspection": "scene_data_inspection.md",
        "Evaluation summary": "evaluation/eval_summary.json",
        "Evaluation table": "evaluation/eval_table.csv",
        "Annotation template": "annotation_template.json",
        "Annotation review": "evaluation/annotation_review.md",
        "Annotation workbench": "evaluation/annotation_workbench/annotation_workbench.html",
        "Annotation finalization": "annotation_finalize_report.md",
        "Annotation review JSON": "evaluation/annotation_review.json",
        "Scene relation report": "scene_relations/scene_relations_report.md",
        "Scene relation edges": "scene_relations/scene_relations_edges.csv",
        "Claim audit": "claim_audit.md",
        "Reproduction report": "reproduction_report.md",
        "Real-run action plan": "real_run_plan/real_run_plan.md",
        "Research report": "research_report.md",
        "Submission checklist": "submission_packet/submission_checklist.md",
        "Reproduction script": "reproduce_run.sh",
        "Portfolio result card": "portfolio_result_card.md",
    }
    artifacts: list[dict[str, str]] = []
    for label, relative_path in candidates.items():
        if (root / relative_path).exists():
            artifacts.append({"label": label, "path": relative_path})
    return artifacts


def _stat_cards(page: PortfolioPage) -> list[str]:
    stats = [
        ("Evidence", page.evidence_level),
        ("Result", page.result_status),
        ("Capture", page.capture_manifest_status),
        ("Capture fails", str(page.capture_manifest_fail_count)),
        ("Query evidence", page.query_evidence_status),
        ("Risk flags", str(page.query_risk_flag_count)),
        ("Backend", page.backend),
        ("Audit", page.audit_status),
        ("Run mode", "dry-run" if page.dry_run else "real"),
    ]
    lines: list[str] = []
    for label, value in stats:
        lines.extend(
            [
                '      <article class="metric">',
                f"        <span>{_escape(value)}</span>",
                f"        <small>{_escape(label)}</small>",
                "      </article>",
            ]
        )
    return lines


def _recommendation_list(items: list[str]) -> str:
    lines = ["        <ol>"]
    for item in items:
        lines.append(f"          <li>{_escape(item)}</li>")
    lines.append("        </ol>")
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "        <p>No metrics were recorded.</p>"
    lines = ['        <table class="table">']
    for key in sorted(metrics):
        value = metrics[key]
        lines.append(
            f"          <tr><th>{_escape(key)}</th><td>{_escape(_format_value(value))}</td></tr>"
        )
    lines.append("        </table>")
    return "\n".join(lines)


def _sharing_readiness_panel(summary: dict[str, Any]) -> str:
    if not summary:
        return (
            "        <p>No submission packet was found. Run "
            "<code>python scripts/create_submission_packet.py --run-dir &lt;run-dir&gt;</code> "
            "before external sharing.</p>"
        )
    lines = ['        <table class="table">']
    rows = [
        ("Status", summary.get("status", "unknown")),
        ("Readiness", summary.get("readiness_level", "unknown")),
        ("Failed checks", summary.get("failed_check_count", 0)),
        ("Warning checks", summary.get("warning_check_count", 0)),
        ("Packet warnings", summary.get("packet_warning_count", 0)),
        ("Pack OK", summary.get("pack_ok")),
        ("Capture manifest", summary.get("capture_manifest_status", "unknown")),
        ("Capture failures", summary.get("capture_manifest_fail_count", 0)),
        ("Query evidence", summary.get("query_evidence_status", "unknown")),
        ("Query counter-evidence", summary.get("query_counter_evidence_count", 0)),
        ("Query risk flags", summary.get("query_risk_flag_count", 0)),
    ]
    for label, value in rows:
        lines.append(f"          <tr><th>{_escape(label)}</th><td>{_escape(_format_value(value))}</td></tr>")
    lines.append("        </table>")
    next_action = summary.get("recommended_next_action")
    if next_action:
        lines.append(f'        <p class="next-action">{_escape(next_action)}</p>')
    blockers = summary.get("top_blockers") if isinstance(summary.get("top_blockers"), list) else []
    if blockers:
        lines.append("        <p class=\"section-label\">Top blockers</p>")
        lines.append(_compact_list(blockers[:3]))
    warnings = summary.get("top_warnings") if isinstance(summary.get("top_warnings"), list) else []
    if warnings:
        lines.append("        <p class=\"section-label\">Top warnings</p>")
        lines.append(_compact_list(warnings[:3]))
    return "\n".join(lines)


def _image_grid(images: list[dict[str, str]]) -> str:
    if not images:
        return "      <p>No visual artifacts were found.</p>"
    lines = ['      <div class="image-grid">']
    for image in images:
        path = _escape(image["path"])
        label = _escape(image["label"])
        lines.extend(
            [
                '        <figure>',
                f'          <a href="{path}"><img src="{path}" alt="{label}"></a>',
                f"          <figcaption>{label}</figcaption>",
                "        </figure>",
            ]
        )
    lines.append("      </div>")
    return "\n".join(lines)


def _artifact_list(artifacts: list[dict[str, str]]) -> str:
    if not artifacts:
        return "      <p>No linked artifacts were found.</p>"
    lines = ['      <ul class="artifacts">']
    for artifact in artifacts:
        label = _escape(artifact["label"])
        path = _escape(artifact["path"])
        lines.append(f'        <li><a href="{path}">{label}</a></li>')
    lines.append("      </ul>")
    return "\n".join(lines)


def _compact_list(items: list[Any]) -> str:
    lines = ["        <ul>"]
    for item in items:
        lines.append(f"          <li>{_escape(item)}</li>")
    lines.append("        </ul>")
    return "\n".join(lines)


def _style_block() -> str:
    return """
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --ink: #18202b;
      --muted: #5b6472;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #e5f5f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    .page { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 42px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding: 28px 0 22px;
      border-bottom: 1px solid var(--line);
    }
    .eyebrow { margin: 0 0 8px; color: var(--accent); font-weight: 700; text-transform: uppercase; font-size: 0.78rem; }
    h1 { margin: 0; font-size: clamp(2rem, 5vw, 4.4rem); line-height: 0.95; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 1.05rem; }
    .summary { max-width: 760px; color: var(--muted); font-size: 1.02rem; }
    .notice { color: #7a3e00; background: #fff4df; border: 1px solid #f1d39a; padding: 10px 12px; border-radius: 6px; width: fit-content; }
    .score {
      min-width: 180px;
      padding: 18px;
      background: var(--accent-soft);
      border: 1px solid #b9ddd7;
      border-radius: 8px;
      text-align: right;
    }
    .score span { display: block; font-size: 2.2rem; font-weight: 800; }
    .score small, .metric small { color: var(--muted); }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .metric span { display: block; font-size: 1.15rem; font-weight: 750; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin: 18px 0; }
    .table { width: 100%; border-collapse: collapse; }
    .table th, .table td { border-top: 1px solid var(--line); padding: 8px 0; text-align: left; vertical-align: top; }
    .table th { color: var(--muted); font-weight: 650; width: 52%; }
    .next-action { margin: 12px 0 0; padding: 10px 12px; border-left: 4px solid var(--accent); background: var(--accent-soft); }
    .section-label { margin: 14px 0 6px; color: var(--muted); font-weight: 700; font-size: 0.82rem; text-transform: uppercase; }
    .image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fbfcfe; }
    img { display: block; width: 100%; height: auto; }
    figcaption { padding: 8px 10px; color: var(--muted); font-size: 0.86rem; overflow-wrap: anywhere; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .artifacts { columns: 2; padding-left: 20px; }
    footer { margin-top: 20px; color: var(--muted); font-size: 0.86rem; }
    @media (max-width: 800px) {
      .hero, .grid { grid-template-columns: 1fr; }
      .metrics, .image-grid { grid-template-columns: 1fr; }
      .score { text-align: left; }
      .artifacts { columns: 1; }
    }
    """.strip()


def _fallback_summary(summary: dict[str, Any]) -> str:
    if summary.get("success") is True:
        return "Pipeline artifacts were generated successfully."
    return "Pipeline summary is missing or incomplete."


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _count_items(raw_items: Any, status: str) -> int:
    if not isinstance(raw_items, list):
        return 0
    return sum(1 for item in raw_items if isinstance(item, dict) and item.get("status") == status)


def _with_query_evidence(
    summary: dict[str, Any],
    packet: dict[str, Any],
    query_evidence: dict[str, Any],
) -> dict[str, Any]:
    counter_evidence_count, risk_flag_count = _query_evidence_counts(query_evidence, packet)
    if "query_evidence_status" not in summary:
        summary["query_evidence_status"] = packet.get("query_evidence_status") or query_evidence.get("status") or "unknown"
    if "query_counter_evidence_count" not in summary:
        summary["query_counter_evidence_count"] = counter_evidence_count
    if "query_risk_flag_count" not in summary:
        summary["query_risk_flag_count"] = risk_flag_count
    return summary


def _with_capture_validation(
    summary: dict[str, Any],
    packet: dict[str, Any],
    capture: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    capture_status = _capture_manifest_status(capture, packet)
    fail_count = _capture_manifest_fail_count(capture, packet)
    summary["capture_manifest_status"] = str(capture.get("status") or capture_status or "missing")
    summary["capture_manifest_fail_count"] = fail_count
    should_block = (not dry_run and not capture) or (not dry_run and capture_status != "ready") or fail_count
    if should_block:
        summary["status"] = "fail"
        summary["readiness_level"] = "blocked"
        _append_unique(summary, "failed_checks", "capture_manifest")
        _append_unique(
            summary,
            "top_blockers",
            (
                "capture_manifest: "
                f"status={summary['capture_manifest_status']}, failures={fail_count}. "
                "Fix capture validation before external sharing."
            ),
        )
        summary["failed_check_count"] = max(
            _safe_int(summary.get("failed_check_count")),
            len(summary.get("failed_checks") or []),
        )
        summary["recommended_next_action"] = "Fix capture validation before external sharing."
    elif capture_status and capture_status != "ready":
        _append_unique(summary, "warning_checks", "capture_manifest")
        _append_unique(
            summary,
            "top_warnings",
            f"capture_manifest: status={summary['capture_manifest_status']}, failures={fail_count}.",
        )
        summary["warning_check_count"] = max(
            _safe_int(summary.get("warning_check_count")),
            len(summary.get("warning_checks") or []),
        )
    return summary


def _page_result_status(
    result_status: str,
    *,
    dry_run: bool,
    capture_present: bool,
    capture_status: str,
    capture_fail_count: int,
) -> str:
    if (not dry_run and not capture_present) or (not dry_run and capture_status != "ready") or capture_fail_count:
        return "blocked"
    return result_status


def _capture_manifest_status(capture: dict[str, Any], source: dict[str, Any]) -> str:
    if capture:
        return str(capture.get("status") or "")
    return str(source.get("capture_manifest_status") or "")


def _capture_manifest_fail_count(capture: dict[str, Any], source: dict[str, Any]) -> int:
    return max(_safe_int(capture.get("fail_count")), _safe_int(source.get("capture_manifest_fail_count")))


def _append_unique(payload: dict[str, Any], key: str, value: str) -> None:
    items = payload.get(key)
    if not isinstance(items, list):
        items = []
        payload[key] = items
    if value not in items:
        items.append(value)


def _query_evidence_counts(query_evidence: dict[str, Any], source: dict[str, Any]) -> tuple[int, int]:
    counter = _safe_int(source.get("query_counter_evidence_count"))
    risk = _safe_int(source.get("query_risk_flag_count"))
    totals = query_evidence.get("totals") if isinstance(query_evidence.get("totals"), dict) else {}
    if not counter:
        counter = _safe_int(totals.get("counter_evidence_count"))
    if not risk:
        risk = _safe_int(totals.get("risk_flag_count"))
    tasks = query_evidence.get("tasks") if isinstance(query_evidence.get("tasks"), list) else []
    if not counter:
        counter = sum(
            _safe_int(task.get("counter_evidence_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    if not risk:
        risk = sum(
            _safe_int(task.get("risk_flag_count"))
            for task in tasks
            if isinstance(task, dict)
        )
    return counter, risk


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
