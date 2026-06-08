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
    metrics: dict[str, Any] = field(default_factory=dict)
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
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")
    scene_name = str(summary.get("scene_name") or scorecard.get("scene_name") or root.name)
    score = scorecard.get("score", "?")
    max_score = scorecard.get("max_score", "?")
    return PortfolioPage(
        run_dir=".",
        scene_name=scene_name,
        backend=str(summary.get("backend") or scorecard.get("backend") or "unknown"),
        dry_run=bool(summary.get("dry_run", scorecard.get("dry_run", False))),
        evidence_level=str(scorecard.get("evidence_level") or "unknown"),
        evidence_score=f"{score}/{max_score}",
        audit_status=str(audit.get("status") or "unknown"),
        summary=str(scorecard.get("summary") or _fallback_summary(summary)),
        metrics=_metrics(scorecard, evaluation),
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
        "Evidence scorecard": "evidence_scorecard.md",
        "Quality gate": "quality_gate.md",
        "Run audit": "run_audit.md",
        "Recommendations": "run_recommendations.md",
        "Scene inspection": "scene_data_inspection.md",
        "Evaluation summary": "evaluation/eval_summary.json",
        "Evaluation table": "evaluation/eval_table.csv",
        "Annotation template": "annotation_template.json",
        "Annotation review": "evaluation/annotation_review.md",
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
