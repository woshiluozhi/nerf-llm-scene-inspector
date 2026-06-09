"""Generate a static project-level portfolio site."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


DEFAULT_REPO_URL = "https://github.com/woshiluozhi/nerf-llm-scene-inspector"


@dataclass
class SiteRunEntry:
    """Compact run summary displayed on the project site."""

    scene_name: str
    status: str
    backend: str
    dry_run: bool
    query_count: int
    audit_score: str
    audit_blocker_count: int
    capture_manifest_fail_count: int
    result_status: str
    submission_readiness_level: str
    query_risk_flag_count: int
    top_k_hit_rate: str
    mean_iou_2d: str
    link: str = ""


@dataclass
class ProjectPortfolioSite:
    """Standalone HTML page for the repository's portfolio front door."""

    repo_root: str
    repo_url: str
    generated_at: str
    hero_image: str = ""
    montage_image: str = ""
    overlay_images: list[dict[str, str]] = field(default_factory=list)
    run_entries: list[SiteRunEntry] = field(default_factory=list)
    run_index_link: str = ""
    run_comparison_link: str = ""

    def to_html(self) -> str:
        """Render the site as static HTML with relative links."""

        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '  <meta charset="utf-8">',
                '  <meta name="viewport" content="width=device-width, initial-scale=1">',
                "  <title>NeRF-LLM Scene Inspector</title>",
                "  <style>",
                _style_block(),
                "  </style>",
                "</head>",
                "<body>",
                '  <main class="site">',
                '    <section class="hero">',
                '      <div class="hero-copy">',
                '        <p class="eyebrow">Research engineering portfolio</p>',
                "        <h1>NeRF-LLM Scene Inspector</h1>",
                "        <p class=\"lede\">Open-vocabulary 3D scene understanding from phone video, "
                "Nerfstudio reconstruction, and LERF-style language-field querying.</p>",
                '        <div class="actions">',
                f'          <a href="{_escape(self.repo_url)}">GitHub repository</a>',
                '          <a href="method_summary.md">Method summary</a>',
                '          <a href="real_run_reproducibility.md">Reproducibility guide</a>',
                "        </div>",
                "      </div>",
                _hero_media(self.hero_image),
                "    </section>",
                '    <section class="band">',
                '      <div class="section-head">',
                "        <h2>What This Demonstrates</h2>",
                "        <p>Implemented research infrastructure, not a claim of new state of the art.</p>",
                "      </div>",
                '      <div class="capabilities">',
                _capability("3D reconstruction", "Nerfstudio wrappers for data processing and baseline NeRF training."),
                _capability("Language fields", "LERF-first semantic query path with OpenNeRF as a secondary adapter."),
                _capability("Query planning", "Deterministic local planner for object, affordance, material, and relation prompts."),
                _capability(
                    "Counter-evidence answers",
                    "Negative prompts are preserved as avoid evidence, with image-space conflict flags for actionable queries.",
                ),
                _capability(
                    "Query evidence audit",
                    "Per-query checks for overlays, localization evidence, fallback mode, confidence, counter-evidence, risk flags, and missing artifacts.",
                ),
                _capability("Scene relations", "Heuristic entity-relation reports from query boxes or 3D points with explicit evidence tags."),
                _capability(
                    "Annotation workbench",
                    "Offline bbox labeling plus finalize tooling for turning browser-edited boxes into refreshed run evidence.",
                ),
                _capability("Experiment matrix", "Ablation-style JSON, CSV, and Markdown summaries across variants and query sets."),
                _capability("Research reports", "Paper-style run summaries that combine metrics, evidence, limitations, and next steps."),
                _capability(
                    "Run result cards",
                    "One-page reviewer summaries with a calibrated takeaway, evidence snapshot, limitations, and safe sharing language.",
                ),
                _capability("Submission packets", "Claim-calibrated CV and outreach checklists for sharing without overclaiming."),
                _capability("Real-run plans", "Command playbooks for upgrading smoke evidence into a real CUDA/Nerfstudio/LERF run."),
                _capability(
                    "Claim audits",
                    "Checks portfolio-facing text for unsupported SOTA, novelty, benchmark, production, or robotics-policy claims.",
                ),
                _capability(
                    "Evidence packaging",
                    "Capture/privacy gates, annotation QA, audits, scorecards, quality gates, and share-safe packs.",
                ),
                "      </div>",
                "    </section>",
                '    <section class="band split">',
                "      <div>",
                "        <h2>Reproducible Workflow</h2>",
                "        <ol class=\"workflow\">",
                "          <li><strong>Capture</strong><span>Slow phone video or overlapping images with structured capture metadata.</span></li>",
                "          <li><strong>Process</strong><span>Run Nerfstudio data preparation and inspect camera pose quality.</span></li>",
                "          <li><strong>Train</strong><span>Fit a baseline NeRF and a LERF-style language field.</span></li>",
                "          <li><strong>Query</strong><span>Generate relevancy overlays, relation graphs, annotation QA, metrics, research reports, and portfolio artifacts.</span></li>",
                "        </ol>",
                "      </div>",
                _code_panel(),
                "    </section>",
                _visual_section(self.overlay_images, self.montage_image),
                _runs_section(self.run_entries, self.run_index_link, self.run_comparison_link),
                '    <section class="band links">',
                "      <h2>Portfolio Materials</h2>",
                "      <ul>",
                '        <li><a href="portfolio_result_card.md">Portfolio result card</a></li>',
                '        <li><a href="project_report.md">Project report template</a></li>',
                '        <li><a href="cv_bullets.md">CV bullets</a></li>',
                '        <li><a href="cold_email_paragraph.md">Cold email paragraph</a></li>',
                '        <li><a href="real_scene_capture_checklist.md">Real-scene capture checklist</a></li>',
                "      </ul>",
                "    </section>",
                f"    <footer>Generated at {_escape(self.generated_at)} from {_escape(self.repo_root)}.</footer>",
                "  </main>",
                "</body>",
                "</html>",
            ]
        )

    def write_html(self, output_path: str | Path) -> Path:
        """Write the rendered site to disk."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html() + "\n", encoding="utf-8")
        return path


def build_project_site(
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    run_index_path: str | Path | None = None,
    repo_url: str = DEFAULT_REPO_URL,
    max_runs: int = 5,
) -> ProjectPortfolioSite:
    """Build a static project portfolio site payload."""

    root = Path(repo_root) if repo_root is not None else project_root()
    output = Path(output_path)
    output_dir = output.parent
    assets_dir = root / "docs" / "assets"
    run_index = _read_json(Path(run_index_path)) if run_index_path else {}
    return ProjectPortfolioSite(
        repo_root=".",
        repo_url=repo_url,
        generated_at=utc_timestamp(),
        hero_image=_optional_relative_link(assets_dir / "query_grid.png", output_dir),
        montage_image=_optional_relative_link(assets_dir / "demo_montage.gif", output_dir),
        overlay_images=_overlay_images(assets_dir, output_dir),
        run_entries=_run_entries(run_index, Path(run_index_path).parent if run_index_path else None, output_dir, max_runs),
        run_index_link=_optional_relative_link(Path(run_index_path), output_dir) if run_index_path else "",
        run_comparison_link=_run_comparison_link(run_index_path, output_dir),
    )


def _run_entries(
    run_index: dict[str, Any],
    runs_root: Path | None,
    output_dir: Path,
    max_runs: int,
) -> list[SiteRunEntry]:
    raw_entries = run_index.get("entries") if isinstance(run_index, dict) else None
    if not isinstance(raw_entries, list):
        return []
    entries: list[SiteRunEntry] = []
    for raw in raw_entries[:max_runs]:
        if not isinstance(raw, dict):
            continue
        run_dir = str(raw.get("run_dir") or "")
        artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
        portfolio_path = str(artifacts.get("portfolio_page") or "")
        link = ""
        if runs_root is not None and run_dir and portfolio_path:
            link = _relative_link(runs_root / run_dir / portfolio_path, output_dir)
        entries.append(
            SiteRunEntry(
                scene_name=str(raw.get("scene_name") or run_dir or "unknown"),
                status="success" if raw.get("success") else "needs review",
                backend=str(raw.get("backend") or "unknown"),
                dry_run=bool(raw.get("dry_run")),
                query_count=_safe_int(raw.get("query_count")),
                audit_score=_display(raw.get("audit_score")),
                audit_blocker_count=_safe_int(raw.get("audit_blocker_count", raw.get("blocker_count"))),
                capture_manifest_fail_count=_safe_int(raw.get("capture_manifest_fail_count")),
                result_status=str(raw.get("result_status") or "unknown"),
                submission_readiness_level=str(raw.get("submission_readiness_level") or "unknown"),
                query_risk_flag_count=_safe_int(raw.get("query_risk_flag_count")),
                top_k_hit_rate=_display_float(raw.get("top_k_hit_rate")),
                mean_iou_2d=_display_float(raw.get("mean_iou_2d")),
                link=link,
            )
        )
    return entries


def _overlay_images(assets_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    candidates = [
        ("Mug relevancy overlay", assets_dir / "mug_overlay.png"),
        ("Metallic tools overlay", assets_dir / "metallic_tools_overlay.png"),
    ]
    images: list[dict[str, str]] = []
    for label, path in candidates:
        link = _optional_relative_link(path, output_dir)
        if link:
            images.append({"label": label, "path": link})
    return images


def _hero_media(hero_image: str) -> str:
    if not hero_image:
        return (
            '      <div class="hero-empty">'
            "<span>Run the dry-run pipeline to generate preview artifacts.</span>"
            "</div>"
        )
    return (
        '      <figure class="hero-media">'
        f'<img src="{_escape(hero_image)}" alt="Query grid preview">'
        "<figcaption>Dry-run visual artifact shape: RGB, relevancy, overlay.</figcaption>"
        "</figure>"
    )


def _visual_section(images: list[dict[str, str]], montage_image: str) -> str:
    parts = [
        '    <section class="band">',
        '      <div class="section-head">',
        "        <h2>Visual Artifacts</h2>",
        "        <p>Checked-in dry-run previews show the artifact format used by real runs.</p>",
        "      </div>",
    ]
    if montage_image:
        parts.extend(
            [
                '      <figure class="wide-figure">',
                f'        <img src="{_escape(montage_image)}" alt="Demo montage">',
                "        <figcaption>Demo montage generated by the project pipeline.</figcaption>",
                "      </figure>",
            ]
        )
    if images:
        parts.append('      <div class="artifact-grid">')
        for image in images:
            parts.extend(
                [
                    "        <figure>",
                    f'          <img src="{_escape(image["path"])}" alt="{_escape(image["label"])}">',
                    f'          <figcaption>{_escape(image["label"])}</figcaption>',
                    "        </figure>",
                ]
            )
        parts.append("      </div>")
    parts.append("    </section>")
    return "\n".join(parts)


def _runs_section(entries: list[SiteRunEntry], run_index_link: str, run_comparison_link: str) -> str:
    lines = [
        '    <section class="band">',
        '      <div class="section-head">',
        "        <h2>Recent Runs</h2>",
        "        <p>Run-scoped summaries can be regenerated locally or after a real GPU experiment.</p>",
        "      </div>",
    ]
    if not entries:
        lines.extend(
            [
                '      <p class="empty">No run index was supplied when this page was generated.</p>',
                "    </section>",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            '      <div class="table-wrap">',
            '        <table class="runs">',
            "          <thead><tr><th>Scene</th><th>Status</th><th>Result</th><th>Submission</th>"
            "<th>Backend</th><th>Mode</th><th>Queries</th><th>Audit</th><th>Audit Blockers</th>"
            "<th>Capture Fails</th><th>Risk Flags</th>"
            "<th>Top-k</th><th>IoU</th><th>Page</th></tr></thead>",
            "          <tbody>",
        ]
    )
    for entry in entries:
        page_cell = f'<a href="{_escape(entry.link)}">open</a>' if entry.link else "n/a"
        mode = "dry-run" if entry.dry_run else "real"
        lines.append(
            "            <tr>"
            f"<td>{_escape(entry.scene_name)}</td>"
            f"<td>{_escape(entry.status)}</td>"
            f"<td>{_escape(entry.result_status)}</td>"
            f"<td>{_escape(entry.submission_readiness_level)}</td>"
            f"<td>{_escape(entry.backend)}</td>"
            f"<td>{mode}</td>"
            f"<td>{entry.query_count}</td>"
            f"<td>{_escape(entry.audit_score)}</td>"
            f"<td>{entry.audit_blocker_count}</td>"
            f"<td>{entry.capture_manifest_fail_count}</td>"
            f"<td>{entry.query_risk_flag_count}</td>"
            f"<td>{_escape(entry.top_k_hit_rate)}</td>"
            f"<td>{_escape(entry.mean_iou_2d)}</td>"
            f"<td>{page_cell}</td>"
            "</tr>"
        )
    lines.extend(["          </tbody>", "        </table>", "      </div>"])
    if run_index_link:
        lines.append(f'      <p class="small-link"><a href="{_escape(run_index_link)}">Full run index JSON</a></p>')
    if run_comparison_link:
        lines.append(
            f'      <p class="small-link"><a href="{_escape(run_comparison_link)}">Run comparison report</a></p>'
        )
    lines.append("    </section>")
    return "\n".join(lines)


def _capability(title: str, description: str) -> str:
    return (
        '        <article class="capability">'
        f"<h3>{_escape(title)}</h3>"
        f"<p>{_escape(description)}</p>"
        "</article>"
    )


def _code_panel() -> str:
    return """      <pre class="commands"><code>python scripts/run_scene_pipeline.py --dry-run --query mug
python scripts/generate_research_report.py --run-dir results/pipeline_runs/desk_scene
python scripts/create_annotation_workbench.py --annotations results/pipeline_runs/desk_scene/annotation_template.json --results results/pipeline_runs/desk_scene/queries --output results/pipeline_runs/desk_scene/evaluation/annotation_workbench
python scripts/finalize_annotations.py --run-dir results/pipeline_runs/desk_scene --filled results/pipeline_runs/desk_scene/evaluation/annotation_workbench/annotation_seed.json --profile smoke --export-pack --zip-pack
python scripts/compare_runs.py --root results/pipeline_runs
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack
python scripts/validate_portfolio_pack.py --pack results/portfolio_pack.zip
python scripts/create_real_run_plan.py --run-dir results/pipeline_runs/desk_scene --input path/to/video.mp4 --type video --submission-packet results/pipeline_runs/desk_scene/submission_packet/submission_packet.json</code></pre>"""


def _run_comparison_link(run_index_path: str | Path | None, output_dir: Path) -> str:
    if not run_index_path:
        return ""
    comparison_path = Path(run_index_path).parent / "run_comparison.md"
    return _optional_relative_link(comparison_path, output_dir)


def _style_block() -> str:
    return """
    :root {
      color-scheme: light;
      --bg: #f6f7f3;
      --ink: #16201c;
      --muted: #59635f;
      --panel: #ffffff;
      --line: #d7dcd5;
      --accent: #176b5d;
      --accent-soft: #e4f0ec;
      --warning: #805500;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    .site { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 44px; }
    .hero {
      min-height: 78vh;
      display: grid;
      grid-template-columns: minmax(0, 0.88fr) minmax(360px, 1.12fr);
      gap: 28px;
      align-items: center;
      border-bottom: 1px solid var(--line);
    }
    .eyebrow { margin: 0 0 10px; color: var(--accent); font-weight: 760; text-transform: uppercase; font-size: 0.78rem; }
    h1 { margin: 0; max-width: 720px; font-size: clamp(2.6rem, 7vw, 6rem); line-height: 0.92; letter-spacing: 0; }
    h2 { margin: 0; font-size: 1.3rem; }
    h3 { margin: 0 0 6px; font-size: 1rem; }
    .lede { max-width: 670px; color: var(--muted); font-size: 1.08rem; }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
    .actions a, .small-link a {
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }
    .hero-media, .wide-figure, .artifact-grid figure {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }
    .hero-media img, .wide-figure img, .artifact-grid img { display: block; width: 100%; height: auto; }
    figcaption { padding: 9px 11px; color: var(--muted); font-size: 0.88rem; }
    .hero-empty {
      min-height: 320px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
    }
    .band { padding: 30px 0; border-bottom: 1px solid var(--line); }
    .section-head { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 16px; }
    .section-head p { margin: 0; color: var(--muted); max-width: 520px; }
    .capabilities { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .capability {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
    }
    .capability p { margin: 0; color: var(--muted); }
    .split { display: grid; grid-template-columns: minmax(0, 0.95fr) minmax(320px, 1.05fr); gap: 28px; align-items: start; }
    .workflow { padding-left: 22px; color: var(--ink); }
    .workflow li { margin: 12px 0; }
    .workflow span { display: block; color: var(--muted); }
    .commands {
      margin: 0;
      overflow-x: auto;
      background: #18201d;
      color: #f2f7f3;
      border-radius: 8px;
      padding: 18px;
      font-size: 0.9rem;
    }
    .wide-figure { margin-bottom: 14px; }
    .artifact-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .runs { width: 100%; border-collapse: collapse; min-width: 780px; }
    .runs th, .runs td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
    .runs th { color: var(--muted); font-size: 0.82rem; text-transform: uppercase; }
    .runs tr:last-child td { border-bottom: 0; }
    .empty { color: var(--warning); background: #fff8e8; border: 1px solid #ead7a7; border-radius: 8px; padding: 12px; }
    .links ul { columns: 2; padding-left: 20px; }
    a { color: var(--accent); }
    footer { margin-top: 22px; color: var(--muted); font-size: 0.86rem; }
    @media (max-width: 880px) {
      .hero, .split { grid-template-columns: 1fr; min-height: auto; padding-top: 36px; }
      .capabilities, .artifact-grid { grid-template-columns: 1fr; }
      .section-head { display: block; }
      .links ul { columns: 1; }
    }
    """.strip()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _optional_relative_link(path: Path, output_dir: Path) -> str:
    if not path.exists():
        return ""
    return _relative_link(path, output_dir)


def _relative_link(path: Path, output_dir: Path) -> str:
    relative = os.path.relpath(path.resolve(), output_dir.resolve())
    return Path(relative).as_posix()


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _display(value: object) -> str:
    return "n/a" if value is None or value == "" else str(value)


def _display_float(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
