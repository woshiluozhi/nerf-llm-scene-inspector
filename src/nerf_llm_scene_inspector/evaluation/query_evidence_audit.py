"""Audit per-query visual and localization evidence in pipeline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


EvidenceAuditStatus = Literal["pass", "warn", "fail"]
EvidenceMode = Literal["3d", "2d_fallback", "render_only", "missing"]
VISUAL_RENDER_KINDS = {
    "rgb",
    "relevancy",
    "overlay",
    "composited",
    "heatmap",
    "mask",
    "semantic",
    "segmentation",
    "depth",
}
FALLBACK_ARTIFACT_KINDS = {"viewer_fallback", "manual_template"}


@dataclass
class QueryEvidenceTask:
    """Evidence summary for one natural-language task directory."""

    task_slug: str
    task: str
    report_path: str
    status: EvidenceAuditStatus
    evidence_mode: EvidenceMode
    support_level: str = ""
    result_count: int = 0
    expanded_queries: list[str] = field(default_factory=list)
    rendered_image_count: int = 0
    existing_rendered_image_count: int = 0
    missing_rendered_image_count: int = 0
    diagnostic_artifact_count: int = 0
    fallback_artifact_count: int = 0
    overlay_count: int = 0
    relevancy_count: int = 0
    rgb_count: int = 0
    bounding_region_count: int = 0
    image_region_count: int = 0
    region_3d_count: int = 0
    candidate_point_count: int = 0
    max_confidence: float | None = None
    mean_confidence: float | None = None
    max_region_score: float | None = None
    counter_evidence_count: int = 0
    counter_evidence_labels: list[str] = field(default_factory=list)
    risk_flag_count: int = 0
    risk_flags: list[str] = field(default_factory=list)
    query_grid_exists: bool = False
    visual_summary_exists: bool = False
    visual_summary_issue_count: int = 0
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class QueryEvidenceAudit:
    """Run-level audit over all scene query reports."""

    ok: bool
    status: EvidenceAuditStatus
    run_dir: str
    queries_dir: str
    timestamp: str
    task_count: int = 0
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    totals: dict[str, Any] = field(default_factory=dict)
    tasks: list[QueryEvidenceTask] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "run_dir": self.run_dir,
            "queries_dir": self.queries_dir,
            "timestamp": self.timestamp,
            "task_count": self.task_count,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "totals": dict(self.totals),
            "tasks": [task.to_dict() for task in self.tasks],
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Query Evidence Audit",
            "",
            f"- Status: {self.status}",
            f"- OK: {self.ok}",
            f"- Run directory: `{self.run_dir}`",
            f"- Queries directory: `{self.queries_dir}`",
            f"- Tasks: {self.task_count}",
            f"- Pass/warn/fail: {self.pass_count}/{self.warn_count}/{self.fail_count}",
            f"- Rendered images: {self.totals.get('rendered_image_count', 0)}",
            f"- Existing rendered images: {self.totals.get('existing_rendered_image_count', 0)}",
            f"- Diagnostic artifacts: {self.totals.get('diagnostic_artifact_count', 0)}",
            f"- Viewer fallback/manual artifacts: {self.totals.get('fallback_artifact_count', 0)}",
            f"- Overlays: {self.totals.get('overlay_count', 0)}",
            f"- Bounding regions: {self.totals.get('bounding_region_count', 0)}",
            f"- Candidate 3D points: {self.totals.get('candidate_point_count', 0)}",
            f"- Counter-evidence items: {self.totals.get('counter_evidence_count', 0)}",
            f"- Risk flags: {self.totals.get('risk_flag_count', 0)}",
            "",
            "## Task Evidence",
            "",
            (
                "| Task | Status | Mode | Support | Results | Overlays | Regions | "
                "3D Points | Max Confidence | Counter | Risk Flags | Grid | Notes |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            *_task_lines(self.tasks),
            "",
            "## Recommendations",
            "",
            *_recommendation_lines(self.tasks, self.warnings),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def audit_query_evidence(
    run_dir: str | Path,
    *,
    queries_dir: str | Path | None = None,
) -> QueryEvidenceAudit:
    """Inspect query reports and summarize whether their evidence is reviewable."""

    root = Path(run_dir)
    query_root = Path(queries_dir) if queries_dir is not None else root / "queries"
    warnings: list[str] = []
    tasks: list[QueryEvidenceTask] = []
    if not query_root.exists() or not query_root.is_dir():
        warnings.append(f"Query results directory does not exist: {_display_path(query_root)}")
        return _audit_report(root, query_root, tasks, warnings)

    report_paths = sorted(query_root.glob("*/scene_query_report.json"))
    if not report_paths:
        warnings.append(f"No scene_query_report.json files found under {_display_path(query_root)}")
        return _audit_report(root, query_root, tasks, warnings)

    for report_path in report_paths:
        tasks.append(_audit_task(report_path, run_dir=root))
    return _audit_report(root, query_root, tasks, warnings)


def write_query_evidence_audit(
    run_dir: str | Path,
    *,
    queries_dir: str | Path | None = None,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> QueryEvidenceAudit:
    """Build and persist query evidence audit JSON and Markdown reports."""

    root = Path(run_dir)
    audit = audit_query_evidence(root, queries_dir=queries_dir)
    audit.to_json(output or root / "query_evidence_audit.json")
    audit.to_markdown(markdown_output or root / "query_evidence_audit.md")
    return audit


def _audit_task(report_path: Path, *, run_dir: Path) -> QueryEvidenceTask:
    task_dir = report_path.parent
    warnings: list[str] = []
    recommendations: list[str] = []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return QueryEvidenceTask(
            task_slug=task_dir.name,
            task=task_dir.name,
            report_path=_relative(report_path, run_dir),
            status="fail",
            evidence_mode="missing",
            warnings=[f"Could not parse scene_query_report.json: {exc}"],
            recommendations=["Regenerate this query report before using it as evidence."],
        )
    if not isinstance(payload, dict):
        return QueryEvidenceTask(
            task_slug=task_dir.name,
            task=task_dir.name,
            report_path=_relative(report_path, run_dir),
            status="fail",
            evidence_mode="missing",
            warnings=["scene_query_report.json is not a JSON object."],
            recommendations=["Regenerate this query report before using it as evidence."],
        )

    results = [item for item in payload.get("query_results") or [] if isinstance(item, dict)]
    rendered_images = [
        view
        for result in results
        for view in result.get("rendered_images") or []
        if isinstance(view, dict)
    ]
    visual_rendered_images = [view for view in rendered_images if _is_visual_render(view)]
    fallback_artifacts = [view for view in rendered_images if _is_fallback_artifact(view)]
    diagnostic_artifact_count = len(rendered_images) - len(visual_rendered_images)
    missing_images = [
        str(view.get("path") or "")
        for view in visual_rendered_images
        if view.get("path") and not _resolve_artifact_path(str(view["path"]), run_dir, task_dir).exists()
    ]
    bounding_regions = [
        region
        for result in results
        for region in result.get("bounding_regions") or []
        if isinstance(region, dict)
    ]
    candidate_points = [
        point
        for result in results
        for point in result.get("candidate_points") or []
        if isinstance(point, dict)
    ]
    confidences = [_safe_float(result.get("confidence")) for result in results]
    confidences = [value for value in confidences if value is not None]
    region_scores = [_safe_float(region.get("score")) for region in bounding_regions]
    region_scores = [value for value in region_scores if value is not None]
    overlay_count = _kind_count(visual_rendered_images, "overlay")
    image_region_count = sum(1 for region in bounding_regions if str(region.get("coordinate_frame") or "image") == "image")
    region_3d_count = sum(1 for region in bounding_regions if _region_has_3d_evidence(region))
    support_level = _support_level(payload)
    counter_evidence = _counter_evidence_items(payload)
    risk_flags = _risk_flags(payload)
    counter_evidence_labels = _counter_evidence_labels(counter_evidence)
    query_grid_exists = (task_dir / "query_grid.png").exists()
    visual_summary_path = task_dir / "query_visual_summary.json"
    visual_summary_exists = visual_summary_path.exists()
    expanded_queries = [str(result.get("query") or "") for result in results if str(result.get("query") or "").strip()]
    existing_overlay_count = _existing_kind_count(visual_rendered_images, "overlay", run_dir, task_dir)

    for warning in payload.get("warnings") or []:
        warnings.append(str(warning))
    for result in results:
        for warning in result.get("warnings") or []:
            warnings.append(f"{result.get('query') or 'query'}: {warning}")
    if fallback_artifacts:
        warnings.append(
            f"{len(fallback_artifacts)} viewer fallback/manual artifact(s) were recorded; these are not visual evidence."
        )
        recommendations.append("Import saved viewer RGB/relevancy outputs or rerun the backend before using this query as evidence.")
    if missing_images:
        warnings.append(f"{len(missing_images)} rendered image reference(s) do not exist on disk.")
        recommendations.append("Regenerate query artifacts or import the missing viewer outputs.")
    if not query_grid_exists:
        warnings.append("query_grid.png is missing for this task.")
        recommendations.append("Regenerate the query grid so reviewers can inspect all expanded prompts together.")
    if not visual_summary_exists:
        warnings.append("query_visual_summary.json is missing for this task.")
        recommendations.append("Regenerate the visual summary to keep expanded prompts machine-readable.")
    visual_summary_issues = _visual_summary_issues(
        visual_summary_path,
        task_dir=task_dir,
        run_dir=run_dir,
        expanded_queries=expanded_queries,
        existing_overlay_count=existing_overlay_count,
    )
    if visual_summary_issues:
        warnings.extend(visual_summary_issues)
        recommendations.append("Regenerate query_visual_summary.json so visual summaries match the query report and files.")
    if overlay_count == 0 and rendered_images:
        warnings.append("Rendered artifacts exist, but no overlay images were recorded.")
        recommendations.append("Create RGB/relevancy overlay images for qualitative review.")
    if support_level and "fallback" in support_level:
        recommendations.append("Treat this answer as fallback evidence until metric 3D points or reviewed boxes are available.")
    if counter_evidence:
        warnings.append(f"answer_summary contains {len(counter_evidence)} counter-evidence item(s).")
        recommendations.append("Review counter-evidence before using this answer as actionable scene guidance.")
    if risk_flags:
        warnings.append(f"answer_summary contains {len(risk_flags)} risk flag(s); inspect spatial conflicts before acting.")
        recommendations.append("Resolve or document risk flags before using this answer for safety-sensitive scene guidance.")

    evidence_mode = _evidence_mode(
        rendered_image_count=len(visual_rendered_images),
        bounding_region_count=len(bounding_regions),
        candidate_point_count=len(candidate_points),
        region_3d_count=region_3d_count,
    )
    status = _task_status(
        result_count=len(results),
        rendered_image_count=len(visual_rendered_images),
        overlay_count=overlay_count,
        evidence_mode=evidence_mode,
        missing_image_count=len(missing_images),
        warnings=warnings,
    )
    if status == "fail" and not recommendations:
        recommendations.append("Rerun query_scene.py or inspect backend/query logs.")

    return QueryEvidenceTask(
        task_slug=task_dir.name,
        task=str(payload.get("task") or task_dir.name),
        report_path=_relative(report_path, run_dir),
        status=status,
        evidence_mode=evidence_mode,
        support_level=support_level,
        result_count=len(results),
        expanded_queries=expanded_queries,
        rendered_image_count=len(visual_rendered_images),
        existing_rendered_image_count=len(visual_rendered_images) - len(missing_images),
        missing_rendered_image_count=len(missing_images),
        diagnostic_artifact_count=diagnostic_artifact_count,
        fallback_artifact_count=len(fallback_artifacts),
        overlay_count=overlay_count,
        relevancy_count=_kind_count(visual_rendered_images, "relevancy"),
        rgb_count=_kind_count(visual_rendered_images, "rgb"),
        bounding_region_count=len(bounding_regions),
        image_region_count=image_region_count,
        region_3d_count=region_3d_count,
        candidate_point_count=len(candidate_points),
        max_confidence=max(confidences) if confidences else None,
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        max_region_score=max(region_scores) if region_scores else None,
        counter_evidence_count=len(counter_evidence),
        counter_evidence_labels=counter_evidence_labels,
        risk_flag_count=len(risk_flags),
        risk_flags=risk_flags,
        query_grid_exists=query_grid_exists,
        visual_summary_exists=visual_summary_exists,
        visual_summary_issue_count=len(visual_summary_issues),
        warnings=_dedupe(warnings),
        recommendations=_dedupe(recommendations),
    )


def _audit_report(
    root: Path,
    query_root: Path,
    tasks: list[QueryEvidenceTask],
    warnings: list[str],
) -> QueryEvidenceAudit:
    fail_count = sum(1 for task in tasks if task.status == "fail")
    warn_count = sum(1 for task in tasks if task.status == "warn")
    pass_count = sum(1 for task in tasks if task.status == "pass")
    if not tasks:
        fail_count = 1
    status: EvidenceAuditStatus = "fail" if fail_count else "warn" if warn_count or warnings else "pass"
    return QueryEvidenceAudit(
        ok=fail_count == 0,
        status=status,
        run_dir=_display_path(root),
        queries_dir=_display_path(query_root),
        timestamp=utc_timestamp(),
        task_count=len(tasks),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        totals=_totals(tasks),
        tasks=tasks,
        warnings=_dedupe(warnings),
    )


def _totals(tasks: list[QueryEvidenceTask]) -> dict[str, Any]:
    mode_counts = {
        mode: sum(1 for task in tasks if task.evidence_mode == mode)
        for mode in ("3d", "2d_fallback", "render_only", "missing")
    }
    max_confidences = [task.max_confidence for task in tasks if task.max_confidence is not None]
    return {
        "result_count": sum(task.result_count for task in tasks),
        "rendered_image_count": sum(task.rendered_image_count for task in tasks),
        "existing_rendered_image_count": sum(task.existing_rendered_image_count for task in tasks),
        "missing_rendered_image_count": sum(task.missing_rendered_image_count for task in tasks),
        "diagnostic_artifact_count": sum(task.diagnostic_artifact_count for task in tasks),
        "fallback_artifact_count": sum(task.fallback_artifact_count for task in tasks),
        "overlay_count": sum(task.overlay_count for task in tasks),
        "relevancy_count": sum(task.relevancy_count for task in tasks),
        "rgb_count": sum(task.rgb_count for task in tasks),
        "bounding_region_count": sum(task.bounding_region_count for task in tasks),
        "candidate_point_count": sum(task.candidate_point_count for task in tasks),
        "counter_evidence_count": sum(task.counter_evidence_count for task in tasks),
        "risk_flag_count": sum(task.risk_flag_count for task in tasks),
        "tasks_with_counter_evidence": sum(1 for task in tasks if task.counter_evidence_count),
        "tasks_with_risk_flags": sum(1 for task in tasks if task.risk_flag_count),
        "query_grid_count": sum(1 for task in tasks if task.query_grid_exists),
        "visual_summary_count": sum(1 for task in tasks if task.visual_summary_exists),
        "visual_summary_issue_count": sum(task.visual_summary_issue_count for task in tasks),
        "mode_counts": mode_counts,
        "mean_task_max_confidence": (sum(max_confidences) / len(max_confidences)) if max_confidences else None,
    }


def _task_status(
    *,
    result_count: int,
    rendered_image_count: int,
    overlay_count: int,
    evidence_mode: EvidenceMode,
    missing_image_count: int,
    warnings: list[str],
) -> EvidenceAuditStatus:
    if result_count == 0 or evidence_mode == "missing":
        return "fail"
    if rendered_image_count == 0 and evidence_mode != "3d":
        return "fail"
    if overlay_count == 0 or missing_image_count or warnings or evidence_mode in {"2d_fallback", "render_only"}:
        return "warn"
    return "pass"


def _evidence_mode(
    *,
    rendered_image_count: int,
    bounding_region_count: int,
    candidate_point_count: int,
    region_3d_count: int,
) -> EvidenceMode:
    if candidate_point_count or region_3d_count:
        return "3d"
    if bounding_region_count:
        return "2d_fallback"
    if rendered_image_count:
        return "render_only"
    return "missing"


def _support_level(payload: dict[str, Any]) -> str:
    summary = payload.get("answer_summary")
    if isinstance(summary, dict):
        return str(summary.get("support_level") or "")
    return ""


def _counter_evidence_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload.get("answer_summary")
    if not isinstance(summary, dict):
        return []
    items = summary.get("counter_evidence")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _counter_evidence_labels(items: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for item in items:
        label = str(item.get("label") or item.get("query") or "").strip()
        if label:
            labels.append(label)
    return _dedupe(labels)


def _risk_flags(payload: dict[str, Any]) -> list[str]:
    summary = payload.get("answer_summary")
    if not isinstance(summary, dict):
        return []
    flags = summary.get("risk_flags")
    if not isinstance(flags, list):
        return []
    return _dedupe([str(flag) for flag in flags if str(flag).strip()])


def _kind_count(rendered_images: list[dict[str, Any]], kind: str) -> int:
    return sum(1 for image in rendered_images if str(image.get("kind") or "") == kind)


def _existing_kind_count(
    rendered_images: list[dict[str, Any]],
    kind: str,
    run_dir: Path,
    task_dir: Path,
) -> int:
    return sum(
        1
        for image in rendered_images
        if str(image.get("kind") or "") == kind
        and image.get("path")
        and _resolve_artifact_path(str(image["path"]), run_dir, task_dir).exists()
    )


def _visual_summary_issues(
    summary_path: Path,
    *,
    task_dir: Path,
    run_dir: Path,
    expanded_queries: list[str],
    existing_overlay_count: int,
) -> list[str]:
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"query_visual_summary.json is not valid JSON: {exc}"]
    if not isinstance(summary, dict):
        return ["query_visual_summary.json must contain a JSON object."]

    issues: list[str] = []
    raw_queries = summary.get("expanded_queries")
    if raw_queries is not None:
        if not isinstance(raw_queries, list):
            issues.append("query_visual_summary.json expanded_queries must be a list.")
        else:
            summary_queries = [str(query) for query in raw_queries if str(query).strip()]
            if summary_queries != expanded_queries:
                issues.append(
                    "query_visual_summary.json expanded_queries do not match scene_query_report.json "
                    f"({summary_queries} != {expanded_queries})."
                )

    raw_overlay_count = summary.get("num_overlay_images")
    if raw_overlay_count is not None:
        if isinstance(raw_overlay_count, bool) or not isinstance(raw_overlay_count, int):
            issues.append("query_visual_summary.json num_overlay_images must be an integer when provided.")
        elif raw_overlay_count != existing_overlay_count:
            issues.append(
                "query_visual_summary.json num_overlay_images does not match existing overlay files "
                f"({raw_overlay_count} != {existing_overlay_count})."
            )

    for field_name in ("scene_query_report", "query_grid", "query_montage"):
        raw_path = summary.get(field_name)
        if raw_path is None or raw_path == "":
            continue
        if not isinstance(raw_path, str):
            issues.append(f"query_visual_summary.json {field_name} must be a string path or null.")
            continue
        if Path(raw_path).is_absolute():
            issues.append(f"query_visual_summary.json {field_name} should be a relative path: {raw_path}")
        if not _resolve_artifact_path(raw_path, run_dir, task_dir).exists():
            issues.append(f"query_visual_summary.json {field_name} path does not exist: {raw_path}")
    return issues


def _is_visual_render(view: dict[str, Any]) -> bool:
    return str(view.get("kind") or "") in VISUAL_RENDER_KINDS


def _is_fallback_artifact(view: dict[str, Any]) -> bool:
    return str(view.get("kind") or "") in FALLBACK_ARTIFACT_KINDS


def _region_has_3d_evidence(region: dict[str, Any]) -> bool:
    frame = str(region.get("coordinate_frame") or "")
    return bool(
        region.get("min_point_3d")
        or region.get("max_point_3d")
        or frame in {"world", "camera"}
    )


def _resolve_artifact_path(raw_path: str, run_dir: Path, task_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [
        path,
        task_dir / path,
        run_dir / path,
        project_root() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_lines(tasks: list[QueryEvidenceTask]) -> list[str]:
    if not tasks:
        return ["| none | fail | missing | n/a | 0 | 0 | 0 | 0 | n/a | 0 | 0 | no | No query reports found. |"]
    lines: list[str] = []
    for task in tasks:
        note = "; ".join(task.warnings[:2]) if task.warnings else ""
        lines.append(
            "| {task} | {status} | {mode} | {support} | {results} | {overlays} | "
            "{regions} | {points} | {confidence} | {counter} | {risk} | {grid} | {note} |".format(
                task=_cell(task.task),
                status=task.status,
                mode=task.evidence_mode,
                support=_cell(task.support_level or "unknown"),
                results=task.result_count,
                overlays=task.overlay_count,
                regions=task.bounding_region_count,
                points=task.candidate_point_count,
                confidence=_display_float(task.max_confidence),
                counter=task.counter_evidence_count,
                risk=task.risk_flag_count,
                grid="yes" if task.query_grid_exists else "no",
                note=_cell(note or "none"),
            )
        )
    return lines


def _recommendation_lines(tasks: list[QueryEvidenceTask], warnings: list[str]) -> list[str]:
    items: list[str] = []
    items.extend(warnings)
    for task in tasks:
        for recommendation in task.recommendations:
            items.append(f"{task.task}: {recommendation}")
    deduped = _dedupe(items)
    if not deduped:
        return ["- None."]
    return [f"- {item}" for item in deduped]


def _display_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cell(value: object) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return _display_path(path)


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
