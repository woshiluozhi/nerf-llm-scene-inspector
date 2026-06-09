"""Compare multiple pipeline runs and rank portfolio candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import utc_timestamp

SelectionStatus = Literal[
    "portfolio_candidate",
    "needs_review",
    "needs_evidence",
    "dry_run_smoke_demo",
    "blocked",
]


@dataclass
class RunComparisonEntry:
    """Ranked comparison record for one pipeline run."""

    rank: int
    run_dir: str
    scene_name: str
    selection_status: SelectionStatus
    portfolio_score: float
    success: bool
    dry_run: bool
    backend: str
    timestamp: str = ""
    evidence_level: str = ""
    evidence_score: int | None = None
    evidence_max_score: int | None = None
    audit_status: str = ""
    audit_blocker_count: int = 0
    audit_score: int | None = None
    capture_manifest_status: str = ""
    capture_manifest_fail_count: int = 0
    result_status: str = ""
    submission_readiness_level: str = ""
    query_evidence_status: str = ""
    query_counter_evidence_count: int = 0
    query_risk_flag_count: int = 0
    scene_quality_score: float | None = None
    pose_coverage_score: float | None = None
    query_count: int = 0
    evaluated_queries: int = 0
    top_k_hit_rate: float | None = None
    mean_iou_2d: float | None = None
    average_relevancy_score: float | None = None
    top_next_action: str = ""
    portfolio_page: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunComparison:
    """Comparison report for all pipeline runs in one root directory."""

    root: str
    generated_at: str
    total_runs: int
    real_run_count: int
    dry_run_count: int
    portfolio_candidate_count: int
    best_run: dict[str, object] | None
    entries: list[RunComparisonEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "real_run_count": self.real_run_count,
            "dry_run_count": self.dry_run_count,
            "portfolio_candidate_count": self.portfolio_candidate_count,
            "best_run": self.best_run,
            "entries": [entry.to_dict() for entry in self.entries],
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
            "# Pipeline Run Comparison",
            "",
            f"- Root: `{self.root}`",
            f"- Generated at: `{self.generated_at}`",
            f"- Total runs: {self.total_runs}",
            f"- Real runs: {self.real_run_count}",
            f"- Dry-run smoke demos: {self.dry_run_count}",
            f"- Portfolio candidates: {self.portfolio_candidate_count}",
            "",
            "## Best Candidate",
            "",
            *_best_run_lines(self.best_run),
            "",
            "## Ranked Runs",
            "",
            (
                "| Rank | Scene | Status | Result | Submission | Mode | Score | Evidence | Audit | Audit Blockers | "
                "Capture | Capture Fails | Query Evidence | Risk Flags | Queries | Evaluated | Top-k | IoU | "
                "Quality | Next Action | Run Dir |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for entry in self.entries:
            lines.append(
                "| {rank} | {scene} | {status} | {result} | {submission} | {mode} | {score} | "
                "{evidence} | {audit} | {audit_blockers} | {capture} | {capture_fails} | {query_evidence} | "
                "{risk_flags} | {queries} | {evaluated} | {topk} | {iou} | {quality} | {action} | `{run_dir}` |".format(
                    rank=entry.rank,
                    scene=_cell(entry.scene_name),
                    status=entry.selection_status,
                    result=_cell(entry.result_status or "unknown"),
                    submission=_cell(entry.submission_readiness_level or "unknown"),
                    mode="dry-run" if entry.dry_run else "real",
                    score=f"{entry.portfolio_score:.1f}",
                    evidence=_cell(entry.evidence_level or "unknown"),
                    audit=_cell(entry.audit_status or "unknown"),
                    audit_blockers=entry.audit_blocker_count,
                    capture=_cell(entry.capture_manifest_status or "unknown"),
                    capture_fails=entry.capture_manifest_fail_count,
                    query_evidence=_cell(entry.query_evidence_status or "unknown"),
                    risk_flags=entry.query_risk_flag_count,
                    queries=entry.query_count,
                    evaluated=entry.evaluated_queries,
                    topk=_display_float(entry.top_k_hit_rate),
                    iou=_display_float(entry.mean_iou_2d),
                    quality=_display_float(entry.scene_quality_score),
                    action=_cell(_short(entry.top_next_action)),
                    run_dir=_cell(entry.run_dir),
                )
            )
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def compare_pipeline_runs(root: str | Path) -> RunComparison:
    """Rank run directories by shareability and evidence quality."""

    root_path = Path(root)
    warnings: list[str] = []
    raw_entries: list[RunComparisonEntry] = []
    if not root_path.exists():
        warnings.append(f"Pipeline runs root does not exist: {root_path}")
    else:
        for summary_path in sorted(root_path.glob("*/pipeline_summary.json")):
            try:
                raw_entries.append(_entry_from_run(summary_path.parent, root_path))
            except Exception as exc:
                warnings.append(f"Could not compare {summary_path.parent}: {exc}")

    raw_entries.sort(key=_sort_key, reverse=True)
    entries = [
        RunComparisonEntry(**{**entry.to_dict(), "rank": index})
        for index, entry in enumerate(raw_entries, start=1)
    ]
    portfolio_candidates = [
        entry for entry in entries if entry.selection_status == "portfolio_candidate"
    ]
    best = entries[0].to_dict() if entries else None
    return RunComparison(
        root=_display_path(root_path),
        generated_at=utc_timestamp(),
        total_runs=len(entries),
        real_run_count=sum(1 for entry in entries if not entry.dry_run),
        dry_run_count=sum(1 for entry in entries if entry.dry_run),
        portfolio_candidate_count=len(portfolio_candidates),
        best_run=best,
        entries=entries,
        warnings=warnings,
    )


def _entry_from_run(run_dir: Path, root: Path) -> RunComparisonEntry:
    summary = _read_json(run_dir / "pipeline_summary.json")
    audit = _read_json(run_dir / "run_audit.json")
    scorecard = _read_json(run_dir / "evidence_scorecard.json")
    capture = _read_json(run_dir / "capture_manifest_validation.json")
    query_evidence = _read_json(run_dir / "query_evidence_audit.json")
    result_card = _read_json(run_dir / "run_result_card.json")
    submission = _read_json(run_dir / "submission_packet" / "submission_packet.json")
    recommendations = _read_json(run_dir / "run_recommendations.json")
    scene = _read_json(run_dir / "scene_data_inspection.json")
    evaluation = _read_json(run_dir / "evaluation" / "eval_summary.json")

    success = bool(summary.get("success"))
    dry_run = bool(summary.get("dry_run"))
    evidence_level = str(scorecard.get("evidence_level") or "")
    audit_status = str(audit.get("status") or "")
    audit_blocker_count = _safe_int(audit.get("blocker_count"))
    capture_status = str(capture.get("status") or "")
    capture_fail_count = _safe_int(capture.get("fail_count"))
    query_evidence_status = str(query_evidence.get("status") or "")
    result_status = str(result_card.get("result_status") or "")
    submission_readiness_level = str(submission.get("readiness_level") or "")
    counter_evidence_count, risk_flag_count = _query_evidence_counts(query_evidence)
    selection_status = _selection_status(
        success=success,
        dry_run=dry_run,
        evidence_level=evidence_level,
        audit_status=audit_status,
        audit_blocker_count=audit_blocker_count,
        capture_status=capture_status,
        capture_fail_count=capture_fail_count,
        result_status=result_status,
        submission_readiness_level=submission_readiness_level,
        query_evidence_status=query_evidence_status,
        query_counter_evidence_count=counter_evidence_count,
        query_risk_flag_count=risk_flag_count,
    )
    return RunComparisonEntry(
        rank=0,
        run_dir=_relative_to_root(run_dir, root),
        scene_name=str(summary.get("scene_name") or scorecard.get("scene_name") or run_dir.name),
        selection_status=selection_status,
        portfolio_score=_portfolio_score(
            success=success,
            dry_run=dry_run,
            selection_status=selection_status,
            evidence_score=_optional_int(scorecard.get("score")),
            evidence_max_score=_optional_int(scorecard.get("max_score")),
            audit_score=_optional_int(audit.get("score")),
            capture_status=capture_status,
            result_status=result_status,
            submission_readiness_level=submission_readiness_level,
            query_evidence_status=query_evidence_status,
            query_counter_evidence_count=counter_evidence_count,
            query_risk_flag_count=risk_flag_count,
            quality_score=_optional_float(scene.get("quality_score")),
            pose_coverage_score=_optional_float(scene.get("pose_coverage_score")),
            top_k_hit_rate=_optional_float(evaluation.get("top_k_hit_rate")),
            mean_iou_2d=_optional_float(evaluation.get("mean_iou_2d")),
            average_relevancy_score=_optional_float(evaluation.get("average_relevancy_score")),
            query_count=len(summary.get("queries") or []),
            evaluated_queries=_safe_int(evaluation.get("num_evaluated_queries")),
        ),
        success=success,
        dry_run=dry_run,
        backend=str(summary.get("backend") or scorecard.get("backend") or ""),
        timestamp=str(summary.get("timestamp") or ""),
        evidence_level=evidence_level,
        evidence_score=_optional_int(scorecard.get("score")),
        evidence_max_score=_optional_int(scorecard.get("max_score")),
        audit_status=audit_status,
        audit_blocker_count=audit_blocker_count,
        audit_score=_optional_int(audit.get("score")),
        capture_manifest_status=capture_status,
        capture_manifest_fail_count=capture_fail_count,
        result_status=result_status,
        submission_readiness_level=submission_readiness_level,
        query_evidence_status=query_evidence_status,
        query_counter_evidence_count=counter_evidence_count,
        query_risk_flag_count=risk_flag_count,
        scene_quality_score=_optional_float(scene.get("quality_score")),
        pose_coverage_score=_optional_float(scene.get("pose_coverage_score")),
        query_count=len(summary.get("queries") or []),
        evaluated_queries=_safe_int(evaluation.get("num_evaluated_queries")),
        top_k_hit_rate=_optional_float(evaluation.get("top_k_hit_rate")),
        mean_iou_2d=_optional_float(evaluation.get("mean_iou_2d")),
        average_relevancy_score=_optional_float(evaluation.get("average_relevancy_score")),
        top_next_action=str(recommendations.get("top_next_action") or ""),
        portfolio_page="portfolio_page.html" if (run_dir / "portfolio_page.html").exists() else "",
    )


def _selection_status(
    *,
    success: bool,
    dry_run: bool,
    evidence_level: str,
    audit_status: str,
    audit_blocker_count: int,
    capture_status: str,
    capture_fail_count: int,
    result_status: str,
    submission_readiness_level: str,
    query_evidence_status: str,
    query_counter_evidence_count: int,
    query_risk_flag_count: int,
) -> SelectionStatus:
    if (
        not success
        or evidence_level == "blocked"
        or audit_status == "blocked"
        or audit_blocker_count
        or capture_status == "blocked"
        or capture_fail_count
        or result_status == "blocked"
        or submission_readiness_level == "blocked"
        or query_evidence_status == "fail"
    ):
        return "blocked"
    if dry_run:
        return "dry_run_smoke_demo"
    if query_risk_flag_count or query_counter_evidence_count or query_evidence_status == "warn":
        return "needs_review"
    if (
        evidence_level == "portfolio_ready_real_run"
        and audit_status == "ready"
        and capture_status == "ready"
        and result_status == "portfolio_ready"
        and submission_readiness_level == "portfolio_ready"
        and query_evidence_status == "pass"
    ):
        return "portfolio_candidate"
    if result_status in {"real_run_review_ready", "needs_evidence"} or submission_readiness_level:
        return "needs_review"
    if evidence_level in {"needs_review", "dry_run_demo_ready"} or audit_status == "needs_review":
        return "needs_review"
    return "needs_evidence"


def _portfolio_score(
    *,
    success: bool,
    dry_run: bool,
    selection_status: SelectionStatus,
    evidence_score: int | None,
    evidence_max_score: int | None,
    audit_score: int | None,
    capture_status: str,
    result_status: str,
    submission_readiness_level: str,
    query_evidence_status: str,
    query_counter_evidence_count: int,
    query_risk_flag_count: int,
    quality_score: float | None,
    pose_coverage_score: float | None,
    top_k_hit_rate: float | None,
    mean_iou_2d: float | None,
    average_relevancy_score: float | None,
    query_count: int,
    evaluated_queries: int,
) -> float:
    score = 0.0
    if success:
        score += 6.0
    score += 40.0 * _ratio(evidence_score, evidence_max_score)
    score += 16.0 * _ratio(audit_score, 100)
    score += 10.0 * _bounded(quality_score)
    score += 8.0 * _bounded(pose_coverage_score)
    score += 7.0 * _bounded(top_k_hit_rate)
    score += 5.0 * _bounded(mean_iou_2d)
    score += 3.0 * _bounded(average_relevancy_score)
    score += 3.0 * min(max(query_count, 0), 3) / 3.0
    score += 2.0 * min(max(evaluated_queries, 0), 3) / 3.0
    if capture_status == "ready":
        score += 5.0
    elif capture_status == "needs_review":
        score += 2.0
    if result_status == "portfolio_ready":
        score += 4.0
    elif result_status == "blocked":
        score -= 20.0
    elif result_status:
        score -= 3.0
    else:
        score -= 4.0
    if submission_readiness_level == "portfolio_ready":
        score += 4.0
    elif submission_readiness_level == "blocked":
        score -= 20.0
    elif submission_readiness_level:
        score -= 4.0
    else:
        score -= 5.0
    if query_evidence_status == "pass":
        score += 4.0
    elif query_evidence_status == "warn":
        score -= 4.0
    elif query_evidence_status == "fail":
        score -= 12.0
    score -= min(10.0, 2.0 * max(query_counter_evidence_count, 0))
    score -= min(25.0, 6.0 * max(query_risk_flag_count, 0))
    if dry_run:
        score = min(score, 65.0)
    if selection_status == "blocked":
        score = min(score, 25.0)
    return round(max(0.0, min(score, 100.0)), 1)


def _sort_key(entry: RunComparisonEntry) -> tuple[int, float, str]:
    tier = {
        "portfolio_candidate": 4,
        "needs_review": 3,
        "needs_evidence": 2,
        "dry_run_smoke_demo": 1,
        "blocked": 0,
    }[entry.selection_status]
    return (tier, entry.portfolio_score, entry.timestamp)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _safe_int(value)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _query_evidence_counts(audit: dict[str, Any]) -> tuple[int, int]:
    totals = audit.get("totals") if isinstance(audit.get("totals"), dict) else {}
    counter = _safe_int(totals.get("counter_evidence_count"))
    risk = _safe_int(totals.get("risk_flag_count"))
    tasks = audit.get("tasks") if isinstance(audit.get("tasks"), list) else []
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


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(value: int | None, total: int | None) -> float:
    if value is None or total is None or total <= 0:
        return 0.0
    return _bounded(float(value) / float(total))


def _bounded(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _best_run_lines(best_run: dict[str, object] | None) -> list[str]:
    if not best_run:
        return ["- No pipeline runs were found."]
    status = str(best_run.get("selection_status") or "unknown")
    lines = [
        f"- Run: `{best_run.get('run_dir')}`",
        f"- Scene: {best_run.get('scene_name')}",
        f"- Status: {status}",
        f"- Score: {best_run.get('portfolio_score')}/100",
        f"- Evidence: {best_run.get('evidence_level') or 'unknown'}",
        f"- Result status: {best_run.get('result_status') or 'unknown'}",
        f"- Submission readiness: {best_run.get('submission_readiness_level') or 'unknown'}",
        f"- Query evidence: {best_run.get('query_evidence_status') or 'unknown'}",
        f"- Query risk flags: {best_run.get('query_risk_flag_count') or 0}",
    ]
    action = str(best_run.get("top_next_action") or "")
    if action:
        lines.append(f"- Next action: {action}")
    if status == "dry_run_smoke_demo":
        lines.append("- Interpretation: useful for workflow demonstration only, not real trained-scene evidence.")
    elif status == "portfolio_candidate":
        lines.append("- Interpretation: strongest current real-run candidate, pending qualitative review.")
    elif status == "needs_review":
        lines.append("- Interpretation: structurally useful, but review warnings before sharing.")
    return lines


def _cell(value: object) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip() or "n/a"


def _short(value: str, limit: int = 96) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _display_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
