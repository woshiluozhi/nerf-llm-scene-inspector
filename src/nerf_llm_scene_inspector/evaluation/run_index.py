"""Index and compare pipeline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class RunIndexEntry:
    """Compact record for one pipeline run."""

    scene_name: str
    run_dir: str
    timestamp: str = ""
    success: bool = False
    dry_run: bool = False
    backend: str = ""
    query_count: int = 0
    audit_status: str = ""
    audit_score: int | None = None
    blocker_count: int = 0
    warning_count: int = 0
    quality_score: float | None = None
    pose_coverage_score: float | None = None
    evaluated_queries: int = 0
    top_k_hit_rate: float | None = None
    mean_iou_2d: float | None = None
    average_relevancy_score: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RunIndex:
    """Portable index over a directory of pipeline runs."""

    root: str
    generated_at: str
    total_runs: int
    successful_runs: int
    ready_runs: int
    entries: list[RunIndexEntry]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "generated_at": self.generated_at,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "ready_runs": self.ready_runs,
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
            "# Pipeline Run Index",
            "",
            f"- Root: `{self.root}`",
            f"- Generated at: `{self.generated_at}`",
            f"- Total runs: {self.total_runs}",
            f"- Successful runs: {self.successful_runs}",
            f"- Ready runs: {self.ready_runs}",
            "",
            "| Scene | Status | Audit | Score | Backend | Dry Run | Queries | Evaluated | Top-k Hit | Mean IoU | Quality | Run Dir |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for entry in self.entries:
            lines.append(
                "| {scene} | {success} | {audit} | {score} | {backend} | {dry_run} | {queries} | "
                "{evaluated} | {topk} | {iou} | {quality} | `{run_dir}` |".format(
                    scene=entry.scene_name,
                    success="success" if entry.success else "failed",
                    audit=entry.audit_status or "unknown",
                    score=_display(entry.audit_score),
                    backend=entry.backend or "unknown",
                    dry_run=entry.dry_run,
                    queries=entry.query_count,
                    evaluated=entry.evaluated_queries,
                    topk=_display_float(entry.top_k_hit_rate),
                    iou=_display_float(entry.mean_iou_2d),
                    quality=_display_float(entry.quality_score),
                    run_dir=entry.run_dir,
                )
            )
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def index_pipeline_runs(root: str | Path) -> RunIndex:
    """Build a compact index over all child directories with pipeline_summary.json."""

    root_path = Path(root)
    warnings: list[str] = []
    entries: list[RunIndexEntry] = []
    if not root_path.exists():
        warnings.append(f"Pipeline runs root does not exist: {root_path}")
    else:
        for summary_path in sorted(root_path.glob("*/pipeline_summary.json")):
            try:
                entries.append(_entry_from_run(summary_path.parent, root_path))
            except Exception as exc:
                warnings.append(f"Could not index {summary_path.parent}: {exc}")
    entries.sort(key=lambda entry: entry.timestamp or "", reverse=True)
    return RunIndex(
        root=_display_path(root_path),
        generated_at=utc_timestamp(),
        total_runs=len(entries),
        successful_runs=sum(1 for entry in entries if entry.success),
        ready_runs=sum(1 for entry in entries if entry.audit_status == "ready"),
        entries=entries,
        warnings=warnings,
    )


def _entry_from_run(run_dir: Path, root: Path) -> RunIndexEntry:
    summary = _read_json(run_dir / "pipeline_summary.json")
    audit = _read_json(run_dir / "run_audit.json")
    scene = _read_json(run_dir / "scene_data_inspection.json")
    evaluation = _read_json(run_dir / "evaluation" / "eval_summary.json")
    scene_name = str(summary.get("scene_name") or run_dir.name)
    return RunIndexEntry(
        scene_name=scene_name,
        run_dir=_relative_to_root(run_dir, root),
        timestamp=str(summary.get("timestamp") or ""),
        success=bool(summary.get("success")),
        dry_run=bool(summary.get("dry_run")),
        backend=str(summary.get("backend") or ""),
        query_count=len(summary.get("queries") or []),
        audit_status=str(audit.get("status") or ""),
        audit_score=_optional_int(audit.get("score")),
        blocker_count=_safe_int(audit.get("blocker_count")),
        warning_count=_safe_int(audit.get("warning_count")),
        quality_score=_optional_float(scene.get("quality_score")),
        pose_coverage_score=_optional_float(scene.get("pose_coverage_score")),
        evaluated_queries=_safe_int(evaluation.get("num_evaluated_queries")),
        top_k_hit_rate=_optional_float(evaluation.get("top_k_hit_rate")),
        mean_iou_2d=_optional_float(evaluation.get("mean_iou_2d")),
        average_relevancy_score=_optional_float(evaluation.get("average_relevancy_score")),
        artifacts=_artifacts(run_dir),
    )


def _artifacts(run_dir: Path) -> dict[str, str]:
    candidates = {
        "pipeline_summary": "pipeline_summary.json",
        "run_audit": "run_audit.md",
        "project_report": "project_report.md",
        "portfolio_card": "portfolio_result_card.md",
        "query_grid": "demo_assets/query_grid.png",
        "evaluation_summary": "evaluation/eval_summary.json",
        "annotation_validation": "evaluation/annotation_validation.json",
    }
    return {name: relative for name, relative in candidates.items() if (run_dir / relative).exists()}


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


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _safe_int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display(value: object) -> str:
    return "n/a" if value is None else str(value)


def _display_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
