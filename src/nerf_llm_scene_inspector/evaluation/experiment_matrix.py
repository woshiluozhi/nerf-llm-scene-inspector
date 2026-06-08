"""Experiment matrix runner and summarizer for repeated scene-inspection runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.config import load_mapping
from nerf_llm_scene_inspector.pipeline import PipelineConfig, run_scene_pipeline
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp


@dataclass
class ExperimentMatrixEntry:
    """One experiment row in a matrix report."""

    experiment_name: str
    scene_name: str
    run_dir: str
    success: bool
    dry_run: bool
    backend: str
    variant: str = ""
    query_count: int = 0
    evaluated_queries: int = 0
    evidence_level: str = ""
    evidence_score: int | None = None
    evidence_max_score: int | None = None
    audit_status: str = ""
    quality_gate_status: str = ""
    top_k_hit_rate: float | None = None
    mean_iou_2d: float | None = None
    average_relevancy_score: float | None = None
    prompt_stable_group_count: int | None = None
    prompt_group_count: int | None = None
    relation_entity_count: int | None = None
    relation_edge_count: int | None = None
    portfolio_score: float = 0.0
    top_next_action: str = ""
    timestamp: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentMatrixReport:
    """Comparison report for a configured set of experiments."""

    matrix_name: str
    output_dir: str
    generated_at: str
    collect_only: bool
    entries: list[ExperimentMatrixEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_experiments(self) -> int:
        return len(self.entries)

    @property
    def successful_experiments(self) -> int:
        return sum(1 for entry in self.entries if entry.success)

    @property
    def best_experiment(self) -> dict[str, Any] | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda entry: entry.portfolio_score).to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_name": self.matrix_name,
            "output_dir": self.output_dir,
            "generated_at": self.generated_at,
            "collect_only": self.collect_only,
            "total_experiments": self.total_experiments,
            "successful_experiments": self.successful_experiments,
            "best_experiment": self.best_experiment,
            "entries": [entry.to_dict() for entry in self.entries],
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(ExperimentMatrixEntry.__dataclass_fields__.keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry.to_dict())
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Experiment Matrix Report",
            "",
            f"- Matrix: `{self.matrix_name}`",
            f"- Output: `{self.output_dir}`",
            f"- Generated: `{self.generated_at}`",
            f"- Collect only: {self.collect_only}",
            f"- Experiments: {self.total_experiments}",
            f"- Successful: {self.successful_experiments}",
            "",
            "## Best Experiment",
            "",
            *_best_lines(self.best_experiment),
            "",
            "## Matrix Table",
            "",
            (
                "| Experiment | Scene | Backend | Mode | Score | Evidence | Audit | "
                "Queries | Evaluated | Top-k | IoU | Prompt Stability | Relation Edges | Run |"
            ),
            "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
            *_entry_lines(self.entries),
            "",
            "## Warnings",
            "",
            *_list_lines(self.warnings),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def run_experiment_matrix(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool | None = None,
    collect_only: bool = False,
    limit: int | None = None,
) -> ExperimentMatrixReport:
    """Run or collect a configured experiment matrix."""

    config_file = Path(config_path)
    raw = load_mapping(config_file)
    matrix_name = str(raw.get("matrix_name") or config_file.stem)
    output = Path(output_dir or raw.get("output_dir") or Path("results") / "experiment_matrix" / matrix_name)
    output.mkdir(parents=True, exist_ok=True)
    base = dict(raw.get("base") or {})
    experiments = _experiment_specs(raw)
    if limit is not None:
        experiments = experiments[:limit]
    warnings: list[str] = []

    if not collect_only:
        for experiment in experiments:
            merged = _merge_dicts(base, experiment)
            if dry_run is not None:
                merged["dry_run"] = dry_run
            try:
                run_scene_pipeline(_pipeline_config_from_mapping(merged, output, matrix_name))
            except Exception as exc:
                warnings.append(f"Experiment {experiment.get('name', 'unnamed')} failed: {exc}")

    entries = []
    for experiment in experiments:
        merged = _merge_dicts(base, experiment)
        entries.append(_entry_from_run(_run_dir_from_mapping(merged, output), merged, output))
    report = ExperimentMatrixReport(
        matrix_name=matrix_name,
        output_dir=str(output),
        generated_at=utc_timestamp(),
        collect_only=collect_only,
        entries=entries,
        warnings=warnings,
    )
    report.to_json(output / "experiment_matrix_summary.json")
    report.to_csv(output / "experiment_matrix_table.csv")
    report.to_markdown(output / "experiment_matrix_report.md")
    return report


def _pipeline_config_from_mapping(raw: dict[str, Any], output: Path, matrix_name: str) -> PipelineConfig:
    name = str(raw.get("name") or "experiment")
    scene_name = str(raw.get("scene_name") or slugify(name))
    queries = _queries_from_mapping(raw)
    return PipelineConfig(
        input_path=raw.get("input") or raw.get("input_path") or "examples",
        scene_name=scene_name,
        data_type=str(raw.get("type") or raw.get("data_type") or "images"),
        backend=str(raw.get("backend") or "lerf"),
        variant=str(raw.get("variant") or "lerf-lite"),
        baseline_method=str(raw.get("baseline_method") or "nerfacto"),
        queries=queries,
        data_root=raw.get("data_root") or output / "data",
        runs_root=raw.get("runs_root") or output / "training_runs",
        output_root=raw.get("output_root") or output / "pipeline_runs",
        annotations_path=raw.get("annotations") or raw.get("annotations_path") or "examples/annotations_example.json",
        prompt_suite_path=raw.get("prompt_suite") or raw.get("prompt_suite_path"),
        capture_manifest_path=raw.get("capture_manifest") or raw.get("capture_manifest_path"),
        config_path=raw.get("config") or raw.get("config_path"),
        max_num_iterations=_optional_int(raw.get("max_num_iterations")),
        num_views=int(raw.get("num_views") or 1),
        top_k=int(raw.get("top_k") or 5),
        min_frames=int(raw.get("min_frames") or 20),
        min_pose_extent=float(raw.get("min_pose_extent") or 0.05),
        dry_run=bool(raw.get("dry_run", True)),
        analyze_relations=bool(raw.get("analyze_relations", False)),
        strict=bool(raw.get("strict", False)),
        skip_prepare=bool(raw.get("skip_prepare", False)),
        skip_baseline=bool(raw.get("skip_baseline", False)),
        skip_language=bool(raw.get("skip_language", False)),
        skip_queries=bool(raw.get("skip_queries", False)),
        skip_demo=bool(raw.get("skip_demo", False)),
        skip_eval=bool(raw.get("skip_eval", False)),
        clean_run_outputs=not bool(raw.get("no_clean_run", False)),
        command=[
            "scripts/run_experiment_matrix.py",
            "--matrix-name",
            matrix_name,
            "--experiment",
            name,
        ],
    )


def _run_dir_from_mapping(raw: dict[str, Any], output: Path) -> Path:
    name = str(raw.get("name") or "experiment")
    scene_name = str(raw.get("scene_name") or slugify(name))
    output_root = Path(raw.get("output_root") or output / "pipeline_runs")
    return output_root / scene_name


def _entry_from_run(run_dir: Path, raw: dict[str, Any], output: Path) -> ExperimentMatrixEntry:
    summary = _read_json(run_dir / "pipeline_summary.json")
    scorecard = _read_json(run_dir / "evidence_scorecard.json")
    audit = _read_json(run_dir / "run_audit.json")
    quality_gate = _read_json(run_dir / "quality_gate.json")
    evaluation = _read_json(run_dir / "evaluation" / "eval_summary.json")
    prompt = _read_json(run_dir / "prompt_sensitivity" / "prompt_sensitivity_summary.json")
    relations = _read_json(run_dir / "scene_relations" / "scene_relations_summary.json")
    recommendations = _read_json(run_dir / "run_recommendations.json")
    success = bool(summary.get("success"))
    return ExperimentMatrixEntry(
        experiment_name=str(raw.get("name") or run_dir.name),
        scene_name=str(summary.get("scene_name") or raw.get("scene_name") or run_dir.name),
        run_dir=_relative(run_dir, output),
        success=success,
        dry_run=bool(summary.get("dry_run", raw.get("dry_run", True))),
        backend=str(summary.get("backend") or raw.get("backend") or ""),
        variant=str(raw.get("variant") or ""),
        query_count=len(summary.get("queries") or raw.get("queries") or []),
        evaluated_queries=_safe_int(evaluation.get("num_evaluated_queries")),
        evidence_level=str(scorecard.get("evidence_level") or ""),
        evidence_score=_optional_int(scorecard.get("score")),
        evidence_max_score=_optional_int(scorecard.get("max_score")),
        audit_status=str(audit.get("status") or ""),
        quality_gate_status=str(quality_gate.get("status") or ""),
        top_k_hit_rate=_optional_float(evaluation.get("top_k_hit_rate")),
        mean_iou_2d=_optional_float(evaluation.get("mean_iou_2d")),
        average_relevancy_score=_optional_float(evaluation.get("average_relevancy_score")),
        prompt_stable_group_count=_optional_int(prompt.get("stable_group_count")),
        prompt_group_count=_optional_int(prompt.get("num_groups")),
        relation_entity_count=_optional_int(relations.get("num_entities")),
        relation_edge_count=_optional_int(relations.get("num_relations")),
        portfolio_score=_portfolio_score(scorecard, evaluation, prompt, relations, success),
        top_next_action=str(recommendations.get("top_next_action") or ""),
        timestamp=str(summary.get("timestamp") or ""),
        error=_failed_step_errors(summary),
    )


def _experiment_specs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = raw.get("experiments") or []
    if not isinstance(experiments, list):
        raise ValueError("experiment matrix config must contain a list field named 'experiments'.")
    specs = [dict(item) for item in experiments if isinstance(item, dict)]
    if not specs:
        raise ValueError("experiment matrix config does not define any experiments.")
    return specs


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _queries_from_mapping(raw: dict[str, Any]) -> list[str]:
    queries = [str(item) for item in raw.get("queries") or [] if str(item).strip()]
    queries_file = raw.get("queries_file")
    if queries_file:
        loaded = load_mapping(queries_file)
        for key in ("queries", "tasks"):
            queries.extend(str(item) for item in loaded.get(key) or [] if str(item).strip())
    if not queries:
        queries = ["mug", "objects that can hold water", "safe place to put a hot cup"]
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        key = query.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(query.strip())
    return deduped


def _portfolio_score(
    scorecard: dict[str, Any],
    evaluation: dict[str, Any],
    prompt: dict[str, Any],
    relations: dict[str, Any],
    success: bool,
) -> float:
    if not success:
        return 0.0
    score = 0.0
    evidence_score = _optional_float(scorecard.get("score"))
    evidence_max = _optional_float(scorecard.get("max_score"))
    if evidence_score is not None and evidence_max:
        score += 55.0 * min(1.0, evidence_score / evidence_max)
    score += 15.0 * min(1.0, max(0.0, _optional_float(evaluation.get("top_k_hit_rate")) or 0.0))
    score += 10.0 * min(1.0, max(0.0, _optional_float(evaluation.get("average_relevancy_score")) or 0.0))
    prompt_groups = _safe_int(prompt.get("num_groups"))
    stable_groups = _safe_int(prompt.get("stable_group_count"))
    if prompt_groups:
        score += 10.0 * min(1.0, stable_groups / prompt_groups)
    if _safe_int(relations.get("num_relations")):
        score += 10.0
    return round(score, 3)


def _failed_step_errors(summary: dict[str, Any]) -> str:
    errors: list[str] = []
    for step in summary.get("steps") or []:
        if isinstance(step, dict) and step.get("status") == "failed":
            errors.append(f"{step.get('name')}: {step.get('error') or 'failed'}")
    return "; ".join(errors)


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


def _best_lines(best: dict[str, Any] | None) -> list[str]:
    if not best:
        return ["- No experiments were summarized."]
    return [
        f"- Experiment: `{best.get('experiment_name')}`",
        f"- Scene: `{best.get('scene_name')}`",
        f"- Score: {best.get('portfolio_score')}",
        f"- Evidence: `{best.get('evidence_level') or 'unknown'}`",
        f"- Run: `{best.get('run_dir')}`",
    ]


def _entry_lines(entries: list[ExperimentMatrixEntry]) -> list[str]:
    if not entries:
        return ["| None |  |  |  |  |  |  |  |  |  |  |  |  |  |"]
    lines: list[str] = []
    for entry in entries:
        stability = _ratio(entry.prompt_stable_group_count, entry.prompt_group_count)
        lines.append(
            "| {name} | {scene} | {backend} | {mode} | {score:.1f} | {evidence} | {audit} | "
            "{queries} | {evaluated} | {topk} | {iou} | {stability} | {edges} | `{run}` |".format(
                name=_cell(entry.experiment_name),
                scene=_cell(entry.scene_name),
                backend=_cell(entry.backend or "unknown"),
                mode="dry-run" if entry.dry_run else "real",
                score=entry.portfolio_score,
                evidence=_cell(entry.evidence_level or "unknown"),
                audit=_cell(entry.audit_status or "unknown"),
                queries=entry.query_count,
                evaluated=entry.evaluated_queries,
                topk=_display_float(entry.top_k_hit_rate),
                iou=_display_float(entry.mean_iou_2d),
                stability=stability,
                edges=entry.relation_edge_count if entry.relation_edge_count is not None else "",
                run=_cell(entry.run_dir),
            )
        )
    return lines


def _ratio(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator in {None, 0}:
        return ""
    return f"{numerator}/{denominator}"


def _display_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]
