"""Research-style report generation from run and matrix artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


@dataclass
class ResearchReport:
    """Structured payload for a portfolio/research report."""

    title: str
    run_dir: str
    scene_name: str
    backend: str
    dry_run: bool
    generated_at: str
    abstract: str
    key_results: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {self.title}",
            "",
            "## Abstract",
            "",
            self.abstract,
            "",
            "## Research Positioning",
            "",
            "- Built on Nerfstudio and LERF-style language fields; this report does not claim a new NeRF architecture.",
            "- Focuses on reproducible open-vocabulary 3D scene querying, qualitative evidence, and lightweight annotation-based evaluation.",
            "- Dry-run artifacts validate pipeline wiring only; real claims require a CUDA-backed trained run and reviewed annotations.",
            "",
            "## Run Snapshot",
            "",
            f"- Scene: `{self.scene_name}`",
            f"- Backend: `{self.backend}`",
            f"- Dry run: {self.dry_run}",
            f"- Run directory: `{self.run_dir}`",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Key Results",
            "",
            *_metric_lines(self.key_results),
            "",
            "## Evidence Summary",
            "",
            *_evidence_lines(self.evidence),
            "",
            "## Reproducibility Artifacts",
            "",
            *_artifact_lines(self.artifacts),
            "",
            "## Limitations",
            "",
            *_list_lines(self.limitations),
            "",
            "## Next Steps",
            "",
            *_list_lines(self.next_steps),
            "",
            "## Warnings",
            "",
            *_list_lines(self.warnings),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_research_report(
    run_dir: str | Path,
    *,
    matrix_summary_path: str | Path | None = None,
    title: str = "NeRF-LLM Scene Inspector Research Report",
) -> ResearchReport:
    """Build a paper-style report from a pipeline run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scorecard = _read_json(root / "evidence_scorecard.json")
    audit = _read_json(root / "run_audit.json")
    recommendations = _read_json(root / "run_recommendations.json")
    quality_gate = _read_json(root / "quality_gate.json")
    evaluation = _read_json(root / "evaluation" / "eval_summary.json")
    prompt = _read_json(root / "prompt_sensitivity" / "prompt_sensitivity_summary.json")
    relations = _read_json(root / "scene_relations" / "scene_relations_summary.json")
    matrix = _read_json(matrix_summary_path) if matrix_summary_path else {}
    scene_name = str(summary.get("scene_name") or scorecard.get("scene_name") or root.name)
    backend = str(summary.get("backend") or scorecard.get("backend") or "unknown")
    dry_run = bool(summary.get("dry_run", scorecard.get("dry_run", False)))
    key_results = _key_results(summary, scorecard, audit, quality_gate, evaluation, prompt, relations, matrix)
    evidence = _evidence(summary, scorecard, evaluation, prompt, relations, matrix)
    limitations = _limitations(dry_run, evaluation, prompt, relations)
    next_steps = _next_steps(recommendations, dry_run)
    warnings = _warnings(audit, quality_gate, recommendations)
    return ResearchReport(
        title=title,
        run_dir=_display_path(root),
        scene_name=scene_name,
        backend=backend,
        dry_run=dry_run,
        generated_at=utc_timestamp(),
        abstract=_abstract(scene_name, backend, dry_run, key_results),
        key_results=key_results,
        evidence=evidence,
        artifacts=_artifacts(root, matrix_summary_path),
        limitations=limitations,
        next_steps=next_steps,
        warnings=warnings,
    )


def write_research_report(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    json_output: str | Path | None = None,
    matrix_summary_path: str | Path | None = None,
    title: str = "NeRF-LLM Scene Inspector Research Report",
) -> ResearchReport:
    """Build and write Markdown plus JSON research report artifacts."""

    root = Path(run_dir)
    report = build_research_report(root, matrix_summary_path=matrix_summary_path, title=title)
    report.to_markdown(output or root / "research_report.md")
    report.to_json(json_output or root / "research_report.json")
    return report


def _key_results(
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    audit: dict[str, Any],
    quality_gate: dict[str, Any],
    evaluation: dict[str, Any],
    prompt: dict[str, Any],
    relations: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    results = {
        "pipeline_success": summary.get("success"),
        "evidence_level": scorecard.get("evidence_level"),
        "evidence_score": _score_text(scorecard.get("score"), scorecard.get("max_score")),
        "audit_status": audit.get("status"),
        "quality_gate_status": quality_gate.get("status"),
        "query_count": len(summary.get("queries") or []),
        "evaluated_queries": evaluation.get("num_evaluated_queries"),
        "top_k_hit_rate": evaluation.get("top_k_hit_rate"),
        "mean_iou_2d": evaluation.get("mean_iou_2d"),
        "average_relevancy_score": evaluation.get("average_relevancy_score"),
    }
    if prompt:
        results["prompt_stability"] = _score_text(
            prompt.get("stable_group_count"),
            prompt.get("num_groups"),
        )
    if relations:
        results["scene_relation_entities"] = relations.get("num_entities")
        results["scene_relation_edges"] = relations.get("num_relations")
    if matrix:
        results["experiment_matrix_name"] = matrix.get("matrix_name")
        results["matrix_successful_experiments"] = _score_text(
            matrix.get("successful_experiments"),
            matrix.get("total_experiments"),
        )
    return results


def _evidence(
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    evaluation: dict[str, Any],
    prompt: dict[str, Any],
    relations: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    return {
        "queries": summary.get("queries") or [],
        "scorecard_summary": scorecard.get("summary", ""),
        "evaluation_metrics_present": bool(evaluation),
        "prompt_sensitivity_present": bool(prompt),
        "scene_relations_present": bool(relations),
        "experiment_matrix_present": bool(matrix),
        "bbox_annotated_queries": evaluation.get("num_bbox_annotated_queries"),
        "qualitative_only_queries": evaluation.get("num_qualitative_only_queries"),
    }


def _artifacts(root: Path, matrix_summary_path: str | Path | None) -> dict[str, str]:
    candidates = {
        "pipeline_summary": "pipeline_summary.json",
        "scene_data_inspection": "scene_data_inspection.md",
        "query_reports": "queries/",
        "prompt_sensitivity": "prompt_sensitivity/prompt_sensitivity_report.md",
        "scene_relations": "scene_relations/scene_relations_report.md",
        "evaluation_summary": "evaluation/eval_summary.json",
        "annotation_review": "evaluation/annotation_review.md",
        "annotation_workbench": "evaluation/annotation_workbench/annotation_workbench.html",
        "evidence_scorecard": "evidence_scorecard.md",
        "quality_gate": "quality_gate.md",
        "run_result_card": "run_result_card.md",
        "portfolio_page": "portfolio_page.html",
        "claim_audit": "claim_audit.md",
        "reproduction_report": "reproduction_report.md",
        "real_run_plan": "real_run_plan/real_run_plan.md",
        "submission_checklist": "submission_packet/submission_checklist.md",
        "portfolio_pack": "../../portfolio_pack.zip",
    }
    artifacts = {name: relative for name, relative in candidates.items() if (root / relative).exists()}
    if matrix_summary_path:
        artifacts["experiment_matrix"] = str(matrix_summary_path).replace("\\", "/")
    return artifacts


def _limitations(
    dry_run: bool,
    evaluation: dict[str, Any],
    prompt: dict[str, Any],
    relations: dict[str, Any],
) -> list[str]:
    limitations = [
        "This is a research engineering system built on upstream Nerfstudio/LERF components.",
        "Reported metrics are lightweight portfolio metrics, not a large benchmark.",
    ]
    if dry_run:
        limitations.append("Dry-run outputs are synthetic and only validate orchestration.")
    if not evaluation or not evaluation.get("num_bbox_annotated_queries"):
        limitations.append("Quantitative localization evidence is limited without manual bbox annotations.")
    if not prompt:
        limitations.append("Prompt wording robustness was not summarized for this run.")
    if relations:
        limitations.append("Scene relations are heuristic and should be treated as qualitative evidence.")
    else:
        limitations.append("Scene-relation evidence was not generated for this run.")
    return limitations


def _next_steps(recommendations: dict[str, Any], dry_run: bool) -> list[str]:
    steps = [
        str(item.get("action"))
        for item in recommendations.get("recommendations") or []
        if isinstance(item, dict) and item.get("action")
    ]
    if steps:
        return steps[:6]
    if dry_run:
        return ["Run the same pipeline on a CUDA machine with a real captured scene."]
    return ["Review qualitative overlays, fill annotations, and rerun portfolio quality gates."]


def _warnings(
    audit: dict[str, Any],
    quality_gate: dict[str, Any],
    recommendations: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if audit.get("status") and audit.get("status") != "ready":
        warnings.append(f"Run audit status is {audit.get('status')}.")
    if quality_gate.get("status") and quality_gate.get("status") != "pass":
        warnings.append(f"Quality gate status is {quality_gate.get('status')}.")
    for item in recommendations.get("recommendations") or []:
        if isinstance(item, dict) and item.get("severity") in {"critical", "high"}:
            warnings.append(str(item.get("action")))
    return warnings


def _abstract(scene_name: str, backend: str, dry_run: bool, key_results: dict[str, Any]) -> str:
    mode = "dry-run smoke" if dry_run else "real-scene"
    evidence = key_results.get("evidence_level") or "unknown evidence"
    query_count = key_results.get("query_count") or 0
    return (
        f"This report summarizes a {mode} run of NeRF-LLM Scene Inspector on `{scene_name}` "
        f"using the `{backend}` backend. The system connects Nerfstudio-style scene processing, "
        "LERF-style language-field querying, deterministic query planning, visualization, "
        f"and lightweight evaluation across {query_count} query tasks. The run is currently "
        f"marked `{evidence}`; interpret dry-run or single-scene metrics as reproducibility "
        "evidence rather than benchmark performance."
    )


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _score_text(value: Any, maximum: Any) -> str:
    if value is None or maximum in {None, 0, "0"}:
        return ""
    return f"{value}/{maximum}"


def _metric_lines(metrics: dict[str, Any]) -> list[str]:
    if not metrics:
        return ["- No metrics were available."]
    return [f"- {key}: `{_format_value(value)}`" for key, value in metrics.items() if value not in {"", None}]


def _evidence_lines(evidence: dict[str, Any]) -> list[str]:
    if not evidence:
        return ["- No structured evidence summary was available."]
    lines = []
    queries = evidence.get("queries") if isinstance(evidence.get("queries"), list) else []
    lines.append(f"- Query tasks: {len(queries)}")
    lines.append(f"- Evaluation metrics present: {evidence.get('evaluation_metrics_present')}")
    lines.append(f"- Prompt sensitivity present: {evidence.get('prompt_sensitivity_present')}")
    lines.append(f"- Scene relations present: {evidence.get('scene_relations_present')}")
    lines.append(f"- Experiment matrix present: {evidence.get('experiment_matrix_present')}")
    if evidence.get("scorecard_summary"):
        lines.append(f"- Scorecard: {evidence.get('scorecard_summary')}")
    return lines


def _artifact_lines(artifacts: dict[str, str]) -> list[str]:
    if not artifacts:
        return ["- No artifacts were indexed."]
    return [f"- {name}: `{path}`" for name, path in artifacts.items()]


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
