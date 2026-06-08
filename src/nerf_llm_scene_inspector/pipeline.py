"""End-to-end scene pipeline orchestration."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.agent.planner import LocalRulePlanner
from nerf_llm_scene_inspector.backends.lerf_backend import LERFBackend
from nerf_llm_scene_inspector.backends.opennerf_backend import OpenNeRFBackend
from nerf_llm_scene_inspector.capture_manifest import copy_or_create_capture_manifest
from nerf_llm_scene_inspector.data_processing import prepare_data
from nerf_llm_scene_inspector.evaluation.evidence_scorecard import build_evidence_scorecard
from nerf_llm_scene_inspector.evaluation.quality_gate import check_run_quality
from nerf_llm_scene_inspector.evaluation.run_audit import audit_pipeline_run
from nerf_llm_scene_inspector.evaluation.run_comparison import compare_pipeline_runs
from nerf_llm_scene_inspector.evaluation.run_index import index_pipeline_runs
from nerf_llm_scene_inspector.evaluation.run_recommendations import build_run_recommendations
from nerf_llm_scene_inspector.preflight import build_real_run_preflight
from nerf_llm_scene_inspector.querying.semantic_query import SemanticQueryEngine
from nerf_llm_scene_inspector.reproducibility import build_reproduction_bundle
from nerf_llm_scene_inspector.scene_validation import inspect_processed_scene
from nerf_llm_scene_inspector.training import train_baseline_nerf, train_language_field
from nerf_llm_scene_inspector.utils.env_check import build_env_report
from nerf_llm_scene_inspector.utils.paths import project_root, slugify, utc_timestamp
from nerf_llm_scene_inspector.utils.provenance import build_provenance
from nerf_llm_scene_inspector.utils.shell import format_command, run_command
from nerf_llm_scene_inspector.visualization.portfolio_page import build_portfolio_page


DEFAULT_PIPELINE_QUERIES = [
    "mug",
    "objects that can hold water",
    "safe place to put a hot cup",
]


@dataclass
class PipelineConfig:
    """Configuration for a reproducible scene pipeline run."""

    input_path: str | Path
    scene_name: str
    data_type: str
    backend: str = "lerf"
    variant: str = "lerf-lite"
    baseline_method: str = "nerfacto"
    queries: list[str] = field(default_factory=lambda: list(DEFAULT_PIPELINE_QUERIES))
    data_root: str | Path = "data/processed"
    runs_root: str | Path = "runs"
    output_root: str | Path = "results/pipeline_runs"
    annotations_path: str | Path = "examples/annotations_example.json"
    capture_manifest_path: str | Path | None = None
    config_path: str | Path | None = None
    max_num_iterations: int | None = None
    num_views: int = 1
    top_k: int = 5
    min_frames: int = 20
    min_pose_extent: float = 0.05
    dry_run: bool = False
    strict: bool = False
    skip_prepare: bool = False
    skip_baseline: bool = False
    skip_language: bool = False
    skip_queries: bool = False
    skip_demo: bool = False
    skip_eval: bool = False
    clean_run_outputs: bool = True
    command: list[str] | None = None


@dataclass
class PipelineStep:
    """One recorded pipeline step."""

    name: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineRunSummary:
    """Persisted summary for a complete scene pipeline run."""

    scene_name: str
    success: bool
    dry_run: bool
    backend: str
    timestamp: str
    paths: dict[str, str]
    queries: list[str]
    steps: list[PipelineStep]
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "success": self.success,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "timestamp": self.timestamp,
            "paths": self.paths,
            "queries": list(self.queries),
            "steps": [step.to_dict() for step in self.steps],
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path


def run_scene_pipeline(config: PipelineConfig) -> PipelineRunSummary:
    """Run or dry-run the project workflow for a single scene."""

    root = project_root()
    processed_dir = Path(config.data_root) / config.scene_name
    baseline_dir = Path(config.runs_root) / f"baseline_{config.scene_name}"
    language_dir = Path(config.runs_root) / f"language_{config.scene_name}"
    run_dir = Path(config.output_root) / config.scene_name
    runs_root = Path(config.output_root)
    query_dir = run_dir / "queries"
    demo_dir = run_dir / "demo_assets"
    eval_dir = run_dir / "evaluation"
    training_dir = run_dir / "training"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.clean_run_outputs:
        for subdir in (query_dir, demo_dir, eval_dir, logs_dir):
            _reset_run_subdir(subdir, run_dir)
    run_queries_path = _write_run_queries_file(run_dir, config.scene_name, config.queries)

    steps: list[PipelineStep] = []
    warnings: list[str] = []
    paths = {
        "processed_data": str(processed_dir),
        "baseline_run": str(baseline_dir),
        "language_run": str(language_dir),
        "pipeline_run": str(run_dir),
        "run_index_json": str(runs_root / "run_index.json"),
        "run_index_markdown": str(runs_root / "run_index.md"),
        "run_comparison_json": str(runs_root / "run_comparison.json"),
        "run_comparison_markdown": str(runs_root / "run_comparison.md"),
        "queries": str(query_dir),
        "demo_assets": str(demo_dir),
        "evaluation": str(eval_dir),
        "annotation_review_json": str(eval_dir / "annotation_review.json"),
        "annotation_review_markdown": str(eval_dir / "annotation_review.md"),
        "annotation_review_contact_sheet": str(eval_dir / "annotation_review_contact_sheet.png"),
        "training": str(training_dir),
        "logs": str(logs_dir),
        "run_queries": str(run_queries_path),
        "capture_manifest_json": str(run_dir / "capture_manifest.json"),
        "capture_manifest_markdown": str(run_dir / "capture_manifest.md"),
        "capture_manifest_validation_json": str(run_dir / "capture_manifest_validation.json"),
        "capture_manifest_validation_markdown": str(run_dir / "capture_manifest_validation.md"),
        "preflight_json": str(run_dir / "preflight_report.json"),
        "preflight_markdown": str(run_dir / "preflight_report.md"),
        "evidence_scorecard_json": str(run_dir / "evidence_scorecard.json"),
        "evidence_scorecard_markdown": str(run_dir / "evidence_scorecard.md"),
        "portfolio_page": str(run_dir / "portfolio_page.html"),
        "run_audit_json": str(run_dir / "run_audit.json"),
        "run_audit_markdown": str(run_dir / "run_audit.md"),
        "run_recommendations_json": str(run_dir / "run_recommendations.json"),
        "run_recommendations_markdown": str(run_dir / "run_recommendations.md"),
        "quality_gate_json": str(run_dir / "quality_gate.json"),
        "quality_gate_markdown": str(run_dir / "quality_gate.md"),
        "reproduction_manifest": str(run_dir / "reproduction_manifest.json"),
        "reproduction_report": str(run_dir / "reproduction_report.md"),
        "reproduce_script": str(run_dir / "reproduce_run.sh"),
        "project_report": str(run_dir / "project_report.md"),
        "portfolio_card": str(run_dir / "portfolio_result_card.md"),
    }

    try:
        (
            capture_manifest_json,
            capture_manifest_md,
            capture_validation_json,
            capture_validation_md,
            capture_validation,
        ) = copy_or_create_capture_manifest(
            output_dir=run_dir,
            input_path=config.input_path,
            input_type=config.data_type,
            scene_name=config.scene_name,
            capture_manifest_path=config.capture_manifest_path,
            min_images=1 if config.dry_run else config.min_frames,
            require_privacy_review=config.strict and not config.dry_run,
        )
        steps.append(
            PipelineStep(
                "capture_manifest",
                "success" if capture_validation.status == "ready" else "warning",
                summary={
                    "status": capture_validation.status,
                    "ok": capture_validation.ok,
                    "fail_count": capture_validation.fail_count,
                    "warn_count": capture_validation.warn_count,
                },
                outputs={
                    "manifest_json": str(capture_manifest_json),
                    "manifest_markdown": str(capture_manifest_md),
                    "validation_json": str(capture_validation_json),
                    "validation_markdown": str(capture_validation_md),
                },
            )
        )
        preflight = build_real_run_preflight(
            input_path=config.input_path,
            input_type=config.data_type,
            capture_manifest_path=capture_manifest_json,
            data_path=processed_dir if config.skip_prepare else None,
            config_path=config.config_path,
            scene_name=config.scene_name,
            backend=config.backend,
            variant=config.variant,
            min_frames=config.min_frames,
            min_pose_extent=config.min_pose_extent,
            require_gpu=config.strict and not config.dry_run,
            check_upstream=not config.dry_run,
            dry_run=config.dry_run,
        )
        preflight_json = preflight.to_json(run_dir / "preflight_report.json")
        preflight_md = preflight.to_markdown(run_dir / "preflight_report.md")
        steps.append(
            PipelineStep(
                "preflight_real_run",
                _preflight_step_status(preflight.status),
                summary={
                    "status": preflight.status,
                    "ready_for_real_run": preflight.ready_for_real_run,
                    "fail_count": preflight.fail_count,
                    "warn_count": preflight.warn_count,
                },
                outputs={"json": str(preflight_json), "markdown": str(preflight_md)},
            )
        )
        if config.strict and preflight.status == "blocked":
            raise RuntimeError("Preflight checks blocked real training. Inspect preflight_report.md.")

        env_report = build_env_report(
            require_gpu=config.strict and not config.dry_run,
            check_upstream=not config.dry_run,
        )
        env_path = run_dir / "environment_report.json"
        env_path.write_text(json.dumps(env_report.to_dict(), indent=2), encoding="utf-8")
        steps.append(
            PipelineStep(
                "check_environment",
                "success" if env_report.ok else "warning",
                summary={"ok": env_report.ok, "strict_failures": env_report.strict_failures},
                outputs={"environment_report": str(env_path)},
            )
        )
        if config.strict and not env_report.ok:
            raise RuntimeError(f"Environment check failed: {env_report.strict_failures}")

        if config.skip_prepare:
            steps.append(PipelineStep("prepare_data", "skipped"))
        else:
            metadata = prepare_data(
                config.input_path,
                processed_dir,
                config.data_type,
                dry_run=config.dry_run,
                command_log_path=logs_dir / "prepare_data_command.json",
            )
            steps.append(
                PipelineStep(
                    "prepare_data",
                    "success",
                    summary=_small_dict(metadata),
                    outputs={
                        "metadata": str(processed_dir / "scene_inspector_metadata.json"),
                        "command_log": str(logs_dir / "prepare_data_command.json"),
                    },
                )
            )

        min_frames = 1 if config.dry_run else config.min_frames
        inspection = inspect_processed_scene(
            processed_dir,
            min_frames=min_frames,
            min_pose_extent=config.min_pose_extent,
        )
        inspection_json = inspection.to_json(run_dir / "scene_data_inspection.json")
        inspection_md = inspection.to_markdown(run_dir / "scene_data_inspection.md")
        warnings.extend(inspection.warnings)
        steps.append(
            PipelineStep(
                "inspect_scene_data",
                "success" if inspection.ready_for_training else "warning",
                summary={
                    "ready_for_training": inspection.ready_for_training,
                    "quality_score": inspection.quality_score,
                    "pose_coverage_score": inspection.pose_coverage_score,
                    "camera_position_extent": inspection.camera_position_extent,
                    "camera_path_length": inspection.camera_path_length,
                    "frame_count": inspection.frame_count,
                    "missing_image_count": inspection.missing_image_count,
                },
                outputs={
                    "json": str(inspection_json),
                    "markdown": str(inspection_md),
                },
            )
        )
        if config.strict and not inspection.ready_for_training:
            raise RuntimeError("Scene data inspection did not pass strict readiness checks.")

        if config.skip_baseline:
            steps.append(PipelineStep("train_baseline_nerf", "skipped"))
        else:
            baseline_summary = train_baseline_nerf(
                processed_dir,
                config.baseline_method,
                baseline_dir,
                max_num_iterations=config.max_num_iterations,
                dry_run=config.dry_run,
                command_log_path=logs_dir / "train_baseline_nerf_command.json",
            )
            baseline_summary_path = _write_step_json(
                training_dir / "baseline_train_summary.json",
                baseline_summary,
            )
            steps.append(
                PipelineStep(
                    "train_baseline_nerf",
                    "success",
                    summary=_small_dict(baseline_summary),
                    outputs={
                        "train_summary": str(baseline_summary_path),
                        "command_log": str(logs_dir / "train_baseline_nerf_command.json"),
                    },
                )
            )

        model_config_path = str(config.config_path) if config.config_path else None
        if config.skip_language:
            steps.append(PipelineStep("train_language_field", "skipped"))
        else:
            language_summary = train_language_field(
                processed_dir,
                config.backend,
                config.variant,
                language_dir,
                max_num_iterations=config.max_num_iterations,
                dry_run=config.dry_run,
                command_log_path=logs_dir / "train_language_field_command.json",
                method_check_log_path=logs_dir / "train_language_field_method_check.json",
            )
            model_config_path = str(language_summary.get("config_path") or model_config_path)
            language_summary_path = _write_step_json(
                training_dir / "language_train_summary.json",
                language_summary,
            )
            steps.append(
                PipelineStep(
                    "train_language_field",
                    "success",
                    summary=_small_dict(language_summary),
                    outputs={
                        "train_summary": str(language_summary_path),
                        "command_log": str(logs_dir / "train_language_field_command.json"),
                    },
                )
            )

        if config.skip_queries or not config.queries:
            steps.append(PipelineStep("query_scene", "skipped"))
        elif not model_config_path:
            steps.append(PipelineStep("query_scene", "failed", error="No model config path available."))
        else:
            query_outputs = _run_queries(
                config=config,
                config_path=model_config_path,
                output_dir=query_dir,
            )
            steps.append(
                PipelineStep(
                    "query_scene",
                    "success",
                    summary={"num_queries": len(query_outputs)},
                    outputs=query_outputs,
                )
            )

        if config.queries:
            annotation_template_result = _run_helper_script(
                [
                    sys.executable,
                    str(root / "scripts" / "create_annotation_template.py"),
                    "--queries",
                    str(run_queries_path),
                    "--results",
                    str(query_dir),
                    "--output",
                    str(run_dir / "annotation_template.json"),
                    "--overwrite",
                ],
                root=root,
                log_path=logs_dir / "create_annotation_template_command.json",
            )
            annotation_template_result.outputs.update(
                {"annotation_template": str(run_dir / "annotation_template.json")}
            )
            steps.append(annotation_template_result)

        if config.skip_demo or not model_config_path:
            steps.append(PipelineStep("generate_demo_assets", "skipped"))
        else:
            demo_result = _run_helper_script(
                [
                    sys.executable,
                    str(root / "scripts" / "generate_demo_assets.py"),
                    "--config",
                    model_config_path,
                    "--backend",
                    config.backend,
                    "--queries",
                    str(run_queries_path),
                    "--output",
                    str(demo_dir),
                    "--report-output",
                    str(run_dir / "project_report.md"),
                    "--portfolio-card-output",
                    str(run_dir / "portfolio_result_card.md"),
                    "--num-views",
                    str(config.num_views),
                    *(["--dry-run"] if config.dry_run else []),
                ],
                root=root,
                log_path=logs_dir / "generate_demo_assets_command.json",
            )
            demo_result.outputs.update(
                {
                    "demo_summary": str(demo_dir / "demo_summary.json"),
                    "query_grid": str(demo_dir / "query_grid.png"),
                    "demo_montage": str(demo_dir / "demo_montage.gif"),
                    "portfolio_card": str(run_dir / "portfolio_result_card.md"),
                }
            )
            steps.append(demo_result)

        if config.skip_eval:
            steps.append(PipelineStep("evaluate_queries", "skipped"))
        else:
            eval_results_dir = query_dir if query_dir.exists() else demo_dir
            eval_result = _run_helper_script(
                [
                    sys.executable,
                    str(root / "scripts" / "evaluate_queries.py"),
                    "--queries",
                    str(run_queries_path),
                    "--annotations",
                    str(config.annotations_path),
                    "--results",
                    str(eval_results_dir),
                    "--output",
                    str(eval_dir),
                    "--report-output",
                    str(run_dir / "project_report.md"),
                    *(["--dry-run"] if config.dry_run else []),
                ],
                root=root,
                log_path=logs_dir / "evaluate_queries_command.json",
            )
            eval_result.outputs.update(
                {
                    "eval_summary": str(eval_dir / "eval_summary.json"),
                    "eval_table": str(eval_dir / "eval_table.csv"),
                    "annotation_validation": str(eval_dir / "annotation_validation.json"),
                    "qualitative_report": str(eval_dir / "qualitative_report.md"),
                    "project_report": str(run_dir / "project_report.md"),
                }
            )
            steps.append(eval_result)
            if eval_result.status == "success":
                review_result = _run_helper_script(
                    [
                        sys.executable,
                        str(root / "scripts" / "review_annotations.py"),
                        "--annotations",
                        str(config.annotations_path),
                        "--results",
                        str(eval_results_dir),
                        "--output",
                        str(eval_dir),
                        "--allow-warnings",
                    ],
                    root=root,
                    log_path=logs_dir / "review_annotations_command.json",
                )
                review_result.outputs.update(
                    {
                        "annotation_review": str(eval_dir / "annotation_review.json"),
                        "annotation_review_markdown": str(eval_dir / "annotation_review.md"),
                        "annotation_review_contact_sheet": str(
                            eval_dir / "annotation_review_contact_sheet.png"
                        ),
                    }
                )
                steps.append(review_result)
            else:
                steps.append(
                    PipelineStep(
                        "review_annotations",
                        "skipped",
                        error="Skipped because evaluate_queries did not complete successfully.",
                    )
                )
    except Exception as exc:
        steps.append(PipelineStep("pipeline", "failed", error=str(exc)))

    success = all(step.status in {"success", "skipped", "warning"} for step in steps)
    summary = PipelineRunSummary(
        scene_name=config.scene_name,
        success=success,
        dry_run=config.dry_run,
        backend=config.backend,
        timestamp=utc_timestamp(),
        paths=paths,
        queries=config.queries,
        steps=steps,
        provenance=build_provenance(command=config.command, repo_root=root).to_dict(),
        warnings=warnings,
    )
    summary_path = run_dir / "pipeline_summary.json"
    summary.to_json(summary_path)
    audit = audit_pipeline_run(run_dir)
    audit_json = audit.to_json(run_dir / "run_audit.json")
    audit_md = audit.to_markdown(run_dir / "run_audit.md")
    steps.append(
        PipelineStep(
            "audit_run",
            _audit_step_status(audit.status),
            summary={
                "status": audit.status,
                "score": audit.score,
                "blocker_count": audit.blocker_count,
                "warning_count": audit.warning_count,
            },
            outputs={"json": str(audit_json), "markdown": str(audit_md)},
        )
    )
    if audit.status == "blocked":
        summary.success = False
    summary.to_json(summary_path)
    recommendations = build_run_recommendations(run_dir)
    recommendations_json = recommendations.to_json(run_dir / "run_recommendations.json")
    recommendations_md = recommendations.to_markdown(run_dir / "run_recommendations.md")
    steps.append(
        PipelineStep(
            "recommend_next_steps",
            "failed" if recommendations.readiness_level == "blocked" else "success",
            summary={
                "readiness_level": recommendations.readiness_level,
                "critical_count": recommendations.critical_count,
                "high_count": recommendations.high_count,
                "top_next_action": recommendations.top_next_action,
            },
            outputs={"json": str(recommendations_json), "markdown": str(recommendations_md)},
        )
    )
    summary.to_json(summary_path)
    scorecard = build_evidence_scorecard(run_dir)
    scorecard_json = scorecard.to_json(run_dir / "evidence_scorecard.json")
    scorecard_md = scorecard.to_markdown(run_dir / "evidence_scorecard.md")
    steps.append(
        PipelineStep(
            "create_evidence_scorecard",
            _scorecard_step_status(scorecard.evidence_level),
            summary={
                "evidence_level": scorecard.evidence_level,
                "score": scorecard.score,
                "max_score": scorecard.max_score,
                "top_recommendations": scorecard.top_recommendations[:3],
            },
            outputs={"json": str(scorecard_json), "markdown": str(scorecard_md)},
        )
    )
    summary.to_json(summary_path)
    quality_gate = check_run_quality(run_dir, profile="smoke")
    quality_gate_json = quality_gate.to_json(run_dir / "quality_gate.json")
    quality_gate_md = quality_gate.to_markdown(run_dir / "quality_gate.md")
    steps.append(
        PipelineStep(
            "quality_gate",
            "success" if quality_gate.passed else "warning",
            summary={
                "profile": quality_gate.profile,
                "status": quality_gate.status,
                "passed": quality_gate.passed,
                "fail_count": quality_gate.fail_count,
                "warn_count": quality_gate.warn_count,
            },
            outputs={"json": str(quality_gate_json), "markdown": str(quality_gate_md)},
        )
    )
    summary.to_json(summary_path)
    portfolio_page = build_portfolio_page(run_dir)
    portfolio_page_path = portfolio_page.write_html(run_dir / "portfolio_page.html")
    steps.append(
        PipelineStep(
            "generate_portfolio_page",
            "success",
            summary={
                "evidence_level": portfolio_page.evidence_level,
                "evidence_score": portfolio_page.evidence_score,
                "image_count": len(portfolio_page.images),
                "artifact_count": len(portfolio_page.artifacts),
            },
            outputs={"html": str(portfolio_page_path)},
        )
    )
    summary.to_json(summary_path)
    run_index = index_pipeline_runs(runs_root)
    run_index.to_json(runs_root / "run_index.json")
    run_index.to_markdown(runs_root / "run_index.md")
    run_comparison = compare_pipeline_runs(runs_root)
    comparison_json = run_comparison.to_json(runs_root / "run_comparison.json")
    comparison_md = run_comparison.to_markdown(runs_root / "run_comparison.md")
    steps.append(
        PipelineStep(
            "compare_runs",
            "success" if run_comparison.total_runs else "warning",
            summary={
                "total_runs": run_comparison.total_runs,
                "portfolio_candidate_count": run_comparison.portfolio_candidate_count,
                "best_run": run_comparison.best_run,
            },
            outputs={"json": str(comparison_json), "markdown": str(comparison_md)},
        )
    )
    summary.to_json(summary_path)
    reproduction = build_reproduction_bundle(run_dir)
    reproduction_manifest = reproduction.to_json(run_dir / "reproduction_manifest.json")
    reproduction_report = reproduction.to_markdown(run_dir / "reproduction_report.md")
    reproduce_script = reproduction.to_shell_script(run_dir / "reproduce_run.sh")
    steps.append(
        PipelineStep(
            "create_reproduction_bundle",
            "success",
            summary={
                "replay_command": reproduction.replay_command,
                "verification_commands": reproduction.verification_commands,
                "artifact_count": len(reproduction.artifacts),
            },
            outputs={
                "manifest": str(reproduction_manifest),
                "markdown": str(reproduction_report),
                "script": str(reproduce_script),
            },
        )
    )
    summary.to_json(summary_path)
    return summary


def _run_queries(
    *,
    config: PipelineConfig,
    config_path: str,
    output_dir: Path,
) -> dict[str, str]:
    backend = (
        LERFBackend(dry_run=config.dry_run, num_views=config.num_views)
        if config.backend == "lerf"
        else OpenNeRFBackend(dry_run=config.dry_run)
    )
    backend.load(config_path)
    engine = SemanticQueryEngine(
        backend=backend,
        planner=LocalRulePlanner(),
        top_k=config.top_k,
        scene_name=config.scene_name,
    )
    outputs: dict[str, str] = {}
    for query in config.queries:
        task_dir = output_dir / slugify(query)
        report = engine.run_task(query, task_dir)
        report_path = report.to_json(task_dir / "scene_query_report.json")
        report_md_path = report.to_markdown(task_dir / "scene_query_report.md")
        outputs[slugify(query)] = str(report_path)
        outputs[f"{slugify(query)}_markdown"] = str(report_md_path)
    return outputs


def _write_run_queries_file(run_dir: Path, scene_name: str, queries: list[str]) -> Path:
    path = run_dir / "queries.yaml"
    lines = [f"scene_name: {json.dumps(scene_name)}", "queries:"]
    lines.extend(f"  - {json.dumps(query)}" for query in queries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_step_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _reset_run_subdir(path: Path, run_dir: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_run_dir = run_dir.resolve()
    try:
        resolved_path.relative_to(resolved_run_dir)
    except ValueError:
        raise RuntimeError(f"Refusing to clean path outside pipeline run directory: {resolved_path}")
    if resolved_path == resolved_run_dir:
        raise RuntimeError(f"Refusing to clean the pipeline run root directly: {resolved_path}")
    shutil.rmtree(resolved_path)


def _run_helper_script(
    command: list[str],
    *,
    root: Path,
    log_path: str | Path | None = None,
) -> PipelineStep:
    result = run_command(command, cwd=root, check=False, log_path=log_path)
    script_name = Path(command[1]).stem if len(command) > 1 else "command"
    status = "success" if result.ok else "failed"
    outputs = {"command_log": str(log_path)} if log_path else {}
    return PipelineStep(
        name=script_name,
        status=status,
        command=format_command(command),
        summary={
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        },
        outputs=outputs,
    )


def _small_dict(raw: dict[str, object]) -> dict[str, Any]:
    keep = {
        "success",
        "dry_run",
        "method",
        "backend",
        "variant",
        "config_path",
        "transforms_path",
        "viewer_command",
        "returncode",
    }
    return {key: raw[key] for key in keep if key in raw}


def _audit_step_status(status: str) -> str:
    if status == "ready":
        return "success"
    if status == "needs_review":
        return "warning"
    return "failed"


def _preflight_step_status(status: str) -> str:
    if status == "ready":
        return "success"
    if status == "needs_attention":
        return "warning"
    return "failed"


def _scorecard_step_status(level: str) -> str:
    if level in {"portfolio_ready_real_run", "dry_run_demo_ready"}:
        return "success"
    return "warning"
