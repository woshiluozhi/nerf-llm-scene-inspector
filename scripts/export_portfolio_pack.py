#!/usr/bin/env python
"""Collect portfolio-facing project and pipeline-run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".html", ".json", ".md", ".sh", ".txt", ".yaml", ".yml"}
TEXT_NAMES = {"LICENSE", "README", "README.md"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/portfolio_pack")
    parser.add_argument(
        "--run-dir",
        help="Optional run-scoped pipeline directory, for example results/pipeline_runs/desk_scene.",
    )
    parser.add_argument("--zip", action="store_true", help="Also create a .zip archive of the pack.")
    parser.add_argument("--allow-missing", action="store_true", help="Exit 0 even if expected files are missing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = _resolve(args.output)
    if output.exists():
        _clean_output(output)
    output.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    missing: list[str] = []
    optional_missing: list[str] = []
    _copy_project_materials(output, copied, missing, optional_missing)

    run_dir = _resolve(args.run_dir) if args.run_dir else None
    run_summary: dict[str, Any] | None = None
    if run_dir is not None:
        run_summary = _copy_run_materials(run_dir, output, copied, missing, optional_missing)
        _copy_run_index(run_dir.parent, output, copied, optional_missing)

    _add_artifact_digests(output, copied)
    archive_path = Path(f"{output}.zip") if args.zip else None
    index = {
        "copied": copied,
        "missing": missing,
        "optional_missing": optional_missing,
        "github": "https://github.com/woshiluozhi/nerf-llm-scene-inspector",
        "run_dir": _display_source_path(run_dir) if run_dir else None,
        "run_summary": run_summary,
        "archive": _display_source_path(Path(archive_path)) if archive_path else None,
        "recommended_demo_command": "python scripts/run_scene_pipeline.py --dry-run --query mug",
    }
    (output / "portfolio_pack_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    if args.zip:
        archive_path = Path(shutil.make_archive(str(output), "zip", output))
        index["archive"] = _display_source_path(archive_path)
        (output / "portfolio_pack_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0 if not missing or args.allow_missing else 1


def _copy_project_materials(
    output: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
    optional_missing: list[str],
) -> None:
    required_project_files = [
        (ROOT / "README.md", "project/README.md"),
        (ROOT / "LICENSE", "project/LICENSE"),
        (ROOT / "CITATION.cff", "project/CITATION.cff"),
        (ROOT / "docs" / "index.html", "project/docs/index.html"),
        (ROOT / "docs" / "portfolio_result_card.md", "project/docs/portfolio_result_card.md"),
        (ROOT / "docs" / "project_report.md", "project/docs/project_report.md"),
        (ROOT / "docs" / "method_summary.md", "project/docs/method_summary.md"),
        (ROOT / "docs" / "cv_bullets.md", "project/docs/cv_bullets.md"),
        (ROOT / "docs" / "cold_email_paragraph.md", "project/docs/cold_email_paragraph.md"),
        (ROOT / "docs" / "real_scene_capture_checklist.md", "project/docs/real_scene_capture_checklist.md"),
        (ROOT / "docs" / "real_run_reproducibility.md", "project/docs/real_run_reproducibility.md"),
        (ROOT / "docs" / "assets" / "query_grid.png", "project/docs/assets/query_grid.png"),
        (ROOT / "docs" / "assets" / "demo_montage.gif", "project/docs/assets/demo_montage.gif"),
    ]
    optional_project_files = [
        (ROOT / "results" / "dry_run_demo_summary.json", "project/results/dry_run_demo_summary.json"),
        (ROOT / "results" / "evaluation" / "eval_summary.json", "project/results/evaluation/eval_summary.json"),
    ]
    for source, relative_destination in required_project_files:
        _copy_share_safe_file(source, output / relative_destination, output, copied, missing, ROOT)
    for source, relative_destination in optional_project_files:
        _copy_share_safe_file(source, output / relative_destination, output, copied, optional_missing, ROOT)


def _copy_run_materials(
    run_dir: Path,
    output: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
    optional_missing: list[str],
) -> dict[str, Any] | None:
    pipeline_summary_path = run_dir / "pipeline_summary.json"
    run_summary = _load_json_if_exists(pipeline_summary_path)
    _copy_pipeline_summary(pipeline_summary_path, output / "run/pipeline_summary.json", output, copied, missing, run_dir)
    run_files = [
        (run_dir / "capture_manifest.json", "run/capture_manifest.json"),
        (run_dir / "capture_manifest.md", "run/capture_manifest.md"),
        (run_dir / "capture_manifest_validation.json", "run/capture_manifest_validation.json"),
        (run_dir / "capture_manifest_validation.md", "run/capture_manifest_validation.md"),
        (run_dir / "preflight_report.json", "run/preflight_report.json"),
        (run_dir / "preflight_report.md", "run/preflight_report.md"),
        (run_dir / "evidence_scorecard.json", "run/evidence_scorecard.json"),
        (run_dir / "evidence_scorecard.md", "run/evidence_scorecard.md"),
        (run_dir / "quality_gate.json", "run/quality_gate.json"),
        (run_dir / "quality_gate.md", "run/quality_gate.md"),
        (run_dir / "claim_audit.json", "run/claim_audit.json"),
        (run_dir / "claim_audit.md", "run/claim_audit.md"),
        (run_dir / "run_result_card.json", "run/run_result_card.json"),
        (run_dir / "run_result_card.md", "run/run_result_card.md"),
        (run_dir / "run_audit.json", "run/run_audit.json"),
        (run_dir / "run_audit.md", "run/run_audit.md"),
        (run_dir / "run_recommendations.json", "run/run_recommendations.json"),
        (run_dir / "run_recommendations.md", "run/run_recommendations.md"),
        (run_dir / "reproduction_manifest.json", "run/reproduction_manifest.json"),
        (run_dir / "reproduction_report.md", "run/reproduction_report.md"),
        (run_dir / "reproduce_run.sh", "run/reproduce_run.sh"),
        (run_dir / "research_report.json", "run/research_report.json"),
        (run_dir / "research_report.md", "run/research_report.md"),
        (run_dir / "real_run_plan" / "real_run_plan.json", "run/real_run_plan/real_run_plan.json"),
        (run_dir / "real_run_plan" / "real_run_plan.md", "run/real_run_plan/real_run_plan.md"),
        (run_dir / "submission_packet" / "submission_packet.json", "run/submission_packet/submission_packet.json"),
        (run_dir / "submission_packet" / "submission_checklist.md", "run/submission_packet/submission_checklist.md"),
        (run_dir / "submission_packet" / "cv_project_entry.md", "run/submission_packet/cv_project_entry.md"),
        (
            run_dir / "submission_packet" / "professor_email_brief.md",
            "run/submission_packet/professor_email_brief.md",
        ),
        (run_dir / "environment_report.json", "run/environment_report.json"),
        (run_dir / "scene_data_inspection.json", "run/scene_data_inspection.json"),
        (run_dir / "scene_data_inspection.md", "run/scene_data_inspection.md"),
        (run_dir / "queries.yaml", "run/queries.yaml"),
        (run_dir / "annotation_template.json", "run/annotation_template.json"),
        (run_dir / "project_report.md", "run/project_report.md"),
        (run_dir / "portfolio_result_card.md", "run/portfolio_result_card.md"),
        (run_dir / "portfolio_page.html", "run/portfolio_page.html"),
        (run_dir / "evaluation" / "eval_summary.json", "run/evaluation/eval_summary.json"),
        (run_dir / "evaluation" / "eval_table.csv", "run/evaluation/eval_table.csv"),
        (run_dir / "evaluation" / "annotation_validation.json", "run/evaluation/annotation_validation.json"),
        (run_dir / "evaluation" / "annotation_review.json", "run/evaluation/annotation_review.json"),
        (run_dir / "evaluation" / "annotation_review.md", "run/evaluation/annotation_review.md"),
        (
            run_dir / "evaluation" / "annotation_review_contact_sheet.png",
            "run/evaluation/annotation_review_contact_sheet.png",
        ),
        (
            run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench.html",
            "run/evaluation/annotation_workbench/annotation_workbench.html",
        ),
        (
            run_dir / "evaluation" / "annotation_workbench" / "annotation_workbench_manifest.json",
            "run/evaluation/annotation_workbench/annotation_workbench_manifest.json",
        ),
        (
            run_dir / "evaluation" / "annotation_workbench" / "annotation_seed.json",
            "run/evaluation/annotation_workbench/annotation_seed.json",
        ),
        (run_dir / "evaluation" / "qualitative_report.md", "run/evaluation/qualitative_report.md"),
        (
            run_dir / "scene_relations" / "scene_relations_summary.json",
            "run/scene_relations/scene_relations_summary.json",
        ),
        (
            run_dir / "scene_relations" / "scene_relations_edges.csv",
            "run/scene_relations/scene_relations_edges.csv",
        ),
        (
            run_dir / "scene_relations" / "scene_relations_report.md",
            "run/scene_relations/scene_relations_report.md",
        ),
        (run_dir / "demo_assets" / "demo_summary.json", "run/demo_assets/demo_summary.json"),
        (run_dir / "demo_assets" / "query_grid.png", "run/demo_assets/query_grid.png"),
        (run_dir / "demo_assets" / "demo_montage.gif", "run/demo_assets/demo_montage.gif"),
    ]
    for source, relative_destination in run_files:
        relation_file = len(source.parts) >= 2 and source.parts[-2] == "scene_relations"
        target_missing = (
            optional_missing
            if relation_file and not _step_succeeded(run_summary, "analyze_scene_relations")
            else missing
        )
        _copy_share_safe_file(
            source,
            output / relative_destination,
            output,
            copied,
            target_missing,
            run_dir,
        )
    _copy_command_logs(run_dir, output, copied)
    _copy_query_reports(run_dir, output, copied)
    _copy_prompt_sensitivity(run_dir, output, copied)
    _copy_annotation_workbench_assets(run_dir, output, copied)
    _copy_share_safe_file(
        run_dir / "training" / "baseline_train_summary.json",
        output / "run/training/baseline_train_summary.json",
        output,
        copied,
        missing if _step_succeeded(run_summary, "train_baseline_nerf") else optional_missing,
        run_dir,
    )
    _copy_share_safe_file(
        run_dir / "training" / "language_train_summary.json",
        output / "run/training/language_train_summary.json",
        output,
        copied,
        missing if _step_succeeded(run_summary, "train_language_field") else optional_missing,
        run_dir,
    )
    for overlay in sorted((run_dir / "demo_assets").rglob("*overlay.png"))[:8]:
        _copy_file(
            overlay,
            output / "run" / "demo_assets" / "overlays" / overlay.parent.name / overlay.name,
            output,
            copied,
            missing,
        )
    for review_image in sorted((run_dir / "evaluation" / "annotation_review_images").glob("*.png"))[:20]:
        _copy_file(
            review_image,
            output / "run" / "evaluation" / "annotation_review_images" / review_image.name,
            output,
            copied,
            missing,
        )
    for source, relative_destination in (
        (run_dir / "annotations_merged.json", "run/annotations_merged.json"),
        (run_dir / "annotation_merge_report.json", "run/annotation_merge_report.json"),
        (run_dir / "annotation_finalize_report.json", "run/annotation_finalize_report.json"),
        (run_dir / "annotation_finalize_report.md", "run/annotation_finalize_report.md"),
    ):
        _copy_share_safe_file(source, output / relative_destination, output, copied, optional_missing, run_dir)
    return _run_summary_excerpt(run_summary, run_dir)


def _copy_run_index(
    runs_root: Path,
    output: Path,
    copied: list[dict[str, Any]],
    optional_missing: list[str],
) -> None:
    for source, relative_destination in (
        (runs_root / "run_index.json", "run_index.json"),
        (runs_root / "run_index.md", "run_index.md"),
        (runs_root / "run_comparison.json", "run_comparison.json"),
        (runs_root / "run_comparison.md", "run_comparison.md"),
    ):
        _copy_share_safe_file(source, output / relative_destination, output, copied, optional_missing, runs_root)


def _copy_file(
    source: Path,
    destination: Path,
    pack_root: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": _display_source_path(source),
                "destination": _relative_display_path(destination, pack_root),
            }
        )
    else:
        missing.append(_display_source_path(source))


def _copy_share_safe_file(
    source: Path,
    destination: Path,
    pack_root: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
    sanitizer_root: Path,
) -> None:
    if not source.exists():
        missing.append(_display_source_path(source))
        return
    payload = _load_json_if_exists(source) if source.suffix.lower() == ".json" else None
    if payload is None:
        if _is_text_like(source):
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                sanitized_text = _sanitize_text_for_portfolio(source.read_text(encoding="utf-8"), sanitizer_root)
                destination.write_text(sanitized_text, encoding="utf-8")
                copied.append(
                    {
                        "source": _display_source_path(source),
                        "destination": _relative_display_path(destination, pack_root),
                    }
                )
                return
            except UnicodeDecodeError:
                pass
        _copy_file(source, destination, pack_root, copied, missing)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_portfolio(payload, sanitizer_root)
    destination.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    copied.append(
        {
            "source": _display_source_path(source),
            "destination": _relative_display_path(destination, pack_root),
        }
    )


def _copy_command_logs(
    run_dir: Path,
    output: Path,
    copied: list[dict[str, Any]],
) -> None:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return
    for source in sorted(logs_dir.glob("*.json")):
        destination = output / "run" / "logs" / source.name
        payload = _load_json_if_exists(source)
        if payload is None:
            _copy_file(source, destination, output, copied, [])
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        sanitized = _sanitize_for_portfolio(payload, run_dir)
        destination.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
        copied.append(
            {
                "source": _display_source_path(source),
                "destination": _relative_display_path(destination, output),
            }
        )


def _copy_query_reports(
    run_dir: Path,
    output: Path,
    copied: list[dict[str, Any]],
) -> None:
    queries_dir = run_dir / "queries"
    if not queries_dir.exists():
        return
    for source in sorted(queries_dir.rglob("*")):
        if source.name not in {"scene_query_report.json", "scene_query_report.md", "query_result.json"}:
            continue
        relative = source.relative_to(run_dir)
        _copy_share_safe_file(source, output / "run" / relative, output, copied, [], run_dir)


def _copy_prompt_sensitivity(
    run_dir: Path,
    output: Path,
    copied: list[dict[str, Any]],
) -> None:
    prompt_dir = run_dir / "prompt_sensitivity"
    if not prompt_dir.exists():
        return
    for source in sorted(prompt_dir.glob("prompt_sensitivity_*")):
        if source.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = source.relative_to(run_dir)
        _copy_share_safe_file(source, output / "run" / relative, output, copied, [], run_dir)


def _copy_annotation_workbench_assets(
    run_dir: Path,
    output: Path,
    copied: list[dict[str, Any]],
) -> None:
    assets_dir = run_dir / "evaluation" / "annotation_workbench" / "assets"
    if not assets_dir.exists():
        return
    for source in sorted(assets_dir.glob("*")):
        if not source.is_file():
            continue
        _copy_file(
            source,
            output / "run" / "evaluation" / "annotation_workbench" / "assets" / source.name,
            output,
            copied,
            [],
        )


def _copy_pipeline_summary(
    source: Path,
    destination: Path,
    pack_root: Path,
    copied: list[dict[str, Any]],
    missing: list[str],
    run_dir: Path,
) -> None:
    if not source.exists():
        missing.append(_display_source_path(source))
        return
    summary = _load_json_if_exists(source)
    if summary is None:
        _copy_file(source, destination, pack_root, copied, missing)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    portable_summary = _sanitize_for_portfolio(summary, run_dir)
    destination.write_text(json.dumps(portable_summary, indent=2), encoding="utf-8")
    copied.append(
        {
            "source": _display_source_path(source),
            "destination": _relative_display_path(destination, pack_root),
        }
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _add_artifact_digests(pack_root: Path, copied: list[dict[str, Any]]) -> None:
    for item in copied:
        destination = item.get("destination")
        if not isinstance(destination, str) or not destination:
            continue
        path = pack_root / destination
        if not path.is_file():
            continue
        item["size_bytes"] = path.stat().st_size
        item["sha256"] = _sha256(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def _run_summary_excerpt(summary: dict[str, Any] | None, run_dir: Path | None = None) -> dict[str, Any] | None:
    if not summary:
        return None
    artifacts = {
        "pipeline_summary": "run/pipeline_summary.json",
        "capture_manifest": "run/capture_manifest.md",
        "capture_manifest_validation": "run/capture_manifest_validation.md",
        "preflight_report": "run/preflight_report.md",
        "evidence_scorecard": "run/evidence_scorecard.md",
        "quality_gate": "run/quality_gate.md",
        "claim_audit": "run/claim_audit.md",
        "run_result_card": "run/run_result_card.md",
        "run_index": "run_index.md",
        "run_audit": "run/run_audit.md",
        "run_recommendations": "run/run_recommendations.md",
        "reproduction_report": "run/reproduction_report.md",
        "reproduce_script": "run/reproduce_run.sh",
        "research_report": "run/research_report.md",
        "real_run_plan": "run/real_run_plan/real_run_plan.md",
        "submission_checklist": "run/submission_packet/submission_checklist.md",
        "command_logs": "run/logs/",
        "environment_report": "run/environment_report.json",
        "scene_data_inspection": "run/scene_data_inspection.md",
        "query_plan": "run/queries.yaml",
        "query_reports": "run/queries/",
        "annotation_template": "run/annotation_template.json",
        "project_report": "run/project_report.md",
        "portfolio_card": "run/portfolio_result_card.md",
        "portfolio_page": "run/portfolio_page.html",
        "evaluation_summary": "run/evaluation/eval_summary.json",
        "annotation_validation": "run/evaluation/annotation_validation.json",
        "annotation_review": "run/evaluation/annotation_review.md",
        "annotation_review_contact_sheet": "run/evaluation/annotation_review_contact_sheet.png",
        "annotation_workbench": "run/evaluation/annotation_workbench/annotation_workbench.html",
        "demo_grid": "run/demo_assets/query_grid.png",
        "demo_montage": "run/demo_assets/demo_montage.gif",
    }
    if run_dir is not None and (run_dir / "annotations_merged.json").exists():
        artifacts["annotations_merged"] = "run/annotations_merged.json"
    if run_dir is not None and (run_dir / "annotation_merge_report.json").exists():
        artifacts["annotation_merge_report"] = "run/annotation_merge_report.json"
    if run_dir is not None and (run_dir / "annotation_finalize_report.md").exists():
        artifacts["annotation_finalize"] = "run/annotation_finalize_report.md"
    if _step_succeeded(summary, "train_baseline_nerf"):
        artifacts["baseline_train_summary"] = "run/training/baseline_train_summary.json"
    if _step_succeeded(summary, "train_language_field"):
        artifacts["language_train_summary"] = "run/training/language_train_summary.json"
    if _step_succeeded(summary, "analyze_prompt_sensitivity"):
        artifacts["prompt_sensitivity"] = "run/prompt_sensitivity/"
    if _step_succeeded(summary, "analyze_scene_relations"):
        artifacts["scene_relations"] = "run/scene_relations/"
    if _step_succeeded(summary, "compare_runs"):
        artifacts["run_comparison"] = "run_comparison.md"
    return {
        "scene_name": summary.get("scene_name"),
        "success": summary.get("success"),
        "dry_run": summary.get("dry_run"),
        "backend": summary.get("backend"),
        "timestamp": summary.get("timestamp"),
        "queries": summary.get("queries"),
        "artifacts": artifacts,
        "provenance": _provenance_excerpt(summary.get("provenance")),
    }


def _step_succeeded(summary: dict[str, Any] | None, step_name: str) -> bool:
    if not summary:
        return False
    for step in summary.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == step_name:
            return step.get("status") == "success"
    return False


def _provenance_excerpt(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "project_version": raw.get("project_version"),
        "python_version": raw.get("python_version"),
        "git_available": raw.get("git_available"),
        "git_commit": raw.get("git_commit"),
        "git_branch": raw.get("git_branch"),
        "git_dirty": raw.get("git_dirty"),
    }


def _sanitize_for_portfolio(value: Any, run_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_portfolio(item, run_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_portfolio(item, run_dir) for item in value]
    if isinstance(value, str):
        return _sanitize_text_for_portfolio(value, run_dir)
    return value


def _sanitize_text_for_portfolio(text: str, run_dir: Path) -> str:
    sanitized = text
    for raw, replacement in _sensitive_path_replacements(run_dir):
        if raw:
            sanitized = sanitized.replace(raw, replacement)
    stripped = sanitized.strip("'\"")
    if not any(character.isspace() for character in stripped):
        executable_name = re.split(r"[\\/]", stripped)[-1].lower()
        if ("/" in stripped or "\\" in stripped) and executable_name in {"python", "python.exe"}:
            return "python"
    sanitized = re.sub(r"'~[\\/][^']*?python(?:\.exe)?'", "python", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'"~[\\/][^"]*?python(?:\.exe)?"', "python", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"~[\\/]\S*?python(?:\.exe)?", "python", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"/tmp/pytest-[^\s\"'`<>|]+", "<pytest-tmp>", sanitized)
    sanitized = re.sub(r"/home/runner/work/[^\s\"'`<>|]+", ".", sanitized)
    sanitized = re.sub(
        r"/opt/hostedtoolcache/Python/[^\s\"'`<>|]*?/bin/python(?:\d(?:\.\d+)?)?",
        "python",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"/tmp/[^\s\"'`<>|]+", "<tmp-path>", sanitized)
    sanitized = re.sub(r"/home/runner/[^\s\"'`<>|]+", "<ci-home-path>", sanitized)
    sanitized = re.sub(r"/opt/hostedtoolcache/[^\s\"'`<>|]+", "<ci-toolcache-path>", sanitized)
    return sanitized


def _sensitive_path_replacements(run_dir: Path) -> list[tuple[str, str]]:
    candidates: list[tuple[Path, str]] = [
        (ROOT.resolve(), "."),
        (Path.home().resolve(), "~"),
        (run_dir.resolve(), "<run-dir>"),
        (run_dir.resolve().parent, "<pipeline-runs-dir>"),
        (run_dir.resolve().parent.parent, "<run-workspace>"),
    ]
    replacements: dict[str, str] = {}
    for path, label in candidates:
        path_texts = {str(path), path.as_posix()}
        for path_text in path_texts:
            replacements.setdefault(path_text, label)
            if "\\" in path_text:
                replacements.setdefault(path_text.replace("\\", "\\\\"), label)
    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _display_source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _relative_display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _clean_output(output: Path) -> None:
    resolved_output = output.resolve()
    resolved_root = ROOT.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        raise RuntimeError(f"Refusing to clean output outside repository: {resolved_output}")
    if resolved_output == resolved_root:
        raise RuntimeError("Refusing to use repository root as export output.")
    shutil.rmtree(resolved_output)


if __name__ == "__main__":
    raise SystemExit(main())
