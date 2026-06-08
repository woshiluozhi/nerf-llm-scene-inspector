"""Create portable reproduction manifests from pipeline run outputs."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


@dataclass
class ReproductionArtifact:
    """One artifact expected inside a reproducible run directory."""

    name: str
    path: str
    exists: bool
    purpose: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ReproductionBundle:
    """Portable instructions and evidence paths for reproducing one pipeline run."""

    run_dir: str
    scene_name: str
    dry_run: bool
    backend: str
    generated_at: str
    source_command: list[str] = field(default_factory=list)
    replay_command: str = ""
    prerequisites: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    artifacts: list[ReproductionArtifact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "scene_name": self.scene_name,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "generated_at": self.generated_at,
            "source_command": list(self.source_command),
            "replay_command": self.replay_command,
            "prerequisites": list(self.prerequisites),
            "verification_commands": list(self.verification_commands),
            "queries": list(self.queries),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
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
            "# Reproduction Report",
            "",
            f"- Scene: {self.scene_name or 'unknown'}",
            f"- Backend: {self.backend or 'unknown'}",
            f"- Dry run: {self.dry_run}",
            f"- Run directory: `{self.run_dir}`",
            "",
            "## Replay Command",
            "",
            "```bash",
            self.replay_command or "# No replay command recorded.",
            "```",
            "",
            "## Verification",
            "",
            *_command_lines(self.verification_commands),
            "",
            "## Key Artifacts",
            "",
            *_artifact_lines(self.artifacts),
            "",
            "## Notes",
            "",
            *_note_lines(self.notes),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def to_shell_script(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "python -m pip install -e \".[dev,video]\"",
            "python scripts/check_env.py --json",
        ]
        if not self.dry_run:
            lines.append("python scripts/check_env.py --check-upstream --require-gpu --verbose")
        if self.replay_command:
            lines.append(self.replay_command)
        lines.extend(self.verification_commands)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_reproduction_bundle(run_dir: str | Path) -> ReproductionBundle:
    """Build replay and verification instructions from a pipeline run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scene_name = str(summary.get("scene_name") or root.name)
    dry_run = bool(summary.get("dry_run"))
    backend = str(summary.get("backend") or "")
    source_command = _source_command(summary)
    return ReproductionBundle(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        dry_run=dry_run,
        backend=backend,
        generated_at=utc_timestamp(),
        source_command=source_command,
        replay_command=_replay_command(source_command, summary),
        prerequisites=_prerequisites(dry_run=dry_run),
        verification_commands=_verification_commands(root),
        queries=[str(query) for query in summary.get("queries") or []],
        artifacts=_artifacts(root),
        notes=_notes(dry_run=dry_run),
    )


def _source_command(summary: dict[str, Any]) -> list[str]:
    raw = summary.get("provenance", {}).get("command") if isinstance(summary.get("provenance"), dict) else []
    return [str(item) for item in raw or []]


def _replay_command(source_command: list[str], summary: dict[str, Any]) -> str:
    if source_command:
        command = list(source_command)
        first = command[0].replace("\\", "/")
        if first.endswith(".py"):
            command = ["python", first, *command[1:]]
        else:
            command[0] = first
        return _format_command(command)
    scene = str(summary.get("scene_name") or "desk_scene")
    queries = [str(query) for query in summary.get("queries") or []]
    command = [
        "python",
        "scripts/run_scene_pipeline.py",
        "--scene-name",
        scene,
        "--backend",
        str(summary.get("backend") or "lerf"),
    ]
    if summary.get("dry_run"):
        command.append("--dry-run")
    for query in queries:
        command.extend(["--query", query])
    return _format_command(command)


def _verification_commands(root: Path) -> list[str]:
    run_dir = _display_run_dir(root)
    runs_root = _display_path(root.parent)
    return [
        _format_command(["python", "scripts/audit_run.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/recommend_next_steps.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/generate_research_report.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_evidence_scorecard.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/check_run_quality.py", "--run-dir", run_dir, "--profile", "smoke"]),
        _format_command(["python", "scripts/generate_research_report.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_run_result_card.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/generate_portfolio_page.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_submission_packet.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/create_real_run_plan.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/audit_claims.py", "--run-dir", run_dir]),
        _format_command(["python", "scripts/compare_runs.py", "--root", runs_root]),
        _format_command(
            [
                "python",
                "scripts/analyze_prompt_sensitivity.py",
                "--suite",
                "examples/prompt_sensitivity.yaml",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/prompt_sensitivity",
            ]
        ),
        _format_command(
            [
                "python",
                "scripts/analyze_scene_relations.py",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/scene_relations",
                "--scene-name",
                root.name,
                "--dry-run",
            ]
        ),
        _format_command(
            [
                "python",
                "scripts/create_annotation_workbench.py",
                "--annotations",
                f"{run_dir}/annotation_template.json",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/evaluation/annotation_workbench",
            ]
        ),
        _format_command(
            [
                "python",
                "scripts/review_annotations.py",
                "--annotations",
                f"{run_dir}/annotation_template.json",
                "--results",
                f"{run_dir}/queries",
                "--output",
                f"{run_dir}/evaluation",
                "--allow-warnings",
            ]
        ),
        _format_command(["python", "scripts/export_portfolio_pack.py", "--run-dir", run_dir, "--zip"]),
        _format_command(["python", "scripts/validate_portfolio_pack.py", "--pack", "results/portfolio_pack"]),
    ]


def _prerequisites(*, dry_run: bool) -> list[str]:
    items = [
        "Python 3.10+",
        "pip install -e .[dev,video]",
        "Git checkout matching the recorded commit when available",
    ]
    if dry_run:
        items.append("No GPU is required for dry-run reproduction.")
    else:
        items.extend(
            [
                "NVIDIA GPU with CUDA-compatible PyTorch",
                "Nerfstudio CLI installed",
                "LERF installed and registered with Nerfstudio",
                "COLMAP and FFmpeg available on PATH",
            ]
        )
    return items


def _artifacts(root: Path) -> list[ReproductionArtifact]:
    candidates = [
        ("pipeline_summary", root / "pipeline_summary.json", "pipeline_summary.json", "Top-level run status and provenance."),
        ("capture_manifest", root / "capture_manifest.md", "capture_manifest.md", "Scene-capture metadata and reproducibility context."),
        (
            "capture_manifest_validation",
            root / "capture_manifest_validation.md",
            "capture_manifest_validation.md",
            "Validation of capture conditions, overlap, static scene, and privacy review.",
        ),
        ("preflight_report", root / "preflight_report.md", "preflight_report.md", "Real-run readiness checks before training."),
        ("environment_report", root / "environment_report.json", "environment_report.json", "Runtime and upstream dependency checks."),
        ("scene_inspection", root / "scene_data_inspection.md", "scene_data_inspection.md", "Processed scene quality and pose readiness."),
        ("run_audit", root / "run_audit.md", "run_audit.md", "Run health audit."),
        ("recommendations", root / "run_recommendations.md", "run_recommendations.md", "Actionable next steps."),
        ("evidence_scorecard", root / "evidence_scorecard.md", "evidence_scorecard.md", "Portfolio evidence quality scorecard."),
        ("quality_gate", root / "quality_gate.md", "quality_gate.md", "Pass/warn/fail run quality gate report."),
        ("claim_audit", root / "claim_audit.md", "claim_audit.md", "Audit report for avoiding unsupported external-facing claims."),
        ("run_result_card", root / "run_result_card.md", "run_result_card.md", "One-page reviewer-facing summary of what this run proves and does not prove."),
        ("portfolio_page", root / "portfolio_page.html", "portfolio_page.html", "Static HTML page for sharing run evidence."),
        (
            "real_run_plan",
            root / "real_run_plan" / "real_run_plan.md",
            "real_run_plan/real_run_plan.md",
            "Action plan for upgrading smoke evidence into a real CUDA/Nerfstudio/LERF run.",
        ),
        ("research_report", root / "research_report.md", "research_report.md", "Paper-style report summarizing method, evidence, limitations, and next steps."),
        (
            "submission_checklist",
            root / "submission_packet" / "submission_checklist.md",
            "submission_packet/submission_checklist.md",
            "Claim-calibrated checklist for CV, portfolio, and professor outreach.",
        ),
        (
            "run_comparison",
            root.parent / "run_comparison.md",
            "../run_comparison.md",
            "Ranked comparison across repeated captures/training attempts.",
        ),
        ("query_grid", root / "demo_assets" / "query_grid.png", "demo_assets/query_grid.png", "Qualitative query visualization."),
        (
            "prompt_sensitivity",
            root / "prompt_sensitivity" / "prompt_sensitivity_report.md",
            "prompt_sensitivity/prompt_sensitivity_report.md",
            "Prompt wording stability report for open-vocabulary query variants.",
        ),
        (
            "scene_relations",
            root / "scene_relations" / "scene_relations_report.md",
            "scene_relations/scene_relations_report.md",
            "Scene-level object relation graph inferred from query boxes or 3D points.",
        ),
        ("evaluation_summary", root / "evaluation" / "eval_summary.json", "evaluation/eval_summary.json", "Quantitative/qualitative metric summary."),
        ("annotation_review", root / "evaluation" / "annotation_review.md", "evaluation/annotation_review.md", "Visual QA report for manual bbox annotations."),
        (
            "annotation_review_contact_sheet",
            root / "evaluation" / "annotation_review_contact_sheet.png",
            "evaluation/annotation_review_contact_sheet.png",
            "Contact sheet of manual bbox annotations over rendered views.",
        ),
        (
            "annotation_workbench",
            root / "evaluation" / "annotation_workbench" / "annotation_workbench.html",
            "evaluation/annotation_workbench/annotation_workbench.html",
            "Offline browser workbench for drawing and exporting manual bbox annotations.",
        ),
        ("portfolio_card", root / "portfolio_result_card.md", "portfolio_result_card.md", "Short project-page result narrative."),
        ("command_logs", root / "logs", "logs", "Subprocess command stdout/stderr records."),
    ]
    artifacts: list[ReproductionArtifact] = []
    for name, path, relative_path, purpose in candidates:
        artifacts.append(
            ReproductionArtifact(
                name=name,
                path=relative_path.replace("\\", "/"),
                exists=path.exists(),
                purpose=purpose,
            )
        )
    return artifacts


def _notes(*, dry_run: bool) -> list[str]:
    notes = [
        "Reproduction commands assume they are run from the repository root.",
        "Run artifacts are intentionally separated from checked-in source files.",
    ]
    if dry_run:
        notes.append("Dry-run reproduction validates pipeline wiring, not trained NeRF/LERF quality.")
    else:
        notes.append("Real-scene reproduction depends on upstream Nerfstudio/LERF/CUDA versions and capture quality.")
    return notes


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item).replace("\\", "/")) for item in command)


def _display_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return "."


def _command_lines(commands: list[str]) -> list[str]:
    if not commands:
        return ["- None."]
    return [f"- `{command}`" for command in commands]


def _artifact_lines(artifacts: list[ReproductionArtifact]) -> list[str]:
    if not artifacts:
        return ["- None."]
    return [
        f"- {artifact.name}: `{artifact.path}` ({'found' if artifact.exists else 'missing'}) - {artifact.purpose}"
        for artifact in artifacts
    ]


def _note_lines(notes: list[str]) -> list[str]:
    if not notes:
        return ["- None."]
    return [f"- {note}" for note in notes]
