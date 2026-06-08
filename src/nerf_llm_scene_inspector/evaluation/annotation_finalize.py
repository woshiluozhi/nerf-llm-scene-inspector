"""Finalize a run after manual annotation workbench edits."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp
from nerf_llm_scene_inspector.utils.shell import format_command, run_command


FinalizeStatus = Literal["success", "failed", "skipped"]


@dataclass
class AnnotationFinalizeStep:
    """One command executed during annotation finalization."""

    name: str
    status: FinalizeStatus
    command: str
    returncode: int | None = None
    log_path: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnnotationFinalizeReport:
    """Structured result for post-workbench annotation finalization."""

    ok: bool
    run_dir: str
    filled_path: str
    profile: str
    generated_at: str
    merged_annotations: str
    merge_report: str
    pack_dir: str = ""
    exported_pack: bool = False
    steps: list[AnnotationFinalizeStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "run_dir": self.run_dir,
            "filled_path": self.filled_path,
            "profile": self.profile,
            "generated_at": self.generated_at,
            "merged_annotations": self.merged_annotations,
            "merge_report": self.merge_report,
            "pack_dir": self.pack_dir,
            "exported_pack": self.exported_pack,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
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
            "# Annotation Finalization Report",
            "",
            f"- OK: {self.ok}",
            f"- Run directory: `{self.run_dir}`",
            f"- Filled annotations: `{self.filled_path}`",
            f"- Merged annotations: `{self.merged_annotations}`",
            f"- Merge report: `{self.merge_report}`",
            f"- Quality profile: `{self.profile}`",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Steps",
            "",
            "| Step | Status | Return Code | Outputs | Log |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for step in self.steps:
            outputs = "<br>".join(f"`{key}`: `{value}`" for key, value in step.outputs.items()) or ""
            log = f"`{step.log_path}`" if step.log_path else ""
            code = "" if step.returncode is None else str(step.returncode)
            lines.append(f"| {step.name} | {step.status} | {code} | {outputs} | {log} |")
        lines.extend(["", "## Errors", "", *_list_lines(self.errors), "", "## Warnings", "", *_list_lines(self.warnings)])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def finalize_workbench_annotations(
    *,
    run_dir: str | Path,
    filled_path: str | Path,
    profile: str = "smoke",
    pack_dir: str | Path | None = None,
    export_pack: bool = False,
    zip_pack: bool = False,
    dry_run_eval: bool = False,
    top_k: int = 5,
    repo_url: str = "",
    continue_on_error: bool = False,
    report_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> AnnotationFinalizeReport:
    """Merge workbench annotations, rerun evaluation, and refresh run-facing artifacts."""

    root = project_root()
    run_root = Path(run_dir)
    logs_dir = run_root / "logs"
    merged = run_root / "annotations_merged.json"
    merge_report = run_root / "annotation_merge_report.json"
    pack = Path(pack_dir) if pack_dir else Path("results/portfolio_pack")
    steps: list[AnnotationFinalizeStep] = []
    errors: list[str] = []
    warnings: list[str] = []

    command_specs = _command_specs(
        run_root=run_root,
        filled=Path(filled_path),
        profile=profile,
        pack=pack,
        use_pack=export_pack or pack_dir is not None,
        export_pack=export_pack,
        zip_pack=zip_pack,
        dry_run_eval=dry_run_eval,
        top_k=top_k,
        repo_url=repo_url,
    )
    for spec in command_specs:
        if errors and not continue_on_error and spec.required_after_failure:
            steps.append(
                AnnotationFinalizeStep(
                    name=spec.name,
                    status="skipped",
                    command=format_command(_display_command(spec.command)),
                    outputs=spec.outputs,
                )
            )
            continue
        step = _run_spec(spec, root=root, logs_dir=logs_dir)
        steps.append(step)
        if step.status == "failed":
            message = f"{spec.name} failed with return code {step.returncode}."
            if step.stderr_tail:
                message += f" stderr: {step.stderr_tail.strip()}"
            if spec.critical:
                errors.append(message)
            else:
                warnings.append(message)
                continue
            if not continue_on_error:
                continue

    ok = not errors
    report = AnnotationFinalizeReport(
        ok=ok,
        run_dir=_display_path(run_root),
        filled_path=_display_path(Path(filled_path)),
        profile=profile,
        generated_at=utc_timestamp(),
        merged_annotations=_display_path(merged),
        merge_report=_display_path(merge_report),
        pack_dir=_display_path(pack) if (export_pack or pack_dir) else "",
        exported_pack=export_pack,
        steps=steps,
        warnings=warnings,
        errors=errors,
    )
    report.to_json(report_output or run_root / "annotation_finalize_report.json")
    report.to_markdown(markdown_output or run_root / "annotation_finalize_report.md")
    return report


@dataclass(frozen=True)
class _CommandSpec:
    name: str
    command: list[str]
    outputs: dict[str, str]
    critical: bool = True
    required_after_failure: bool = True


def _command_specs(
    *,
    run_root: Path,
    filled: Path,
    profile: str,
    pack: Path,
    use_pack: bool,
    export_pack: bool,
    zip_pack: bool,
    dry_run_eval: bool,
    top_k: int,
    repo_url: str,
) -> list[_CommandSpec]:
    py = sys.executable
    eval_dir = run_root / "evaluation"
    merged = run_root / "annotations_merged.json"
    merge_report = run_root / "annotation_merge_report.json"
    commands = [
        _CommandSpec(
            "merge_annotation_workbench",
            [
                py,
                "scripts/merge_annotation_workbench.py",
                "--template",
                str(run_root / "annotation_template.json"),
                "--filled",
                str(filled),
                "--output",
                str(merged),
                "--queries",
                str(run_root / "queries.yaml"),
                "--results",
                str(run_root / "queries"),
                "--report-output",
                str(merge_report),
                "--overwrite",
            ],
            {"annotations": str(merged), "merge_report": str(merge_report)},
        ),
        _CommandSpec(
            "evaluate_queries",
            [
                py,
                "scripts/evaluate_queries.py",
                "--queries",
                str(run_root / "queries.yaml"),
                "--annotations",
                str(merged),
                "--results",
                str(run_root / "queries"),
                "--output",
                str(eval_dir),
                "--report-output",
                str(run_root / "project_report.md"),
                "--top-k",
                str(top_k),
                *(["--dry-run"] if dry_run_eval else []),
            ],
            {
                "eval_summary": str(eval_dir / "eval_summary.json"),
                "eval_table": str(eval_dir / "eval_table.csv"),
                "annotation_validation": str(eval_dir / "annotation_validation.json"),
                "qualitative_report": str(eval_dir / "qualitative_report.md"),
            },
        ),
        _CommandSpec(
            "review_annotations",
            [
                py,
                "scripts/review_annotations.py",
                "--annotations",
                str(merged),
                "--results",
                str(run_root / "queries"),
                "--output",
                str(eval_dir),
                "--allow-warnings",
            ],
            {
                "annotation_review": str(eval_dir / "annotation_review.json"),
                "annotation_review_markdown": str(eval_dir / "annotation_review.md"),
                "contact_sheet": str(eval_dir / "annotation_review_contact_sheet.png"),
            },
        ),
        _CommandSpec(
            "audit_run",
            [py, "scripts/audit_run.py", "--run-dir", str(run_root)],
            {"run_audit": str(run_root / "run_audit.json"), "run_audit_markdown": str(run_root / "run_audit.md")},
            critical=False,
        ),
        _CommandSpec(
            "recommend_next_steps",
            [py, "scripts/recommend_next_steps.py", "--run-dir", str(run_root)],
            {
                "run_recommendations": str(run_root / "run_recommendations.json"),
                "run_recommendations_markdown": str(run_root / "run_recommendations.md"),
            },
            critical=False,
        ),
        _CommandSpec(
            "create_evidence_scorecard",
            [py, "scripts/create_evidence_scorecard.py", "--run-dir", str(run_root)],
            {
                "evidence_scorecard": str(run_root / "evidence_scorecard.json"),
                "evidence_scorecard_markdown": str(run_root / "evidence_scorecard.md"),
            },
            critical=False,
        ),
        _CommandSpec(
            "check_run_quality",
            [
                py,
                "scripts/check_run_quality.py",
                "--run-dir",
                str(run_root),
                "--profile",
                profile,
                *(["--pack", str(pack)] if use_pack and not export_pack else []),
                *(["--no-require-pack"] if (export_pack or (not use_pack and profile != "portfolio")) else []),
            ],
            {"quality_gate": str(run_root / "quality_gate.json"), "quality_gate_markdown": str(run_root / "quality_gate.md")},
            critical=False,
        ),
        _CommandSpec(
            "generate_research_report",
            [py, "scripts/generate_research_report.py", "--run-dir", str(run_root)],
            {"research_report": str(run_root / "research_report.md"), "research_report_json": str(run_root / "research_report.json")},
            critical=False,
        ),
        _CommandSpec(
            "generate_portfolio_page",
            [py, "scripts/generate_portfolio_page.py", "--run-dir", str(run_root)],
            {"portfolio_page": str(run_root / "portfolio_page.html")},
            critical=False,
        ),
        _CommandSpec(
            "create_real_run_plan",
            [py, "scripts/create_real_run_plan.py", "--run-dir", str(run_root)],
            {"real_run_plan": str(run_root / "real_run_plan" / "real_run_plan.json")},
            critical=False,
        ),
        _CommandSpec(
            "create_reproduction_bundle",
            [py, "scripts/create_reproduction_bundle.py", "--run-dir", str(run_root)],
            {"reproduction_manifest": str(run_root / "reproduction_manifest.json")},
            critical=False,
        ),
    ]
    if export_pack:
        commands.extend(
            [
                _CommandSpec(
                    "export_portfolio_pack",
                    [
                        py,
                        "scripts/export_portfolio_pack.py",
                        "--run-dir",
                        str(run_root),
                        "--output",
                        str(pack),
                        *(["--zip"] if zip_pack else []),
                    ],
                    {"portfolio_pack": str(pack), "portfolio_pack_zip": str(pack) + ".zip"},
                    critical=False,
                ),
                _CommandSpec(
                    "validate_portfolio_pack",
                    [py, "scripts/validate_portfolio_pack.py", "--pack", str(pack)],
                    {"portfolio_pack_validation": str(pack / "portfolio_pack_validation.json")},
                    critical=False,
                ),
                _CommandSpec(
                    "refresh_quality_gate_with_pack",
                    [
                        py,
                        "scripts/check_run_quality.py",
                        "--run-dir",
                        str(run_root),
                        "--profile",
                        profile,
                        "--pack",
                        str(pack),
                    ],
                    {
                        "quality_gate": str(run_root / "quality_gate.json"),
                        "quality_gate_markdown": str(run_root / "quality_gate.md"),
                    },
                    critical=False,
                ),
            ]
        )
    commands.extend(
        [
            _CommandSpec(
                "audit_claims",
                [
                    py,
                    "scripts/audit_claims.py",
                    "--run-dir",
                    str(run_root),
                    *(["--pack", str(pack)] if use_pack else []),
                ],
                {"claim_audit": str(run_root / "claim_audit.json"), "claim_audit_markdown": str(run_root / "claim_audit.md")},
                critical=False,
            ),
            _CommandSpec(
                "create_submission_packet",
                [
                    py,
                    "scripts/create_submission_packet.py",
                    "--run-dir",
                    str(run_root),
                    *(["--pack", str(pack)] if use_pack else []),
                    *(["--repo-url", repo_url] if repo_url else []),
                ],
                {"submission_packet": str(run_root / "submission_packet" / "submission_packet.json")},
                critical=False,
            ),
            _CommandSpec(
                "create_run_readiness",
                [
                    py,
                    "scripts/create_run_readiness.py",
                    "--run-dir",
                    str(run_root),
                    *(["--pack", str(pack)] if use_pack else []),
                ],
                {
                    "run_readiness": str(run_root / "run_readiness.json"),
                    "run_readiness_markdown": str(run_root / "run_readiness.md"),
                },
                critical=False,
            ),
            _CommandSpec(
                "create_run_result_card",
                [py, "scripts/create_run_result_card.py", "--run-dir", str(run_root)],
                {"run_result_card": str(run_root / "run_result_card.json"), "run_result_card_markdown": str(run_root / "run_result_card.md")},
                critical=False,
            ),
            _CommandSpec(
                "refresh_research_report",
                [py, "scripts/generate_research_report.py", "--run-dir", str(run_root)],
                {"research_report": str(run_root / "research_report.md"), "research_report_json": str(run_root / "research_report.json")},
                critical=False,
            ),
            _CommandSpec(
                "refresh_portfolio_page",
                [py, "scripts/generate_portfolio_page.py", "--run-dir", str(run_root)],
                {"portfolio_page": str(run_root / "portfolio_page.html")},
                critical=False,
            ),
            _CommandSpec(
                "refresh_reproduction_bundle",
                [py, "scripts/create_reproduction_bundle.py", "--run-dir", str(run_root)],
                {"reproduction_manifest": str(run_root / "reproduction_manifest.json")},
                critical=False,
            ),
            _CommandSpec(
                "index_runs",
                [py, "scripts/index_runs.py", "--root", str(run_root.parent)],
                {"run_index": str(run_root.parent / "run_index.json"), "run_index_markdown": str(run_root.parent / "run_index.md")},
                critical=False,
            ),
            _CommandSpec(
                "compare_runs",
                [py, "scripts/compare_runs.py", "--root", str(run_root.parent)],
                {"run_comparison": str(run_root.parent / "run_comparison.json"), "run_comparison_markdown": str(run_root.parent / "run_comparison.md")},
                critical=False,
            ),
        ]
    )
    if export_pack:
        commands.extend(
            [
                _CommandSpec(
                    "final_export_portfolio_pack",
                    [
                        py,
                        "scripts/export_portfolio_pack.py",
                        "--run-dir",
                        str(run_root),
                        "--output",
                        str(pack),
                        *(["--zip"] if zip_pack else []),
                    ],
                    {"portfolio_pack": str(pack), "portfolio_pack_zip": str(pack) + ".zip"},
                    critical=False,
                ),
                _CommandSpec(
                    "final_validate_portfolio_pack",
                    [py, "scripts/validate_portfolio_pack.py", "--pack", str(pack)],
                    {"portfolio_pack_validation": str(pack / "portfolio_pack_validation.json")},
                    critical=False,
                ),
            ]
        )
        if zip_pack:
            zip_path = Path(f"{pack}.zip")
            zip_validation_path = zip_path.with_name(f"{zip_path.stem}_validation.json")
            commands.append(
                _CommandSpec(
                    "final_archive_portfolio_pack",
                    [
                        py,
                        "-c",
                        "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])",
                        str(pack),
                        str(pack),
                    ],
                    {"portfolio_pack_zip": str(pack) + ".zip"},
                    critical=False,
                )
            )
            commands.append(
                _CommandSpec(
                    "final_validate_portfolio_zip",
                    [
                        py,
                        "scripts/validate_portfolio_pack.py",
                        "--pack",
                        str(zip_path),
                    ],
                    {"portfolio_pack_zip_validation": str(zip_validation_path)},
                    critical=False,
                )
            )
    return commands


def _run_spec(spec: _CommandSpec, *, root: Path, logs_dir: Path) -> AnnotationFinalizeStep:
    log_path = logs_dir / f"finalize_{spec.name}_command.json"
    result = run_command(spec.command, cwd=root, log_path=log_path, check=False)
    return AnnotationFinalizeStep(
        name=spec.name,
        status="success" if result.ok else "failed",
        command=format_command(_display_command(spec.command)),
        returncode=result.returncode,
        log_path=_display_path(log_path),
        outputs={key: _display_path(Path(value)) for key, value in spec.outputs.items()},
        stdout_tail=result.stdout[-1000:],
        stderr_tail=result.stderr[-1000:],
    )


def _display_command(command: list[str]) -> list[str]:
    if not command:
        return []
    displayed = list(command)
    displayed[0] = "python" if Path(displayed[0]).name.lower().startswith("python") else displayed[0]
    return [item.replace("\\", "/") for item in displayed]


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]
