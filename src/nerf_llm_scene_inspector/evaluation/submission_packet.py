"""Professor/CV submission packet generation from validated run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.evaluation.portfolio_validation import validate_portfolio_pack
from nerf_llm_scene_inspector.utils.paths import utc_timestamp


ChecklistStatus = Literal["pass", "warn", "fail"]
ReadinessLevel = Literal[
    "blocked",
    "needs_pack_validation",
    "shareable_smoke_demo",
    "real_run_review_ready",
    "portfolio_ready",
]


@dataclass
class SubmissionChecklistItem:
    """One claim-readiness check for sharing a run externally."""

    name: str
    status: ChecklistStatus
    evidence: str
    action: str = ""
    artifact: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SubmissionPacket:
    """Structured share packet for CV, portfolio, and professor outreach."""

    run_dir: str
    scene_name: str
    backend: str
    dry_run: bool
    readiness_level: ReadinessLevel
    share_decision: str
    generated_at: str
    readiness_summary: dict[str, Any] = field(default_factory=dict)
    repo_url: str = ""
    ci_url: str = ""
    pack_dir: str = ""
    pack_ok: bool | None = None
    recommended_links: dict[str, str] = field(default_factory=dict)
    allowed_claims: list[str] = field(default_factory=list)
    avoid_claims: list[str] = field(default_factory=list)
    checklist: list[SubmissionChecklistItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checklist"] = [item.to_dict() for item in self.checklist]
        return payload

    def to_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def to_markdown(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Submission Checklist",
            "",
            f"- Scene: `{self.scene_name}`",
            f"- Backend: `{self.backend}`",
            f"- Dry run: {self.dry_run}",
            f"- Readiness: `{self.readiness_level}`",
            f"- Share decision: {self.share_decision}",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Readiness Summary",
            "",
            *_readiness_summary_lines(self.readiness_summary),
            "",
            "## Recommended Links",
            "",
            *_dict_lines(self.recommended_links),
            "",
            "## Allowed Claims",
            "",
            *_list_lines(self.allowed_claims),
            "",
            "## Claims To Avoid",
            "",
            *_list_lines(self.avoid_claims),
            "",
            "## Checklist",
            "",
            *_checklist_lines(self.checklist),
            "",
            "## Warnings",
            "",
            *_list_lines(self.warnings),
            "",
            "## Next Actions",
            "",
            *_list_lines(self.next_actions),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def write_cv_entry(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# CV Project Entry",
            "",
            "- Built NeRF-LLM Scene Inspector, a reproducible research engineering system for open-vocabulary 3D scene inspection from phone video using Nerfstudio-style reconstruction and LERF-style language-field querying.",
            "- Implemented deterministic query planning, semantic relevancy artifacts, scene-relation analysis, annotation QA, evaluation summaries, research reports, and share-safe portfolio packaging.",
        ]
        if self.dry_run:
            lines.append(
                "- Current shared artifacts are a CPU-only dry-run smoke demo; real trained-scene claims require a CUDA/Nerfstudio/LERF run."
            )
        else:
            lines.append(
                "- Validated a real-scene run with run-level quality gates, reproducibility manifests, and portfolio packaging."
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def write_email_brief(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_mode = "CPU-only dry-run smoke demo" if self.dry_run else "real captured-scene run"
        lines = [
            "# Professor Outreach Brief",
            "",
            (
                "I built NeRF-LLM Scene Inspector as a research engineering project connecting "
                "Nerfstudio reconstruction, LERF-style language-embedded radiance fields, and "
                "natural-language 3D scene querying. The current share packet is a "
                f"{run_mode} with structured reports, visual query artifacts, evaluation scaffolding, "
                "and reproducibility checks."
            ),
            "",
            "I am not claiming a new NeRF architecture or state-of-the-art benchmark result; the emphasis is reproducible implementation and research-ready tooling for language-queryable physical scene representations.",
        ]
        if self.repo_url:
            lines.extend(["", f"Repository: {self.repo_url}"])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_submission_packet(
    run_dir: str | Path,
    *,
    pack_dir: str | Path | None = None,
    pack_validation_path: str | Path | None = None,
    repo_url: str = "",
    ci_url: str = "",
) -> SubmissionPacket:
    """Build a calibrated external-sharing packet from one run directory."""

    root = Path(run_dir)
    summary = _read_json(root / "pipeline_summary.json")
    scorecard = _read_json(root / "evidence_scorecard.json")
    quality = _read_json(root / "quality_gate.json")
    audit = _read_json(root / "run_audit.json")
    recommendations = _read_json(root / "run_recommendations.json")
    annotations = _read_json(root / "evaluation" / "annotation_validation.json")
    research = _read_json(root / "research_report.json")
    claim_audit = _read_json(root / "claim_audit.json")
    pack_validation = _pack_validation(pack_dir, pack_validation_path)

    scene_name = str(summary.get("scene_name") or scorecard.get("scene_name") or root.name)
    backend = str(summary.get("backend") or scorecard.get("backend") or research.get("backend") or "unknown")
    dry_run = bool(summary.get("dry_run", scorecard.get("dry_run", False)))
    checklist = _checklist(
        root,
        summary,
        scorecard,
        quality,
        audit,
        annotations,
        claim_audit,
        pack_validation,
        ci_url,
    )
    readiness = _readiness(dry_run, summary, scorecard, quality, pack_validation, checklist)
    warnings = _warnings(quality, audit, annotations, claim_audit, pack_validation)
    next_actions = _next_actions(recommendations, readiness, pack_validation)
    return SubmissionPacket(
        run_dir=_display_run_dir(root),
        scene_name=scene_name,
        backend=backend,
        dry_run=dry_run,
        readiness_level=readiness,
        share_decision=_share_decision(readiness),
        generated_at=utc_timestamp(),
        readiness_summary=_readiness_summary(
            readiness=readiness,
            checklist=checklist,
            warnings=warnings,
            next_actions=next_actions,
            pack_validation=pack_validation,
        ),
        repo_url=repo_url,
        ci_url=ci_url,
        pack_dir=_display_pack_path(Path(pack_dir)) if pack_dir else "",
        pack_ok=_pack_ok(pack_validation),
        recommended_links=_recommended_links(repo_url, ci_url, pack_dir),
        allowed_claims=_allowed_claims(dry_run, readiness),
        avoid_claims=_avoid_claims(dry_run),
        checklist=checklist,
        warnings=warnings,
        next_actions=next_actions,
    )


def write_submission_packet(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    pack_dir: str | Path | None = None,
    pack_validation_path: str | Path | None = None,
    repo_url: str = "",
    ci_url: str = "",
) -> SubmissionPacket:
    """Build and write JSON/Markdown share-packet artifacts."""

    root = Path(run_dir)
    output = Path(output_dir) if output_dir else root / "submission_packet"
    packet = build_submission_packet(
        root,
        pack_dir=pack_dir,
        pack_validation_path=pack_validation_path,
        repo_url=repo_url,
        ci_url=ci_url,
    )
    packet.to_json(output / "submission_packet.json")
    packet.to_markdown(output / "submission_checklist.md")
    packet.write_cv_entry(output / "cv_project_entry.md")
    packet.write_email_brief(output / "professor_email_brief.md")
    return packet


def _checklist(
    root: Path,
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    quality: dict[str, Any],
    audit: dict[str, Any],
    annotations: dict[str, Any],
    claim_audit: dict[str, Any],
    pack_validation: dict[str, Any],
    ci_url: str,
) -> list[SubmissionChecklistItem]:
    return [
        _item(
            "pipeline_success",
            summary.get("success") is True,
            f"pipeline_summary.success={summary.get('success')}",
            "Rerun or debug the pipeline before sharing.",
            "pipeline_summary.json",
        ),
        _item(
            "research_report",
            (root / "research_report.md").exists() and (root / "research_report.json").exists(),
            "research_report.md/json present",
            "Run generate_research_report.py.",
            "research_report.md",
        ),
        _item(
            "portfolio_page",
            (root / "portfolio_page.html").exists(),
            "portfolio_page.html present",
            "Run generate_portfolio_page.py after reports are refreshed.",
            "portfolio_page.html",
        ),
        _item(
            "reproduction_bundle",
            (root / "reproduction_manifest.json").exists() and (root / "reproduction_report.md").exists(),
            "reproduction manifest/report present",
            "Run create_reproduction_bundle.py.",
            "reproduction_report.md",
        ),
        _scorecard_item(scorecard),
        _quality_item(quality),
        _audit_item(audit),
        _annotation_item(annotations),
        _claim_audit_item(claim_audit),
        _pack_item(pack_validation),
        _path_leak_item(pack_validation),
        SubmissionChecklistItem(
            "claim_calibration",
            "warn" if bool(summary.get("dry_run", scorecard.get("dry_run", False))) else "pass",
            "dry-run artifacts must be described as smoke evidence only"
            if bool(summary.get("dry_run", scorecard.get("dry_run", False)))
            else "real-run mode recorded",
            "Run a real CUDA-backed scene before claiming real trained-scene performance."
            if bool(summary.get("dry_run", scorecard.get("dry_run", False)))
            else "",
            "pipeline_summary.json",
        ),
        SubmissionChecklistItem(
            "ci_status",
            "pass" if ci_url else "warn",
            ci_url or "CI URL not recorded in packet",
            "Attach the latest successful GitHub Actions run URL before professor outreach.",
            "",
        ),
    ]


def _scorecard_item(scorecard: dict[str, Any]) -> SubmissionChecklistItem:
    if not scorecard:
        return SubmissionChecklistItem(
            "evidence_scorecard",
            "fail",
            "evidence_scorecard.json missing",
            "Run create_evidence_scorecard.py.",
            "evidence_scorecard.json",
        )
    score = scorecard.get("score")
    maximum = scorecard.get("max_score")
    return SubmissionChecklistItem(
        "evidence_scorecard",
        "pass" if scorecard.get("evidence_level") in {"dry_run_demo_ready", "portfolio_ready_real_run"} else "warn",
        f"level={scorecard.get('evidence_level')}, score={score}/{maximum}",
        "" if scorecard.get("evidence_level") in {"dry_run_demo_ready", "portfolio_ready_real_run"} else "Review evidence_scorecard.md recommendations.",
        "evidence_scorecard.md",
    )


def _quality_item(quality: dict[str, Any]) -> SubmissionChecklistItem:
    if not quality:
        return SubmissionChecklistItem("quality_gate", "fail", "quality_gate.json missing", "Run check_run_quality.py.", "quality_gate.json")
    status = str(quality.get("status") or "")
    return SubmissionChecklistItem(
        "quality_gate",
        "fail" if status == "fail" else ("warn" if status == "warn" else "pass"),
        f"profile={quality.get('profile')}, status={status}, passed={quality.get('passed')}",
        "Resolve failed quality-gate criteria." if status == "fail" else "Review warnings before sharing." if status == "warn" else "",
        "quality_gate.md",
    )


def _audit_item(audit: dict[str, Any]) -> SubmissionChecklistItem:
    status = str(audit.get("status") or "")
    if not audit:
        return SubmissionChecklistItem("run_audit", "fail", "run_audit.json missing", "Run audit_run.py.", "run_audit.md")
    return SubmissionChecklistItem(
        "run_audit",
        "pass" if status == "ready" else "warn",
        f"status={status}, score={audit.get('score')}",
        "Review run_audit.md warnings before using as evidence." if status != "ready" else "",
        "run_audit.md",
    )


def _annotation_item(annotations: dict[str, Any]) -> SubmissionChecklistItem:
    warnings = annotations.get("warnings") if isinstance(annotations.get("warnings"), list) else []
    if not annotations:
        return SubmissionChecklistItem(
            "annotations",
            "warn",
            "annotation validation missing",
            "Run validate_annotations.py/review_annotations.py before reporting metrics.",
            "evaluation/annotation_validation.json",
        )
    return SubmissionChecklistItem(
        "annotations",
        "warn" if warnings else "pass",
        f"ok={annotations.get('ok')}, warnings={len(warnings)}",
        "Resolve annotation warnings before reporting quantitative localization numbers." if warnings else "",
        "evaluation/annotation_validation.json",
    )


def _claim_audit_item(claim_audit: dict[str, Any]) -> SubmissionChecklistItem:
    if not claim_audit:
        return SubmissionChecklistItem(
            "claim_audit",
            "warn",
            "claim audit missing",
            "Run audit_claims.py before external sharing.",
            "claim_audit.json",
        )
    status = str(claim_audit.get("status") or "")
    if status == "fail" or claim_audit.get("ok") is False:
        checklist_status: ChecklistStatus = "fail"
        action = "Fix unsupported external-facing claims before sharing."
    elif status == "warn":
        checklist_status = "warn"
        action = "Review claim_audit.md warnings before professor outreach."
    else:
        checklist_status = "pass"
        action = ""
    return SubmissionChecklistItem(
        "claim_audit",
        checklist_status,
        f"status={status}, fails={claim_audit.get('fail_count', 0)}, warnings={claim_audit.get('warn_count', 0)}",
        action,
        "claim_audit.md",
    )


def _pack_item(pack_validation: dict[str, Any]) -> SubmissionChecklistItem:
    if not pack_validation:
        return SubmissionChecklistItem(
            "portfolio_pack",
            "warn",
            "portfolio pack was not validated for this packet",
            "Run finalize_annotations.py with --export-pack --zip-pack, then regenerate this packet with --pack.",
            "results/portfolio_pack",
        )
    return SubmissionChecklistItem(
        "portfolio_pack",
        "pass" if pack_validation.get("ok") is True else "fail",
        f"ok={pack_validation.get('ok')}, warnings={len(pack_validation.get('warnings') or [])}, errors={len(pack_validation.get('errors') or [])}",
        "Fix pack validation errors before sharing externally." if pack_validation.get("ok") is not True else "",
        "portfolio_pack_validation.json",
    )


def _path_leak_item(pack_validation: dict[str, Any]) -> SubmissionChecklistItem:
    leaks = pack_validation.get("path_leaks") if isinstance(pack_validation.get("path_leaks"), list) else []
    if not pack_validation:
        return SubmissionChecklistItem(
            "path_leaks",
            "warn",
            "no portfolio pack validation available",
            "Validate the pack to check for local path leakage.",
            "portfolio_pack_validation.json",
        )
    return SubmissionChecklistItem(
        "path_leaks",
        "fail" if leaks else "pass",
        f"path_leaks={len(leaks)}",
        "Regenerate the portfolio pack after fixing local path leakage." if leaks else "",
        "portfolio_pack_validation.json",
    )


def _item(
    name: str,
    ok: bool,
    evidence: str,
    action: str,
    artifact: str,
) -> SubmissionChecklistItem:
    return SubmissionChecklistItem(name, "pass" if ok else "fail", evidence, "" if ok else action, artifact)


def _readiness(
    dry_run: bool,
    summary: dict[str, Any],
    scorecard: dict[str, Any],
    quality: dict[str, Any],
    pack_validation: dict[str, Any],
    checklist: list[SubmissionChecklistItem],
) -> ReadinessLevel:
    if any(item.status == "fail" and item.name not in {"ci_status"} for item in checklist):
        return "blocked"
    if not pack_validation or pack_validation.get("ok") is not True:
        return "needs_pack_validation"
    if dry_run:
        return "shareable_smoke_demo"
    if (
        summary.get("success") is True
        and scorecard.get("evidence_level") == "portfolio_ready_real_run"
        and quality.get("status") == "pass"
    ):
        return "portfolio_ready"
    return "real_run_review_ready"


def _share_decision(readiness: ReadinessLevel) -> str:
    decisions = {
        "blocked": "Do not share yet; one or more required evidence artifacts failed.",
        "needs_pack_validation": "Regenerate and validate the portfolio pack before external sharing.",
        "shareable_smoke_demo": "Share as a CPU-only research-engineering smoke demo with clear dry-run wording.",
        "real_run_review_ready": "Share selectively after reviewing warnings; avoid benchmark-style claims.",
        "portfolio_ready": "Ready for portfolio sharing with the recorded evidence and limitations.",
    }
    return decisions[readiness]


def _readiness_summary(
    *,
    readiness: ReadinessLevel,
    checklist: list[SubmissionChecklistItem],
    warnings: list[str],
    next_actions: list[str],
    pack_validation: dict[str, Any],
) -> dict[str, Any]:
    failed_items = [item for item in checklist if item.status == "fail"]
    warning_items = [item for item in checklist if item.status == "warn"]
    status = "fail" if failed_items else "warn" if warning_items or warnings else "pass"
    top_warnings = [_item_summary(item) for item in warning_items[:5]]
    top_warnings.extend(f"packet_warning: {warning}" for warning in warnings[:5])
    return {
        "status": status,
        "readiness_level": readiness,
        "failed_check_count": len(failed_items),
        "warning_check_count": len(warning_items),
        "packet_warning_count": len(warnings),
        "failed_checks": [item.name for item in failed_items],
        "warning_checks": [item.name for item in warning_items],
        "top_blockers": [_item_summary(item) for item in failed_items[:5]],
        "top_warnings": _dedupe(top_warnings)[:5],
        "pack_ok": _pack_ok(pack_validation),
        "recommended_next_action": next_actions[0] if next_actions else _default_next_action(readiness),
    }


def _item_summary(item: SubmissionChecklistItem) -> str:
    summary = f"{item.name}: {item.evidence}"
    if item.action:
        summary += f" Action: {item.action}"
    if item.artifact:
        summary += f" Artifact: {item.artifact}"
    return summary


def _default_next_action(readiness: ReadinessLevel) -> str:
    if readiness == "blocked":
        return "Resolve failed checklist items, then regenerate the submission packet."
    if readiness == "needs_pack_validation":
        return "Export and validate the portfolio pack, then regenerate the submission packet with --pack."
    if readiness == "shareable_smoke_demo":
        return "Share only as a dry-run smoke demo, or run a real CUDA-backed scene for stronger evidence."
    if readiness == "real_run_review_ready":
        return "Review warning-level items before sending externally."
    return "Attach the repository, portfolio pack, and latest successful CI URL when sharing."


def _allowed_claims(dry_run: bool, readiness: ReadinessLevel) -> list[str]:
    claims = [
        "Built a reproducible research engineering project on Nerfstudio and LERF-style language fields.",
        "Implemented open-vocabulary 3D scene query artifacts, evaluation scaffolding, and portfolio packaging.",
    ]
    if dry_run:
        claims.append("The checked artifacts demonstrate CPU-safe pipeline wiring and artifact formats.")
    else:
        claims.append("Ran the pipeline on a captured scene with recorded quality gates and reproducibility artifacts.")
    if readiness == "portfolio_ready":
        claims.append("The run passed the portfolio quality profile for the recorded evidence.")
    return claims


def _avoid_claims(dry_run: bool) -> list[str]:
    claims = [
        "Do not claim a new NeRF architecture.",
        "Do not claim state-of-the-art detection, segmentation, or 3D grounding performance.",
        "Do not present lightweight single-scene metrics as benchmark results.",
    ]
    if dry_run:
        claims.append("Do not describe dry-run overlays as trained LERF outputs from a real scene.")
    return claims


def _warnings(
    quality: dict[str, Any],
    audit: dict[str, Any],
    annotations: dict[str, Any],
    claim_audit: dict[str, Any],
    pack_validation: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if quality.get("status") == "warn":
        warnings.append("Quality gate has warning-level criteria.")
    if audit.get("status") and audit.get("status") != "ready":
        warnings.append(f"Run audit status is {audit.get('status')}.")
    for item in annotations.get("warnings") or []:
        warnings.append(f"Annotation warning: {item}")
    if claim_audit.get("status") == "warn":
        warnings.append("Claim audit has warning-level findings.")
    if claim_audit.get("status") == "fail":
        warnings.append("Claim audit has failure-level findings.")
    for item in pack_validation.get("warnings") or []:
        warnings.append(f"Pack warning: {item}")
    return warnings


def _next_actions(
    recommendations: dict[str, Any],
    readiness: ReadinessLevel,
    pack_validation: dict[str, Any],
) -> list[str]:
    actions = [
        str(item.get("action"))
        for item in recommendations.get("recommendations") or []
        if isinstance(item, dict) and item.get("action")
    ]
    if readiness == "needs_pack_validation":
        actions.insert(
            0,
            "Finalize annotations with --export-pack --zip-pack, then regenerate this submission packet with --pack.",
        )
    if pack_validation.get("warnings"):
        actions.append("Review portfolio pack validation warnings before sending the packet externally.")
    return _dedupe(actions)[:8]


def _recommended_links(
    repo_url: str,
    ci_url: str,
    pack_dir: str | Path | None,
) -> dict[str, str]:
    links = {
        "research_report": "research_report.md",
        "run_result_card": "run_result_card.md",
        "portfolio_page": "portfolio_page.html",
        "reproduction_report": "reproduction_report.md",
        "evidence_scorecard": "evidence_scorecard.md",
        "quality_gate": "quality_gate.md",
    }
    if repo_url:
        links["repository"] = repo_url
    if ci_url:
        links["ci_run"] = ci_url
    if pack_dir:
        links["portfolio_pack"] = _display_pack_path(Path(pack_dir))
    return {key: value for key, value in links.items() if value}


def _pack_validation(
    pack_dir: str | Path | None,
    pack_validation_path: str | Path | None,
) -> dict[str, Any]:
    if pack_validation_path:
        return _read_json(pack_validation_path)
    if not pack_dir:
        return {}
    pack = Path(pack_dir)
    if not pack.exists():
        return {"ok": False, "errors": [f"Pack path does not exist: {_display_pack_path(pack)}"]}
    return validate_portfolio_pack(pack).to_dict()


def _pack_ok(pack_validation: dict[str, Any]) -> bool | None:
    if not pack_validation:
        return None
    return bool(pack_validation.get("ok"))


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _display_run_dir(path: Path) -> str:
    return path.name or "run"


def _display_pack_path(path: Path) -> str:
    return path.name or "portfolio_pack"


def _dict_lines(items: dict[str, str]) -> list[str]:
    if not items:
        return ["- None recorded."]
    return [f"- {key}: `{value}`" for key, value in items.items()]


def _readiness_summary_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return ["- No readiness summary was recorded."]
    lines = [
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Readiness level: `{summary.get('readiness_level', 'unknown')}`",
        f"- Failed checks: {summary.get('failed_check_count', 0)}",
        f"- Warning checks: {summary.get('warning_check_count', 0)}",
        f"- Packet warnings: {summary.get('packet_warning_count', 0)}",
        f"- Pack OK: `{summary.get('pack_ok')}`",
        f"- Recommended next action: {summary.get('recommended_next_action') or 'Review the checklist.'}",
    ]
    blockers = summary.get("top_blockers") if isinstance(summary.get("top_blockers"), list) else []
    if blockers:
        lines.append("- Top blockers:")
        lines.extend(f"  - {item}" for item in blockers)
    warnings = summary.get("top_warnings") if isinstance(summary.get("top_warnings"), list) else []
    if warnings:
        lines.append("- Top warnings:")
        lines.extend(f"  - {item}" for item in warnings)
    return lines


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _checklist_lines(items: list[SubmissionChecklistItem]) -> list[str]:
    if not items:
        return ["- No checklist items were generated."]
    return [
        f"- `{item.status}` {item.name}: {item.evidence}"
        + (f" Action: {item.action}" if item.action else "")
        + (f" Artifact: `{item.artifact}`" if item.artifact else "")
        for item in items
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
