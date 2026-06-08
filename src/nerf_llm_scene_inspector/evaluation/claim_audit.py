"""Audit project and run-facing text for calibrated research claims."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import project_root, utc_timestamp


FindingSeverity = Literal["fail", "warn"]
CheckStatus = Literal["pass", "warn", "fail"]

TEXT_SUFFIXES = {".html", ".json", ".md", ".txt", ".yaml", ".yml"}
PROJECT_CLAIM_FILES = [
    "README.md",
    "docs/project_report.md",
    "docs/method_summary.md",
    "docs/portfolio_result_card.md",
    "docs/cv_bullets.md",
    "docs/cold_email_paragraph.md",
    "docs/real_run_reproducibility.md",
    "docs/real_scene_capture_checklist.md",
    "docs/index.html",
]
RUN_CLAIM_FILES = [
    "project_report.md",
    "portfolio_result_card.md",
    "portfolio_page.html",
    "research_report.md",
    "run_result_card.md",
    "run_result_card.json",
    "reproduction_report.md",
    "real_run_plan/real_run_plan.md",
    "submission_packet/submission_checklist.md",
    "submission_packet/cv_project_entry.md",
    "submission_packet/professor_email_brief.md",
]
PACK_CLAIM_ROOTS = ["project", "run"]


@dataclass(frozen=True)
class ClaimRule:
    """Regex rule for one risky external-facing claim pattern."""

    name: str
    pattern: re.Pattern[str]
    severity: FindingSeverity
    reason: str
    recommendation: str


@dataclass
class ClaimFinding:
    """One risky claim occurrence found in external-facing text."""

    file: str
    line: int
    rule: str
    severity: FindingSeverity
    text: str
    reason: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ClaimCheck:
    """One required claim-calibration check."""

    name: str
    status: CheckStatus
    detail: str
    artifact: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ClaimAuditReport:
    """Structured report for claim safety before CV or professor outreach."""

    ok: bool
    status: CheckStatus
    generated_at: str
    root: str
    run_dir: str = ""
    pack_dir: str = ""
    dry_run: bool | None = None
    scanned_files: list[str] = field(default_factory=list)
    findings: list[ClaimFinding] = field(default_factory=list)
    checks: list[ClaimCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fail_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "fail") + sum(
            1 for check in self.checks if check.status == "fail"
        )

    @property
    def warn_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warn") + sum(
            1 for check in self.checks if check.status == "warn"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "generated_at": self.generated_at,
            "root": self.root,
            "run_dir": self.run_dir,
            "pack_dir": self.pack_dir,
            "dry_run": self.dry_run,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "scanned_files": list(self.scanned_files),
            "findings": [finding.to_dict() for finding in self.findings],
            "checks": [check.to_dict() for check in self.checks],
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
            "# Claim Audit",
            "",
            f"- Status: `{self.status}`",
            f"- OK: {self.ok}",
            f"- Failed items: {self.fail_count}",
            f"- Warning items: {self.warn_count}",
            f"- Dry run: {self.dry_run}",
            f"- Scanned files: {len(self.scanned_files)}",
            f"- Generated: `{self.generated_at}`",
            "",
            "## Required Checks",
            "",
            *_check_lines(self.checks),
            "",
            "## Risky Claim Findings",
            "",
            *_finding_lines(self.findings),
            "",
            "## Warnings",
            "",
            *_list_lines(self.warnings),
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


CLAIM_RULES = [
    ClaimRule(
        "sota_or_benchmark_claim",
        re.compile(r"\b(state[- ]of[- ]the[- ]art|sota|benchmark[- ]leading)\b", re.IGNORECASE),
        "fail",
        "State-of-the-art or benchmark-leading wording requires strong benchmark evidence.",
        "Replace with implementation-focused wording or explicitly state that no SOTA claim is made.",
    ),
    ClaimRule(
        "novel_architecture_claim",
        re.compile(
            r"\b(novel|new)\s+(NeRF|radiance field|language[- ]field|architecture|method|model)\b",
            re.IGNORECASE,
        ),
        "fail",
        "Novel architecture or method wording overstates this repository's contribution.",
        "Describe the project as research engineering built on Nerfstudio/LERF unless a real method contribution is documented.",
    ),
    ClaimRule(
        "outperforms_claim",
        re.compile(
            r"\b(outperforms?|beats?|surpasses?|exceeds?)\b.{0,80}\b(baseline|benchmark|method|SOTA|state[- ]of[- ]the[- ]art)\b",
            re.IGNORECASE,
        ),
        "fail",
        "Performance-superiority wording needs benchmark evidence that this project does not currently provide.",
        "Use qualitative/evaluation-scaffold wording or add a real benchmark section with evidence.",
    ),
    ClaimRule(
        "production_ready_claim",
        re.compile(r"\b(production[- ]ready|guarantees?|fully autonomous|fully automated)\b", re.IGNORECASE),
        "warn",
        "Production or guarantee wording is stronger than the current research-engineering evidence.",
        "Use reproducible, best-effort, or prototype wording unless the claim is backed by deployment evidence.",
    ),
    ClaimRule(
        "robotics_policy_claim",
        re.compile(r"\b(robotics manipulation policy|robot policy|manipulation policy)\b", re.IGNORECASE),
        "fail",
        "The project does not implement a robotics manipulation policy.",
        "Frame robotics as future work or motivation, not an implemented capability.",
    ),
]

NEGATION_MARKERS = [
    "not ",
    "not a ",
    "not an ",
    "do not",
    "does not",
    "did not",
    "must not",
    "without claiming",
    "without making",
    "no claim",
    "not claim",
    "not claiming",
    "avoid claim",
    "claims to avoid",
    "rather than claiming",
    "rather than",
    "not presented as",
    "not a claim",
    "unsupported",
    "unqualified",
    "claim audit",
    "claim-audit",
    "claims.",
    "claims before sharing",
]


def audit_claims(
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    pack_dir: str | Path | None = None,
) -> ClaimAuditReport:
    """Audit external-facing project/run/pack files for overclaiming risk."""

    root_path = Path(root) if root else project_root()
    run_path = Path(run_dir) if run_dir else None
    pack_path = Path(pack_dir) if pack_dir else None
    dry_run = _dry_run(run_path)
    files = _collect_files(root_path=root_path, run_dir=run_path, pack_dir=pack_path)
    findings = _scan_claims(files)
    checks = _required_checks(root_path=root_path, run_dir=run_path, pack_dir=pack_path, dry_run=dry_run)
    warnings = _warnings(files, pack_path)
    status = _status(findings, checks)
    return ClaimAuditReport(
        ok=status == "pass",
        status=status,
        generated_at=utc_timestamp(),
        root=_display_path(root_path),
        run_dir=_display_path(run_path) if run_path else "",
        pack_dir=_display_path(pack_path) if pack_path else "",
        dry_run=dry_run,
        scanned_files=[_display_file(path, root_path) for path in files],
        findings=findings,
        checks=checks,
        warnings=warnings,
    )


def write_claim_audit(
    *,
    root: str | Path | None = None,
    run_dir: str | Path | None = None,
    pack_dir: str | Path | None = None,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> ClaimAuditReport:
    """Build and write JSON/Markdown claim-audit artifacts."""

    report = audit_claims(root=root, run_dir=run_dir, pack_dir=pack_dir)
    if output:
        report.to_json(output)
    elif run_dir:
        report.to_json(Path(run_dir) / "claim_audit.json")
    if markdown_output:
        report.to_markdown(markdown_output)
    elif run_dir:
        report.to_markdown(Path(run_dir) / "claim_audit.md")
    return report


def _collect_files(
    *,
    root_path: Path,
    run_dir: Path | None,
    pack_dir: Path | None,
) -> list[Path]:
    files: list[Path] = []
    for relative in PROJECT_CLAIM_FILES:
        candidate = root_path / relative
        if candidate.exists() and candidate.is_file():
            files.append(candidate)
    if run_dir and run_dir.exists():
        for relative in RUN_CLAIM_FILES:
            candidate = run_dir / relative
            if candidate.exists() and candidate.is_file():
                files.append(candidate)
    if pack_dir and pack_dir.exists():
        for subroot in PACK_CLAIM_ROOTS:
            base = pack_dir / subroot
            if not base.exists():
                continue
            for candidate in sorted(base.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES:
                    files.append(candidate)
    return _dedupe_paths(files)


def _scan_claims(files: list[Path]) -> list[ClaimFinding]:
    findings: list[ClaimFinding] = []
    root = project_root()
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for rule in CLAIM_RULES:
                if not rule.pattern.search(line):
                    continue
                if _safe_context(line):
                    continue
                findings.append(
                    ClaimFinding(
                        file=_display_file(path, root),
                        line=line_number,
                        rule=rule.name,
                        severity=rule.severity,
                        text=line.strip()[:300],
                        reason=rule.reason,
                        recommendation=rule.recommendation,
                    )
                )
    return findings


def _required_checks(
    *,
    root_path: Path,
    run_dir: Path | None,
    pack_dir: Path | None,
    dry_run: bool | None,
) -> list[ClaimCheck]:
    project_text = _joined_text([root_path / relative for relative in PROJECT_CLAIM_FILES])
    checks = [
        _text_check(
            "upstream_attribution",
            project_text,
            ["built on nerfstudio", "lerf"],
            "Project docs attribute the work to Nerfstudio/LERF foundations.",
            "README.md",
            "State that the project is built on Nerfstudio and LERF.",
        ),
        _text_check(
            "no_new_architecture_disclaimer",
            project_text,
            ["not a new nerf architecture"],
            "Project docs explicitly avoid claiming a new NeRF architecture.",
            "README.md",
            "Add a clear disclaimer that this is not a new NeRF architecture.",
        ),
        _text_check(
            "dry_run_disclaimer",
            project_text,
            ["dry-run outputs are synthetic"],
            "Project docs explain dry-run artifacts are synthetic.",
            "README.md",
            "Add a dry-run disclaimer before using preview images externally.",
        ),
        _text_check(
            "gpu_requirement",
            project_text,
            ["nvidia gpu"],
            "Project docs state that real training requires an NVIDIA GPU.",
            "README.md",
            "Document the GPU requirement for real Nerfstudio/LERF training.",
        ),
        _text_check(
            "sota_disclaimer",
            project_text,
            ["not a state-of-the-art", "not state-of-the-art", "do not claim state-of-the-art"],
            "Project docs explicitly avoid SOTA claims.",
            "README.md",
            "Add explicit no-SOTA wording near the project positioning.",
            match_any=True,
        ),
    ]
    if run_dir:
        checks.extend(_run_checks(run_dir, dry_run=dry_run))
    if pack_dir:
        checks.append(
            ClaimCheck(
                "pack_validation_present",
                "pass" if (pack_dir / "portfolio_pack_validation.json").exists() else "warn",
                "portfolio_pack_validation.json is present."
                if (pack_dir / "portfolio_pack_validation.json").exists()
                else "Portfolio pack validation was not found in the pack directory.",
                "portfolio_pack_validation.json",
                "Run validate_portfolio_pack.py before external sharing.",
            )
        )
    return checks


def _run_checks(run_dir: Path, *, dry_run: bool | None) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    submission = _read_json(run_dir / "submission_packet" / "submission_packet.json")
    avoid_claims = " ".join(str(item).lower() for item in submission.get("avoid_claims") or [])
    checks.append(
        ClaimCheck(
            "submission_packet_claims",
            "pass" if "state-of-the-art" in avoid_claims and "trained lerf outputs" in avoid_claims else "warn",
            "Submission packet records claims to avoid."
            if avoid_claims
            else "Submission packet is missing or does not list avoid-claims.",
            "submission_packet/submission_packet.json",
            "Regenerate create_submission_packet.py before outreach.",
        )
    )
    if dry_run is True:
        run_text = _joined_text([run_dir / relative for relative in RUN_CLAIM_FILES])
        checks.append(
            _text_check(
                "dry_run_run_artifact_disclaimer",
                run_text,
                ["dry-run", "smoke"],
                "Run-facing artifacts mark the current evidence as dry-run/smoke.",
                "pipeline_summary.json",
                "Regenerate reports and submission packet so dry-run status is visible.",
                warn_only=False,
            )
        )
    return checks


def _text_check(
    name: str,
    text: str,
    required_phrases: list[str],
    pass_detail: str,
    artifact: str,
    action: str,
    *,
    warn_only: bool = False,
    match_any: bool = False,
) -> ClaimCheck:
    lowered = text.lower()
    ok = any(phrase in lowered for phrase in required_phrases) if match_any else all(
        phrase in lowered for phrase in required_phrases
    )
    return ClaimCheck(
        name=name,
        status="pass" if ok else ("warn" if warn_only else "fail"),
        detail=pass_detail if ok else "Missing required claim-calibration wording.",
        artifact=artifact,
        action="" if ok else action,
    )


def _status(findings: list[ClaimFinding], checks: list[ClaimCheck]) -> CheckStatus:
    if any(finding.severity == "fail" for finding in findings) or any(
        check.status == "fail" for check in checks
    ):
        return "fail"
    if findings or any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _safe_context(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def _warnings(files: list[Path], pack_dir: Path | None) -> list[str]:
    warnings: list[str] = []
    if not files:
        warnings.append("No claim-audit files were found.")
    if pack_dir and not pack_dir.exists():
        warnings.append(f"Pack directory does not exist: {_display_path(pack_dir)}")
    return warnings


def _dry_run(run_dir: Path | None) -> bool | None:
    if not run_dir:
        return None
    summary = _read_json(run_dir / "pipeline_summary.json")
    if "dry_run" not in summary:
        return None
    return bool(summary.get("dry_run"))


def _joined_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(chunks)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _display_file(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return _display_path(path)


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def _check_lines(checks: list[ClaimCheck]) -> list[str]:
    if not checks:
        return ["- No required checks were generated."]
    return [
        f"- `{check.status}` {check.name}: {check.detail}"
        + (f" Artifact: `{check.artifact}`" if check.artifact else "")
        + (f" Action: {check.action}" if check.action else "")
        for check in checks
    ]


def _finding_lines(findings: list[ClaimFinding]) -> list[str]:
    if not findings:
        return ["- No risky claim findings."]
    return [
        f"- `{finding.severity}` {finding.rule} at `{finding.file}:{finding.line}`: {finding.text} Recommendation: {finding.recommendation}"
        for finding in findings
    ]


def _list_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]
