"""Prompt-sensitivity analysis for open-vocabulary scene queries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from nerf_llm_scene_inspector.backends.base import BoundingRegion, QueryResult
from nerf_llm_scene_inspector.config import load_mapping
from nerf_llm_scene_inspector.evaluation.metrics import bbox_iou
from nerf_llm_scene_inspector.utils.paths import slugify, utc_timestamp


@dataclass
class PromptGroup:
    """A set of prompts intended to retrieve the same concept or affordance."""

    name: str
    prompts: list[str]
    description: str = ""
    expected_behavior: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptVariantRow:
    """Per-prompt evidence used in the prompt-sensitivity table."""

    group_name: str
    prompt: str
    status: str
    confidence: float | None = None
    top_label: str = ""
    top_score: float | None = None
    source_view: str = ""
    bbox_2d: tuple[float, float, float, float] | None = None
    result_path: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptGroupAnalysis:
    """Aggregate stability metrics for one prompt group."""

    group_name: str
    description: str
    expected_behavior: str
    prompts: list[str]
    num_prompts: int
    num_results: int
    missing_prompts: list[str]
    mean_confidence: float | None
    confidence_std: float | None
    confidence_range: float | None
    mean_pairwise_top1_iou: float | None
    comparable_box_pairs: int
    view_agreement_rate: float | None
    dominant_view: str
    stability_label: str
    recommendation: str
    rows: list[PromptVariantRow]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        return payload


@dataclass
class PromptSensitivityReport:
    """Prompt-sensitivity analysis across all prompt groups in a suite."""

    scene_name: str
    suite_path: str
    results_dir: str
    timestamp: str
    min_mean_confidence: float
    min_box_consistency_iou: float
    min_view_agreement: float
    groups: list[PromptGroupAnalysis]
    warnings: list[str] = field(default_factory=list)

    @property
    def stable_group_count(self) -> int:
        return sum(group.stability_label == "stable" for group in self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "suite_path": self.suite_path,
            "results_dir": self.results_dir,
            "timestamp": self.timestamp,
            "min_mean_confidence": self.min_mean_confidence,
            "min_box_consistency_iou": self.min_box_consistency_iou,
            "min_view_agreement": self.min_view_agreement,
            "stable_group_count": self.stable_group_count,
            "num_groups": len(self.groups),
            "groups": [group.to_dict() for group in self.groups],
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    def to_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "group_name",
            "prompt",
            "status",
            "confidence",
            "top_label",
            "top_score",
            "source_view",
            "bbox_2d",
            "result_path",
            "warnings",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for group in self.groups:
                for row in group.rows:
                    payload = row.to_dict()
                    payload["bbox_2d"] = _format_bbox(row.bbox_2d)
                    payload["warnings"] = "; ".join(row.warnings)
                    writer.writerow(payload)
        return output

    def to_markdown(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Prompt Sensitivity Report: {self.scene_name}",
            "",
            "This report checks whether semantically related prompts produce consistent",
            "query evidence. It is a robustness diagnostic, not a benchmark claim.",
            "",
            "## Summary",
            "",
            f"- Stable groups: {self.stable_group_count}/{len(self.groups)}",
            f"- Results directory: `{self.results_dir}`",
            f"- Suite: `{self.suite_path}`",
            "",
            "| Group | Status | Results | Mean confidence | Mean top-1 IoU | View agreement |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for group in self.groups:
            lines.append(
                "| {name} | {status} | {num_results}/{num_prompts} | {confidence} | {iou} | {view} |".format(
                    name=group.group_name,
                    status=group.stability_label,
                    num_results=group.num_results,
                    num_prompts=group.num_prompts,
                    confidence=_display_float(group.mean_confidence),
                    iou=_display_float(group.mean_pairwise_top1_iou),
                    view=_display_float(group.view_agreement_rate),
                )
            )
        for group in self.groups:
            lines.extend(_group_markdown_lines(group))
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output


@dataclass
class _ResultRecord:
    result: QueryResult
    path: Path


def load_prompt_suite(path: str | Path) -> tuple[str, list[PromptGroup]]:
    """Load a prompt-sensitivity YAML suite."""

    raw = load_mapping(path)
    scene_name = str(raw.get("scene_name") or "")
    groups_raw = raw.get("groups") or []
    if not isinstance(groups_raw, list):
        raise ValueError("Prompt suite must contain a 'groups' list.")
    groups: list[PromptGroup] = []
    for index, item in enumerate(groups_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt suite group {index} must be a mapping.")
        prompts = [str(prompt).strip() for prompt in item.get("prompts", []) if str(prompt).strip()]
        if not prompts:
            raise ValueError(f"Prompt suite group {index} has no prompts.")
        groups.append(
            PromptGroup(
                name=str(item.get("name") or f"group_{index + 1}"),
                description=str(item.get("description") or ""),
                expected_behavior=str(item.get("expected_behavior") or ""),
                prompts=prompts,
            )
        )
    if not groups:
        raise ValueError("Prompt suite does not contain any groups.")
    return scene_name, groups


def prompt_suite_queries(path: str | Path) -> list[str]:
    """Return all unique prompts in a suite while preserving order."""

    _, groups = load_prompt_suite(path)
    seen: set[str] = set()
    queries: list[str] = []
    for group in groups:
        for prompt in group.prompts:
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                queries.append(prompt)
    return queries


def analyze_prompt_sensitivity(
    *,
    suite_path: str | Path,
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    scene_name: str | None = None,
    dry_run: bool = False,
    min_mean_confidence: float = 0.55,
    min_box_consistency_iou: float = 0.25,
    min_view_agreement: float = 0.67,
) -> PromptSensitivityReport:
    """Analyze prompt robustness from existing query_result.json artifacts."""

    suite_scene_name, groups = load_prompt_suite(suite_path)
    records = _load_result_records(Path(results_dir))
    warnings: list[str] = []
    if dry_run and not records:
        records = _synthetic_records(groups, Path(results_dir))
        warnings.append("Dry-run synthetic prompt sensitivity results were generated.")
    if not records:
        warnings.append("No query_result.json files were found; all groups will be insufficient.")
    by_query = _best_record_by_query(records)
    analyses = [
        _analyze_group(
            group,
            by_query=by_query,
            min_mean_confidence=min_mean_confidence,
            min_box_consistency_iou=min_box_consistency_iou,
            min_view_agreement=min_view_agreement,
        )
        for group in groups
    ]
    report = PromptSensitivityReport(
        scene_name=scene_name or suite_scene_name or "unknown_scene",
        suite_path=str(suite_path),
        results_dir=str(results_dir),
        timestamp=utc_timestamp(),
        min_mean_confidence=min_mean_confidence,
        min_box_consistency_iou=min_box_consistency_iou,
        min_view_agreement=min_view_agreement,
        groups=analyses,
        warnings=warnings,
    )
    if output_dir:
        output = Path(output_dir)
        report.to_json(output / "prompt_sensitivity_summary.json")
        report.to_csv(output / "prompt_sensitivity_table.csv")
        report.to_markdown(output / "prompt_sensitivity_report.md")
    return report


def _load_result_records(results_dir: Path) -> list[_ResultRecord]:
    if not results_dir.exists():
        return []
    records: list[_ResultRecord] = []
    for path in sorted(results_dir.rglob("query_result.json")):
        try:
            records.append(_ResultRecord(result=QueryResult.from_json(path), path=path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return records


def _best_record_by_query(records: list[_ResultRecord]) -> dict[str, _ResultRecord]:
    best: dict[str, _ResultRecord] = {}
    for record in records:
        key = record.result.query.strip().lower()
        if not key:
            continue
        if key not in best or _record_rank(record) > _record_rank(best[key]):
            best[key] = record
    return best


def _record_rank(record: _ResultRecord) -> tuple[int, float]:
    return int(_is_direct_prompt_record(record)), _result_strength(record.result) or -1.0


def _is_direct_prompt_record(record: _ResultRecord) -> bool:
    query_slug = slugify(record.result.query)
    parent = record.path.parent.name
    grandparent = record.path.parent.parent.name if record.path.parent.parent else ""
    return parent == query_slug and grandparent == query_slug


def _analyze_group(
    group: PromptGroup,
    *,
    by_query: dict[str, _ResultRecord],
    min_mean_confidence: float,
    min_box_consistency_iou: float,
    min_view_agreement: float,
) -> PromptGroupAnalysis:
    rows: list[PromptVariantRow] = []
    top_regions: list[BoundingRegion] = []
    confidences: list[float] = []
    missing: list[str] = []
    for prompt in group.prompts:
        record = by_query.get(prompt.lower())
        if not record:
            missing.append(prompt)
            rows.append(PromptVariantRow(group_name=group.name, prompt=prompt, status="missing"))
            continue
        row, top_region = _variant_row(group.name, prompt, record)
        rows.append(row)
        if row.confidence is not None:
            confidences.append(row.confidence)
        if top_region is not None:
            top_regions.append(top_region)
    mean_confidence = round(mean(confidences), 4) if confidences else None
    confidence_std = round(pstdev(confidences), 4) if len(confidences) > 1 else 0.0 if confidences else None
    confidence_range = round(max(confidences) - min(confidences), 4) if confidences else None
    mean_iou, comparable_pairs = _mean_pairwise_top1_iou(top_regions)
    view_agreement, dominant_view = _view_agreement(top_regions)
    stability_label = _stability_label(
        num_results=len(confidences),
        num_prompts=len(group.prompts),
        mean_confidence=mean_confidence,
        mean_iou=mean_iou,
        comparable_pairs=comparable_pairs,
        view_agreement=view_agreement,
        min_mean_confidence=min_mean_confidence,
        min_box_consistency_iou=min_box_consistency_iou,
        min_view_agreement=min_view_agreement,
    )
    return PromptGroupAnalysis(
        group_name=group.name,
        description=group.description,
        expected_behavior=group.expected_behavior,
        prompts=list(group.prompts),
        num_prompts=len(group.prompts),
        num_results=len(confidences),
        missing_prompts=missing,
        mean_confidence=mean_confidence,
        confidence_std=confidence_std,
        confidence_range=confidence_range,
        mean_pairwise_top1_iou=mean_iou,
        comparable_box_pairs=comparable_pairs,
        view_agreement_rate=view_agreement,
        dominant_view=dominant_view,
        stability_label=stability_label,
        recommendation=_recommendation(stability_label, missing, comparable_pairs, mean_iou),
        rows=rows,
    )


def _variant_row(
    group_name: str,
    prompt: str,
    record: _ResultRecord,
) -> tuple[PromptVariantRow, BoundingRegion | None]:
    top_region = _top_region(record.result)
    confidence = _result_strength(record.result)
    return (
        PromptVariantRow(
            group_name=group_name,
            prompt=prompt,
            status="found",
            confidence=confidence,
            top_label=top_region.label if top_region else "",
            top_score=top_region.score if top_region else None,
            source_view=top_region.source_view if top_region and top_region.source_view else "",
            bbox_2d=top_region.bbox_2d if top_region else None,
            result_path=str(record.path),
            warnings=list(record.result.warnings),
        ),
        top_region,
    )


def _top_region(result: QueryResult) -> BoundingRegion | None:
    if not result.bounding_regions:
        return None
    return sorted(
        result.bounding_regions,
        key=lambda region: region.score if region.score is not None else -1.0,
        reverse=True,
    )[0]


def _result_strength(result: QueryResult) -> float | None:
    scores = [region.score for region in result.bounding_regions if region.score is not None]
    if scores:
        return float(max(scores))
    return result.confidence


def _mean_pairwise_top1_iou(regions: list[BoundingRegion]) -> tuple[float | None, int]:
    values: list[float] = []
    for left, right in combinations(regions, 2):
        if left.bbox_2d is None or right.bbox_2d is None:
            continue
        if left.source_view and right.source_view and left.source_view != right.source_view:
            continue
        values.append(bbox_iou(left.bbox_2d, right.bbox_2d))
    if not values:
        return None, 0
    return round(mean(values), 4), len(values)


def _view_agreement(regions: list[BoundingRegion]) -> tuple[float | None, str]:
    views = [region.source_view for region in regions if region.source_view]
    if not views:
        return None, ""
    counts = {view: views.count(view) for view in sorted(set(views))}
    dominant_view, dominant_count = max(counts.items(), key=lambda item: item[1])
    return round(dominant_count / len(views), 4), dominant_view


def _stability_label(
    *,
    num_results: int,
    num_prompts: int,
    mean_confidence: float | None,
    mean_iou: float | None,
    comparable_pairs: int,
    view_agreement: float | None,
    min_mean_confidence: float,
    min_box_consistency_iou: float,
    min_view_agreement: float,
) -> str:
    if num_results < min(2, num_prompts) or num_results < num_prompts:
        return "insufficient_evidence"
    if mean_confidence is None or mean_confidence < min_mean_confidence:
        return "needs_review"
    if view_agreement is not None and view_agreement < min_view_agreement:
        return "needs_review"
    if comparable_pairs == 0:
        return "needs_spatial_evidence"
    if mean_iou is None or mean_iou < min_box_consistency_iou:
        return "needs_review"
    return "stable"


def _recommendation(
    label: str,
    missing_prompts: list[str],
    comparable_pairs: int,
    mean_iou: float | None,
) -> str:
    if missing_prompts:
        return "Run missing prompt variants before interpreting prompt stability."
    if label == "stable":
        return "Prompt variants are consistent enough for this lightweight diagnostic."
    if comparable_pairs == 0:
        return "Render or import comparable relevancy maps so top regions can be compared."
    if mean_iou is not None and mean_iou < 0.25:
        return "Inspect overlays; prompt wording may be shifting localization."
    return "Review confidence, view agreement, and overlays before using this as evidence."


def _synthetic_records(groups: list[PromptGroup], results_dir: Path) -> list[_ResultRecord]:
    records: list[_ResultRecord] = []
    for group_index, group in enumerate(groups):
        for prompt_index, prompt in enumerate(group.prompts):
            offset = float(prompt_index * 3)
            bbox = (40.0 + offset, 50.0 + offset, 130.0 + offset, 160.0 + offset)
            result = QueryResult(
                query=prompt,
                backend_name="dry-run",
                config_path="dry-run",
                bounding_regions=[
                    BoundingRegion(
                        label=group.name,
                        score=round(0.85 - prompt_index * 0.04, 4),
                        coordinate_frame="image",
                        bbox_2d=bbox,
                        source_view=f"view_{group_index:04d}",
                        notes="Synthetic prompt-sensitivity dry-run region.",
                    )
                ],
                confidence=round(0.85 - prompt_index * 0.04, 4),
                warnings=["Synthetic dry-run prompt-sensitivity result."],
            )
            path = results_dir / group.name / f"prompt_{prompt_index:02d}" / "query_result.json"
            records.append(_ResultRecord(result=result, path=path))
    return records


def _group_markdown_lines(group: PromptGroupAnalysis) -> list[str]:
    lines = [
        "",
        f"## {group.group_name}",
        "",
        group.description or "No description provided.",
        "",
        f"- Status: `{group.stability_label}`",
        f"- Recommendation: {group.recommendation}",
    ]
    if group.missing_prompts:
        lines.append(f"- Missing prompts: {', '.join(group.missing_prompts)}")
    lines.extend(
        [
            "",
            "| Prompt | Status | Confidence | Top label | Source view | BBox |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in group.rows:
        lines.append(
            "| {prompt} | {status} | {confidence} | {label} | {view} | {bbox} |".format(
                prompt=row.prompt,
                status=row.status,
                confidence=_display_float(row.confidence),
                label=row.top_label or "n/a",
                view=row.source_view or "n/a",
                bbox=_format_bbox(row.bbox_2d) or "n/a",
            )
        )
    return lines


def _format_bbox(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return ""
    return "[" + ", ".join(f"{value:.1f}" for value in bbox) + "]"


def _display_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
