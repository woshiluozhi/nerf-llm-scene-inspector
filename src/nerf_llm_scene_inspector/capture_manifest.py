"""Capture manifest helpers for real-scene reproducibility."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from nerf_llm_scene_inspector.utils.paths import utc_timestamp


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
CaptureCheckStatus = Literal["pass", "warn", "fail"]
CaptureValidationStatus = Literal["ready", "needs_review", "blocked"]


@dataclass
class CaptureManifest:
    """Human-readable metadata about a real scene capture."""

    scene_name: str
    input_path: str
    input_type: str
    capture_device: str = "unknown"
    scene_type: str = "unknown"
    lighting: str = "unknown"
    camera_motion: str = "unknown"
    duration_seconds: float | None = None
    approximate_frame_count: int | None = None
    static_scene: bool | None = None
    high_overlap: bool | None = None
    privacy_reviewed: bool = False
    contains_people: bool | None = None
    contains_private_text: bool | None = None
    notes: str = ""
    created_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CaptureManifest":
        return cls(
            scene_name=str(raw.get("scene_name") or ""),
            input_path=str(raw.get("input_path") or ""),
            input_type=str(raw.get("input_type") or ""),
            capture_device=str(raw.get("capture_device") or "unknown"),
            scene_type=str(raw.get("scene_type") or "unknown"),
            lighting=str(raw.get("lighting") or "unknown"),
            camera_motion=str(raw.get("camera_motion") or "unknown"),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            approximate_frame_count=_optional_int(raw.get("approximate_frame_count")),
            static_scene=_optional_bool(raw.get("static_scene")),
            high_overlap=_optional_bool(raw.get("high_overlap")),
            privacy_reviewed=bool(raw.get("privacy_reviewed", False)),
            contains_people=_optional_bool(raw.get("contains_people")),
            contains_private_text=_optional_bool(raw.get("contains_private_text")),
            notes=str(raw.get("notes") or ""),
            created_at=str(raw.get("created_at") or utc_timestamp()),
        )

    def to_dict(self) -> dict[str, object]:
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
            f"# Capture Manifest: {self.scene_name or 'unknown'}",
            "",
            "This manifest records scene-capture context that affects NeRF/LERF quality.",
            "",
            f"- Input: `{self.input_path}`",
            f"- Input type: {self.input_type}",
            f"- Capture device: {self.capture_device}",
            f"- Scene type: {self.scene_type}",
            f"- Lighting: {self.lighting}",
            f"- Camera motion: {self.camera_motion}",
            f"- Duration seconds: {_display(self.duration_seconds)}",
            f"- Approximate frame count: {_display(self.approximate_frame_count)}",
            f"- Static scene: {_display(self.static_scene)}",
            f"- High overlap: {_display(self.high_overlap)}",
            f"- Privacy reviewed: {self.privacy_reviewed}",
            f"- Contains people: {_display(self.contains_people)}",
            f"- Contains private text: {_display(self.contains_private_text)}",
            "",
            "## Notes",
            "",
            self.notes or "None.",
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


@dataclass
class CaptureManifestCheck:
    """One capture-manifest validation check."""

    name: str
    status: CaptureCheckStatus
    detail: str
    recommendation: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class CaptureManifestValidation:
    """Validation result for capture metadata."""

    status: CaptureValidationStatus
    ok: bool
    manifest_path: str
    checks: list[CaptureManifestCheck]
    timestamp: str = field(default_factory=utc_timestamp)

    @property
    def fail_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ok": self.ok,
            "manifest_path": self.manifest_path,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "checks": [check.to_dict() for check in self.checks],
            "timestamp": self.timestamp,
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
            "# Capture Manifest Validation",
            "",
            f"- Status: {self.status}",
            f"- OK: {self.ok}",
            f"- Failed checks: {self.fail_count}",
            f"- Warning checks: {self.warn_count}",
            f"- Manifest: `{self.manifest_path}`",
            "",
            "## Checks",
            "",
        ]
        lines.extend(_check_lines(self.checks))
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path


def build_capture_manifest(
    *,
    input_path: str | Path,
    input_type: str,
    scene_name: str,
    capture_device: str = "unknown",
    scene_type: str = "unknown",
    lighting: str = "unknown",
    camera_motion: str = "unknown",
    duration_seconds: float | None = None,
    approximate_frame_count: int | None = None,
    static_scene: bool | None = None,
    high_overlap: bool | None = None,
    privacy_reviewed: bool = False,
    contains_people: bool | None = None,
    contains_private_text: bool | None = None,
    notes: str = "",
) -> CaptureManifest:
    """Create capture metadata, inferring simple file counts when possible."""

    inferred_frame_count = approximate_frame_count
    if inferred_frame_count is None and input_type == "images":
        inferred_frame_count = _count_images(Path(input_path))
    return CaptureManifest(
        scene_name=scene_name,
        input_path=str(input_path),
        input_type=input_type,
        capture_device=capture_device,
        scene_type=scene_type,
        lighting=lighting,
        camera_motion=camera_motion,
        duration_seconds=duration_seconds,
        approximate_frame_count=inferred_frame_count,
        static_scene=static_scene,
        high_overlap=high_overlap,
        privacy_reviewed=privacy_reviewed,
        contains_people=contains_people,
        contains_private_text=contains_private_text,
        notes=notes,
    )


def load_capture_manifest(path: str | Path) -> CaptureManifest:
    """Load a capture manifest JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Capture manifest must be a JSON object.")
    return CaptureManifest.from_dict(raw)


def validate_capture_manifest(
    manifest: CaptureManifest | str | Path,
    *,
    manifest_path: str | Path = "",
    min_images: int = 50,
    require_privacy_review: bool = False,
) -> CaptureManifestValidation:
    """Validate capture metadata for real-run reproducibility."""

    if isinstance(manifest, CaptureManifest):
        payload = manifest
        display_path = str(manifest_path) if manifest_path else ""
    else:
        display_path = str(manifest)
        payload = load_capture_manifest(manifest)

    checks = [
        _required_text_check("scene_name", payload.scene_name),
        _input_path_check(payload),
        _input_type_check(payload.input_type),
        _required_text_check(
            "capture_device",
            payload.capture_device,
            recommendation="Record the phone/camera model or at least the capture device family.",
        ),
        _required_text_check(
            "lighting",
            payload.lighting,
            recommendation="Record lighting conditions, for example bright diffuse indoor light.",
        ),
        _bool_check(
            "static_scene",
            payload.static_scene,
            positive_detail="Scene was marked static.",
            negative_recommendation="Use a static scene for COLMAP and NeRF/LERF training.",
        ),
        _bool_check(
            "high_overlap",
            payload.high_overlap,
            positive_detail="Capture was marked high-overlap.",
            negative_recommendation="Capture with slow motion and high frame overlap.",
        ),
        _privacy_check(payload, require_privacy_review=require_privacy_review),
    ]
    checks.append(_frame_count_check(payload, min_images=min_images))
    status = _overall_status(checks)
    return CaptureManifestValidation(
        status=status,
        ok=status == "ready",
        manifest_path=display_path,
        checks=checks,
    )


def write_capture_manifest_bundle(
    manifest: CaptureManifest,
    output_dir: str | Path,
    *,
    min_images: int = 50,
    require_privacy_review: bool = False,
) -> tuple[Path, Path, Path, Path, CaptureManifestValidation]:
    """Write manifest, markdown, validation JSON, and validation markdown."""

    root = Path(output_dir)
    manifest_json = manifest.to_json(root / "capture_manifest.json")
    manifest_md = manifest.to_markdown(root / "capture_manifest.md")
    validation = validate_capture_manifest(
        manifest,
        manifest_path=manifest_json,
        min_images=min_images,
        require_privacy_review=require_privacy_review,
    )
    validation_json = validation.to_json(root / "capture_manifest_validation.json")
    validation_md = validation.to_markdown(root / "capture_manifest_validation.md")
    return manifest_json, manifest_md, validation_json, validation_md, validation


def copy_or_create_capture_manifest(
    *,
    output_dir: str | Path,
    input_path: str | Path,
    input_type: str,
    scene_name: str,
    capture_manifest_path: str | Path | None = None,
    min_images: int = 50,
    require_privacy_review: bool = False,
) -> tuple[Path, Path, Path, Path, CaptureManifestValidation]:
    """Copy an existing manifest into a run dir, or create a scaffold from inputs."""

    output = Path(output_dir)
    if capture_manifest_path:
        manifest = load_capture_manifest(capture_manifest_path)
        if not manifest.input_path:
            manifest.input_path = str(input_path)
        if not manifest.input_type:
            manifest.input_type = input_type
        if not manifest.scene_name:
            manifest.scene_name = scene_name
    else:
        manifest = build_capture_manifest(
            input_path=input_path,
            input_type=input_type,
            scene_name=scene_name,
            notes="Auto-generated scaffold. Fill capture fields before presenting real results.",
        )
    paths = write_capture_manifest_bundle(
        manifest,
        output,
        min_images=min_images,
        require_privacy_review=require_privacy_review,
    )
    if capture_manifest_path:
        source = Path(capture_manifest_path)
        destination = output / "capture_manifest_source.json"
        if source.exists() and source.resolve() != paths[0].resolve():
            shutil.copy2(source, destination)
    return paths


def _required_text_check(
    name: str,
    value: str,
    *,
    recommendation: str = "Fill this field before sharing a real-scene result.",
) -> CaptureManifestCheck:
    if value.strip() and value.strip().lower() != "unknown":
        return CaptureManifestCheck(name=name, status="pass", detail=value.strip())
    return CaptureManifestCheck(
        name=name,
        status="warn",
        detail="Missing or unknown.",
        recommendation=recommendation,
    )


def _input_type_check(input_type: str) -> CaptureManifestCheck:
    if input_type in {"video", "images"}:
        return CaptureManifestCheck("input_type", "pass", input_type)
    return CaptureManifestCheck("input_type", "fail", input_type, "Use video or images.")


def _input_path_check(manifest: CaptureManifest) -> CaptureManifestCheck:
    path = Path(manifest.input_path)
    if not manifest.input_path:
        return CaptureManifestCheck("input_path", "fail", "Missing input path.")
    if not path.exists():
        return CaptureManifestCheck(
            "input_path",
            "warn",
            f"Input path does not exist locally: {path}",
            "Keep a private copy of raw data and record how to regenerate processed data.",
        )
    if manifest.input_type == "images" and not path.is_dir():
        return CaptureManifestCheck("input_path", "fail", f"Images input is not a directory: {path}")
    if manifest.input_type == "video" and not path.is_file():
        return CaptureManifestCheck("input_path", "fail", f"Video input is not a file: {path}")
    return CaptureManifestCheck("input_path", "pass", str(path))


def _bool_check(
    name: str,
    value: bool | None,
    *,
    positive_detail: str,
    negative_recommendation: str,
) -> CaptureManifestCheck:
    if value is True:
        return CaptureManifestCheck(name, "pass", positive_detail)
    if value is False:
        return CaptureManifestCheck(name, "warn", "Field is false.", negative_recommendation)
    return CaptureManifestCheck(name, "warn", "Field is unknown.", negative_recommendation)


def _privacy_check(
    manifest: CaptureManifest,
    *,
    require_privacy_review: bool,
) -> CaptureManifestCheck:
    if manifest.privacy_reviewed:
        detail = "Privacy reviewed."
        if manifest.contains_people:
            detail += " Contains people."
        if manifest.contains_private_text:
            detail += " Contains private text."
        return CaptureManifestCheck("privacy_review", "pass", detail)
    status: CaptureCheckStatus = "fail" if require_privacy_review else "warn"
    return CaptureManifestCheck(
        "privacy_review",
        status,
        "Privacy review not confirmed.",
        "Confirm the capture does not expose private people, screens, addresses, or documents before sharing.",
    )


def _frame_count_check(manifest: CaptureManifest, *, min_images: int) -> CaptureManifestCheck:
    if manifest.input_type != "images":
        return CaptureManifestCheck(
            "approximate_frame_count",
            "warn" if manifest.approximate_frame_count is None else "pass",
            _display(manifest.approximate_frame_count),
            "For video, record duration and approximate extracted frames after ns-process-data.",
        )
    count = manifest.approximate_frame_count
    if count is None:
        return CaptureManifestCheck(
            "approximate_frame_count",
            "warn",
            "Unknown image count.",
            "Record the number of captured images.",
        )
    if count < min_images:
        return CaptureManifestCheck(
            "approximate_frame_count",
            "warn",
            f"{count} images; recommended minimum is {min_images}.",
            "Capture more overlapping images for a stronger real-scene run.",
        )
    return CaptureManifestCheck("approximate_frame_count", "pass", f"{count} images.")


def _count_images(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def _overall_status(checks: list[CaptureManifestCheck]) -> CaptureValidationStatus:
    if any(check.status == "fail" for check in checks):
        return "blocked"
    if any(check.status == "warn" for check in checks):
        return "needs_review"
    return "ready"


def _check_lines(checks: list[CaptureManifestCheck]) -> list[str]:
    if not checks:
        return ["- None."]
    lines: list[str] = []
    for check in checks:
        lines.append(f"- [{check.status.upper()}] {check.name}: {check.detail}")
        if check.recommendation:
            lines.append(f"  Recommendation: {check.recommendation}")
    return lines


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        if normalized in {"unknown", "none", "null"}:
            return None
    return bool(value)


def _display(value: object) -> str:
    return "unknown" if value is None else str(value)
