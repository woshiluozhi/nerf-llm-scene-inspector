"""Create GIF or MP4 montages from query artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_gif(image_paths: list[str | Path], output_path: str | Path, duration_ms: int = 900) -> Path:
    """Create an animated GIF from image paths."""

    if not image_paths:
        raise ValueError("No images provided for GIF generation.")
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    first, rest = frames[0], frames[1:]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first.save(out, save_all=True, append_images=rest, duration=duration_ms, loop=0)
    return out


def make_mp4_or_gif(image_paths: list[str | Path], output_path: str | Path, fps: int = 2) -> Path:
    """Create MP4 with imageio when available, otherwise create a GIF."""

    out = Path(output_path)
    if out.suffix.lower() == ".gif":
        return make_gif(image_paths, out, duration_ms=max(100, int(1000 / fps)))
    try:
        import imageio.v2 as imageio  # type: ignore
    except ModuleNotFoundError:
        return make_gif(image_paths, out.with_suffix(".gif"), duration_ms=max(100, int(1000 / fps)))
    frames = [imageio.imread(path) for path in image_paths]
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    return out


def make_query_grid(
    image_paths: list[str | Path],
    output_path: str | Path,
    *,
    max_columns: int = 2,
    target_width: int = 720,
) -> Path | None:
    """Create a compact static grid from rendered query overlay images."""

    existing = [Path(path) for path in image_paths if Path(path).exists()]
    if not existing:
        return None

    thumbs: list[tuple[Path, Image.Image]] = []
    for path in existing:
        image = Image.open(path).convert("RGB")
        scale = target_width / image.width
        thumb = image.resize((target_width, int(image.height * scale)))
        thumbs.append((path, thumb))

    columns = min(max_columns, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = target_width
    cell_h = max(image.height for _path, image in thumbs) + 34
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, image) in enumerate(thumbs):
        col = index % columns
        row = index // columns
        x = col * cell_w
        y = row * cell_h
        label = path.parent.name.replace("_", " ")
        draw.text((x + 10, y + 10), label[:80], fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, y + 34))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
