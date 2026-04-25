from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


NEAR_WHITE_RGB = (244, 244, 244)
BACKGROUND_DISTANCE = 28
OUTPUT_PADDING = 6
GIF_ROTATION_DEGREES = (-2.0, 2.0)
GIF_FRAME_DURATION_MS = 180


@dataclass(frozen=True)
class SliceRecord:
    index: int
    row: int
    col: int
    path: str
    gif_path: str
    width: int
    height: int
    bytes: int
    gif_bytes: int


def parse_grid(value: str) -> tuple[int, int] | None:
    if value == "auto":
        return None

    normalized = value.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("grid must be 'auto' or like '4x3', '3x3'.")

    cols, rows = (int(parts[0]), int(parts[1]))
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive.")
    return cols, rows


def detect_grid(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    ratio = width / height

    if abs(ratio - 1) < 0.08:
        return 3, 3
    if ratio > 1:
        return 4, 3
    return 3, 4


def iter_boxes(
    width: int,
    height: int,
    cols: int,
    rows: int,
) -> Iterable[tuple[int, int, int, int, int, int, int]]:
    index = 1
    for row in range(rows):
        for col in range(cols):
            left = round(width * col / cols)
            top = round(height * row / rows)
            right = round(width * (col + 1) / cols)
            bottom = round(height * (row + 1) / rows)
            yield index, row + 1, col + 1, left, top, right, bottom
            index += 1


def expand_to_square(
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[int, int, int, int]:
    box_width = right - left
    box_height = bottom - top
    side = max(box_width, box_height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    square_left = round(center_x - side / 2)
    square_top = round(center_y - side / 2)
    square_right = square_left + side
    square_bottom = square_top + side

    return square_left, square_top, square_right, square_bottom


def make_square_slice(
    image: Image.Image,
    cell_box: tuple[int, int, int, int],
    square_box: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = cell_box
    square_left, square_top, square_right, square_bottom = square_box
    side = square_right - square_left
    background = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    square = Image.new(image.mode, (side, side), background)

    crop_left = max(0, square_left)
    crop_top = max(0, square_top)
    crop_right = min(image.width, square_right)
    crop_bottom = min(image.height, square_bottom)
    if crop_left < crop_right and crop_top < crop_bottom:
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        square.paste(crop, (crop_left - square_left, crop_top - square_top))

    inner_box = (
        left - square_left,
        top - square_top,
        right - square_left,
        bottom - square_top,
    )
    return keep_components_intersecting_box(square, inner_box, side)


def is_near_white(pixel: tuple[int, ...]) -> bool:
    return pixel[0] >= NEAR_WHITE_RGB[0] and pixel[1] >= NEAR_WHITE_RGB[1] and pixel[2] >= NEAR_WHITE_RGB[2]


def estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    samples: list[tuple[int, int, int]] = []
    step = max(1, min(width, height) // 24)

    for x in range(0, width, step):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, height - 1])
    for y in range(0, height, step):
        samples.append(pixels[0, y])
        samples.append(pixels[width - 1, y])

    channels = list(zip(*samples))
    return tuple(sorted(channel)[len(channel) // 2] for channel in channels)


def is_background(pixel: tuple[int, ...], background: tuple[int, int, int]) -> bool:
    if is_near_white(pixel):
        return True
    distance = max(abs(pixel[i] - background[i]) for i in range(3))
    return distance <= BACKGROUND_DISTANCE and min(pixel[:3]) >= 215


def keep_components_intersecting_box(
    image: Image.Image,
    keep_box: tuple[int, int, int, int],
    side: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background = estimate_background_color(image)
    mask = [[not is_background(pixels[x, y], background) for x in range(width)] for y in range(height)]
    visited: set[tuple[int, int]] = set()
    background = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    output = Image.new(image.mode, (side, side), background)
    source_pixels = image.load()
    output_pixels = output.load()
    keep_left, keep_top, keep_right, keep_bottom = keep_box

    for y in range(height):
        for x in range(width):
            if (x, y) in visited or not mask[y][x]:
                continue

            component: list[tuple[int, int]] = []
            stack = [(x, y)]
            visited.add((x, y))

            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if (
                        (nx, ny) in visited
                        or not (0 <= nx < width and 0 <= ny < height)
                        or not mask[ny][nx]
                    ):
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))

            inside_count = sum(
                1
                for px, py in component
                if keep_left <= px < keep_right and keep_top <= py < keep_bottom
            )
            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            component_left = min(xs)
            component_right = max(xs)
            component_top = min(ys)
            component_bottom = max(ys)
            center_x = (component_left + component_right) / 2
            center_y = (component_top + component_bottom) / 2
            center_in_cell = keep_left <= center_x < keep_right and keep_top <= center_y < keep_bottom
            inside_ratio = inside_count / len(component)
            component_width = component_right - component_left + 1
            component_height = component_bottom - component_top + 1
            inset_x = max(12, round((keep_right - keep_left) * 0.08))
            inset_y = max(12, round((keep_bottom - keep_top) * 0.08))
            safe_left = keep_left + inset_x
            safe_top = keep_top + inset_y
            safe_right = keep_right - inset_x
            safe_bottom = keep_bottom - inset_y
            safe_inside_count = sum(
                1
                for px, py in component
                if safe_left <= px < safe_right and safe_top <= py < safe_bottom
            )
            average_brightness = sum(
                pixels[px, py][0] + pixels[px, py][1] + pixels[px, py][2]
                for px, py in component
            ) / (3 * len(component))
            touches_canvas_edge = (
                component_left == 0
                or component_top == 0
                or component_right == width - 1
                or component_bottom == height - 1
            )
            near_cell_edge = (
                abs(component_left - keep_left) <= 3
                or abs(component_right - (keep_right - 1)) <= 3
                or abs(component_top - keep_top) <= 3
                or abs(component_bottom - (keep_bottom - 1)) <= 3
            )
            thin_boundary_artifact = (
                near_cell_edge
                and safe_inside_count == 0
                and (
                    component_width <= width * 0.08
                    or component_height <= height * 0.055
                )
            )
            light_line_artifact = (
                safe_inside_count == 0
                and average_brightness >= 225
                and component_height <= max(5, round(height * 0.018))
                and component_width >= width * 0.18
            )
            light_outer_artifact = (
                safe_inside_count == 0
                and len(component) <= 120
                and average_brightness >= 225
            )
            keep_component = inside_count >= 8 and (center_in_cell or inside_ratio >= 0.5)
            if (
                (touches_canvas_edge and safe_inside_count == 0 and inside_ratio < 0.85)
                or thin_boundary_artifact
                or light_line_artifact
                or light_outer_artifact
            ):
                keep_component = False
            if keep_component:
                for px, py in component:
                    output_pixels[px, py] = source_pixels[px, py]

    return output


def foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background = estimate_background_color(image)

    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            if not is_background(pixels[x, y], background):
                xs.append(x)
                ys.append(y)

    if not xs:
        return None
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def tightened_side(image: Image.Image, padding: int = OUTPUT_PADDING) -> int:
    bbox = foreground_bbox(image)
    if bbox is None:
        return image.width

    left, top, right, bottom = bbox
    content_width = right - left
    content_height = bottom - top
    return max(content_width, content_height) + padding * 2


def center_and_tighten(
    image: Image.Image,
    padding: int = OUTPUT_PADDING,
    target_side: int | None = None,
) -> Image.Image:
    bbox = foreground_bbox(image)
    if bbox is None:
        return image

    left, top, right, bottom = bbox
    side = target_side or tightened_side(image, padding)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(center_x - side / 2)
    crop_top = round(center_y - side / 2)
    crop_right = crop_left + side
    crop_bottom = crop_top + side

    background = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    output = Image.new(image.mode, (side, side), background)
    src_left = max(0, crop_left)
    src_top = max(0, crop_top)
    src_right = min(image.width, crop_right)
    src_bottom = min(image.height, crop_bottom)

    if src_left < src_right and src_top < src_bottom:
        crop = image.crop((src_left, src_top, src_right, src_bottom))
        output.paste(crop, (src_left - crop_left, src_top - crop_top))

    return output


def is_light_gray_artifact(pixel: tuple[int, ...]) -> bool:
    r, g, b = pixel[:3]
    brightness = (r + g + b) / 3
    chroma = max(r, g, b) - min(r, g, b)
    return 185 <= brightness <= 242 and chroma <= 18


def clean_light_horizontal_artifacts(image: Image.Image) -> Image.Image:
    output = image.copy()
    rgb = output.convert("RGB")
    width, height = rgb.size
    rgb_pixels = rgb.load()
    output_pixels = output.load()
    white = (255, 255, 255, 255) if "A" in output.getbands() else (255, 255, 255)

    for y in range(round(height * 0.04), round(height * 0.28)):
        xs = [x for x in range(width) if is_light_gray_artifact(rgb_pixels[x, y])]
        if len(xs) < width * 0.22:
            continue

        for yy in range(max(0, y - 1), min(height, y + 2)):
            for x in xs:
                if is_light_gray_artifact(rgb_pixels[x, yy]):
                    output_pixels[x, yy] = white

    return output


def save_slice(image: Image.Image, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)
    return path.stat().st_size


def make_gif_frames(image: Image.Image) -> list[Image.Image]:
    base = image.convert("RGBA")
    frames: list[Image.Image] = []

    for degrees in GIF_ROTATION_DEGREES:
        frame = base.rotate(
            degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255, 255),
        )
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    return frames


def save_gif(image: Image.Image, path: Path) -> int:
    frames = make_gif_frames(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path.stat().st_size


def split_image(input_path: Path, output_dir: Path, grid: tuple[int, int] | None) -> dict[str, object]:
    image = Image.open(input_path)
    cols, rows = grid or detect_grid(image)
    slices_dir = output_dir / "slices"
    gifs_dir = output_dir / "gifs"
    records: list[SliceRecord] = []
    pieces: list[tuple[int, int, int, Image.Image]] = []

    for index, row, col, left, top, right, bottom in iter_boxes(image.width, image.height, cols, rows):
        square_box = expand_to_square(left, top, right, bottom)
        piece = make_square_slice(image, (left, top, right, bottom), square_box)
        pieces.append((index, row, col, piece))

    target_side = max(tightened_side(piece) for _, _, _, piece in pieces)

    for index, row, col, piece in pieces:
        piece = clean_light_horizontal_artifacts(center_and_tighten(piece, target_side=target_side))
        path = slices_dir / f"{index:02d}.png"
        gif_path = gifs_dir / f"{index:02d}.gif"
        file_size = save_slice(piece, path)
        gif_size = save_gif(piece, gif_path)
        records.append(
            SliceRecord(
                index=index,
                row=row,
                col=col,
                path=str(path),
                gif_path=str(gif_path),
                width=piece.width,
                height=piece.height,
                bytes=file_size,
                gif_bytes=gif_size,
            )
        )

    manifest = {
        "input": str(input_path),
        "grid": {"cols": cols, "rows": rows},
        "count": len(records),
        "outputs": [asdict(record) for record in records],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a 9-grid or 12-grid image into PNG files.")
    parser.add_argument("input", type=Path, help="Path to the source grid image.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for generated files. Defaults to ./output.",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=None,
        help="Grid as COLSxROWS, for example 4x3 or 3x3. Defaults to auto.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = split_image(args.input.expanduser(), args.output, args.grid)
    print(
        f"Generated {manifest['count']} slices from a "
        f"{manifest['grid']['cols']}x{manifest['grid']['rows']} grid into {args.output / 'slices'}"
    )


if __name__ == "__main__":
    main()
