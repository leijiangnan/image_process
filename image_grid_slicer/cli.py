from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


NEAR_WHITE_RGB = (244, 244, 244)
BACKGROUND_DISTANCE = 28
OUTPUT_PADDING = 12
SOURCE_MARGIN_RATIO = 0.22
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


def detect_black_grid_boxes(
    image: Image.Image,
    cols: int,
    rows: int,
) -> list[tuple[int, int, int, int, int, int, int]] | None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    def is_dark(pixel: tuple[int, int, int]) -> bool:
        return max(pixel) <= 45

    def runs(values: list[bool]) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(values + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if index - start >= 2:
                    found.append((start, index))
                start = None
        return found

    dark_columns = [
        sum(1 for y in range(height) if is_dark(pixels[x, y])) >= height * 0.72
        for x in range(width)
    ]
    dark_rows = [
        sum(1 for x in range(width) if is_dark(pixels[x, y])) >= width * 0.72
        for y in range(height)
    ]
    column_runs = runs(dark_columns)
    row_runs = runs(dark_rows)
    if len(column_runs) != cols + 1 or len(row_runs) != rows + 1:
        return None

    boxes: list[tuple[int, int, int, int, int, int, int]] = []
    index = 1
    for row in range(rows):
        for col in range(cols):
            left = column_runs[col][1] + 1
            top = row_runs[row][1] + 1
            right = column_runs[col + 1][0] - 1
            bottom = row_runs[row + 1][0] - 1
            if left >= right or top >= bottom:
                return None
            boxes.append((index, row + 1, col + 1, left, top, right, bottom))
            index += 1
    return boxes


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

    keep_box = (
        left - square_left,
        top - square_top,
        right - square_left,
        bottom - square_top,
    )
    return keep_content_regions(square, keep_box)


def make_plain_square_slice(image: Image.Image, cell_box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = cell_box
    width = right - left
    height = bottom - top
    side = max(width, height)
    background = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    output = Image.new(image.mode, (side, side), background)
    crop = image.crop((left, top, right, bottom))
    output.paste(crop, ((side - width) // 2, (side - height) // 2))
    return output


def expand_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    margin: int,
) -> tuple[int, int, int, int]:
    return left - margin, top - margin, right + margin, bottom + margin


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


def keep_content_regions(image: Image.Image, keep_box: tuple[int, int, int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background_color = estimate_background_color(image)
    mask = [[not is_background(pixels[x, y], background_color) for x in range(width)] for y in range(height)]
    visited: set[tuple[int, int]] = set()
    white = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    output = Image.new(image.mode, image.size, white)
    keep_left, keep_top, keep_right, keep_bottom = keep_box
    copy_boxes: list[tuple[int, int, int, int]] = [
        (
            max(0, keep_left),
            max(0, keep_top),
            min(width, keep_right),
            min(height, keep_bottom),
        )
    ]

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
            if inside_count < 8:
                continue

            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            component_left = min(xs)
            component_right = max(xs) + 1
            component_top = min(ys)
            component_bottom = max(ys) + 1
            outside_cell = (
                component_left < keep_left
                or component_top < keep_top
                or component_right > keep_right
                or component_bottom > keep_bottom
            )
            component_width = component_right - component_left
            component_height = component_bottom - component_top
            center_x = (component_left + component_right) / 2
            center_y = (component_top + component_bottom) / 2
            center_in_cell = keep_left <= center_x < keep_right and keep_top <= center_y < keep_bottom
            inside_ratio = inside_count / len(component)

            if not (center_in_cell or inside_ratio >= 0.5):
                continue
            if outside_cell and len(component) < 180 and min(component_width, component_height) < 12:
                continue

            padding = 10
            copy_boxes.append(
                (
                    max(0, component_left - padding),
                    max(0, component_top - padding),
                    min(width, component_right + padding),
                    min(height, component_bottom + padding),
                )
            )

    for box in copy_boxes:
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        output.paste(image.crop(box), box[:2])
    output = remove_right_edge_intrusions(output, keep_box)

    if not copy_boxes:
        return output

    left = min(box[0] for box in copy_boxes)
    top = min(box[1] for box in copy_boxes)
    right = max(box[2] for box in copy_boxes)
    bottom = max(box[3] for box in copy_boxes)
    return center_crop_to_content(output, (left, top, right, bottom))


def remove_right_edge_intrusions(image: Image.Image, keep_box: tuple[int, int, int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background_color = estimate_background_color(image)
    mask = [[not is_background(pixels[x, y], background_color) for x in range(width)] for y in range(height)]
    visited: set[tuple[int, int]] = set()
    output = image.copy()
    output_pixels = output.load()
    white = (255, 255, 255, 255) if "A" in output.getbands() else (255, 255, 255)
    keep_left, keep_top, keep_right, keep_bottom = keep_box
    edge_band = max(12, round((keep_right - keep_left) * 0.055))

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

            xs = [point[0] for point in component]
            ys = [point[1] for point in component]
            component_left = min(xs)
            component_right = max(xs) + 1
            component_top = min(ys)
            component_bottom = max(ys) + 1
            component_width = component_right - component_left
            component_height = component_bottom - component_top
            average_chroma = sum(
                max(pixels[px, py]) - min(pixels[px, py])
                for px, py in component
            ) / len(component)
            near_right_edge = component_right >= keep_right - edge_band
            isolated_sliver = (
                len(component) <= 650
                and component_left >= keep_right - edge_band * 4
                and (component_width <= edge_band * 2 or component_height <= edge_band * 4)
            )
            colorful = average_chroma >= 35

            if not (near_right_edge and isolated_sliver and colorful):
                continue

            padding = 5
            for yy in range(max(0, component_top - padding), min(height, component_bottom + padding)):
                for xx in range(max(0, component_left - padding), min(width, component_right + padding)):
                    output_pixels[xx, yy] = white

    return output


def center_crop_to_content(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = bbox
    width, height = image.size
    side = max(right - left, bottom - top) + OUTPUT_PADDING * 2
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(center_x - side / 2)
    crop_top = round(center_y - side / 2)
    crop_right = crop_left + side
    crop_bottom = crop_top + side
    white = (255, 255, 255, 255) if "A" in image.getbands() else (255, 255, 255)
    output = Image.new(image.mode, (side, side), white)

    src_left = max(0, crop_left)
    src_top = max(0, crop_top)
    src_right = min(width, crop_right)
    src_bottom = min(height, crop_bottom)
    if src_left < src_right and src_top < src_bottom:
        crop = image.crop((src_left, src_top, src_right, src_bottom))
        output.paste(crop, (src_left - crop_left, src_top - crop_top))

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
    detected_boxes = detect_black_grid_boxes(image, cols, rows)
    boxes = detected_boxes or list(iter_boxes(image.width, image.height, cols, rows))

    for index, row, col, left, top, right, bottom in boxes:
        if detected_boxes:
            piece = make_plain_square_slice(image, (left, top, right, bottom))
            pieces.append((index, row, col, piece))
            continue

        margin = 0 if detected_boxes else round(max(right - left, bottom - top) * SOURCE_MARGIN_RATIO)
        expanded_box = expand_box(left, top, right, bottom, margin)
        square_box = expand_to_square(*expanded_box)
        piece = make_square_slice(image, (left, top, right, bottom), square_box)
        pieces.append((index, row, col, piece))

    for index, row, col, piece in pieces:
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
