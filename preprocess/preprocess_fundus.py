#!/usr/bin/env python3
"""Fundus image preprocessing for self-supervised pretraining.

Pipeline:
1. Scan input images
2. Deduplicate with perceptual hash (pHash)
3. Crop black background (threshold + morphological opening to ignore speckles)
4. Pad to square with black (0), then resize to 224x224 (keep aspect ratio)
5. Save as {dataset}__{original_stem}.jpg + manifest CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, ImageFilter
import imagehash
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ProcessResult:
    status: str
    src_path: str
    output_name: str = ""
    source_dataset: str = ""
    phash: str = ""
    orig_width: int = 0
    orig_height: int = 0
    crop_box: str = ""
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess fundus images for SSL pretraining.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input image folder (test on one folder first, e.g. dataset/AIROGS/0).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("pretrain_data"),
        help="Output folder for processed images.",
    )
    parser.add_argument(
        "--source_name",
        type=str,
        default="",
        help="Dataset source tag written to manifest (default: input folder name).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Grayscale threshold for non-black pixels when cropping.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding pixels added around detected ROI.",
    )
    parser.add_argument(
        "--open_size",
        type=int,
        default=31,
        help="Morphological opening kernel size to remove sparse bright noise before crop.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output square size.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG output quality.",
    )
    parser.add_argument(
        "--hash_size",
        type=int,
        default=16,
        help="pHash size; larger means stricter deduplication.",
    )
    parser.add_argument(
        "--hash_threshold",
        type=int,
        default=0,
        help="Hamming distance threshold for duplicates. 0 means exact pHash match only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images (0 = all). Useful for quick test runs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders under input_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest/output in output_dir.",
    )
    return parser.parse_args()


def iter_image_paths(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def crop_black_background(
    img: Image.Image,
    threshold: int,
    padding: int,
    open_size: int = 31,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop to fundus ROI.

    Uses thresholding + morphological opening so sparse bright noise
    (e.g. top-edge speckles) does not inflate the bounding box.
    Mask is computed on a downscaled image for speed, then mapped back.
    """
    gray = img.convert("L")
    # Downscale for fast morphology; map bbox back to full resolution.
    max_side = 512
    scale = 1.0
    work = gray
    if max(gray.size) > max_side:
        scale = max_side / max(gray.size)
        work = gray.resize(
            (max(1, int(gray.width * scale)), max(1, int(gray.height * scale))),
            Image.Resampling.BILINEAR,
        )

    mask = work.point(lambda p: 255 if p > threshold else 0)
    # Kernel must be odd for Min/MaxFilter.
    k = max(3, open_size if open_size % 2 == 1 else open_size + 1)
    # Scale kernel with downscale so effective FOV cleaning stays similar.
    if scale < 1.0:
        k = max(3, int(round(k * scale)))
        if k % 2 == 0:
            k += 1
    opened = mask.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    bbox = opened.getbbox()
    if bbox is None:
        # Fallback without morphology.
        bbox = mask.getbbox()
    if bbox is None:
        return img.copy(), (0, 0, img.width, img.height)

    left, top, right, bottom = bbox
    if scale < 1.0:
        inv = 1.0 / scale
        left = int(left * inv)
        top = int(top * inv)
        right = int(right * inv)
        bottom = int(bottom * inv)

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)
    # Guard against empty/inverted boxes.
    if right <= left or bottom <= top:
        return img.copy(), (0, 0, img.width, img.height)

    crop_box = (left, top, right, bottom)
    return img.crop(crop_box), crop_box


def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    """Pad image to square with constant fill (default black) without distorting shape."""
    w, h = img.size
    if w == h:
        return img.copy()

    side = max(w, h)
    if img.mode == "RGB":
        fill_color = (fill, fill, fill)
    elif img.mode == "L":
        fill_color = fill
    else:
        fill_color = (fill, fill, fill)

    canvas = Image.new(img.mode, (side, side), fill_color)
    offset = ((side - w) // 2, (side - h) // 2)
    canvas.paste(img, offset)
    return canvas


def resize_square(img: Image.Image, size: int) -> Image.Image:
    """Pad to square with black, then resize. Preserves eyeball aspect ratio."""
    squared = pad_to_square(img, fill=0)
    return squared.resize((size, size), Image.Resampling.LANCZOS)


def compute_phash(img: Image.Image, hash_size: int) -> imagehash.ImageHash:
    return imagehash.phash(img, hash_size=hash_size)


def hamming_distance(hash_a: imagehash.ImageHash, hash_b: imagehash.ImageHash) -> int:
    return hash_a - hash_b


def is_duplicate(
    current_hash: imagehash.ImageHash,
    seen_hashes: list[imagehash.ImageHash],
    threshold: int,
) -> bool:
    for seen in seen_hashes:
        if hamming_distance(current_hash, seen) <= threshold:
            return True
    return False


def sanitize_name(name: str) -> str:
    """Keep filesystem-friendly characters for output filenames."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in allowed else "_" for ch in name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def make_output_name(source_name: str, src_path: Path) -> str:
    """Name format: {dataset}__{original_stem}.jpg"""
    dataset = sanitize_name(source_name)
    stem = sanitize_name(src_path.stem)
    return f"{dataset}__{stem}.jpg"


def load_resume_state(output_dir: Path) -> tuple[list[imagehash.ImageHash], set[str]]:
    manifest_path = output_dir / "manifest.csv"
    seen_hashes: list[imagehash.ImageHash] = []
    processed_src: set[str] = set()

    if not manifest_path.exists():
        return seen_hashes, processed_src

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_src.add(row["src_path"])
            if row["status"] == "saved" and row["phash"]:
                seen_hashes.append(imagehash.hex_to_hash(row["phash"]))
    return seen_hashes, processed_src


def write_manifest_row(manifest_path: Path, fieldnames: list[str], row: ProcessResult) -> None:
    file_exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def save_config(output_dir: Path, args: argparse.Namespace) -> None:
    config = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "source_name": args.source_name or args.input_dir.name,
        "threshold": args.threshold,
        "padding": args.padding,
        "open_size": args.open_size,
        "size": args.size,
        "jpeg_quality": args.jpeg_quality,
        "hash_size": args.hash_size,
        "hash_threshold": args.hash_threshold,
        "recursive": args.recursive,
    }
    with (output_dir / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] input_dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    source_name = args.source_name or input_dir.name
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "status",
        "src_path",
        "output_name",
        "source_dataset",
        "phash",
        "orig_width",
        "orig_height",
        "crop_box",
        "message",
    ]

    seen_hashes: list[imagehash.ImageHash] = []
    processed_src: set[str] = set()
    if args.resume:
        seen_hashes, processed_src = load_resume_state(output_dir)

    image_paths = list(iter_image_paths(input_dir, args.recursive))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    stats = {
        "total": len(image_paths),
        "saved": 0,
        "duplicate": 0,
        "skipped_resume": 0,
        "failed": 0,
    }

    save_config(output_dir, args)

    for src_path in tqdm(image_paths, desc="Preprocessing", unit="img"):
        src_str = str(src_path.resolve())
        if args.resume and src_str in processed_src:
            stats["skipped_resume"] += 1
            continue

        try:
            original = load_rgb_image(src_path)
            cropped, crop_box = crop_black_background(
                original, args.threshold, args.padding, open_size=args.open_size
            )
            resized = resize_square(cropped, args.size)
            phash = compute_phash(resized, args.hash_size)

            if is_duplicate(phash, seen_hashes, args.hash_threshold):
                stats["duplicate"] += 1
                write_manifest_row(
                    manifest_path,
                    fieldnames,
                    ProcessResult(
                        status="duplicate",
                        src_path=src_str,
                        source_dataset=source_name,
                        phash=str(phash),
                        orig_width=original.width,
                        orig_height=original.height,
                        crop_box=str(crop_box),
                        message="skipped by pHash deduplication",
                    ),
                )
                continue

            output_name = make_output_name(source_name, src_path)
            output_path = images_dir / output_name
            if output_path.exists() and not args.resume:
                stats["failed"] += 1
                write_manifest_row(
                    manifest_path,
                    fieldnames,
                    ProcessResult(
                        status="failed",
                        src_path=src_str,
                        output_name=output_name,
                        source_dataset=source_name,
                        phash=str(phash),
                        orig_width=original.width,
                        orig_height=original.height,
                        crop_box=str(crop_box),
                        message="output filename already exists",
                    ),
                )
                continue

            resized.save(output_path, format="JPEG", quality=args.jpeg_quality, optimize=True)

            seen_hashes.append(phash)
            stats["saved"] += 1

            write_manifest_row(
                manifest_path,
                fieldnames,
                ProcessResult(
                    status="saved",
                    src_path=src_str,
                    output_name=output_name,
                    source_dataset=source_name,
                    phash=str(phash),
                    orig_width=original.width,
                    orig_height=original.height,
                    crop_box=str(crop_box),
                    message="ok",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - keep batch running on bad files
            stats["failed"] += 1
            write_manifest_row(
                manifest_path,
                fieldnames,
                ProcessResult(
                    status="failed",
                    src_path=src_str,
                    source_dataset=source_name,
                    message=str(exc),
                ),
            )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Images dir  : {images_dir}")
    print(f"Manifest    : {manifest_path}")
    print(f"Saved       : {stats['saved']}")
    print(f"Duplicate   : {stats['duplicate']}")
    print(f"Failed      : {stats['failed']}")
    print(f"Skipped     : {stats['skipped_resume']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
