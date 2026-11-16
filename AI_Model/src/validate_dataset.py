"""
Stage 3 (Step 1): Non-Destructive Lunar Source Image Validation Script.

Inspects all source images in dataset/raw_images/ without modifying, resizing,
moving, or deleting any original file.

Checks performed:
1. File readability and header integrity (via tifffile for TIFFs, Pillow for others).
2. Full-resolution image dimensions (width, height, channels, bit-depth).
3. Image format verification against configured supported formats.
4. File size verification (non-empty).
5. Exact duplicate detection via SHA-256 hashing (plus cross-check against Stage 2
   image_metadata.csv hashes to confirm original files were never altered).
6. Minimum dimension thresholds (without rejecting low-contrast lunar lighting).
7. Writes a structured audit log to dataset/processed/reports/validation_report.csv.

Usage (from AI_Model/ directory):
    python src/validate_dataset.py
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
import tifffile

# Reuse Stage 2 helpers cleanly without duplicating code
from collect_images import compute_sha256, load_config

Image.MAX_IMAGE_PIXELS = None

VALIDATION_REPORT_HEADERS = [
    "source_filename",
    "relative_path",
    "validation_status",
    "rejection_reason",
    "image_format",
    "width",
    "height",
    "channels",
    "dtype",
    "file_size_bytes",
    "sha256_hash",
    "stage2_hash_match",
    "checked_at_utc",
]


def load_stage2_metadata_lookup(metadata_csv: Path) -> Dict[str, dict]:
    """Load Stage 2 image_metadata.csv keyed by filename for read-only hash verification."""
    lookup: Dict[str, dict] = {}
    if not metadata_csv.exists():
        return lookup

    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fname = (row.get("filename") or "").strip()
            if fname:
                lookup[fname] = row
    return lookup


def inspect_image_header(file_path: Path) -> Tuple[bool, str, dict]:
    """
    Inspect image dimensions, channels, dtype, and format without loading the
    entire raster into memory.
    """
    meta = {
        "format": "",
        "width": 0,
        "height": 0,
        "channels": 0,
        "dtype": "",
    }

    suffix = file_path.suffix.lower().lstrip(".")

    # Use tifffile for TIFF / LROC Pyramid TIFFs so multi-page strips & huge dimensions
    # are inspected safely from header tags without full-frame allocation.
    if suffix in ("tif", "tiff"):
        try:
            with tifffile.TiffFile(file_path) as tif:
                if not tif.pages:
                    return False, "CORRUPTED_TIFF_NO_PAGES", meta
                # Page 0 is the full-resolution base layer in LROC pyramid TIFFs
                base_page = tif.pages[0]
                shape = base_page.shape
                if len(shape) == 2:
                    height, width = shape
                    channels = 1
                elif len(shape) == 3:
                    # Handle (H, W, C) vs (C, H, W)
                    if shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
                        channels, height, width = shape
                    else:
                        height, width, channels = shape
                else:
                    return False, f"UNSUPPORTED_TIFF_SHAPE ({shape})", meta

                meta["format"] = "TIFF"
                meta["width"] = int(width)
                meta["height"] = int(height)
                meta["channels"] = int(channels)
                meta["dtype"] = str(base_page.dtype)
                return True, "", meta
        except (tifffile.TiffFileError, OSError, ValueError, RuntimeError) as exc:
            return False, f"CORRUPTED_OR_UNREADABLE_TIFF: {exc}", meta

    # Fallback to Pillow header inspection for PNG/JPG
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            meta["format"] = (img.format or suffix).upper()
            meta["width"] = int(width)
            meta["height"] = int(height)
            meta["channels"] = int(bands)
            meta["dtype"] = "uint8"
            return True, "", meta
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"CORRUPTED_OR_UNREADABLE_IMAGE: {exc}", meta


def validate_raw_dataset(cfg: dict, ai_model_root: Path) -> Tuple[List[dict], dict]:
    """Validate all files in raw_images_dir and return detailed rows and summary counts."""
    paths_cfg = cfg.get("paths", {})
    val_cfg = cfg.get("validation", {})

    raw_dir = ai_model_root / paths_cfg.get("raw_images_dir", "dataset/raw_images")
    metadata_csv = ai_model_root / paths_cfg.get(
        "image_metadata_csv", "dataset/image_metadata.csv"
    )
    report_csv = ai_model_root / paths_cfg.get(
        "validation_report_csv", "dataset/processed/reports/validation_report.csv"
    )

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw images directory not found: {raw_dir}")

    supported_formats = {
        fmt.lower().lstrip(".")
        for fmt in val_cfg.get("supported_formats", ["tif", "tiff", "png", "jpg", "jpeg"])
    }
    min_width = int(val_cfg.get("min_width", 512))
    min_height = int(val_cfg.get("min_height", 512))
    check_duplicates = bool(val_cfg.get("enable_duplicate_detection", True))
    verify_stage2 = bool(val_cfg.get("verify_against_stage2_hashes", True))

    stage2_lookup = load_stage2_metadata_lookup(metadata_csv) if verify_stage2 else {}
    seen_hashes: Dict[str, str] = {}
    report_rows: List[dict] = []

    counts = {
        "total_files": 0,
        "valid_images": 0,
        "rejected_images": 0,
        "duplicate_images": 0,
        "hash_mismatches": 0,
    }

    candidate_files = sorted([p for p in raw_dir.iterdir() if p.is_file()])

    for file_path in candidate_files:
        counts["total_files"] += 1
        checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rel_path = file_path.relative_to(ai_model_root).as_posix()
        file_size = file_path.stat().st_size
        ext = file_path.suffix.lower().lstrip(".")

        print(f"Inspecting [{counts['total_files']}/{len(candidate_files)}]: {file_path.name}")

        # 1. Check file size
        if file_size == 0:
            counts["rejected_images"] += 1
            report_rows.append(
                {
                    "source_filename": file_path.name,
                    "relative_path": rel_path,
                    "validation_status": "REJECTED",
                    "rejection_reason": "EMPTY_FILE_0_BYTES",
                    "image_format": ext.upper(),
                    "width": "",
                    "height": "",
                    "channels": "",
                    "dtype": "",
                    "file_size_bytes": 0,
                    "sha256_hash": "",
                    "stage2_hash_match": "N/A",
                    "checked_at_utc": checked_at,
                }
            )
            print("    -> [REJECTED] Empty file (0 bytes). Original left untouched.")
            continue

        # 2. Check supported extension/format
        if ext not in supported_formats:
            counts["rejected_images"] += 1
            report_rows.append(
                {
                    "source_filename": file_path.name,
                    "relative_path": rel_path,
                    "validation_status": "REJECTED",
                    "rejection_reason": f"UNSUPPORTED_FORMAT ({ext})",
                    "image_format": ext.upper(),
                    "width": "",
                    "height": "",
                    "channels": "",
                    "dtype": "",
                    "file_size_bytes": file_size,
                    "sha256_hash": "",
                    "stage2_hash_match": "N/A",
                    "checked_at_utc": checked_at,
                }
            )
            print(f"    -> [REJECTED] Unsupported format '{ext}'. Original left untouched.")
            continue

        # 3. Compute SHA-256 hash and verify against Stage 2 metadata
        sha256_hash = compute_sha256(file_path)
        stage2_match_status = "NOT_IN_STAGE2_CSV"
        if file_path.name in stage2_lookup:
            recorded_hash = (stage2_lookup[file_path.name].get("image_hash") or "").strip()
            if recorded_hash:
                if recorded_hash == sha256_hash:
                    stage2_match_status = "MATCH"
                else:
                    stage2_match_status = "MISMATCH"
                    counts["hash_mismatches"] += 1

        # 4. Inspect header & dimensions (low contrast / shadows are NOT rejected)
        readable, err_reason, meta = inspect_image_header(file_path)
        if not readable:
            counts["rejected_images"] += 1
            report_rows.append(
                {
                    "source_filename": file_path.name,
                    "relative_path": rel_path,
                    "validation_status": "REJECTED",
                    "rejection_reason": err_reason,
                    "image_format": meta["format"] or ext.upper(),
                    "width": meta["width"] or "",
                    "height": meta["height"] or "",
                    "channels": meta["channels"] or "",
                    "dtype": meta["dtype"],
                    "file_size_bytes": file_size,
                    "sha256_hash": sha256_hash,
                    "stage2_hash_match": stage2_match_status,
                    "checked_at_utc": checked_at,
                }
            )
            print(f"    -> [REJECTED] {err_reason}. Original left untouched.")
            continue

        # 5. Minimum dimension check
        if meta["width"] < min_width or meta["height"] < min_height:
            counts["rejected_images"] += 1
            reason = (
                f"DIMENSIONS_TOO_SMALL ({meta['width']}x{meta['height']} "
                f"< {min_width}x{min_height})"
            )
            report_rows.append(
                {
                    "source_filename": file_path.name,
                    "relative_path": rel_path,
                    "validation_status": "REJECTED",
                    "rejection_reason": reason,
                    "image_format": meta["format"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "channels": meta["channels"],
                    "dtype": meta["dtype"],
                    "file_size_bytes": file_size,
                    "sha256_hash": sha256_hash,
                    "stage2_hash_match": stage2_match_status,
                    "checked_at_utc": checked_at,
                }
            )
            print(f"    -> [REJECTED] {reason}. Original left untouched.")
            continue

        # 6. Exact duplicate check via SHA-256
        if check_duplicates and sha256_hash in seen_hashes:
            counts["duplicate_images"] += 1
            counts["rejected_images"] += 1
            first_file = seen_hashes[sha256_hash]
            reason = f"EXACT_DUPLICATE_OF ({first_file})"
            report_rows.append(
                {
                    "source_filename": file_path.name,
                    "relative_path": rel_path,
                    "validation_status": "DUPLICATE",
                    "rejection_reason": reason,
                    "image_format": meta["format"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "channels": meta["channels"],
                    "dtype": meta["dtype"],
                    "file_size_bytes": file_size,
                    "sha256_hash": sha256_hash,
                    "stage2_hash_match": stage2_match_status,
                    "checked_at_utc": checked_at,
                }
            )
            print(f"    -> [DUPLICATE] Matches '{first_file}'. Original left untouched.")
            continue

        seen_hashes[sha256_hash] = file_path.name
        counts["valid_images"] += 1
        report_rows.append(
            {
                "source_filename": file_path.name,
                "relative_path": rel_path,
                "validation_status": "VALID",
                "rejection_reason": "",
                "image_format": meta["format"],
                "width": meta["width"],
                "height": meta["height"],
                "channels": meta["channels"],
                "dtype": meta["dtype"],
                "file_size_bytes": file_size,
                "sha256_hash": sha256_hash,
                "stage2_hash_match": stage2_match_status,
                "checked_at_utc": checked_at,
            }
        )
        print(
            f"    -> [VALID] {meta['width']}x{meta['height']} ({meta['dtype']}, "
            f"{file_size / (1024 * 1024):.2f} MB) | Stage2 Hash: {stage2_match_status}"
        )

    # Write validation_report.csv
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VALIDATION_REPORT_HEADERS)
        writer.writeheader()
        writer.writerows(report_rows)

    return report_rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: Validate raw lunar surface images non-destructively."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to dataset_config.yaml relative to AI_Model/.",
    )
    return parser.parse_args()


def main() -> int:
    ai_model_root = Path(__file__).resolve().parent.parent
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ai_model_root / config_path

    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=" * 64)
    print("Lunar Boulder Hazard Mapper - Stage 3 Dataset Validation")
    print("=" * 64)

    try:
        _, counts = validate_raw_dataset(cfg, ai_model_root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    report_rel = cfg.get("paths", {}).get(
        "validation_report_csv", "dataset/processed/reports/validation_report.csv"
    )

    print("\n" + "=" * 64)
    print("Validation Summary:")
    print(f"  Total Source Files Checked : {counts['total_files']}")
    print(f"  Valid Source Images        : {counts['valid_images']}")
    print(f"  Rejected / Duplicate Files : {counts['rejected_images']}")
    print(f"  Exact Duplicates Detected  : {counts['duplicate_images']}")
    print(f"  Stage 2 Hash Mismatches    : {counts['hash_mismatches']}")
    print(f"  Report Saved To            : {report_rel}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())