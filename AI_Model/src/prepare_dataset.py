"""
Stage 3 (Step 2): Leak-Free Dataset Splitting & Memory-Efficient Lunar TIFF Tiling.

Key design guarantees:
1. Never modifies, resizes, moves, or overwrites original files in dataset/raw_images/.
2. Splits VALID source images into train / val / test BEFORE generating any tiles,
   guaranteeing zero spatial data leakage across splits.
3. Uses tifffile memory-mapping / Zarr windowed slicing so multi-hundred-megabyte
   LROC NAC strips (e.g., 5064 x 52224) are tiled window-by-window without loading
   the full uncompressed raster into system RAM.
4. Handles uneven image dimensions cleanly (either boundary-snapped full 1024x1024
   tiles or exact edge crops) and records exact x_offset, y_offset, and dimensions
   in dataset/dataset_manifest.csv for Stage 4 Roboflow annotation.
5. Writes factual execution metrics to dataset/processed/reports/dataset_summary.json.

Usage (from AI_Model/ directory):
    python src/prepare_dataset.py
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import tifffile

from collect_images import load_config
from validate_dataset import validate_raw_dataset

Image.MAX_IMAGE_PIXELS = None

MANIFEST_HEADERS = [
    "tile_id",
    "tile_filename",
    "tile_path",
    "original_filename",
    "original_image_hash",
    "split",
    "tile_width",
    "tile_height",
    "x_offset",
    "y_offset",
    "overlap_percentage",
    "source_image_width",
    "source_image_height",
]


def sanitize_stem(filename: str) -> str:
    """Create a filesystem- and Roboflow-safe identifier stem from an original filename."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "lunar_source"


def compute_axis_offsets(
    full_length: int,
    tile_length: int,
    overlap_pct: float,
    snap_edge_tiles: bool,
) -> List[Tuple[int, int]]:
    """
    Compute (offset, actual_tile_length) pairs along a single axis (X or Y).

    - When snap_edge_tiles is True, if the final step extends past full_length,
      the final tile shifts inward to (full_length - tile_length) so every tile
      has uniform dimensions (e.g., 1024x1024) without thin border slivers.
    - When snap_edge_tiles is False, the final tile stops at full_length with
      actual_tile_length = full_length - offset.
    - If full_length <= tile_length, produces a single window (0, full_length).
    """
    if full_length <= tile_length:
        return [(0, full_length)]

    overlap_fraction = max(0.0, min(0.95, overlap_pct / 100.0))
    stride = max(1, int(round(tile_length * (1.0 - overlap_fraction))))

    windows: List[Tuple[int, int]] = []
    pos = 0
    while pos < full_length:
        if pos + tile_length >= full_length:
            if snap_edge_tiles:
                snapped_pos = max(0, full_length - tile_length)
                if not windows or windows[-1][0] != snapped_pos:
                    windows.append((snapped_pos, tile_length))
            else:
                windows.append((pos, full_length - pos))
            break

        windows.append((pos, tile_length))
        pos += stride

    return windows


def assign_source_splits(
    valid_records: List[dict],
    split_cfg: dict,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Deterministically assign each VALID source image to 'train', 'val', or 'test'
    BEFORE generating tiles.
    """
    seed = int(split_cfg.get("random_seed", 42))
    mode = str(split_cfg.get("mode", "counts")).lower()

    # Sort deterministically by filename before shuffling with fixed seed
    ordered = sorted(valid_records, key=lambda r: r["source_filename"])
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    if total == 0:
        return {}, {"train": [], "val": [], "test": []}

    if mode == "counts":
        n_train = int(split_cfg.get("train_count", 6))
        n_val = int(split_cfg.get("val_count", 1))
        n_test = int(split_cfg.get("test_count", 1))
        if n_train + n_val + n_test != total:
            # Fallback proportionally if dataset size differs from configured counts
            mode = "ratios"

    if mode != "counts":
        r_train = float(split_cfg.get("train_ratio", 0.75))
        r_val = float(split_cfg.get("val_ratio", 0.125))
        n_train = max(1, int(round(total * r_train))) if total >= 3 else total
        remaining = max(0, total - n_train)
        n_val = max(1, int(round(total * r_val))) if remaining >= 2 else (1 if remaining == 1 else 0)
        n_test = max(0, total - n_train - n_val)

    file_to_split: Dict[str, str] = {}
    split_to_files: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for idx, rec in enumerate(ordered):
        fname = rec["source_filename"]
        if idx < n_train:
            split_name = "train"
        elif idx < n_train + n_val:
            split_name = "val"
        else:
            split_name = "test"
        file_to_split[fname] = split_name
        split_to_files[split_name].append(fname)

    return file_to_split, split_to_files


def convert_tile_to_uint8(tile_array: np.ndarray, normalize_16bit: bool) -> np.ndarray:
    """
    Convert a tile array to 8-bit BGR/grayscale suitable for PNG/JPEG saving
    while preserving local lunar boulder and shadow contrast.
    """
    arr = np.asarray(tile_array)

    # Handle channel-first (C, H, W) if encountered
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4:
        arr = np.transpose(arr, (1, 2, 0))

    # Drop alpha channel if RGBA
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if arr.dtype == np.uint8:
        out = arr
    elif normalize_16bit and np.issubdtype(arr.dtype, np.integer):
        non_zero = arr[arr > 0]
        if non_zero.size > 0:
            low, high = np.percentile(non_zero, (1.0, 99.0))
            if high > low:
                scaled = np.clip((arr.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
                out = scaled.astype(np.uint8)
            else:
                out = np.clip(arr >> 8, 0, 255).astype(np.uint8)
        else:
            out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > arr_min:
            out = ((arr.astype(np.float32) - arr_min) * (255.0 / (arr_max - arr_min))).astype(
                np.uint8
            )
        else:
            out = np.zeros(arr.shape, dtype=np.uint8)

    # Convert RGB to BGR for cv2.imencode if 3-channel
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out


def save_tile_unicode_safe(tile_uint8: np.ndarray, dest_path: Path, ext: str, jpeg_quality: int) -> bool:
    """Save an image tile to a Windows path safely using cv2.imencode + Path.write_bytes."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ext.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        ok, encoded = cv2.imencode(".jpg", tile_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        ok, encoded = cv2.imencode(".png", tile_uint8, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    if not ok:
        return False
    dest_path.write_bytes(encoded.tobytes())
    return True


def open_source_raster_view(file_path: Path, use_memmap: bool):
    """
    Open a memory-efficient view of a source image without loading the full raster
    into RAM when memory-mapping or Zarr slicing is supported by the TIFF structure.
    Returns (array_like, cleanup_callback, access_mode_str).
    """
    suffix = file_path.suffix.lower().lstrip(".")
    if suffix in ("tif", "tiff"):
        # Attempt 1: Direct memory-map of Page 0 (zero-copy OS virtual memory paging)
        if use_memmap:
            try:
                mm = tifffile.memmap(str(file_path), page=0, mode="r")
                return mm, (lambda: None), "tifffile.memmap"
            except Exception:
                pass

        # Attempt 2: Zarr windowed store on Page 0 (reads only requested strip/tile chunks)
        tif = tifffile.TiffFile(file_path)
        try:
            import zarr  # optional if installed

            store = tif.aszarr(level=0)
            z_arr = zarr.open(store, mode="r")
            return z_arr, tif.close, "tifffile.aszarr"
        except Exception:
            # Attempt 3: Read Page 0 array (for compressed single-strip TIFFs) and close immediately
            arr = tif.pages[0].asarray()
            tif.close()
            return arr, (lambda: None), "tifffile.page0_array"

    # Non-TIFF fallback
    raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
    decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError(f"Failed to decode image: {file_path}")
    return decoded, (lambda: None), "opencv_array"


def extract_window(raster, x_off: int, y_off: int, width: int, height: int) -> np.ndarray:
    """Slice a [y_off : y_off + height, x_off : x_off + width] window from 2D or 3D raster."""
    if raster.ndim == 2:
        return np.array(raster[y_off : y_off + height, x_off : x_off + width], copy=True)
    if raster.ndim == 3:
        if raster.shape[0] <= 4 and raster.shape[1] > 4 and raster.shape[2] > 4:
            return np.array(raster[:, y_off : y_off + height, x_off : x_off + width], copy=True)
        return np.array(raster[y_off : y_off + height, x_off : x_off + width, :], copy=True)
    raise ValueError(f"Unsupported raster dimensions: {raster.shape}")


def prepare_dataset(cfg: dict, ai_model_root: Path, clean_existing: bool = False) -> dict:
    """Execute full Stage 3 pipeline: validate, split by source image, tile, and write reports."""
    paths_cfg = cfg.get("paths", {})
    split_cfg = cfg.get("splitting", {})
    tile_cfg = cfg.get("tiling", {})
    mem_cfg = cfg.get("memory", {})

    raw_dir = ai_model_root / paths_cfg.get("raw_images_dir", "dataset/raw_images")
    tiles_root = ai_model_root / paths_cfg.get("tiles_dir", "dataset/processed/tiles")
    reports_dir = ai_model_root / paths_cfg.get("reports_dir", "dataset/processed/reports")
    manifest_csv = ai_model_root / paths_cfg.get(
        "dataset_manifest_csv", "dataset/dataset_manifest.csv"
    )
    summary_json = ai_model_root / paths_cfg.get(
        "dataset_summary_json", "dataset/processed/reports/dataset_summary.json"
    )

    tile_width = int(tile_cfg.get("tile_width", 1024))
    tile_height = int(tile_cfg.get("tile_height", 1024))
    overlap_pct = float(tile_cfg.get("overlap_percentage", 10.0))
    snap_edge_tiles = bool(tile_cfg.get("snap_edge_tiles", True))
    output_format = str(tile_cfg.get("output_format", "png")).lower().lstrip(".")
    jpeg_quality = int(tile_cfg.get("jpeg_quality", 95))
    skip_blank_tiles = bool(tile_cfg.get("skip_blank_tiles", True))
    min_non_zero_fraction = float(tile_cfg.get("min_non_zero_fraction", 0.05))

    use_memmap = bool(mem_cfg.get("use_memmap", True))
    normalize_16bit = bool(mem_cfg.get("normalize_16bit_to_8bit", True))

    # Step 1: Run non-destructive source validation first
    print("[Step 1/4] Validating raw source images (read-only)...")
    validation_rows, val_counts = validate_raw_dataset(cfg, ai_model_root)
    valid_records = [r for r in validation_rows if r["validation_status"] == "VALID"]

    if not valid_records:
        raise RuntimeError("No VALID source images found in dataset/raw_images/. Aborting.")

    # Optional cleanup of previous processed tiles only (NEVER touches raw_images)
    for split_name in ("train", "val", "test"):
        split_dir = tiles_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if clean_existing:
            for existing_tile in split_dir.iterdir():
                if existing_tile.is_file():
                    existing_tile.unlink()

    # Step 2: Source-level train/val/test split (prevents spatial data leakage)
    print("\n[Step 2/4] Assigning source images to train / val / test splits...")
    file_to_split, split_to_files = assign_source_splits(valid_records, split_cfg)
    for split_name, files in split_to_files.items():
        print(f"  {split_name.upper():<5} ({len(files)} source images): {files}")

    # Step 3: Windowed tiling per source image
    print("\n[Step 3/4] Generating memory-efficient image tiles...")
    manifest_rows: List[dict] = []
    processing_errors: List[dict] = []
    source_dimensions_record: List[dict] = []

    tile_counts = {"train": 0, "val": 0, "test": 0}
    skipped_blank_count = 0

    for rec in sorted(valid_records, key=lambda r: r["source_filename"]):
        src_filename = rec["source_filename"]
        src_path = raw_dir / src_filename
        src_hash = rec["sha256_hash"]
        src_width = int(rec["width"])
        src_height = int(rec["height"])
        assigned_split = file_to_split[src_filename]
        safe_stem = sanitize_stem(src_filename)

        source_dimensions_record.append(
            {
                "filename": src_filename,
                "width": src_width,
                "height": src_height,
                "split": assigned_split,
                "sha256_hash": src_hash,
            }
        )

        x_windows = compute_axis_offsets(src_width, tile_width, overlap_pct, snap_edge_tiles)
        y_windows = compute_axis_offsets(src_height, tile_height, overlap_pct, snap_edge_tiles)
        expected_windows = len(x_windows) * len(y_windows)

        print(
            f"\n  Source: {src_filename} ({src_width}x{src_height}) -> "
            f"Split: [{assigned_split.upper()}] | Grid: {len(x_windows)}x{len(y_windows)} "
            f"({expected_windows} candidate tiles)"
        )

        cleanup_fn = lambda: None
        try:
            raster, cleanup_fn, access_mode = open_source_raster_view(src_path, use_memmap)
            print(f"    Access mode: {access_mode}")

            saved_for_source = 0
            for y_off, cur_h in y_windows:
                for x_off, cur_w in x_windows:
                    window = extract_window(raster, x_off, y_off, cur_w, cur_h)

                    # Skip empty nodata border tiles if configured
                    if skip_blank_tiles:
                        non_zero_ratio = float(np.count_nonzero(window)) / float(window.size)
                        if non_zero_ratio < min_non_zero_fraction:
                            skipped_blank_count += 1
                            continue

                    tile_uint8 = convert_tile_to_uint8(window, normalize_16bit)
                    tile_id = f"{safe_stem}_x{x_off:05d}_y{y_off:05d}"
                    tile_filename = f"{tile_id}.{output_format}"
                    tile_abs_path = tiles_root / assigned_split / tile_filename
                    tile_rel_path = tile_abs_path.relative_to(ai_model_root).as_posix()

                    saved_ok = save_tile_unicode_safe(
                        tile_uint8, tile_abs_path, output_format, jpeg_quality
                    )
                    if not saved_ok:
                        processing_errors.append(
                            {
                                "source_filename": src_filename,
                                "tile_id": tile_id,
                                "error": "Failed to encode/save tile image.",
                            }
                        )
                        continue

                    saved_for_source += 1
                    tile_counts[assigned_split] += 1

                    manifest_rows.append(
                        {
                            "tile_id": tile_id,
                            "tile_filename": tile_filename,
                            "tile_path": tile_rel_path,
                            "original_filename": src_filename,
                            "original_image_hash": src_hash,
                            "split": assigned_split,
                            "tile_width": cur_w,
                            "tile_height": cur_h,
                            "x_offset": x_off,
                            "y_offset": y_off,
                            "overlap_percentage": overlap_pct,
                            "source_image_width": src_width,
                            "source_image_height": src_height,
                        }
                    )

            print(f"    Saved {saved_for_source} tiles into dataset/processed/tiles/{assigned_split}/")

        except Exception as exc:
            err_msg = f"Failed processing '{src_filename}': {exc}"
            print(f"    [ERROR] {err_msg}")
            processing_errors.append(
                {
                    "source_filename": src_filename,
                    "tile_id": "",
                    "error": err_msg,
                }
            )
        finally:
            cleanup_fn()
            gc.collect()

    # Step 4: Write dataset_manifest.csv and dataset_summary.json
    print("\n[Step 4/4] Writing dataset_manifest.csv and dataset_summary.json...")
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_tiles = sum(tile_counts.values())
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_image_counts": {
            "total_source_images": val_counts["total_files"],
            "valid_source_images": val_counts["valid_images"],
            "rejected_source_images": val_counts["rejected_images"],
            "duplicate_images": val_counts["duplicate_images"],
        },
        "tile_counts": {
            "total_tiles_generated": total_tiles,
            "training_tiles": tile_counts["train"],
            "validation_tiles": tile_counts["val"],
            "testing_tiles": tile_counts["test"],
            "skipped_blank_border_tiles": skipped_blank_count,
        },
        "split_assignments": split_to_files,
        "evaluation_limitation_note": (
            "Dataset contains only 8 source LROC strips (6 train / 1 val / 1 test). "
            "While source-level splitting prevents spatial leakage, 1 validation and 1 test "
            "strip provide limited geographic diversity for statistically broad generalization metrics."
        ),
        "original_image_dimensions": source_dimensions_record,
        "processing_errors": processing_errors,
        "processing_configuration": {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "overlap_percentage": overlap_pct,
            "snap_edge_tiles": snap_edge_tiles,
            "output_format": output_format,
            "random_seed": int(split_cfg.get("random_seed", 42)),
            "use_memmap": use_memmap,
            "normalize_16bit_to_8bit": normalize_16bit,
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: Split source lunar images and generate memory-safe tiles."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to dataset_config.yaml relative to AI_Model/.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated tiles in processed/tiles/ before regenerating (never touches raw_images/).",
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
    print("Lunar Boulder Hazard Mapper - Stage 3 Dataset Preparation")
    print("=" * 64)

    try:
        summary = prepare_dataset(cfg, ai_model_root, clean_existing=args.clean)
    except Exception as exc:
        print(f"[ERROR] Dataset preparation failed: {exc}")
        return 1

    t_counts = summary["tile_counts"]
    print("\n" + "=" * 64)
    print("Dataset Preparation Complete:")
    print(f"  Valid Source Images   : {summary['source_image_counts']['valid_source_images']}")
    print(f"  Total Tiles Generated : {t_counts['total_tiles_generated']}")
    print(f"    - Train Tiles       : {t_counts['training_tiles']}")
    print(f"    - Validation Tiles  : {t_counts['validation_tiles']}")
    print(f"    - Test Tiles        : {t_counts['testing_tiles']}")
    print(f"  Blank Tiles Skipped   : {t_counts['skipped_blank_border_tiles']}")
    print(f"  Processing Errors     : {len(summary['processing_errors'])}")
    print("  Manifest Written To   : dataset/dataset_manifest.csv")
    print("  Summary Written To    : dataset/processed/reports/dataset_summary.json")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())