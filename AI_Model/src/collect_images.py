"""
Stage 2: Lunar Surface Image Collection, Validation & Metadata Logging Script.

Capabilities:
1. Downloads genuine lunar surface imagery from verified direct URLs listed in
   a CSV manifest (or passed via CLI) while preserving original resolution and format.
2. Also supports ingesting and validating genuine lunar images downloaded manually
   from LROC / NASA PDS portals (--local-dir).
3. Streams downloads safely to temporary '.part' files to recover cleanly from
   interrupted connections and prevents partial file corruption.
4. Validates every file using Pillow and OpenCV (verifying readability, image format,
   and minimum dimensions without rejecting low-contrast lunar lighting).
5. Computes SHA-256 hashes to prevent duplicate images across runs.
6. Never overwrites existing files unless explicitly authorized (--overwrite or config).
7. Appends structured, factual records to dataset/image_metadata.csv without
   fabricating missing metadata fields.

Usage (from AI_Model/ directory):
    python src/collect_images.py --max-images 5
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
import yaml

# Allow large scientific LROC NAC images without triggering Pillow decompression bomb errors
Image.MAX_IMAGE_PIXELS = None

METADATA_HEADERS = [
    "image_id",
    "filename",
    "source_name",
    "original_source_url",
    "download_date",
    "image_width",
    "image_height",
    "image_format",
    "file_size",
    "image_hash",
    "instrument_name",
    "mission_name",
    "product_id",
    "validation_status",
]


def load_config(config_path: Path) -> dict:
    """Load YAML configuration settings."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in chunks for duplicate detection."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def ensure_metadata_csv(csv_path: Path) -> None:
    """Create metadata CSV with headers if it does not already exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS)
            writer.writeheader()


def load_existing_records(csv_path: Path) -> Tuple[Set[str], Set[str], int]:
    """
    Read existing metadata CSV to retrieve known SHA-256 hashes, known source URLs,
    and the highest existing numeric image index.
    """
    known_hashes: Set[str] = set()
    known_urls: Set[str] = set()
    max_index = 0

    if not csv_path.exists():
        return known_hashes, known_urls, max_index

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_hash = (row.get("image_hash") or "").strip()
            if img_hash:
                known_hashes.add(img_hash)

            src_url = (row.get("original_source_url") or "").strip()
            if src_url and row.get("validation_status") == "VALID":
                known_urls.add(src_url)

            img_id = (row.get("image_id") or "").strip()
            if "_" in img_id:
                suffix = img_id.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))

    return known_hashes, known_urls, max_index


def get_next_sequence_index(output_dir: Path, prefix: str, csv_max_index: int) -> int:
    """Determine the next available sequential index across both disk and CSV."""
    disk_max = 0
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file() and item.stem.startswith(f"{prefix}_"):
                suffix = item.stem.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    disk_max = max(disk_max, int(suffix))
    return max(csv_max_index, disk_max) + 1


def append_metadata_row(csv_path: Path, row: Dict[str, str]) -> None:
    """Append a single validated or rejected metadata record to CSV."""
    ensure_metadata_csv(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_HEADERS)
        sanitized_row = {key: row.get(key, "") for key in METADATA_HEADERS}
        writer.writerow(sanitized_row)


def is_valid_url(url: str) -> bool:
    """Validate basic HTTP/HTTPS URL structure."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def download_file_with_progress(
    url: str,
    temp_path: Path,
    timeout: int,
    chunk_size: int,
    max_retries: int,
    user_agent: str,
) -> Tuple[bool, str]:
    """
    Download a file from a direct URL into a temporary .part file with progress bar
    and retry handling for interrupted connections.
    """
    headers = {"User-Agent": user_agent}

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=timeout
            ) as response:
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    return (
                        False,
                        "URL returned an HTML webpage instead of a direct image file.",
                    )

                total_bytes = int(response.headers.get("Content-Length", 0))
                downloaded_bytes = 0

                with temp_path.open("wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        file_handle.write(chunk)
                        downloaded_bytes += len(chunk)

                        if total_bytes > 0:
                            percent = (downloaded_bytes / total_bytes) * 100
                            mb_done = downloaded_bytes / (1024 * 1024)
                            mb_total = total_bytes / (1024 * 1024)
                            print(
                                f"\r    Downloading: {percent:6.2f}% "
                                f"({mb_done:.2f}/{mb_total:.2f} MB)",
                                end="",
                                flush=True,
                            )
                        else:
                            mb_done = downloaded_bytes / (1024 * 1024)
                            print(
                                f"\r    Downloading: {mb_done:.2f} MB received",
                                end="",
                                flush=True,
                            )

                print()  # Newline after progress output
                if downloaded_bytes == 0:
                    return False, "Downloaded file is empty (0 bytes)."
                return True, "OK"

        except requests.exceptions.Timeout:
            print()
            msg = f"Connection timed out after {timeout}s (attempt {attempt}/{max_retries})."
        except requests.exceptions.HTTPError as exc:
            print()
            status = exc.response.status_code if exc.response is not None else "Unknown"
            # Do not retry permanent client errors (e.g., 404 Not Found, 403 Forbidden)
            if isinstance(status, int) and 400 <= status < 500:
                return False, f"HTTP {status} error for URL."
            msg = f"HTTP {status} error (attempt {attempt}/{max_retries})."
        except requests.exceptions.RequestException as exc:
            print()
            msg = f"Network error: {exc} (attempt {attempt}/{max_retries})."

        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        if attempt < max_retries:
            time.sleep(2 * attempt)
        else:
            return False, msg

    return False, "Download failed after maximum retries."


def validate_image_file(
    file_path: Path,
    min_width: int,
    min_height: int,
    supported_formats: List[str],
) -> Tuple[bool, str, dict]:
    """
    Validate image integrity, format, and dimensions using both Pillow and OpenCV.
    Does NOT reject low-contrast or shadowed lunar images.
    """
    info = {
        "width": "",
        "height": "",
        "format": "",
        "file_size": str(file_path.stat().st_size) if file_path.exists() else "",
    }

    if not file_path.exists() or file_path.stat().st_size == 0:
        return False, "REJECTED_EMPTY_FILE", info

    # Step 1: Verify header & structure with Pillow
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            width, height = img.size
            pil_format = (img.format or "").lower()
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "REJECTED_CORRUPTED_OR_UNREADABLE", info

    normalized_supported = {fmt.lower().lstrip(".") for fmt in supported_formats}
    # Pillow reports JPEG as 'jpeg' and TIFF as 'tiff'
    if pil_format not in normalized_supported:
        return False, f"REJECTED_UNSUPPORTED_FORMAT ({pil_format})", info

    info["width"] = str(width)
    info["height"] = str(height)
    info["format"] = pil_format.upper()

    # Step 2: Check minimum spatial dimensions
    if width < min_width or height < min_height:
        return (
            False,
            f"REJECTED_TOO_SMALL ({width}x{height} < {min_width}x{min_height})",
            info,
        )

    # Step 3: Confirm OpenCV can decode pixel data cleanly without altering resolution
    # Use np.fromfile + cv2.imdecode for full Windows Unicode path support
    try:
        raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            return False, "REJECTED_OPENCV_DECODE_FAILED", info
    except Exception:
        return False, "REJECTED_OPENCV_EXCEPTION", info

    return True, "VALID", info


def normalize_extension(pil_format: str) -> str:
    """Map Pillow format string to a clean lowercase file extension."""
    fmt = pil_format.lower()
    if fmt == "jpeg":
        return "jpg"
    if fmt == "tiff":
        return "tif"
    return fmt


def read_url_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Read URL entries and optional authentic metadata from CSV manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {manifest_path}\n"
            "Add verified LROC/PDS direct image URLs to this file first."
        )

    entries: List[Dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url or url.startswith("#"):
                continue
            entries.append(
                {
                    "url": url,
                    "source_name": (row.get("source_name") or "").strip(),
                    "instrument_name": (row.get("instrument_name") or "").strip(),
                    "mission_name": (row.get("mission_name") or "").strip(),
                    "product_id": (row.get("product_id") or "").strip(),
                }
            )
    return entries


def read_local_directory(local_dir: Path, supported_formats: List[str]) -> List[Dict[str, str]]:
    """Collect candidate image paths from a local staging directory."""
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    allowed_exts = {f".{fmt.lower().lstrip('.')}" for fmt in supported_formats}
    entries: List[Dict[str, str]] = []
    for path in sorted(local_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed_exts:
            entries.append(
                {
                    "local_path": str(path),
                    "url": "",
                    "source_name": "Local Verified Import",
                    "instrument_name": "",
                    "mission_name": "",
                    "product_id": path.stem,
                }
            )
    return entries


def process_candidates(
    candidates: List[Dict[str, str]],
    output_dir: Path,
    metadata_csv: Path,
    prefix: str,
    max_images: int,
    min_width: int,
    min_height: int,
    supported_formats: List[str],
    deduplicate: bool,
    overwrite: bool,
    timeout: int,
    chunk_size: int,
    max_retries: int,
    user_agent: str,
) -> dict:
    """Download/ingest, validate, deduplicate, save, and log metadata for lunar images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv(metadata_csv)

    known_hashes, known_urls, csv_max_index = load_existing_records(metadata_csv)
    next_idx = get_next_sequence_index(output_dir, prefix, csv_max_index)

    stats = {
        "processed": 0,
        "saved": 0,
        "duplicates": 0,
        "rejected": 0,
        "failed_downloads": 0,
    }

    temp_dir = output_dir / ".tmp_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, item in enumerate(candidates, start=1):
            if stats["saved"] >= max_images:
                print(f"\n[INFO] Reached target max_images limit ({max_images}). Stopping.")
                break

            stats["processed"] += 1
            url = item.get("url", "").strip()
            local_path_str = item.get("local_path", "").strip()
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            print(f"\n[{idx}/{len(candidates)}] Processing: {url or local_path_str}")

            # 1. URL validation & URL-level duplicate check
            if url:
                if not is_valid_url(url):
                    print("    [REJECTED] Invalid URL format.")
                    stats["rejected"] += 1
                    append_metadata_row(
                        metadata_csv,
                        {
                            "source_name": item.get("source_name", ""),
                            "original_source_url": url,
                            "download_date": timestamp,
                            "instrument_name": item.get("instrument_name", ""),
                            "mission_name": item.get("mission_name", ""),
                            "product_id": item.get("product_id", ""),
                            "validation_status": "REJECTED_INVALID_URL",
                        },
                    )
                    continue

                if deduplicate and not overwrite and url in known_urls:
                    print("    [SKIPPED] Duplicate source URL already recorded.")
                    stats["duplicates"] += 1
                    continue

                temp_file = temp_dir / f"download_{int(time.time() * 1000)}.part"
                ok, download_msg = download_file_with_progress(
                    url=url,
                    temp_path=temp_file,
                    timeout=timeout,
                    chunk_size=chunk_size,
                    max_retries=max_retries,
                    user_agent=user_agent,
                )
                if not ok:
                    print(f"    [FAILED] {download_msg}")
                    stats["failed_downloads"] += 1
                    temp_file.unlink(missing_ok=True)
                    append_metadata_row(
                        metadata_csv,
                        {
                            "source_name": item.get("source_name", ""),
                            "original_source_url": url,
                            "download_date": timestamp,
                            "instrument_name": item.get("instrument_name", ""),
                            "mission_name": item.get("mission_name", ""),
                            "product_id": item.get("product_id", ""),
                            "validation_status": f"DOWNLOAD_FAILED: {download_msg}",
                        },
                    )
                    continue
            else:
                src_local = Path(local_path_str)
                temp_file = temp_dir / f"local_{int(time.time() * 1000)}_{src_local.name}"
                shutil.copy2(src_local, temp_file)

            # 2. Image integrity & dimension validation
            is_valid, status_msg, img_info = validate_image_file(
                file_path=temp_file,
                min_width=min_width,
                min_height=min_height,
                supported_formats=supported_formats,
            )

            file_hash = compute_sha256(temp_file) if temp_file.exists() else ""

            if not is_valid:
                print(f"    [REJECTED] {status_msg}")
                stats["rejected"] += 1
                temp_file.unlink(missing_ok=True)
                append_metadata_row(
                    metadata_csv,
                    {
                        "source_name": item.get("source_name", ""),
                        "original_source_url": url,
                        "download_date": timestamp,
                        "image_width": img_info["width"],
                        "image_height": img_info["height"],
                        "image_format": img_info["format"],
                        "file_size": img_info["file_size"],
                        "image_hash": file_hash,
                        "instrument_name": item.get("instrument_name", ""),
                        "mission_name": item.get("mission_name", ""),
                        "product_id": item.get("product_id", ""),
                        "validation_status": status_msg,
                    },
                )
                continue

            # 3. Content-hash duplicate detection
            if deduplicate and file_hash in known_hashes:
                print(f"    [DUPLICATE] Identical image hash ({file_hash[:12]}...) already exists.")
                stats["duplicates"] += 1
                temp_file.unlink(missing_ok=True)
                continue

            # 4. Assign unique sequential filename & protect existing files
            ext = normalize_extension(img_info["format"])
            image_id = f"{prefix}_{next_idx:03d}"
            final_filename = f"{image_id}.{ext}"
            final_path = output_dir / final_filename

            if final_path.exists() and not overwrite:
                print(
                    f"    [PROTECTED] Target file {final_filename} already exists and "
                    "overwrite is disabled. Skipping."
                )
                temp_file.unlink(missing_ok=True)
                continue

            shutil.move(str(temp_file), str(final_path))
            known_hashes.add(file_hash)
            if url:
                known_urls.add(url)
            next_idx += 1
            stats["saved"] += 1

            append_metadata_row(
                metadata_csv,
                {
                    "image_id": image_id,
                    "filename": final_filename,
                    "source_name": item.get("source_name", ""),
                    "original_source_url": url,
                    "download_date": timestamp,
                    "image_width": img_info["width"],
                    "image_height": img_info["height"],
                    "image_format": img_info["format"],
                    "file_size": img_info["file_size"],
                    "image_hash": file_hash,
                    "instrument_name": item.get("instrument_name", ""),
                    "mission_name": item.get("mission_name", ""),
                    "product_id": item.get("product_id", ""),
                    "validation_status": "VALID",
                },
            )

            print(
                f"    [SAVED] {final_filename} | "
                f"{img_info['width']}x{img_info['height']} {img_info['format']} | "
                f"SHA256: {file_hash[:12]}..."
            )

    finally:
        # Clean up temporary staging folder if empty
        if temp_dir.exists():
            for leftover in temp_dir.iterdir():
                leftover.unlink(missing_ok=True)
            temp_dir.rmdir()

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line options overriding YAML defaults if specified."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Collect and validate genuine lunar surface images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/collection_config.yaml",
        help="Path to collection YAML config relative to AI_Model/.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional override path to source_urls.csv manifest.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="Direct verified image URL (can be passed multiple times).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Optional directory of manually downloaded LROC/PDS images to validate & import.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of valid images to collect in this run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files (disabled by default).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for lunar surface image collection."""
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

    col_cfg = cfg.get("collection", {})
    net_cfg = cfg.get("network", {})
    val_cfg = cfg.get("validation", {})

    max_images = args.max_images if args.max_images is not None else col_cfg.get("max_images", 20)
    overwrite = args.overwrite or bool(col_cfg.get("overwrite_existing", False))
    output_dir = ai_model_root / col_cfg.get("output_dir", "dataset/raw_images")
    metadata_csv = ai_model_root / col_cfg.get("metadata_csv", "dataset/image_metadata.csv")
    prefix = col_cfg.get("filename_prefix", "lunar_image")

    min_width = int(val_cfg.get("min_width", 256))
    min_height = int(val_cfg.get("min_height", 256))
    supported_formats = val_cfg.get("supported_formats", ["jpg", "jpeg", "png", "tif", "tiff"])
    deduplicate = bool(val_cfg.get("enable_duplicate_detection", True))

    timeout = int(net_cfg.get("timeout_seconds", 30))
    chunk_size = int(net_cfg.get("chunk_size_bytes", 8192))
    max_retries = int(net_cfg.get("max_retries", 3))
    user_agent = net_cfg.get("user_agent", "LunarBoulderHazardMapper-ResearchClient/1.0")

    candidates: List[Dict[str, str]] = []

    if args.url:
        for single_url in args.url:
            candidates.append(
                {
                    "url": single_url.strip(),
                    "source_name": "",
                    "instrument_name": "",
                    "mission_name": "",
                    "product_id": "",
                }
            )
    elif args.local_dir:
        local_path = Path(args.local_dir)
        if not local_path.is_absolute():
            local_path = ai_model_root / local_path
        try:
            candidates = read_local_directory(local_path, supported_formats)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1
    else:
        manifest_rel = args.manifest or col_cfg.get("source_manifest", "config/source_urls.csv")
        manifest_path = Path(manifest_rel)
        if not manifest_path.is_absolute():
            manifest_path = ai_model_root / manifest_path
        try:
            candidates = read_url_manifest(manifest_path)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1

    if not candidates:
        print("=" * 60)
        print("[INFO] No candidate image URLs or local files found to process.")
        print(f"       Add verified LROC/PDS direct image URLs to:")
        print(f"       {ai_model_root / col_cfg.get('source_manifest', 'config/source_urls.csv')}")
        print("       Or pass --url <DIRECT_IMAGE_URL> or --local-dir <FOLDER_PATH>.")
        print("=" * 60)
        return 0

    print("=" * 60)
    print("Lunar Boulder Hazard Mapper - Stage 2 Image Collection")
    print("=" * 60)
    print(f"Target Output Dir : {output_dir}")
    print(f"Metadata CSV      : {metadata_csv}")
    print(f"Max Images Limit  : {max_images}")
    print(f"Min Dimensions    : {min_width}x{min_height} px")
    print(f"Duplicate Check   : {'Enabled' if deduplicate else 'Disabled'}")
    print(f"Overwrite Mode    : {'Enabled' if overwrite else 'Disabled (Safe)'}")
    print("=" * 60)

    stats = process_candidates(
        candidates=candidates,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
        prefix=prefix,
        max_images=max_images,
        min_width=min_width,
        min_height=min_height,
        supported_formats=supported_formats,
        deduplicate=deduplicate,
        overwrite=overwrite,
        timeout=timeout,
        chunk_size=chunk_size,
        max_retries=max_retries,
        user_agent=user_agent,
    )

    print("\n" + "=" * 60)
    print("Collection Summary:")
    print(f"  Candidates Checked : {stats['processed']}")
    print(f"  Valid Images Saved : {stats['saved']}")
    print(f"  Duplicates Skipped : {stats['duplicates']}")
    print(f"  Rejected Images    : {stats['rejected']}")
    print(f"  Download Failures  : {stats['failed_downloads']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())