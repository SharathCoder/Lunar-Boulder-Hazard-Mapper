import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_DIR = Path("dataset/raw_images")
CSV_FILE = Path("dataset/image_metadata.csv")

FIELDS = [
    "image_id", "filename", "source_name",
    "original_source_url", "download_date",
    "image_width", "image_height", "image_format",
    "file_size", "image_hash", "instrument_name",
    "mission_name", "product_id", "validation_status"
]


def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def generate_metadata():

    image_files = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ])

    rows = []

    for index, path in enumerate(image_files, start=1):

        print(f"\nChecking: {path.name}")

        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                width, height = img.size
                image_format = img.format

            row = {
                "image_id": f"IMG_{index:03d}",
                "filename": path.name,
                "source_name": "",
                "original_source_url": "",
                "download_date": "",
                "image_width": width,
                "image_height": height,
                "image_format": image_format,
                "file_size": path.stat().st_size,
                "image_hash": calculate_hash(path),
                "instrument_name": "",
                "mission_name": "",
                "product_id": "",
                "validation_status": "VALID"
            }

            print(f"[VALID] {width} x {height} | {image_format}")

        except Exception as error:

            print(f"[REJECTED] {error}")

            row = {
                "image_id": f"IMG_{index:03d}",
                "filename": path.name,
                "source_name": "",
                "original_source_url": "",
                "download_date": "",
                "image_width": "",
                "image_height": "",
                "image_format": "",
                "file_size": path.stat().st_size,
                "image_hash": "",
                "instrument_name": "",
                "mission_name": "",
                "product_id": "",
                "validation_status": "REJECTED"
            }

        rows.append(row)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===================================")
    print("METADATA GENERATION COMPLETED")
    print("===================================")
    print(f"Total images: {len(rows)}")
    print(f"Valid: {sum(r['validation_status'] == 'VALID' for r in rows)}")
    print(f"Rejected: {sum(r['validation_status'] == 'REJECTED' for r in rows)}")
    print(f"Saved to: {CSV_FILE}")


if __name__ == "__main__":
    generate_metadata()