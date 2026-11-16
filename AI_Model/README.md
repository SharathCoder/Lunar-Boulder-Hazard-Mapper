---

## Stage 3: Lunar Image Dataset Preparation & Tiling

### 1. Dataset Preparation Workflow
Raw LROC NAC Pyramid TIFF strips have very large dimensions (e.g., `5064 × 52224` pixels) that cannot be fed directly into a YOLO network. Stage 3 transforms them into training-ready tiles in four non-destructive steps:
1. **Read-Only Source Validation (`src/validate_dataset.py`):** Verifies TIFF headers, dimensions, file sizes, and SHA-256 hashes against Stage 2 `image_metadata.csv`. Original files in `dataset/raw_images/` are never modified, resized, or overwritten.
2. **Duplicate Detection:** Computes SHA-256 checksums across all raw files so exact duplicates are flagged in `validation_report.csv` without deleting source files.
3. **Leak-Free Source Splitting:** Splits the 8 original source images first (`6 train / 1 val / 1 test` using `random_seed: 42`) so all tiles from any single LROC strip stay exclusively inside one split. *(Note: Because only 8 source strips are currently available, validation and test metrics reflect 1 strip each and carry limited geographic sample diversity.)*
4. **Memory-Efficient Windowed Tiling (`src/prepare_dataset.py`):** Uses `tifffile` memory-mapping (`memmap`) / Zarr slicing on full-resolution Page 0 to extract `1024 × 1024` tiles with `10%` overlap (`stride = 922 px`). Uneven edges snap cleanly inward so border tiles preserve uniform `1024 × 1024` geometry.

### 2. Configuration (`config/dataset_config.yaml`)
- `tiling.tile_width` / `tile_height`: Tile size in pixels (default `1024 × 1024`).
- `tiling.overlap_percentage`: Adjacent tile overlap (default `10.0%`) to avoid cutting boulders in half at tile borders.
- `splitting`: Configures `train_count: 6`, `val_count: 1`, `test_count: 1`, and `random_seed: 42`.
- `memory.use_memmap`: Enables low-RAM memory-mapped access for multi-thousand-pixel strips.

### 3. How to Execute Stage 3 Scripts (Windows PowerShell)
Run from inside `AI_Model/` with `.venv` activated:
```powershell
# Step 1: Run non-destructive source image validation
python src/validate_dataset.py

# Step 2: Split source images and generate 1024x1024 tiles + manifest + reports
python src/prepare_dataset.py