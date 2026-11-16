"""
Environment and Hardware Verification Script for Lunar Boulder Hazard Mapper (AI_Model).

Verifies:
1. Python runtime version compatibility.
2. Required project directory structure using cross-platform relative paths.
3. Core library installations (PyTorch, Torchvision, OpenCV, Ultralytics YOLO, NumPy, Pillow, PyYAML).
4. CUDA / GPU availability with automatic fallback guidance for CPU-only execution.

Usage (run from inside the AI_Model directory):
    python src/verify_env.py
"""

from pathlib import Path
import platform
import sys


def verify_python_version(min_major: int = 3, min_minor: int = 9) -> bool:
    """Verify that the active Python interpreter meets minimum version requirements."""
    version_info = sys.version_info
    is_valid = (version_info.major, version_info.minor) >= (min_major, min_minor)
    status = "OK" if is_valid else "WARNING"
    print(
        f"[{status}] Python Version : {platform.python_version()} "
        f"({platform.system()} {platform.release()})"
    )
    if not is_valid:
        print(f"       -> Recommended Python >= {min_major}.{min_minor} for Ultralytics YOLO.")
    return is_valid


def verify_directory_structure(base_dir: Path) -> bool:
    """Verify that all required AI_Model subdirectories exist using relative paths."""
    required_dirs = ["config", "dataset", "outputs", "src", "weights"]
    all_present = True

    print("\n--- Directory Structure Check ---")
    print(f"Base AI_Model Path : {base_dir.resolve()}")

    for folder_name in required_dirs:
        folder_path = base_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            print(f"[OK] Found folder  : {folder_name}/")
        else:
            print(f"[MISSING] Folder   : {folder_name}/ (Creating automatically...)")
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"[CREATED] Folder   : {folder_name}/")
            except OSError as exc:
                print(f"[ERROR] Could not create {folder_name}/: {exc}")
                all_present = False

    return all_present


def verify_dependencies() -> dict:
    """Check if required packages are installed and importable."""
    print("\n--- Dependency Verification ---")
    packages = {
        "torch": "PyTorch",
        "torchvision": "Torchvision",
        "cv2": "OpenCV (opencv-python)",
        "ultralytics": "Ultralytics YOLO",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "yaml": "PyYAML",
    }

    imported_modules = {}
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "installed (version attr N/A)")
            print(f"[OK] {display_name:<24} : {version}")
            imported_modules[module_name] = module
        except ImportError as exc:
            print(f"[ERROR] {display_name:<21} : NOT INSTALLED ({exc})")
            imported_modules[module_name] = None

    return imported_modules


def verify_compute_device(torch_module) -> str:
    """Check CUDA GPU availability and configure a safe fallback device string."""
    print("\n--- Hardware / Compute Device Check ---")
    if torch_module is None:
        print("[ERROR] Cannot check compute device because PyTorch is not installed.")
        return "unavailable"

    if torch_module.cuda.is_available():
        device_count = torch_module.cuda.device_count()
        gpu_name = torch_module.cuda.get_device_name(0)
        cuda_version = getattr(torch_module.version, "cuda", "Unknown")
        print("[OK] CUDA GPU Available       : True")
        print(f"     Device Count             : {device_count}")
        print(f"     Primary GPU (cuda:0)     : {gpu_name}")
        print(f"     PyTorch CUDA Version     : {cuda_version}")
        return "cuda:0"

    print("[INFO] CUDA GPU Available     : False")
    print("[OK] Fallback Compute Device  : cpu")
    print(
        "       -> No NVIDIA CUDA GPU detected. Training and inference will run on CPU.\n"
        "       -> Note: CPU inference works out of the box. For faster model training in\n"
        "          Stage 3, you may use a smaller backbone (e.g., yolov8n.pt) or cloud GPU."
    )
    return "cpu"


def verify_yolo_runtime(ultralytics_module) -> bool:
    """Confirm Ultralytics YOLO class is accessible without downloading weights yet."""
    print("\n--- Ultralytics YOLO Runtime Check ---")
    if ultralytics_module is None:
        print("[ERROR] Ultralytics is not installed.")
        return False

    try:
        yolo_cls = getattr(ultralytics_module, "YOLO")
        print(f"[OK] Ultralytics YOLO class   : {yolo_cls.__name__} ready")
        return True
    except AttributeError as exc:
        print(f"[ERROR] Failed to access YOLO class: {exc}")
        return False


def main() -> int:
    """Run all Stage 1 environment checks and return an exit status code."""
    print("=" * 60)
    print("Lunar Boulder Hazard Mapper - AI_Model Environment Check")
    print("=" * 60)

    # Resolve AI_Model root dynamically relative to AI_Model/src/verify_env.py
    ai_model_root = Path(__file__).resolve().parent.parent

    py_ok = verify_python_version()
    dirs_ok = verify_directory_structure(ai_model_root)
    modules = verify_dependencies()
    device = verify_compute_device(modules.get("torch"))
    yolo_ok = verify_yolo_runtime(modules.get("ultralytics"))

    all_packages_ok = all(mod is not None for mod in modules.values())
    overall_ready = py_ok and dirs_ok and all_packages_ok and yolo_ok and device != "unavailable"

    print("\n" + "=" * 60)
    if overall_ready:
        print(f"STATUS: READY FOR STAGE 2 (Active compute device: '{device}')")
        print("=" * 60)
        return 0

    print("STATUS: SETUP INCOMPLETE - Please fix the [ERROR] items above.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())