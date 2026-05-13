"""Upload Arabic handwriting datasets to Kaggle using the Python API."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username", help="Kaggle username")
    parser.add_argument("--skip-data", action="store_true", help="Skip re-uploading processed data")
    parser.add_argument("--ml-dir", help="Path to ml/ directory", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    ml_dir = Path(args.ml_dir) if args.ml_dir else script_dir.parent
    staging = Path.home() / "kaggle_staging"
    username = args.username

    try:
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Kaggle auth failed: {e}")
        print("Make sure ~/.kaggle/kaggle.json exists.")
        sys.exit(1)

    # Find best checkpoint (highest step number)
    ckpt_dir = ml_dir / "outputs" / "arabic_v1" / "checkpoints"
    ckpts = [f for f in ckpt_dir.glob("model-*") if f.is_file()]
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)
    best = max(ckpts, key=lambda f: int(f.name.split("-")[1]))
    step = best.name.split("-")[1]
    print(f"Best checkpoint: {best.name} (step {step})")

    # Always include model-6389 as the warm-start reference
    warm_start = ckpt_dir / "model-6389"

    # Clean and build staging
    if staging.exists():
        shutil.rmtree(staging)

    def write_meta(folder: Path, title: str, ds_id: str):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dataset-metadata.json").write_text(
            json.dumps({"title": title, "id": ds_id, "licenses": [{"name": "CC0-1.0"}]})
        )

    def upload(folder: Path, title: str, ds_id: str, message: str):
        write_meta(folder, title, ds_id)
        print(f"\n--- Uploading {ds_id} ---")
        try:
            api.dataset_create_version(str(folder), message, quiet=False, dir_mode="zip")
            print(f"    Done.")
        except Exception as e:
            print(f"    ERROR uploading {ds_id}: {e}")

    # Dataset 1: processed data
    if not args.skip_data:
        data_staging = staging / "data"
        src = ml_dir / "data" / "processed"
        dst = data_staging / "processed"
        shutil.copytree(str(src), str(dst))
        upload(data_staging, "Arabic Handwriting Processed Data",
               f"{username}/arabic-handwriting-data", "update processed data")
    else:
        print("--- Skipping data (--skip-data) ---")

    # Dataset 2: checkpoint
    ckpt_staging = staging / "checkpoints" / "files"
    ckpt_staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, ckpt_staging / best.name)
    if warm_start.exists() and warm_start != best:
        shutil.copy2(warm_start, ckpt_staging / warm_start.name)
        print(f"Also including warm-start checkpoint: {warm_start.name}")
    config_src = ml_dir / "outputs" / "arabic_v1" / "config" / "config.yaml"
    shutil.copy2(config_src, ckpt_staging / "config.yaml")
    upload(staging / "checkpoints", "Arabic Handwriting Checkpoints",
           f"{username}/arabic-handwriting-checkpoints", f"best checkpoint model-{step}")

    # Dataset 3: code
    code_staging = staging / "code" / "files"
    code_staging.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(ml_dir / "src"), str(code_staging / "src"))
    shutil.copytree(str(ml_dir / "config"), str(code_staging / "config"))
    upload(staging / "code", "Arabic Handwriting Code",
           f"{username}/arabic-handwriting-code", "update src")

    print(f"\nAll done. model-{step} is now in the Kaggle checkpoint dataset.")
    print("Next: open Kaggle notebook, update patience to [1000,700,500], run all cells.")

if __name__ == "__main__":
    main()
