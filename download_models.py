"""
download_models.py — Helper untuk mengunduh file model dari Google Drive
=========================================================================
Jalankan SEKALI untuk mengunduh semua file yang dibutuhkan demo.

Prasyarat:
    pip install gdown

Cara pakai:
    python download_models.py --checkpoint <FILE_ID_PT> --bloom <FOLDER_ID>

Atau manual (tanpa script ini): ikuti panduan DEMO_GUIDE.md.
"""

import argparse
import os
import sys
import subprocess

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "main_bloom_mtl_v2")
BLOOM_DIR      = os.path.join(MODELS_DIR, "bloom-560m-local")
RESULTS_DIR    = os.path.join(MODELS_DIR, "results_v2")


def _check_gdown():
    try:
        import gdown
    except ImportError:
        print("gdown tidak terinstal. Menjalankan: pip install gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        print("✅ gdown terinstal.")


def download_file(file_id: str, output_path: str):
    """Unduh satu file dari Google Drive."""
    import gdown
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Mengunduh → {output_path}")
    gdown.download(url, output_path, quiet=False, fuzzy=True)


def download_folder(folder_id: str, output_dir: str):
    """Unduh seluruh folder dari Google Drive."""
    import gdown
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"  Mengunduh folder → {output_dir}")
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)


def main():
    parser = argparse.ArgumentParser(
        description="Unduh file model dari Google Drive untuk demo"
    )
    parser.add_argument(
        "--checkpoint-id",
        help="Google Drive File ID untuk best_model_hm.pt "
             "(misal: 1AbCdEfGhIjKlMn...)",
    )
    parser.add_argument(
        "--bloom-folder-id",
        help="Google Drive Folder ID untuk bloom-560m-local/",
    )
    parser.add_argument(
        "--threshold-id",
        help="Google Drive File ID untuk threshold_info_v2.json (opsional)",
    )
    args = parser.parse_args()

    _check_gdown()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BLOOM_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print(" Download Helper — Hate Speech Demo Models")
    print("="*60)

    if args.checkpoint_id:
        out = os.path.join(CHECKPOINT_DIR, "best_model_hm.pt")
        if os.path.exists(out):
            print(f"  ℹ️  {out} sudah ada, skip.")
        else:
            download_file(args.checkpoint_id, out)
    else:
        print(
            "  ⚠️  --checkpoint-id tidak diberikan.\n"
            "     Unduh best_model_hm.pt secara manual dari Google Drive.\n"
            f"     Simpan di: {CHECKPOINT_DIR}/best_model_hm.pt"
        )

    if args.bloom_folder_id:
        if os.path.exists(os.path.join(BLOOM_DIR, "tokenizer.json")):
            print(f"  ℹ️  bloom-560m-local/ sudah ada, skip.")
        else:
            download_folder(args.bloom_folder_id, BLOOM_DIR)
    else:
        print(
            "  ⚠️  --bloom-folder-id tidak diberikan.\n"
            "     Unduh folder bloom-560m-local/ secara manual.\n"
            f"     Simpan di: {BLOOM_DIR}/"
        )

    if args.threshold_id:
        out = os.path.join(MODELS_DIR, "threshold_info_v2.json")
        if os.path.exists(out):
            print(f"  ℹ️  threshold_info_v2.json sudah ada, skip.")
        else:
            download_file(args.threshold_id, out)
    else:
        print(
            "  ⚠️  --threshold-id tidak diberikan.\n"
            "     Nilai threshold default (0.4429) akan digunakan otomatis."
        )

    print("\n" + "="*60)
    print(" Verifikasi file:")
    checks = [
        (os.path.join(CHECKPOINT_DIR, "best_model_hm.pt"), "best_model_hm.pt (~4.2 GB)"),
        (os.path.join(BLOOM_DIR, "tokenizer.json"),     "bloom tokenizer.json"),
        (os.path.join(BLOOM_DIR, "config.json"),         "bloom config.json"),
    ]
    all_ok = True
    for path, label in checks:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"  ✅ {label}  ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {label}  → TIDAK DITEMUKAN di {path}")
            all_ok = False

    if all_ok:
        print("\n✅ Semua file tersedia. Jalankan: python app.py")
    else:
        print(
            "\n⚠️  Ada file yang belum tersedia.\n"
            "   Unduh manual sesuai panduan DEMO_GUIDE.md."
        )
    print("="*60)


if __name__ == "__main__":
    main()
