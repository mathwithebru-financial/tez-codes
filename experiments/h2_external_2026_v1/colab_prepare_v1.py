"""Fetch a pinned preparation bundle, stage it, and run no-fit T4 checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "fe594ff77b0310b53529702a20d09c8edcf3cf5f"
MANIFEST_SHA256 = "21abe6da20959ab958498b24bf0d4307595e4e251231e6005f9297c4ca8d4144"
PREFIX = ("https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
          + SOURCE_COMMIT + "/experiments/h2_external_2026_v1/")
NAMES = {"README.md", "protocol_plan_v1.json", "stage_external_plan_v1.py",
         "run_stl_controls_v1.py", "prepare_external_sequences_v1.py"}
PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
H2_PYTHON = Path("/content/h2_t4_runtime_v1/venv/bin/python")
CACHE = Path("/content/h2_external_2026_bundle_v1") / SOURCE_COMMIT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name):
    with urllib.request.urlopen(PREFIX + name, timeout=45) as response:
        return response.read()


def save_identical_or_new(path, data):
    if path.is_symlink():
        raise ValueError("Kod yolu sembolik bağlantı olamaz.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("Mevcut farklı kod dosyasının üzerine yazılmadı: " + path.name)
        return
    with path.open("xb") as handle:
        handle.write(data)
    if path.read_bytes() != data:
        raise OSError("İndirilen kodun dosya kontrolü başarısız.")


def checked_json(command):
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode:
        print(result.stdout[-5000:], flush=True)
        print(result.stderr[-9000:], flush=True)
        raise RuntimeError("Hazırlık kontrolü durdu; eski eğitim hücrelerini yeniden çalıştırmadan bu çıktıyı paylaş.")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout[-9000:], flush=True)
        raise RuntimeError("Hazırlık çıktısı doğrulanamadı; bu çıktıyı paylaş.")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1:] == ["--help"]:
            print(__doc__)
            return
        raise ValueError("Bu hazırlık başlatıcısının parametresi yoktur.")
    if platform.system() != "Linux" or not Path("/content").is_dir():
        raise RuntimeError("Bu başlatıcı Colab Linux ortamı içindir.")
    if not Path("/content/drive/MyDrive").is_dir() or not H2_PYTHON.is_file():
        raise RuntimeError("Drive bağlantısı ve ayrı H2 ortamı önce hazır olmalıdır.")

    print("Hazırlık kodları indiriliyor ve kimlikleri doğrulanıyor...", flush=True)
    manifest_bytes = fetch("bundle_manifest_v1.json")
    if digest(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("Kod paketi kimliği eşleşmedi; çalıştırılmadı.")
    manifest = json.loads(manifest_bytes)
    if set(manifest.get("files", {})) != NAMES:
        raise ValueError("Kod paketinde beklenmeyen dosya var.")
    payload = {"bundle_manifest_v1.json": manifest_bytes}
    for name in sorted(NAMES):
        data = fetch(name)
        expected = manifest["files"][name]
        if len(data) != expected["size_bytes"] or digest(data) != expected["sha256"]:
            raise ValueError("Kod dosyasının kimliği eşleşmedi: " + name)
        payload[name] = data
    for parent in (CACHE.parent, CACHE):
        if parent.is_symlink():
            raise ValueError("Geçici kod klasörü sembolik bağlantı olamaz.")
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        save_identical_or_new(CACHE / name, data)

    print("Ayrı çalışma klasörü ve eski girdi kimlikleri hazırlanıyor...", flush=True)
    staged = checked_json([sys.executable, "-B", str(CACHE / "stage_external_plan_v1.py"),
                           "--bundle-root", str(CACHE)])
    if staged.get("status") != "H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA" or staged.get("n_input") != 22:
        raise RuntimeError("Ayrı proje hazırlığı beklenen durumu üretmedi.")
    print("Altı modelin boyutları, başlangıçları ve sentetik ileri/geri hesabı kontrol ediliyor; eğitim yapılmıyor...", flush=True)
    prepared = checked_json([str(H2_PYTHON), "-B", str(PROJECT / "implementation/run_stl_controls_v1.py"),
                             "--project-root", str(PROJECT), "--prepare-only"])
    smoke = prepared.get("synthetic_checks", {})
    if (prepared.get("status") != "STL_PREPARATION_PASS_NO_TRAINING_NO_NEW_DATA"
            or smoke.get("models_checked") != 6 or smoke.get("optimizer_steps") != 0):
        raise RuntimeError("Model hazırlık denetimi beklenen sonucu üretmedi.")
    if (any(staged.get(key) is not False for key in
            ("training_started", "new_period_downloaded", "test_arrays_loaded"))
            or any(prepared.get(key) is not False for key in
                   ("training_started", "file_writes", "test_arrays_loaded", "new_period_accessed"))):
        raise RuntimeError("Hazırlık çıktısındaki işlem sınırları doğrulanamadı.")
    print(json.dumps({
        "status": "H2_EXTERNAL_PREPARATION_COMPLETE_NO_TRAINING",
        "project_root": str(PROJECT), "source_commit": SOURCE_COMMIT,
        "protocol_plan_sha256": staged["protocol_plan_sha256"],
        "n_verified_inputs": 22, "n_stl_fits_planned": 6,
        "synthetic_models_checked": 6,
        "parameters_per_stl_model": 27268,
        "parameters_per_stl_task_pair": 54536,
        "paired_branch_initialization_check": "PASS",
        "synthetic_forward_backward": "PASS",
        "stage_receipt_reused": staged["stage_receipt_reused"],
        "full_package_freeze_matches_old_runtime": staged["runtime_environment"]["freeze_matches_old_runtime"],
        "training_started": False, "test_arrays_loaded": False,
        "new_period_accessed": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
