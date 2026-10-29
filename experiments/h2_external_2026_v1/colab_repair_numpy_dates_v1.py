"""Stage a recorded NumPy date compatibility fix and prepare the same snapshot.

The original before-data lock and original implementations remain unchanged.
Only new versioned implementations and a truthful technical amendment are added.
No model training, model evaluation, package install or price request is made.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "1945fa41126b9a449d9e6d98be3f84ade5749716"
PREFIX = ("https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
          + SOURCE_COMMIT + "/experiments/h2_external_2026_v1/")
PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
H2_PYTHON = Path("/content/h2_t4_runtime_v1/venv/bin/python")
LOCK_SHA = "a155ad428996a6c1d40ca6a755212206ca9092a7b0fdc76200c3a1f063bdb274"
RAW_SHA = "a9c581a851d69498cd48cc5ee8bda9422c927b25c9a81ee1f154d9a74d95f889"
FILES = {
    "prepare_external_sequences_v2.py": {
        "sha256": "b9839295895729e7020503e05953f150067d138c5053331a6ce85078af97029e",
        "size_bytes": 19217,
    },
    "evaluate_external_h2_numpy_dates_v1.py": {
        "sha256": "443b72f4e3fef205eb00faa47173d0c99d4b4689fec295b11b96fea262d575f1",
        "size_bytes": 10603,
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def no_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Çalışma yolu sembolik bağlantı içeriyor; işlem durduruldu.")


def original_identities():
    no_symlinks(PROJECT)
    lock_path = PROJECT / "analysis_locked_v1.json"
    no_symlinks(lock_path)
    lock_bytes = lock_path.read_bytes()
    if digest(lock_bytes) != LOCK_SHA:
        raise RuntimeError("Özgün analiz kaydı değişmiş; işlem durduruldu.")
    lock = json.loads(lock_bytes)
    if lock.get("status") != "ANALYSIS_LOCKED_BEFORE_NEW_DATA":
        raise RuntimeError("Özgün analiz kaydı doğrulanamadı.")
    implementations = lock.get("implementation_sha256", {})
    if not implementations:
        raise RuntimeError("Özgün kod kimlikleri bulunamadı.")
    for name, expected in implementations.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("Özgün kod yolu geçersiz.")
        path = PROJECT / "implementation" / name
        no_symlinks(path)
        if digest(path.read_bytes()) != expected:
            raise RuntimeError("Özgün kod korunmamış: " + name)
    raw = PROJECT / "new_period/acquisition_v1/raw_prices.csv"
    no_symlinks(raw)
    if digest(raw.read_bytes()) != RAW_SHA:
        raise RuntimeError("Önceden indirilen veri kimliği eşleşmedi; yeniden indirilmedi.")
    return lock


def main():
    if sys.argv[1:]:
        raise ValueError("Bu başlatıcının parametresi yoktur.")
    if not H2_PYTHON.is_file():
        raise RuntimeError("Ayrı H2 Python ortamı bulunamadı; eğitim hücresini yeniden çalıştırma.")
    original_identities()
    implementation = PROJECT / "implementation"
    no_symlinks(implementation)
    print("Özgün analiz kaydı, eski kodlar ve indirilen veri eşleşiyor.", flush=True)
    print("Tarih türü düzeltmesinin ayrı sürümleri hazırlanıyor...", flush=True)
    payload = {}
    for name, identity in FILES.items():
        with urllib.request.urlopen(PREFIX + name, timeout=45) as response:
            data = response.read()
        if len(data) != identity["size_bytes"] or digest(data) != identity["sha256"]:
            raise RuntimeError("Düzeltilmiş kod kimliği eşleşmedi: " + name)
        path = implementation / name
        no_symlinks(path)
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise RuntimeError("Mevcut farklı sürüm korunarak işlem durduruldu: " + name)
        payload[name] = data
    for name, data in payload.items():
        path = implementation / name
        no_symlinks(path)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
        if path.read_bytes() != data:
            raise RuntimeError("Düzeltilmiş kodun okuma kontrolü başarısız: " + name)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    print("Teknik düzeltme kaydediliyor ve aynı verinin hazırlığı tamamlanıyor...", flush=True)
    result = subprocess.run([
        str(H2_PYTHON), "-B", str(implementation / "prepare_external_sequences_v2.py"),
        "--project-root", str(PROJECT),
        "--raw-file", str(PROJECT / "new_period/acquisition_v1/raw_prices.csv"),
        "--acquisition-record", str(PROJECT / "new_period/acquisition_v1/acquisition_record.json"),
    ], capture_output=True, text=True, env=env)
    if result.returncode:
        print(result.stdout[-6000:], flush=True)
        print(result.stderr[-10000:], flush=True)
        raise RuntimeError("Hazırlık durdu; eğitim veya indirmeyi yeniden başlatmadan bu çıktıyı paylaş.")
    summary = json.loads(result.stdout)
    path = PROJECT / "new_period/prepared_v1/prepared_manifest_v1.json"
    no_symlinks(path)
    data = path.read_bytes()
    manifest = json.loads(data)
    expected_status = "EXTERNAL_SEQUENCES_PREPARED_NO_MODEL_EVALUATION"
    if summary.get("status") != expected_status or manifest.get("status") != expected_status:
        raise RuntimeError("Hazırlık durumu doğrulanamadı.")
    if (any(manifest.get(key) is not False for key in
            ("models_evaluated", "predictions_produced", "old_test_arrays_loaded", "old_files_modified"))
            or manifest.get("scalers", {}).get("refitted") is not False
            or manifest.get("scalers", {}).get("state_unchanged_after_transform") is not True):
        raise RuntimeError("Hazırlık işlem sınırları doğrulanamadı.")
    original_identities()
    summary.update({"prepared_manifest_sha256": digest(data),
        **{key: manifest[key] for key in
           ("period", "calendar", "price_basis_overlap_audit", "historical_reproduction", "technical_amendment")},
        "scalers_refitted": False, "scaler_state_unchanged": True,
        "training_started": False, "prices_downloaded_again": False,
        "original_analysis_lock_unchanged": True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
