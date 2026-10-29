"""Stage three reviewed files and lock the analysis; no fitting or price access."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "4d1e221fd5433f9a1bd9a0beb2448f9e0c6195aa"
PREFIX = ("https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
          + SOURCE_COMMIT + "/experiments/h2_external_2026_v1/")
PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
H2_PYTHON = Path("/content/h2_t4_runtime_v1/venv/bin/python")
STL_SHA256 = "c1c546a9233f0ce7e77dffe33850b58ed5028e4d695597871f3c216d8f3c5f44"
PLAN_SHA256 = "0073b3bcc80a342d25f1034fef66d5eb353a9e4e5c14ae3719f57e98de81a659"
FILES = {
    "acquire_external_prices_v1.py": {
        "sha256": "258fc4ed7311a4f4936e0bca3b8102c470761f5ef3bdf532491d364c4e2d4342",
        "size_bytes": 35895,
    },
    "evaluate_external_h2_v1.py": {
        "sha256": "773f01d1c9d5cd8b535d2c8640c86b0b039ab9391d6265fb734543861d953a0f",
        "size_bytes": 49241,
    },
    "lock_external_analysis_v1.py": {
        "sha256": "1e2d4b827ad3118288756f88f5be53a036429eb68afd63b63d1c2edc8d0f4ea1",
        "size_bytes": 42710,
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reject_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Çalışma yolu sembolik bağlantı içeriyor; işlem durduruldu.")


def main():
    if sys.argv[1:]:
        raise ValueError("Bu başlatıcının parametresi yoktur.")
    reject_symlinks(PROJECT)
    if not PROJECT.is_dir() or not H2_PYTHON.is_file():
        raise RuntimeError("Drive bağlantısı ve ayrı H2 Python ortamı hazır olmalıdır.")
    plan_path = PROJECT / "protocol_plan_v1.json"
    frozen_path = PROJECT / "runs/stl_run_v1/selected_stl_checkpoints_frozen.json"
    for path in (plan_path, frozen_path):
        reject_symlinks(path)
    if digest(plan_path.read_bytes()) != PLAN_SHA256:
        raise RuntimeError("Protokol kimliği eşleşmedi; işlem durduruldu.")
    frozen_bytes = frozen_path.read_bytes()
    if digest(frozen_bytes) != STL_SHA256:
        raise RuntimeError("Altı modelin tamamlanma kaydı paylaşılan sonuçla eşleşmedi.")
    frozen = json.loads(frozen_bytes)
    if (frozen.get("status") != "STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA"
            or frozen.get("n_actual_fits") != 6
            or frozen.get("plan_sha256") != PLAN_SHA256
            or frozen.get("test_arrays_loaded") is not False
            or frozen.get("new_period_accessed") is not False):
        raise RuntimeError("Altı modelin tamamlanma durumu doğrulanamadı.")
    if not (PROJECT / "analysis_locked_v1.json").exists() and (PROJECT / "new_period").exists():
        raise RuntimeError("Yeni dönem klasörü zaten var; veri öncesi kayıt oluşturulmadı.")
    implementation = PROJECT / "implementation"
    reject_symlinks(implementation)
    if not implementation.is_dir():
        raise RuntimeError("Doğrulanmış hazırlık kodları bulunamadı.")

    print("Altı eğitimin tamamlanma kaydı doğrulandı.", flush=True)
    print("Analiz kodları indiriliyor ve dosya kimlikleri kontrol ediliyor...", flush=True)
    payload = {}
    for name, expected in FILES.items():
        with urllib.request.urlopen(PREFIX + name, timeout=45) as response:
            data = response.read()
        if len(data) != expected["size_bytes"] or digest(data) != expected["sha256"]:
            raise RuntimeError("Kod kimliği eşleşmedi: " + name)
        path = implementation / name
        reject_symlinks(path)
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise RuntimeError("Mevcut farklı kod korunarak işlem durduruldu: " + name)
        payload[name] = data
    for name, data in payload.items():
        path = implementation / name
        reject_symlinks(path)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
        if path.read_bytes() != data:
            raise RuntimeError("Kaydedilen kodun okuma kontrolü başarısız: " + name)

    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    print("15 model dosyası ve eğitim kayıtları denetleniyor; analiz kuralları kaydediliyor...", flush=True)
    result = subprocess.run(
        [str(H2_PYTHON), "-B", str(implementation / "lock_external_analysis_v1.py"),
         "--project-root", str(PROJECT), "--lock"],
        capture_output=True, text=True, env=environment,
    )
    if result.returncode:
        print(result.stdout[-6000:], flush=True)
        print(result.stderr[-9000:], flush=True)
        raise RuntimeError("Analiz kaydı oluşturulamadı; eğitimi yeniden başlatmadan bu çıktıyı paylaş.")
    locked = json.loads(result.stdout)
    if (locked.get("status") != "ANALYSIS_LOCKED_BEFORE_NEW_DATA"
            or locked.get("n_existing_checkpoints") != 9
            or locked.get("n_stl_checkpoints") != 6
            or locked.get("new_period_accessed_by_this_code") is not False
            or locked.get("training_started_by_this_code") is not False):
        raise RuntimeError("Analiz kaydı beklenen doğrulamaları üretmedi.")
    print(json.dumps(locked, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
