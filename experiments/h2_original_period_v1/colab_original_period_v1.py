"""Stage the original-period comparison, freeze its settings, then evaluate.

Uses the original 584 test examples and fifteen already selected models. It
downloads reviewed code only, performs no fitting, and fetches no prices.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "d8a810837f55f935f5239492d426b7073874850d"
PREFIX = ("https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
          + SOURCE_COMMIT + "/experiments/h2_original_period_v1/")
PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_original_period_v1")
SOURCE_PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
H2_PYTHON = Path("/content/h2_t4_runtime_v1/venv/bin/python")
PLAN_SHA = "f952059f03f84e69732964c58a6fcb1f14b1b7cc72c5b2e97a1a6358de361641"
FILES = {
    "protocol_original_period_v1.json": {
        "sha256": PLAN_SHA, "size_bytes": 12876,
        "destination": "protocol_original_period_v1.json",
    },
    "evaluate_original_period_v1.py": {
        "sha256": "aeece22bd61ea34855ca94eee3957c38ff92ce563b316946f5223b9ee6427446",
        "size_bytes": 45288,
        "destination": "implementation/evaluate_original_period_v1.py",
    },
    "README_ORIGINAL_PERIOD_V1.md": {
        "sha256": "12df63ac49c33a41e7c2e7097cf5b6565877431ccccdfdffba3578145748143c",
        "size_bytes": 2725, "destination": "README_ORIGINAL_PERIOD_V1.md",
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Çalışma yolu sembolik bağlantı içeriyor.")


def environment():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def main():
    if sys.argv[1:]:
        raise ValueError("Bu başlatıcının parametresi yoktur.")
    if not H2_PYTHON.is_file():
        raise RuntimeError("Ayrı H2 Python ortamı bulunamadı; eğitim hücresini yeniden çalıştırmadan bu çıktıyı paylaş.")
    no_symlinks(PROJECT)
    no_symlinks(SOURCE_PROJECT)
    if not SOURCE_PROJECT.is_dir():
        raise RuntimeError("Sabit modellerin ve özgün girdilerin bulunduğu klasör görünmüyor.")
    if PROJECT == SOURCE_PROJECT or PROJECT.is_relative_to(SOURCE_PROJECT) or SOURCE_PROJECT.is_relative_to(PROJECT):
        raise RuntimeError("Yeni sonuç klasörü kaynak klasörden ayrı olmalıdır.")
    print("Özgün tez dönemi için sabit kodlar hazırlanıyor: 584 örnek, 2022–2024 test dönemi.", flush=True)
    payload = {}
    for name, identity in FILES.items():
        with urllib.request.urlopen(PREFIX + name, timeout=45) as response:
            data = response.read()
        if len(data) != identity["size_bytes"] or sha(data) != identity["sha256"]:
            raise RuntimeError("Kod kimliği eşleşmedi: " + name)
        path = PROJECT / identity["destination"]
        no_symlinks(path)
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise RuntimeError("Mevcut farklı dosya korunarak işlem durduruldu: " + name)
        payload[name] = data
    owner = {"purpose": "H2_ORIGINAL_PERIOD_SUPPLEMENT_V1",
             "project_root": str(PROJECT), "source_project_root": str(SOURCE_PROJECT),
             "protocol_sha256": PLAN_SHA}
    if PROJECT.exists() and not PROJECT.is_dir():
        raise RuntimeError("Yeni proje yolu bir klasör olmalıdır.")
    marker = PROJECT / "original_period_project_owner_v1.json"
    no_symlinks(marker)
    if marker.exists():
        if json.loads(marker.read_bytes()) != owner:
            raise RuntimeError("Mevcut klasörün sahiplik kaydı farklı; değiştirilmedi.")
    elif PROJECT.exists() and any(PROJECT.iterdir()):
        raise RuntimeError("Sahiplik kaydı olmayan mevcut klasör korunarak işlem durduruldu.")
    PROJECT.mkdir(exist_ok=True)
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(owner, handle, ensure_ascii=False, indent=2)
    implementation = PROJECT / "implementation"
    no_symlinks(implementation)
    implementation.mkdir(exist_ok=True)
    for name, data in payload.items():
        path = PROJECT / FILES[name]["destination"]
        no_symlinks(path)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
        if path.read_bytes() != data:
            raise RuntimeError("Kaydedilen kodun okuma kontrolü başarısız: " + name)

    script = implementation / "evaluate_original_period_v1.py"
    command = [str(H2_PYTHON), "-u", "-B", str(script), "--project-root", str(PROJECT)]
    print("15 modelin dosyaları ve özgün veri kimlikleri kontrol ediliyor; karşılaştırma kuralları kaydediliyor...", flush=True)
    locked = subprocess.run(command + ["--lock"], capture_output=True, text=True, env=environment())
    if locked.returncode:
        print(locked.stdout[-5000:], flush=True)
        print(locked.stderr[-10000:], flush=True)
        raise RuntimeError("Özgün dönem kaydı oluşturulamadı; eğitim veya indirmeyi başlatmadan çıktıyı paylaş.")
    record = json.loads(locked.stdout)
    if (record.get("status") != "ORIGINAL_PERIOD_ANALYSIS_FIXED_BEFORE_CURRENT_TEST_INFERENCE"
            or record.get("n_selected_checkpoints") != 15
            or any(record.get(key) is not False for key in
                   ("original_test_arrays_loaded_by_this_code", "training_started", "old_results_written",
                    "external_2026_data_or_output_files_opened_by_this_code"))):
        raise RuntimeError("Özgün dönem karşılaştırma kaydı doğrulanamadı.")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    print("Önce kayıtlı eğitim/doğrulama sonuçları, sonra aynı 584 test örneği değerlendiriliyor; eğitim yapılmıyor.", flush=True)
    with subprocess.Popen(command + ["--evaluate"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=environment()) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError("Özgün dönem değerlendirmesi durdu; yeniden çalıştırmadan son çıktıyı paylaş.")
    output = PROJECT / "runs/original_test_v1"
    manifest_path = output / "evaluation_manifest_v1.json"
    no_symlinks(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "H2_ORIGINAL_PERIOD_EVALUATION_COMPLETE":
        raise RuntimeError("Özgün dönem sonuç durumu doğrulanamadı.")

    def artifact(name):
        path = output / name
        no_symlinks(path)
        data = path.read_bytes()
        if sha(data) != manifest["artifacts"][name]["sha256"]:
            raise RuntimeError("Sonuç dosyası kimliği eşleşmedi: " + name)
        return data.decode("utf-8")

    report = json.loads(artifact("analysis_report_v1.json"))
    seed_rows = list(csv.DictReader(artifact("primary_losses_seed_summary_v1.csv").splitlines()))
    gap_rows = list(csv.DictReader(artifact("generalization_gaps_seed_summary_v1.csv").splitlines()))
    print(json.dumps({"status": manifest["status"],
        "evaluation_manifest_sha256": sha(manifest_bytes), "period": report["period"],
        "train_validation_replay": report["train_validation_replay"],
        "legacy_original_test_replay": report["legacy_original_test_replay"],
        "comparisons": report["comparisons"],
        "four_asset_seed_summary": [r for r in seed_rows if r["asset"] == "four_asset_mean"],
        "generalization_gaps_descriptive": gap_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
