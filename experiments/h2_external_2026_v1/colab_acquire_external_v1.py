"""Prepare an isolated acquisition environment, then acquire the locked snapshot.

No model training or prediction is performed. The existing locked acquisition
implementation supplies every scientific acquisition and coverage rule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
H2_ROOT = Path("/content/h2_t4_runtime_v1")
H2_PYTHON = H2_ROOT / "venv/bin/python"
ACQ_ROOT = Path("/content/h2_external_acq_runtime_v1")
ACQ_PYTHON = ACQ_ROOT / "venv/bin/python"
LOCK_SHA256 = "a155ad428996a6c1d40ca6a755212206ca9092a7b0fdc76200c3a1f063bdb274"
ACQUIRER_SHA256 = "258fc4ed7311a4f4936e0bca3b8102c470761f5ef3bdf532491d364c4e2d4342"
PREPROCESSOR_SHA256 = "8648220271dbed9d537e86f42f944b52ce69755a77a846d84a4fadecab501179"
PINS = {"numpy": "2.3.5", "pandas": "2.2.3", "yfinance": "0.2.66"}
TRAINING_PINS = {"torch": "2.11.0+cu128", "numpy": "2.3.5", "pandas": "2.2.3",
                 "scipy": "1.17.0", "scikit-learn": "1.6.1"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("Çalışma klasörü sembolik bağlantı içeriyor.")


def environment():
    result = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        result.pop(key, None)
    result.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return result


def run(command, *, env=None):
    result = subprocess.run(command, capture_output=True, text=True,
                            env=environment() if env is None else env)
    if result.returncode:
        print(result.stdout[-5000:], flush=True)
        print(result.stderr[-10000:], flush=True)
        raise RuntimeError("İşlem durdu; yeniden indirmeyi veya eğitimi başlatmadan bu çıktıyı paylaş.")
    return result.stdout


def verify_lock():
    no_symlinks(PROJECT)
    lock_path = PROJECT / "analysis_locked_v1.json"
    no_symlinks(lock_path)
    data = lock_path.read_bytes()
    if sha(data) != LOCK_SHA256:
        raise RuntimeError("Analiz kaydının kimliği paylaşılan kayıtla eşleşmedi.")
    lock = json.loads(data)
    if lock.get("status") != "ANALYSIS_LOCKED_BEFORE_NEW_DATA":
        raise RuntimeError("Veri öncesi analiz kaydı doğrulanamadı.")
    implementations = lock.get("implementation_sha256", {})
    if implementations.get("acquire_external_prices_v1.py") != ACQUIRER_SHA256:
        raise RuntimeError("İndirme kodu sabitlenen sürümle eşleşmedi.")
    for name, expected in implementations.items():
        if Path(name).name != name:
            raise RuntimeError("Analiz kaydındaki kod yolu geçersiz.")
        path = PROJECT / "implementation" / name
        no_symlinks(path)
        if sha(path.read_bytes()) != expected:
            raise RuntimeError("Sabitlenen kod değişmiş: " + name)
    return lock


def training_inventory():
    probe = r'''
import importlib.metadata as m, json, platform, sys
pins = {name:m.version(name) for name in
        ("torch", "numpy", "pandas", "scipy", "scikit-learn")}
inventory = sorted((d.metadata.get("Name", ""),d.version) for d in m.distributions())
print(json.dumps({"python":platform.python_version(), "prefix":sys.prefix,
                  "packages":pins,"inventory":inventory}))
'''
    result = json.loads(run([str(H2_PYTHON), "-B", "-c", probe]))
    if (result.get("python") != "3.12.13"
            or result.get("prefix") != str(H2_ROOT / "venv")
            or result.get("packages") != TRAINING_PINS):
        raise RuntimeError("Korunan H2 eğitim ortamı beklenen sürümlerle eşleşmedi.")
    return result


def verify_historical_preprocessing():
    probe = r'''
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1])
path=root/"implementation/prepare_external_sequences_v1.py"
source=path.read_bytes()
if hashlib.sha256(source).hexdigest()!=sys.argv[2]:
    raise RuntimeError("Historical preprocessing source changed.")
namespace={"__name__":"h2_pre_acquisition_frozen_preprocessing","__file__":str(path)}
exec(compile(source,str(path),"exec"),namespace)
root=namespace["project_root_checked"](root)
namespace["verify_analysis_lock"](root)
_,identities=namespace["verify_frozen_inputs"](root)
inputs=root/"inputs"
old=namespace["read_prices"](inputs/"data/raw/raw_prices.csv","frozen old prices",
    identities["data/raw/raw_prices.csv"],allow_all_missing_rows=True)
bundle,before=namespace["load_original_scalers"](inputs/"data/sequences/baseline/scalers.pkl",
    identities["data/sequences/baseline/scalers.pkl"])
proof=namespace["historical_reproduction"](old,bundle,inputs)
if namespace["scaler_snapshot"](bundle)!=before:
    raise RuntimeError("Frozen scaler state changed during historical reproduction.")
for relative,identity in identities.items():
    namespace["verify_identity"](namespace["checked_relative"](inputs,relative),identity)
print(json.dumps(proof,allow_nan=False))
'''
    proof = json.loads(run([str(H2_PYTHON), "-B", "-c", probe,
                            str(PROJECT), PREPROCESSOR_SHA256]))
    if (proof.get("status") != "ORIGINAL_TRAIN_VALIDATION_BASELINE_REPRODUCED"
            or proof.get("old_test_arrays_loaded") is not False
            or set(proof.get("splits", {})) != {"train", "val"}):
        raise RuntimeError("Eski eğitim/doğrulama verilerinin yeniden hesaplanması doğrulanamadı; fiyat indirilmedi.")
    return proof


def prepare_environment():
    no_symlinks(ACQ_ROOT)
    owner = {"purpose": "H2_EXTERNAL_PRICE_ACQUISITION_V1", "python": "3.12.13",
             "uv": "0.13.4", "pins": PINS, "analysis_lock_sha256": LOCK_SHA256}
    ACQ_ROOT.mkdir(exist_ok=True)
    marker = ACQ_ROOT / "acquisition_runtime_owner.json"
    no_symlinks(marker)
    if marker.exists():
        if json.loads(marker.read_bytes()) != owner:
            raise RuntimeError("Mevcut indirme ortamı farklı ayarlara ait; değiştirilmedi.")
    else:
        if any(ACQ_ROOT.iterdir()):
            raise RuntimeError("Sahiplik kaydı olmayan indirme klasörü korunarak işlem durduruldu.")
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(owner, handle, ensure_ascii=False, indent=2)
    tools = H2_ROOT / "uv_tool"
    no_symlinks(tools)
    if not (tools / "uv/__main__.py").is_file():
        raise RuntimeError("Önceden doğrulanan kurulum aracı bulunamadı.")
    uv_env = environment()
    uv_env.update({"PYTHONPATH": str(tools), "UV_CACHE_DIR": str(ACQ_ROOT / "cache"),
                   "UV_PYTHON_DOWNLOADS": "never", "UV_NO_PROGRESS": "1"})
    uv = [sys.executable, "-m", "uv"]
    version = run(uv + ["--version"], env=uv_env).split()
    if len(version) < 2 or version[:2] != ["uv", "0.13.4"]:
        raise RuntimeError("Kurulum aracı sürümü eşleşmedi.")
    venv = ACQ_ROOT / "venv"
    no_symlinks(venv)
    if not venv.exists():
        run([str(H2_PYTHON), "-B", "-m", "venv", "--without-pip", str(venv)])
    if not ACQ_PYTHON.is_file():
        raise RuntimeError("Ayrı indirme ortamının Python yürütücüsü bulunamadı.")
    isolated = json.loads(run([str(ACQ_PYTHON), "-B", "-c",
        "import json,platform,sys; print(json.dumps({'python':platform.python_version(),"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"]))
    if (isolated.get("python") != "3.12.13" or isolated.get("prefix") != str(venv)
            or isolated.get("prefix") == isolated.get("base_prefix")):
        raise RuntimeError("Paket kurulmadan önce ayrı Python ortamı doğrulanamadı.")
    print("Ayrı indirme ortamına sabit paketler kuruluyor...", flush=True)
    run(uv + ["pip", "install", "--no-config", "--python", str(ACQ_PYTHON),
              "--default-index", "https://pypi.org/simple",
              *[f"{name}=={version}" for name, version in PINS.items()]], env=uv_env)
    run(uv + ["pip", "check", "--python", str(ACQ_PYTHON)], env=uv_env)
    probe = r'''
import importlib.metadata as m, json, platform, sys
print(json.dumps({"python":platform.python_version(),"prefix":sys.prefix,
                  "base_prefix":sys.base_prefix,
                  "packages":{k:m.version(k) for k in ("numpy","pandas","yfinance")}}))
'''
    result = json.loads(run([str(ACQ_PYTHON), "-B", "-c", probe]))
    if (result.get("python") != "3.12.13" or result.get("prefix") != str(venv)
            or result.get("base_prefix") == result.get("prefix") or result.get("packages") != PINS):
        raise RuntimeError("Ayrı indirme ortamı veya paket sürümleri doğrulanamadı.")
    return result


def main():
    if sys.argv[1:]:
        raise ValueError("Bu başlatıcının parametresi yoktur.")
    if not H2_PYTHON.is_file():
        raise RuntimeError("Ayrı H2 Python ortamı bulunamadı; eğitim hücresini yeniden çalıştırma.")
    verify_lock()
    before = training_inventory()
    protected_record = H2_ROOT / "installed_packages_freeze.txt"
    no_symlinks(protected_record)
    freeze_before = protected_record.read_bytes()
    print("Özgün eğitim/doğrulama girdileri eski ham veriden yeniden hesaplanarak kontrol ediliyor...", flush=True)
    historical = verify_historical_preprocessing()
    print("Özgün eğitim/doğrulama ön işleme kontrolü geçti; yeni fiyatlar henüz indirilmedi.", flush=True)
    acquisition_runtime = prepare_environment()
    if training_inventory() != before or protected_record.read_bytes() != freeze_before:
        raise RuntimeError("Eğitim ortamının değişmediği doğrulanamadı; fiyat indirilmedi.")
    print("Ayrı indirme ortamı hazır; eğitim ortamının paket kayıtları eşleşiyor.", flush=True)
    verify_lock()
    print("Sabit tarih aralığındaki kapanışlar indiriliyor ve gözlem sayıları denetleniyor...", flush=True)
    acquired = json.loads(run([
        str(ACQ_PYTHON), "-B", str(PROJECT / "implementation/acquire_external_prices_v1.py"),
        "--project-root", str(PROJECT), "--acquire",
    ]))
    if (acquired.get("status") != "ACQUISITION_COMPLETE"
            or acquired.get("analysis_lock_sha256") != LOCK_SHA256
            or any(acquired.get(k) is not False for k in
                   ("models_evaluated", "old_test_arrays_loaded", "old_project_writes"))):
        raise RuntimeError("İndirme çıktısı beklenen kayıtları üretmedi; bu çıktıyı paylaş.")
    if training_inventory() != before or protected_record.read_bytes() != freeze_before:
        raise RuntimeError("İndirme tamamlandı ancak eğitim ortamı kayıtları değişmiş; yeniden indirmeden çıktıyı paylaş.")
    acquired.update({"training_environment_package_inventory_unchanged": True,
                     "acquisition_environment": acquisition_runtime,
                     "historical_preprocessing_reproduction": historical,
                     "training_started": False, "preprocessing_completed": False})
    print(json.dumps(acquired, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
