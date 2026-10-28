"""Prepare an isolated H2 interpreter on Colab; no Drive writes or training.

The notebook kernel remains unchanged. Later H2 commands must use the printed
venv/bin/python executable. Python and direct dependency pins are fixed here;
the actual CUDA driver, cuDNN and complete freeze are recorded after install.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys

UV_VERSION = "0.13.4"
PYTHON_VERSION = "3.12.13"
PYTHON_BUILD = "20260807"
TORCH_VERSION = "2.11.0+cu128"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
PINS = ["numpy==2.3.5", "pandas==2.2.3", "scipy==1.17.0", "scikit-learn==1.6.1"]
RUNTIME_ROOT = Path("/content/h2_t4_runtime_v1")


def run(command, *, env=None, capture=False):
    return subprocess.run(command, check=True, env=env, text=True,
                          stdout=subprocess.PIPE if capture else None)


def clean_environment():
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def require_uv_version(version_text):
    # Official wheels may append a platform or build identifier to --version.
    # Keep the exact numeric pin; the informational suffix is not a version.
    fields = version_text.split()
    if len(fields) < 2 or fields[0] != "uv" or fields[1] != UV_VERSION:
        raise RuntimeError(f"Beklenmeyen kurulum aracı: {version_text}")


def main():
    if platform.system() != "Linux" or not Path("/content").is_dir():
        raise RuntimeError("Bu kurulum Colab Linux çalışma zamanı içindir.")
    names = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture=True).stdout.strip().splitlines()
    if len(names) != 1 or "T4" not in names[0].upper():
        raise RuntimeError(f"Tek bir T4 GPU bekleniyor; görülen aygıtlar: {names}")

    root = RUNTIME_ROOT
    root.mkdir(parents=True, exist_ok=True)
    tools_root, venv = root / "uv_tool", root / "venv"
    tools_root.mkdir(exist_ok=True)
    marker = root / "h2_runtime_owner.json"
    owner = {"purpose": "H2_T4_ENVIRONMENT_V1", "python": PYTHON_VERSION,
             "python_build": PYTHON_BUILD, "uv": UV_VERSION,
             "torch": TORCH_VERSION, "packages": PINS}
    if marker.exists():
        if json.loads(marker.read_text()) != owner:
            raise RuntimeError("Mevcut ayrı ortam başka ayarlara ait; üzerine yazılmadı.")
    elif venv.exists():
        raise RuntimeError("Sahiplik kaydı olmayan mevcut venv bulundu; üzerine yazılmadı.")
    else:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(owner, handle, indent=2)

    # --target installs only the bootstrap tool into a new task-specific folder;
    # it does not replace the notebook kernel's Python packages.
    if not (tools_root / "uv" / "__main__.py").is_file():
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--quiet", "--target", str(tools_root), f"uv=={UV_VERSION}"],
            env=clean_environment())
    uv_env = clean_environment()
    uv_env.update({"PYTHONPATH": str(tools_root),
                   "UV_PYTHON_INSTALL_DIR": str(root / "python"),
                   "UV_PYTHON_BIN_DIR": str(root / "python_bin"),
                   "UV_CACHE_DIR": str(root / "cache"),
                   "UV_PYTHON_CPYTHON_BUILD": PYTHON_BUILD,
                   "UV_PYTHON_INSTALL_BIN": "0"})
    uv = [sys.executable, "-m", "uv"]
    installed_uv = run(uv + ["--version"], env=uv_env, capture=True).stdout.strip()
    require_uv_version(installed_uv)
    if not venv.exists():
        print("Ayrı Python 3.12.13 ortamı oluşturuluyor...", flush=True)
        run(uv + ["venv", "--no-config", "--managed-python", "--python", PYTHON_VERSION, str(venv)], env=uv_env)
    python = venv / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("Ayrı ortamın Python yürütücüsü bulunamadı.")
    exact_python = run([str(python), "-c", "import platform; print(platform.python_version())"],
                       env=clean_environment(), capture=True).stdout.strip()
    if exact_python != PYTHON_VERSION:
        raise RuntimeError(f"Python sürümü farklı: {exact_python}")

    print("H2 için PyTorch CUDA 12.8 ve sabit paketler kuruluyor; indirme birkaç dakika sürebilir.", flush=True)
    run(uv + ["pip", "install", "--no-config", "--python", str(python),
              "--default-index", CUDA_INDEX, f"torch=={TORCH_VERSION}"], env=uv_env)
    run(uv + ["pip", "install", "--no-config", "--python", str(python),
              "--default-index", "https://pypi.org/simple", *PINS], env=uv_env)
    run(uv + ["pip", "check", "--python", str(python)], env=uv_env)

    probe = r'''
import importlib.metadata as m
import json, platform, torch
expected = {"torch":"2.11.0+cu128", "numpy":"2.3.5", "pandas":"2.2.3",
            "scipy":"1.17.0", "scikit-learn":"1.6.1"}
actual = {name:m.version(name) for name in expected}
assert actual == expected, (actual, expected)
assert platform.python_version() == "3.12.13"
assert torch.cuda.is_available(), "CUDA erişimi yok"
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
assert "T4" in name.upper(), name
assert torch.cuda.get_device_capability(0) == (7,5)
x = torch.ones((2,2), device="cuda", dtype=torch.float32)
product = x @ x
torch.cuda.synchronize()
assert torch.isfinite(product).all().item() and torch.equal(product, 2*x)
print(json.dumps({"status":"H2_T4_ORTAM_HAZIR", "python":platform.python_version(),
                  "packages":actual,"cuda_runtime":torch.version.cuda,
                  "cudnn_version":torch.backends.cudnn.version(),"gpu":name,
                  "cuda_tensor_smoke":"PASS",
                  "training_started":False},ensure_ascii=False))
'''
    result = run([str(python), "-c", probe], env=clean_environment(), capture=True)
    verified = json.loads(result.stdout)
    (root / "environment_verified.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n")
    freeze = run(uv + ["pip", "freeze", "--python", str(python)], env=uv_env, capture=True).stdout
    (root / "installed_packages_freeze.txt").write_text(freeze)
    print(json.dumps(verified, ensure_ascii=False, indent=2))
    print("H2_PYTHON:", python)
    print("Colab hücresinin Python sürümü değişmedi; H2 bu ayrı yürütücüyle çalışacak.")
    print("Eğitim başlatılmadı. Drive dosyaları değiştirilmedi.")


if __name__ == "__main__":
    main()
