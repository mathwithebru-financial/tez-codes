#!/usr/bin/env python3
"""Stage verified H2 inputs and lock a separate T4 protocol; never fit a model.

Only allowlisted bytes are copied from the immutable CPU archive. Existing
destination files are reused only when their exact bytes match. The CUDA probe
uses synthetic zeros and backward only: it creates no optimizer or predictions
from actual train/validation/test arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
import warnings
import zipfile

sys.dont_write_bytecode = True

ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1")
DRIVE_MOUNT = Path("/content/drive")
H2PY = Path("/content/h2_t4_runtime_v1/venv/bin/python")
INSTALL_ROOT = Path("/content/h2_t4_runtime_v1")
CPU_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_v1")
V4_ROOT = Path("/content/drive/MyDrive/tez_transformer_v4_repro")
CPU_ZIP = CPU_ROOT / "H2_Experiment_v1_20261028.zip"
CPU_ZIP_SHA = "c5aaabac6331add3057df008b324e5a449380985f070e453a1bdebff02ecad6c"
CPU_LOCK_SHA = "edc01a81a4e235f2bf46d5fb9fe8a7c2f626ee65eb3a3d418d5ab56f179e9fb3"
SOURCE_COMMIT = "5d4424a512659d6ec110ef9b5cda4e33792b0475"
RAW_BASE = ("https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
            + SOURCE_COMMIT + "/experiments/h2_colab_t4_v1/")
SOURCE_HASHES = {
    "implementation/run_h2_training_t4_v1.py": "de71f182000c57197919b1139212c9e2607dc60b8c40abf1e2e910ed2ed19cdc",
    "implementation/evaluate_h2_t4_v1.py": "e0c00890564c208df59e1c8ae6c5373c8a175a3675cdb0ab11aecbd8c564f27e",
    "protocol_template_t4_v1.json": "6e613678e7b66cbf6521aa5249eba5f86b46e1a206b99f15296e58bfd402b9cd",
}
REFERENCE_HASHES = {
    "bartlett_loss_differentials_v1.npz": "12a47461ddb6e4d4bc31cdbdd4e2414cb6cd960e870864ad50534d82aee52145",
    "bartlett_dm_hln_results_v1.csv": "c2335c42b01d8fb6a47f5913e46b558e3109d37726b3ebd3e414bd87f4cbd8ab",
}
PINS = {"torch": "2.11.0+cu128", "numpy": "2.3.5", "pandas": "2.2.3",
        "scipy": "1.17.0", "scikit-learn": "1.6.1"}
DETERMINISM = {"cublas_workspace_config": ":4096:8", "deterministic_algorithms": True,
               "cudnn_deterministic": True, "cudnn_benchmark": False,
               "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
               "float32_matmul_precision": "highest", "amp_enabled": False}
INPUTS = {"scripts/06_best_model_multiseed_v4.py", "config/schema_v4.json",
          "data/raw/raw_prices.csv", "data/sequences/baseline/scalers.pkl",
          "data/processed/selection_baseline_denominators_v4.json",
          "data/sequences/baseline/lb10/sequence_meta.json"}
INPUTS |= {"data/sequences/baseline/lb10/" + name + "_" + split + ".npy"
           for split in ("train", "val", "test")
           for name in ("X", "y", "anchor_dates", "target_realization_dates")}
INPUTS |= {"data/sequences/baseline/lb10/y_" + split + "_raw.npy" for split in ("train", "val", "test")}
PROTOCOL_NAME = "protocol_locked_t4_v1.json"
READINESS_NAME = "readiness_t4_v1.json"
PROPOSED_RUN = ROOT / "runs/t4_full_run_v1"

GPU_PROBE = r'''
import importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("h2_t4_frozen_runner", root/"implementation/run_h2_training_t4_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch = m.configure_torch(1)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_name(0)=="Tesla T4" and torch.cuda.get_device_capability(0)==(7,5)
data, scaler, denominators = m.load_train_validation(root/"inputs")
del data, scaler
namespace = m.source_namespace(root/"inputs", denominators)
results = []
for arm in m.ARMS:
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    model = m.build_model(namespace,arm).to("cuda").train()
    before = {name:p.detach().clone() for name,p in model.named_parameters()}
    x = torch.zeros((2,10,8),device="cuda",dtype=torch.float32)
    output = model(x)
    assert output.shape==(2,8) and torch.isfinite(output).all().item()
    loss = 0.7*torch.square(output[:,:4]).mean() + 0.3*(0.5*torch.abs(output[:,4:])).mean()
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad)
    assert all(torch.equal(before[name],p.detach()) and torch.isfinite(p).all().item() for name,p in model.named_parameters())
    results.append({"arm":arm["id"],"parameters":arm["parameters"],"finite_forward":True,
                    "finite_gradients":True,"parameters_unchanged":True})
    del model, before, output, loss, x
torch.cuda.synchronize()
print(json.dumps({"status":"SYNTHETIC_T4_FORWARD_BACKWARD_PASS","arms":results,"n_arms":3,
                  "synthetic_input_shape":[2,10,8],"optimizer_steps":0,"models_fitted":False,
                  "train_validation_inputs_verified":True,"scaler_refitted":False,
                  "real_data_predictions":False,"training_started":False,"test_arrays_loaded":False}))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_relative(value):
    value = str(value)
    if "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("Platform-ambiguous archive path refused.")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(part in {".", ".."} for part in p.parts):
        raise ValueError("Unsafe archive-relative path refused.")
    return p.as_posix()


def reject_symlinks(path):
    path = Path(path).absolute()
    for candidate in [path, *path.parents]:
        if candidate.is_symlink():
            raise ValueError(f"Symlink path refused: {path}")


def destination(root, relative, protected):
    path = root / safe_relative(relative)
    reject_symlinks(root)
    reject_symlinks(path)
    resolved, base = path.resolve(), root.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError("Destination must be a file inside the new isolated root.")
    for old in protected:
        old = Path(old).resolve()
        if base == old or base.is_relative_to(old) or old.is_relative_to(base):
            raise ValueError("The new root overlaps a protected old project.")
        if resolved == old or resolved.is_relative_to(old) or old.is_relative_to(resolved):
            raise ValueError("A protected old project cannot be a destination.")
    return path


def require_same_or_absent(path, data):
    reject_symlinks(path)
    if path.exists() and (not path.is_file() or path.read_bytes() != data):
        raise FileExistsError(f"Existing different file was preserved, not overwritten: {path}")


def write_same_or_fail(path, data):
    require_same_or_absent(path, data)
    if path.exists():
        return "EXACT_BYTES_REUSED"
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlinks(path)
    with path.open("xb") as handle:
        handle.write(data)
    return "NEW_FILE_CREATED"


def download_verified(relative, expected):
    request = Request(RAW_BASE + safe_relative(relative), headers={"User-Agent": "H2-T4-fixed-source-staging-v1"})
    with urlopen(request, timeout=45) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024 or sha(data) != expected:
        raise ValueError(f"Immutable source SHA256 differs: {relative}")
    return data


def read_archive_payloads(path, archive_sha, lock_sha, identities, references):
    """Read the byte-verified ZIP in memory; never call extract/extractall."""
    if set(identities) != INPUTS or len(identities) != 21:
        raise ValueError("Exactly the 21 allowlisted input identities are required.")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Unexpected CPU archive size.")
    archive_bytes = path.read_bytes()
    if sha(archive_bytes) != archive_sha:
        raise ValueError("CPU ZIP SHA256 differs; ZIP was not opened.")
    copied = {}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members refused.")
        for item in infos:
            safe_relative(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
                raise ValueError("Symlink/encrypted ZIP member refused.")
        lock_bytes = bundle.read("H2_v1/full_run_v2/locked_training_manifest.json")
        if sha(lock_bytes) != lock_sha:
            raise ValueError("Accepted CPU training manifest SHA256 differs.")
        if json.loads(lock_bytes)["input_identities"] != identities:
            raise ValueError("CPU ZIP and T4 template disagree about original frozen inputs.")
        for relative, identity in identities.items():
            data = bundle.read("H2_v1/inputs/" + safe_relative(relative))
            if len(data) != identity["size_bytes"] or sha(data) != identity["sha256"]:
                raise ValueError(f"CPU input identity differs: {relative}")
            copied["inputs/" + relative] = data
        for name, expected in references.items():
            data = bundle.read("H2_v1/reference/" + safe_relative(name))
            if sha(data) != expected:
                raise ValueError(f"Old reference identity differs: {name}")
            copied["reference/" + name] = data
    return copied


def child_environment():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    return env


def child_json(arguments, *, timeout=180):
    try:
        result = subprocess.run([str(H2PY), *arguments], check=True, env=child_environment(),
                                capture_output=True, text=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        for captured in (getattr(exc, "stdout", None), getattr(exc, "stderr", None)):
            if captured:
                if isinstance(captured, bytes):
                    captured = captured.decode("utf-8", errors="replace")
                print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")
        raise RuntimeError("The isolated no-fit runtime/preflight child failed; its output is shown above.") from exc
    return json.loads(result.stdout)


def verify_installer_report(report):
    if (report.get("status") != "H2_T4_ORTAM_HAZIR" or report.get("python") != "3.12.13"
            or report.get("packages") != PINS or report.get("cuda_runtime") != "12.8"
            or report.get("gpu") != "Tesla T4" or report.get("cuda_tensor_smoke") != "PASS"
            or report.get("training_started") is not False
            or not isinstance(report.get("cudnn_version"), int) or report["cudnn_version"] <= 0):
        raise ValueError("The isolated environment's verified installer report does not match.")


def verify_runtime(runtime, installer):
    if (runtime.get("python_version") != "3.12.13" or runtime.get("packages") != PINS
            or runtime.get("cuda_runtime") != "12.8" or runtime.get("gpu_count") != 1
            or runtime.get("gpu_name") != "Tesla T4" or runtime.get("device_capability") != [7, 5]
            or runtime.get("cudnn_version") != installer["cudnn_version"]
            or runtime.get("determinism") != DETERMINISM
            or not isinstance(runtime.get("driver_version"), str) or not runtime["driver_version"].strip()):
        raise ValueError("The actual frozen-code runtime collector differs from the verified isolated T4 environment.")


def locked_protocol(template, runtime, staging, preflight, existing=None):
    protocol = json.loads(json.dumps(template))
    protocol.update({"status": "LOCKED_BEFORE_T4_FULL_TRAINING", "runtime_environment": runtime,
                     "staging": staging, "preflight": preflight,
                     "next_gate": "Run all nine fixed-design T4 fits in the proposed new run directory; test arrays remain gated until checkpoint and train/validation reproduction checks pass."})
    if existing is not None:
        timestamp = existing.get("locked_at_utc")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("Existing protocol has no valid preserved lock timestamp.")
        protocol["locked_at_utc"] = timestamp
        if protocol != existing:
            raise ValueError("Existing locked protocol core fields differ; it was preserved unchanged.")
    else:
        protocol["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    return protocol


def main_stage():
    if (platform.system() != "Linux" or platform.python_version() != "3.12.13"
            or Path(sys.executable).absolute() != H2PY or not H2PY.is_file()):
        raise RuntimeError("Run this stager with the verified /content/h2_t4_runtime_v1/venv/bin/python only.")
    if not os.path.ismount(DRIVE_MOUNT) or not (DRIVE_MOUNT / "MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive before staging; no login or mount is performed here.")
    for path in (ROOT, ROOT / "inputs", ROOT / "runs", CPU_ZIP):
        reject_symlinks(path)
    for directory in (ROOT, ROOT / "inputs", ROOT / "runs"):
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"An expected project directory is not a directory: {directory}")
    if PROPOSED_RUN.exists():
        raise FileExistsError("The proposed run already exists; staging will not reuse, modify or restart it.")
    installer_bytes = (INSTALL_ROOT / "environment_verified.json").read_bytes()
    freeze_bytes = (INSTALL_ROOT / "installed_packages_freeze.txt").read_bytes()
    installer = json.loads(installer_bytes)
    verify_installer_report(installer)
    if not freeze_bytes.strip():
        raise ValueError("The verified installed-packages freeze is missing or empty.")

    # All immutable downloads, old archive checks and existing-byte conflicts
    # are resolved before the first new destination file is written.
    sources = {relative: download_verified(relative, digest) for relative, digest in SOURCE_HASHES.items()}
    template = json.loads(sources["protocol_template_t4_v1.json"])
    if (template.get("protocol_id") != "H2_CONTROLLED_SHARING_T4_V1"
            or template.get("project_root") != str(ROOT)
            or set(template.get("protected_roots", [])) != {str(CPU_ROOT), str(V4_ROOT)}
            or template["cpu_reference_execution"]["archive_sha256"] != CPU_ZIP_SHA
            or template["cpu_reference_execution"]["accepted_training_manifest_sha256"] != CPU_LOCK_SHA
            or template["analysis"]["training_implementation_sha256"] != SOURCE_HASHES["implementation/run_h2_training_t4_v1.py"]
            or template["analysis"]["evaluation_implementation_sha256"] != SOURCE_HASHES["implementation/evaluate_h2_t4_v1.py"]):
        raise ValueError("The immutable T4 template does not match the staged fixed design.")
    payloads = read_archive_payloads(CPU_ZIP, CPU_ZIP_SHA, CPU_LOCK_SHA, template["input_identities"], REFERENCE_HASHES)
    planned = {**sources, **payloads,
               "environment/environment_verified.json": installer_bytes,
               "environment/installed_packages_freeze.txt": freeze_bytes}
    protected = [CPU_ROOT, V4_ROOT]
    for relative, data in planned.items():
        require_same_or_absent(destination(ROOT, relative, protected), data)
    for relative in (PROTOCOL_NAME, READINESS_NAME, "preflight_t4_no_fit_v1.json",
                     "environment/runtime_environment_t4_v1.json"):
        destination(ROOT, relative, protected)
    for relative, data in planned.items():
        write_same_or_fail(destination(ROOT, relative, protected), data)

    runner = ROOT / "implementation/run_h2_training_t4_v1.py"
    runtime = child_json([str(runner), "--runtime-report"])
    verify_runtime(runtime, installer)
    # Reuse an old lock only after all substantive fields and fresh checks agree.
    protocol_path = ROOT / PROTOCOL_NAME
    existing = json.loads(protocol_path.read_bytes()) if protocol_path.exists() else None
    staging = {"implementation_sha256": sha(Path(__file__).read_bytes()), "source_commit": SOURCE_COMMIT,
               "source_hashes": SOURCE_HASHES, "cpu_archive_sha256": CPU_ZIP_SHA,
               "cpu_training_manifest_sha256": CPU_LOCK_SHA, "reference_sha256": REFERENCE_HASHES,
               "installer_report_sha256": sha(installer_bytes), "installed_packages_freeze_sha256": sha(freeze_bytes)}
    if existing is not None:
        pending = json.loads(json.dumps(existing))
        pending["preflight"] = template["preflight"]
        locked_protocol(template, runtime, staging, template["preflight"], pending)
    cpu = child_json([str(runner), "--input-root", str(ROOT / "inputs"), "--self-test"])
    if (cpu.get("status") != "CPU_MODEL_AND_RUNTIME_GUARD_SELF_TEST_PASS"
            or cpu.get("parameter_counts") != [54536, 28456, 54586]
            or cpu.get("raw_metric_validation_score_check") != 2.5
            or cpu.get("cuda_tested") is not False or cpu.get("models_trained") is not False
            or cpu.get("file_writes") is not False):
        raise ValueError("The frozen runner CPU semantic self-test failed.")
    gpu = child_json(["-c", GPU_PROBE, str(ROOT)])
    if (gpu.get("status") != "SYNTHETIC_T4_FORWARD_BACKWARD_PASS" or gpu.get("n_arms") != 3
            or gpu.get("optimizer_steps") != 0 or gpu.get("models_fitted") is not False
            or gpu.get("train_validation_inputs_verified") is not True or gpu.get("scaler_refitted") is not False
            or gpu.get("real_data_predictions") is not False or gpu.get("test_arrays_loaded") is not False):
        raise ValueError("The no-fit synthetic CUDA compatibility check failed.")
    preflight = {"status": "PASS_NO_MODEL_FITTING", "cpu_semantic_check": cpu,
                 "synthetic_cuda_forward_backward": gpu, "n_input_identities_verified": 21,
                 "test_files_copied_as_opaque_bytes_only": True, "test_arrays_loaded": False,
                 "training_started": False, "models_fitted": False}
    protocol = locked_protocol(template, runtime, staging, preflight, existing)
    protocol_bytes = encoded(protocol)
    readiness = {"status": "H2_T4_GIRDI_VE_PROTOKOL_HAZIR", "n_input": 21, "n_runs_planned": 9,
                 "protocol_sha256": sha(protocol_bytes), "source_commit": SOURCE_COMMIT,
                 "staging_implementation_sha256": staging["implementation_sha256"],
                 "proposed_run_root": str(PROPOSED_RUN), "training_started": False,
                 "test_arrays_loaded": False, "models_fitted": False}
    final_files = {PROTOCOL_NAME: protocol_bytes, "preflight_t4_no_fit_v1.json": encoded(preflight),
                   "environment/runtime_environment_t4_v1.json": encoded(runtime), READINESS_NAME: encoded(readiness)}
    for relative, data in final_files.items():
        require_same_or_absent(destination(ROOT, relative, protected), data)
    # Recheck exact copied bytes after probes and before publishing readiness.
    for relative, data in planned.items():
        path = destination(ROOT, relative, protected)
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"A staged file changed or disappeared during preflight: {relative}")
    reject_symlinks(ROOT / "runs")
    (ROOT / "runs").mkdir(exist_ok=True)
    if PROPOSED_RUN.exists():
        raise RuntimeError("The proposed run appeared during staging; no training was started here.")
    for relative, data in final_files.items():
        write_same_or_fail(destination(ROOT, relative, protected), data)
    print(json.dumps(readiness, ensure_ascii=False, allow_nan=False))


def self_test():
    """Temporary fixtures only: no Drive, network, CUDA or experiment execution."""
    compile(GPU_PROBE, "<synthetic_cuda_probe>", "exec")
    rejected = 0
    with tempfile.TemporaryDirectory(prefix="h2_t4_staging_fixture_") as folder:
        tmp = Path(folder)
        root = tmp / "new"
        root.mkdir()
        protected = [tmp / "old_cpu", tmp / "old_v4"]
        target = destination(root, "inputs/example.bin", protected)
        assert write_same_or_fail(target, b"original") == "NEW_FILE_CREATED"
        assert write_same_or_fail(target, b"original") == "EXACT_BYTES_REUSED"
        try:
            write_same_or_fail(target, b"changed")
        except FileExistsError:
            rejected += 1
        else:
            raise AssertionError("Mismatched existing file was accepted.")
        assert target.read_bytes() == b"original"
        for unsafe in ("../escape", "/absolute", "a\\b", "C:/escape"):
            try:
                destination(root, unsafe, protected)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Unsafe destination accepted.")
        (root / "linked").symlink_to(tmp / "old_cpu", target_is_directory=True)
        try:
            destination(root, "linked/file", protected)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Symlink destination accepted.")
        identities = {name: {"sha256": sha(name.encode()), "size_bytes": len(name.encode())} for name in INPUTS}
        lock_bytes = encoded({"input_identities": identities})
        refs = {"one.npz": b"synthetic-npz-bytes", "two.csv": b"synthetic-csv-bytes"}
        archive = tmp / "synthetic.zip"
        with zipfile.ZipFile(archive, "x") as z:
            z.writestr("H2_v1/full_run_v2/locked_training_manifest.json", lock_bytes)
            for name in INPUTS:
                z.writestr("H2_v1/inputs/" + name, name.encode())
            for name, data in refs.items():
                z.writestr("H2_v1/reference/" + name, data)
            z.writestr("H2_v1/unrelated_file_not_copied.txt", b"must not be copied")
        payload = read_archive_payloads(archive, sha(archive.read_bytes()), sha(lock_bytes), identities,
                                        {name: sha(data) for name, data in refs.items()})
        assert len(payload) == 23 and not any("unrelated" in name for name in payload)
        try:
            read_archive_payloads(archive, "0" * 64, sha(lock_bytes), identities, {})
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Wrong archive identity accepted.")
        for member in ("../traversal", "/absolute"):
            unsafe_zip = tmp / ("unsafe" + str(rejected) + ".zip")
            with zipfile.ZipFile(unsafe_zip, "x") as z:
                z.writestr(member, b"bad")
            try:
                read_archive_payloads(unsafe_zip, sha(unsafe_zip.read_bytes()), sha(lock_bytes), identities, {})
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Unsafe ZIP member accepted.")
        duplicate = tmp / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "x") as z:
                z.writestr("same", b"1"); z.writestr("same", b"2")
        try:
            read_archive_payloads(duplicate, sha(duplicate.read_bytes()), sha(lock_bytes), identities, {})
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Duplicate ZIP members accepted.")
        old = locked_protocol({"core": "same"}, {"runtime": "synthetic"}, {"code": "same"}, {"status": "PASS"})
        assert locked_protocol({"core": "same"}, {"runtime": "synthetic"}, {"code": "same"}, {"status": "PASS"}, old) == old
        try:
            locked_protocol({"core": "changed"}, {"runtime": "synthetic"}, {"code": "same"}, {"status": "PASS"}, old)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Changed locked core accepted.")
    print(json.dumps({"status": "STAGER_TEMP_FIXTURE_SELF_TEST_PASS", "mismatch_path_zip_cases_rejected": rejected,
                      "copied_allowlist_size": 23, "real_drive_access": False, "cuda_tested": False,
                      "training_started": False, "test_arrays_loaded": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Temporary synthetic fixtures only; no Drive/network/CUDA.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main_stage()


if __name__ == "__main__":
    main()
