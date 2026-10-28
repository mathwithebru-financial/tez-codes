#!/usr/bin/env python3
"""Package the completed isolated H2 experiment without executing its code.

Only a completed, hash-consistent nine-fit run and completed evaluation may be
packaged. Existing source bytes are preserved. No dependencies, private transfer
helpers, cached files, temporary downloads or credentials are collected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

ARCHIVE_NAME = "H2_Experiment_v1_20261028.zip"
ARCHIVE_ROOT = "H2_v1"
MANIFEST_NAME = "SHA256SUMS.json"
ARMS = ("NS_base", "FS_same_width", "FS_budget_control")
SEEDS = (123, 777, 2026)
EXPECTED_INPUTS = {
    "scripts/06_best_model_multiseed_v4.py", "config/schema_v4.json",
    "data/raw/raw_prices.csv", "data/sequences/baseline/scalers.pkl",
    "data/processed/selection_baseline_denominators_v4.json",
    "data/sequences/baseline/lb10/sequence_meta.json",
}
# The explicit names avoid deserializing any NumPy file while packaging.
EXPECTED_INPUTS |= {"data/sequences/baseline/lb10/" + name + "_" + split + ".npy"
                    for split in ("train", "val", "test")
                    for name in ("X", "y", "anchor_dates", "target_realization_dates")}
EXPECTED_INPUTS |= {"data/sequences/baseline/lb10/y_" + split + "_raw.npy"
                    for split in ("train", "val", "test")}
DENIED_PARTS = {"deps", ".git", "__pycache__", "private_fetchlogs", "private_library_helpers", ".venv"}
ALLOWED_RUN_SUFFIXES = {".json", ".csv", ".pt", ".npz", ".txt", ".log", ".md", ".png", ".pdf", ".svg", ".sha256"}
TEXT_SUFFIXES = {".py", ".json", ".csv", ".txt", ".log", ".md", ".svg", ".sha256"}
PRIVATE_CONTENT = re.compile(
    rb"https?://[^\s\"'<>]*(?:blob\.core\.windows\.net|amazonaws\.com|oaiusercontent\.com)[^\s\"'<>]*"
    rb"|(?:X-Amz-Signature|X-Goog-Signature|SharedAccessSignature|[?&]sig)=|Authorization:\s*Bearer\s+",
    re.IGNORECASE,
)

REPLAY_README = r"""H2 experiment v1: integrity and replay

This archive is self-contained and preserves the recorded input, implementation,
checkpoint and result bytes. Historical absolute paths inside provenance JSON
files describe the original run. Pass the current extracted paths explicitly.
The old 584-date test period had already been examined; this is a supplementary
exploratory experiment, not a new independent confirmation.

The first attempt was rejected before test access because one training history
failed the paired-seed/hash integrity checks. rejected_attempt_v1/ retains only
technical audit records. Those files are not accepted scientific results. All
nine fits were repeated in full_run_v2 with the same frozen code and protocol.

1. Extract the archive and change into its H2_v1 directory. Python 3.12 was used.

cd /absolute/path/to/H2_v1
H2_ROOT="$(pwd -P)"

2. Verify every archived payload against its SHA256 manifest. The manifest does
not include itself (a cryptographic self-hash is not possible).

python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path.cwd().resolve()
manifest = json.loads((root / "SHA256SUMS.json").read_text())
for relative, record in manifest["files"].items():
    path = root / relative
    assert path.is_file() and path.stat().st_size == record["size_bytes"], relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], relative
print("ALL_ARCHIVED_PAYLOAD_HASHES_VERIFIED", len(manifest["files"]))
PY

3. Create an isolated Python environment and install the recorded dependencies.
This does not use any environment from the original computer.

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu"
python -m pip install -r requirements_h2_v1.txt

4. Verify the packaged frozen run without opening its test arrays or writing
result files. The statistics self-test reads the old reference loss series.
Keep implementation/ and full_run_v2/ directly under the same H2_v1 root:
the evaluator intentionally rejects run roots outside that project.

python implementation/evaluate_h2_v1.py \
  --input-root "$H2_ROOT/inputs" --run-root "$H2_ROOT/full_run_v2" \
  --reference-npz "$H2_ROOT/reference/bartlett_loss_differentials_v1.npz" \
  --reference-csv "$H2_ROOT/reference/bartlett_dm_hln_results_v1.csv" --verify-only

Packaged evaluation_v1 results already exist. Do not use them as an output
directory. If a fresh analysis is needed, retain the frozen run and choose
a new path, for example --output "$H2_ROOT/full_run_v2/evaluation_replay_001"
instead of --verify-only. This opens test arrays only after all frozen gates.

5. To repeat all nine fits, leave full_run_v2 unchanged. new_run_001 must not
exist; the trainer refuses an existing output root and any output within
inputs/. The protocol fixes 100 epochs, three seeds and all three arms.

python implementation/run_h2_training_v1.py \
  --input-root "$H2_ROOT/inputs" --run-root "$H2_ROOT/new_run_001" \
  --protocol "$H2_ROOT/protocol_locked_v1.json" --device cpu --threads 1 --workers 3

After all nine fits finish, verify that new run, then evaluate into its own
new subdirectory. The two commands are:

python implementation/evaluate_h2_v1.py \
  --input-root "$H2_ROOT/inputs" --run-root "$H2_ROOT/new_run_001" \
  --reference-npz "$H2_ROOT/reference/bartlett_loss_differentials_v1.npz" \
  --reference-csv "$H2_ROOT/reference/bartlett_dm_hln_results_v1.csv" --verify-only

python implementation/evaluate_h2_v1.py \
  --input-root "$H2_ROOT/inputs" --run-root "$H2_ROOT/new_run_001" \
  --reference-npz "$H2_ROOT/reference/bartlett_loss_differentials_v1.npz" \
  --reference-csv "$H2_ROOT/reference/bartlett_dm_hln_results_v1.csv" \
  --output "$H2_ROOT/new_run_001/evaluation_v1"

The frozen implementation extracts only verified model/metric definitions from
the archived Stage 06 source. Do not execute that original pipeline directly:
it contains the original project's top-level setup. CPU/PyTorch versions and
all seeds are recorded; hardware or dependency changes can alter fitted values.
The scientific interpretation and proposed thesis additions are in the Turkish
result report included beside this file. No thesis or original v4 result file
was modified by this supplementary experiment or by this packaging operation.
"""


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_relative(relative):
    relative = str(relative)
    if "\\" in relative or "\x00" in relative or ":" in relative:
        raise ValueError("Platform-ambiguous archive-relative path refused.")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in {"", ".", ".."} for p in path.parts):
        raise ValueError(f"Unsafe archive-relative path: {relative}")
    if any(p in DENIED_PARTS or p.startswith("private_") for p in path.parts):
        raise ValueError(f"Private or excluded path refused: {relative}")
    if path.suffix in {".pyc", ".pyo", ".tmp", ".part"}:
        raise ValueError(f"Temporary or cache file refused: {relative}")
    return path.as_posix()


def regular_child(root, relative):
    relative = safe_relative(relative)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink input refused: {relative}")
    resolved = current.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Required regular file missing: {relative}")
    return resolved


def check_record(path, record):
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Recorded artifact identity differs: {path.name}")


def preparation_file(base, name):
    # The preparation checkout has repo/experiments/h2_v1/. The delivered
    # self-contained hierarchy stores these same exact bytes at its root.
    relative = name if (base / name).is_file() else "repo/experiments/h2_v1/" + name
    return regular_child(base, relative)


def completed_run_gate(base, run_name, evaluation_name):
    """Inspect JSON and hash bytes only; never import or execute experiment code."""
    inputs = base / "inputs"
    run = base / run_name
    lock_path = regular_child(run, "locked_training_manifest.json")
    frozen_path = regular_child(run, "selected_checkpoints_frozen.json")
    lock, frozen = read_json(lock_path), read_json(frozen_path)
    if lock.get("status") != "LOCKED_BEFORE_TRAINING" or lock.get("effective_epochs") != 100:
        raise ValueError("A locked 100-epoch full training run is required.")
    if lock.get("execution_mode") != "H2_EXPLORATORY_FIXED_DESIGN":
        raise ValueError("Pilot runs cannot be packaged as an H2 result.")
    if frozen.get("status") != "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS":
        raise ValueError("All nine fits must finish and freeze before packaging.")
    if frozen.get("locked_training_manifest_sha256") != sha256(lock_path):
        raise ValueError("Frozen run references another training manifest.")
    if frozen.get("test_arrays_loaded") is not False or lock.get("test_arrays_loaded") is not False:
        raise ValueError("Training must certify no test-array loading.")
    identities = lock["input_identities"]
    if set(identities) != EXPECTED_INPUTS or len(identities) != 21:
        raise ValueError("Exactly the 21 frozen input identities are required.")
    for relative, record in identities.items():
        check_record(regular_child(inputs, relative), record)
    protocol_path = preparation_file(base, "protocol_locked_v1.json")
    protocol = read_json(protocol_path)
    if sha256(protocol_path) != lock["protocol_draft_sha256"] or protocol != lock["protocol_draft"]:
        raise ValueError("The standalone locked protocol differs from the pretraining lock.")
    training_script = regular_child(base, "implementation/run_h2_training_v1.py")
    evaluation_script = regular_child(base, "implementation/evaluate_h2_v1.py")
    if sha256(training_script) != lock["training_implementation_sha256"]:
        raise ValueError("Training implementation changed after the pretraining lock.")
    if sha256(training_script) != protocol["analysis"]["training_implementation_sha256"]:
        raise ValueError("Protocol training implementation identity differs.")
    if sha256(evaluation_script) != protocol["analysis"]["evaluation_implementation_sha256"]:
        raise ValueError("Evaluation implementation changed after the protocol lock.")
    records = frozen.get("completed_runs", [])
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != 9 or {(r["arm"]["id"], r["seed"]) for r in records} != expected:
        raise ValueError("Frozen completion must contain all nine unique arm/seed fits.")
    for record in records:
        if record.get("status") != "SUCCESS_NO_TEST_ACCESS" or record.get("epochs_ran") != 100:
            raise ValueError("An arm/seed fit is incomplete.")
        for key in ("best_checkpoint", "final_checkpoint", "history"):
            item = record[key]
            if sha256(regular_child(run, item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen {key} identity changed.")
    status = read_json(regular_child(run, "status.json"))
    if status.get("status") != "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS" or status.get("completed") != 9:
        raise ValueError("Training status does not certify nine completed fits.")
    evaluation = regular_child(run, safe_relative(evaluation_name) + "/h2_evaluation_summary_v1.json")
    summary = read_json(evaluation)
    if summary.get("status") != "H2_EVALUATION_COMPLETE_EXPLORATORY":
        raise ValueError("A completed exploratory evaluation is required before packaging.")
    if summary.get("test_arrays_loaded_only_after_gate") is not True or summary.get("old_project_writes") is not False:
        raise ValueError("Evaluation provenance gate is incomplete.")
    provenance = summary["provenance"]
    if (provenance["locked_training_manifest_sha256"] != sha256(lock_path)
            or provenance["selected_checkpoints_frozen_sha256"] != sha256(frozen_path)):
        raise ValueError("Evaluation summary references another frozen training run.")
    if summary["evaluation_implementation_sha256"] != sha256(evaluation_script):
        raise ValueError("Evaluation summary references another implementation.")
    if not summary.get("artifacts"):
        raise ValueError("Completed evaluation has no recorded artifacts.")
    for relative, identity in summary["artifacts"].items():
        check_record(regular_child(evaluation.parent, relative), identity)
    return lock, frozen, summary


def collect_sources(base, report, reference_root, run_name, evaluation_name):
    lock, frozen, summary = completed_run_gate(base, run_name, evaluation_name)
    files = {}
    pinned_hashes = {"inputs/" + relative: record["sha256"]
                     for relative, record in lock["input_identities"].items()}
    pinned_hashes.update({
        "implementation/run_h2_training_v1.py": lock["training_implementation_sha256"],
        "implementation/evaluate_h2_v1.py": summary["evaluation_implementation_sha256"],
        "protocol_locked_v1.json": lock["protocol_draft_sha256"],
        run_name + "/locked_training_manifest.json": frozen["locked_training_manifest_sha256"],
        run_name + "/selected_checkpoints_frozen.json": summary["provenance"]["selected_checkpoints_frozen_sha256"],
    })
    for record in frozen["completed_runs"]:
        for key in ("best_checkpoint", "final_checkpoint", "history"):
            pinned_hashes[run_name + "/" + record[key]["path"]] = record[key]["sha256"]
    for relative, record in summary["artifacts"].items():
        pinned_hashes[run_name + "/" + evaluation_name + "/" + relative] = record["sha256"]
    summary_relative = run_name + "/" + evaluation_name + "/h2_evaluation_summary_v1.json"
    pinned_hashes[summary_relative] = sha256(regular_child(base, summary_relative))

    def add(relative, source):
        relative = safe_relative(relative)
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Required packaging file missing or symlinked: {relative}")
        if relative in files:
            raise ValueError(f"Duplicate archive path: {relative}")
        files[relative] = source.resolve()

    for relative in sorted(EXPECTED_INPUTS):
        add("inputs/" + relative, regular_child(base / "inputs", relative))
    for relative in ("implementation/run_h2_training_v1.py", "implementation/evaluate_h2_v1.py", "preflight_report.json"):
        add(relative, regular_child(base, relative))
    for name in ("protocol_draft.json", "protocol_locked_v1.json", "README.md", "requirements_h2_v1.txt"):
        add(name, preparation_file(base, name))
    add("package_h2_v1.py", Path(__file__).resolve())
    add(safe_relative(report.name), report)
    add("rejected_attempt_v1_notice.json", regular_child(base, "rejected_attempt_v1_notice.json"))
    add(run_name + "_console.log", regular_child(base, run_name + "_console.log"))
    statistics = read_json(regular_child(base, run_name + "/" + evaluation_name + "/statistics_self_test_v1.json"))
    if statistics.get("status") != "STATISTICS_SELF_TEST_PASS" or statistics.get("reference_comparisons") != 22:
        raise ValueError("Successful statistics reference self-test is required.")
    for name, key in (("bartlett_loss_differentials_v1.npz", "reference_npz_sha256"),
                      ("bartlett_dm_hln_results_v1.csv", "reference_csv_sha256")):
        pinned_hashes["reference/" + name] = statistics[key]
        add("reference/" + name, regular_child(reference_root, name))
    run = base / run_name
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in completed run refused: {path.name}")
        if path.is_dir():
            continue
        relative = path.relative_to(base).as_posix()
        safe_relative(relative)
        if path.suffix not in ALLOWED_RUN_SUFFIXES:
            raise ValueError(f"Unexpected file type in completed run: {relative}")
        add(relative, path)
    # Preserve the rejected attempt as a pre-test technical record, never as
    # primary results and never together with its invalid checkpoints.
    rejected = base / "full_run_v1"
    if not rejected.is_dir():
        rejected = base / "rejected_attempt_v1"  # Delivered hierarchy on replay.
    rejected_status = read_json(regular_child(rejected, "status.json"))
    if (rejected_status.get("status") != "FAILED_NO_TEST_ACCESS"
            or rejected_status.get("test_arrays_loaded") is not False):
        raise ValueError("The first rejected attempt must certify failure before test access.")
    for relative in ("status.json", "locked_training_manifest.json",
                     "runs/FS_same_width_seed2026/history.csv"):
        add("rejected_attempt_v1/" + relative, regular_child(rejected, relative))
    for arm in ARMS:
        for seed in SEEDS:
            relative = f"runs/{arm}_seed{seed}/run_record.json"
            add("rejected_attempt_v1/" + relative, regular_child(rejected, relative))
    return files, pinned_hashes


def payload_records(files, generated, pinned_hashes):
    records = {}
    if MANIFEST_NAME in files:
        raise ValueError("The manifest filename is reserved.")
    if set(pinned_hashes) - set(files):
        raise ValueError("A frozen artifact is missing from the packaging selection.")
    for relative, source in files.items():
        safe_relative(relative)
        data = source.read_bytes()
        if source.suffix.lower() in TEXT_SUFFIXES and PRIVATE_CONTENT.search(data):
            raise ValueError(f"Private URL or credential-like content refused in: {relative}")
        records[relative] = {"sha256": digest_bytes(data), "size_bytes": len(data)}
        if relative in pinned_hashes and records[relative]["sha256"] != pinned_hashes[relative]:
            raise ValueError(f"Frozen artifact changed after completion gate: {relative}")
    for relative, data in generated.items():
        safe_relative(relative)
        if relative in records or relative == MANIFEST_NAME:
            raise ValueError(f"Duplicate or reserved generated path: {relative}")
        records[relative] = {"sha256": digest_bytes(data), "size_bytes": len(data)}
    return dict(sorted(records.items()))


def archive_info(relative):
    info = zipfile.ZipInfo(ARCHIVE_ROOT + "/" + safe_relative(relative), (2026, 10, 28, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def verify_archive(archive):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or bundle.testzip() is not None:
            raise ValueError("Archive contains duplicate names or a corrupt member.")
        manifest = json.loads(bundle.read(ARCHIVE_ROOT + "/" + MANIFEST_NAME))
        expected = {ARCHIVE_ROOT + "/" + name for name in manifest["files"]}
        if set(names) != expected | {ARCHIVE_ROOT + "/" + MANIFEST_NAME}:
            raise ValueError("Archive members differ from the exact SHA manifest payload.")
        for relative, record in manifest["files"].items():
            safe_relative(relative)
            data = bundle.read(ARCHIVE_ROOT + "/" + relative)
            if len(data) != record["size_bytes"] or digest_bytes(data) != record["sha256"]:
                raise ValueError(f"Archived payload identity failed: {relative}")
    return len(manifest["files"])


def write_archive(destination, files, generated, pinned_hashes=None):
    """Write one new archive; preserve every pre-existing file exactly."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Archive already exists; no overwrite permitted: {destination}")
    records = payload_records(files, generated, pinned_hashes or {})
    manifest = {"format": "H2_V1_ARCHIVE_SHA256_MANIFEST", "archive_root": ARCHIVE_ROOT,
                "scope": "Every payload file, excluding this manifest itself.",
                "source_bytes_preserved": True, "files": records}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary_created = False
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            temporary_created = True
            for relative, record in records.items():
                data = files[relative].read_bytes() if relative in files else generated[relative]
                if len(data) != record["size_bytes"] or digest_bytes(data) != record["sha256"]:
                    raise ValueError(f"Source changed while packaging: {relative}")
                bundle.writestr(archive_info(relative), data)
            bundle.writestr(archive_info(MANIFEST_NAME), manifest_bytes)
        count = verify_archive(temporary)
        # A same-directory hard link publishes atomically and fails if another
        # process created the destination; replace() would silently overwrite.
        os.link(temporary, destination)
        return count
    finally:
        if temporary_created and temporary.exists():
            temporary.unlink()


def self_test():
    """Temporary synthetic bytes only; never inspect or package actual results."""
    with tempfile.TemporaryDirectory(prefix="h2_packager_synthetic_") as folder:
        root = Path(folder)
        one, two = root / "one.txt", root / "two.bin"
        one.write_bytes(b"synthetic text\n")
        two.write_bytes(bytes(range(256)))
        originals = {one: one.read_bytes(), two: two.read_bytes()}
        target = root / "synthetic.zip"
        count = write_archive(target, {"one.txt": one, "inputs/two.bin": two}, {"REPLAY_README.txt": b"synthetic replay\n"})
        if count != 3 or verify_archive(target) != 3:
            raise AssertionError("Synthetic archive verification failed.")
        try:
            write_archive(target, {"one.txt": one}, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Archive overwrite guard failed.")
        for path, data in originals.items():
            if path.read_bytes() != data:
                raise AssertionError("A source file changed during packaging.")
        blocked = root / f"blocked.zip.tmp.{os.getpid()}"
        blocked.write_bytes(b"pre-existing temporary bytes must survive\n")
        try:
            write_archive(root / "blocked.zip", {"one.txt": one}, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Pre-existing temporary path guard failed.")
        if blocked.read_bytes() != b"pre-existing temporary bytes must survive\n":
            raise AssertionError("Pre-existing temporary file was changed.")
        try:
            write_archive(root / "changed.zip", {"one.txt": one}, {}, {"one.txt": "0" * 64})
        except ValueError:
            pass
        else:
            raise AssertionError("Frozen source identity guard failed.")
        for unsafe in ("../escape", "/absolute", "deps/file.py", "private_fetchlogs/a.json", "x/__pycache__/a.pyc"):
            try:
                safe_relative(unsafe)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe path guard failed.")
        if len(EXPECTED_INPUTS) != 21:
            raise AssertionError("Expected frozen input count differs.")
    print(json.dumps({"status": "PACKAGER_SYNTHETIC_SELF_TEST_PASS", "source_bytes_unchanged": True,
                      "manifest_hash_checks": True, "overwrite_and_path_guards": True,
                      "actual_experiment_packaged": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--delivery-dir", type=Path)
    parser.add_argument("--evaluation-name", default="evaluation_v1")
    parser.add_argument("--run-name", default="full_run_v2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.report is None or args.reference_root is None:
        parser.error("--report and --reference-root are required for real packaging.")
    base = args.base_root.expanduser().resolve()
    run_name = safe_relative(args.run_name)
    if len(PurePosixPath(run_name).parts) != 1 or not run_name.startswith("full_run_v"):
        parser.error("--run-name must be one dedicated full_run_vN directory name.")
    report = args.report.expanduser().resolve()
    reference = args.reference_root.expanduser().resolve()
    delivery = (args.delivery_dir or base / "delivery").expanduser().resolve()
    if (not delivery.is_relative_to(base) or delivery == base or delivery.is_relative_to(base / "inputs")
            or delivery.is_relative_to(base / run_name) or delivery.is_relative_to(base / "full_run_v1")
            or delivery.is_relative_to(base / "implementation") or delivery.is_relative_to(base / "rejected_attempt_v1")):
        raise ValueError("Delivery must be a separate subdirectory of this isolated H2 project.")
    files, pinned_hashes = collect_sources(base, report, reference, run_name, args.evaluation_name)
    replay = REPLAY_README.replace("full_run_v2", run_name)
    rejected_note = ("REJECTED ATTEMPT: PRE-TEST TECHNICAL AUDIT RECORD ONLY\n"
                     "The first attempt failed before any test arrays were opened.\n"
                     "FS_same_width seed2026 history had 99 rows despite the completed fit record; "
                     "its frozen hash and paired batch-order checks failed.\n"
                     "Cause was not established. These records are not primary scientific results.\n"
                     "All nine fits were repeated with the unchanged frozen code and protocol in " + run_name + ".\n")
    generated = {"REPLAY_README.txt": replay.encode("utf-8"),
                 "rejected_attempt_v1/README.txt": rejected_note.encode("utf-8")}
    destination = delivery / ARCHIVE_NAME
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists() or checksum_path.exists():
        raise FileExistsError("A delivery archive/checksum already exists; no overwrite permitted.")
    delivery.mkdir(parents=True, exist_ok=True)
    count = write_archive(destination, files, generated, pinned_hashes)
    checksum = sha256(destination)
    with checksum_path.open("x", encoding="ascii") as handle:
        handle.write(checksum + "  " + destination.name + "\n")
    print(json.dumps({"status": "H2_DELIVERY_ARCHIVE_COMPLETE", "archive": str(destination),
                      "sha256": checksum, "payload_files": count, "source_files_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
