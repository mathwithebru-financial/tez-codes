"""Prepare a fresh P1 V2 replay directory from the public repo. No model execution."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RAW_SHA = "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
MAP_SHA = "b11aacffad4e5e2087c3cb68d0b9d136712ffe460f0d13aca57283d396747a91"
EXCLUDED = {
    "2010-02-15": "INSUFFICIENT_INITIAL_FEATURE_HISTORY",
    "2020-07-09": "TRAIN_LABEL_AFTER_FIRST_VALIDATION_INFO_CUTOFF",
    "2022-10-05": "SELECTION_LABEL_AFTER_FIRST_TEST_INFO_CUTOFF",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date_rows(dates):
    """Rebuild the recorded date map from date labels and the original splits."""
    samples = Counter()
    for u in range(30, len(dates)):
        target = dates[u]
        split = "train" if target <= "2020-07-09" else "val" if target <= "2022-10-05" else "test"
        reason = EXCLUDED.get(target, "")
        yield {
            "split": split,
            "original_split_sample_index": samples[split],
            "target_date": target,
            "return_target_start_label": dates[u-1],
            "return_target_end_label": target,
            "vol20_first_return_label": dates[u-19],
            "vol20_last_return_label": target,
            "reference_input_first": dates[u-10],
            "reference_input_last": dates[u-1],
            "proposed_lag1_input_first": dates[u-11],
            "proposed_lag1_input_last": dates[u-2],
            "feature_window_eligible": u-11 >= 20,
            "training_or_selection_eligible": not reason,
            "boundary_diagnostic_only": target == "2022-10-05",
            "proposed_eligible": not reason,
            "exclusion_reason": reason,
            "proposed_horizon_common_rows_from_last_input_to_target": 2,
            "wall_clock_cutoff": "NOT_VERIFIED_OR_SELECTED",
            "first_availability_assumption": "UNVERIFIED",
            "proposal_executed_as_model_experiment": False,
        }
        samples[split] += 1


def prepare(workspace):
    workspace = workspace.resolve()
    if workspace.exists() or workspace.is_relative_to(REPOSITORY):
        raise ValueError("Choose a new directory outside the repository.")
    raw = REPOSITORY / "data/raw/raw_prices.csv"
    if digest(raw) != RAW_SHA:
        raise ValueError("The root raw file hash differs from the recorded input.")
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        dates = [row["Date"] for row in csv.DictReader(handle)]
    rows = list(date_rows(dates))
    counts = Counter(row["split"] for row in rows if row["proposed_eligible"])
    if counts != {"train": 2712, "val": 583, "test": 584}:
        raise ValueError("Unexpected split counts.")
    shutil.copytree(HERE, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    inputs = workspace / "inputs"
    inputs.mkdir()
    snapshots = workspace / "pipeline/source_snapshots"
    snapshots.mkdir()
    sources = [(raw, inputs / raw.name)]
    for name in ("01_rebuild_from_frozen_raw_v4.py", "02_preprocessing_v4.py"):
        sources.append((REPOSITORY / "scripts" / name, snapshots / name))
    records = []
    for source, destination in sources:
        shutil.copyfile(source, destination)
        records.append({
            "source_repository_path": source.relative_to(REPOSITORY).as_posix(),
            "destination_relative_path": destination.relative_to(workspace).as_posix(),
            "destination_sha256": digest(destination),
        })
    manifest = {"origin": "fresh_public_workspace", "status": "PUBLIC_INPUTS_STAGED", "records": records}
    (inputs / "data_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    map_path = workspace / "protocol/proposed_lag1_date_map.csv"
    with map_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    if digest(map_path) != MAP_SHA:
        raise ValueError("Reconstructed date map differs from the original P1 V2 bytes.")
    print(json.dumps({"dates": len(dates), "eligible": counts, "map_sha256": MAP_SHA}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    prepare(parser.parse_args().workspace)
