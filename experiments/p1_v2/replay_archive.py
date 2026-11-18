"""Replay the P1 V2 archival evidence in an extracted, read-only directory.

Run one mode per process. Only the two explicitly named SHAP modes differentiate
model outputs with respect to inputs. No mode trains, selects or updates a model.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import types
from pathlib import Path, PurePosixPath, PureWindowsPath


HISTORICAL_MANIFEST_SHA = "a07bf37f9bcabba338552e4854178fb9d2d8028a50b33d0954fc684fd8131ab7"
BASE_NAME = "thesis_rebuild_20261113"
RUN_NAME = "p1_v2_mse_mae_20261113_full"
SHAP_NAME = "shap_stability_20261113"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_tree(root):
    """Verify every member and separately pin the original scientific members."""
    old_path = root / "provenance_20261117/MANIFEST.json"
    require(digest(old_path) == HISTORICAL_MANIFEST_SHA, "Historical manifest identity mismatch")
    old, current = read_json(old_path), read_json(root / "MANIFEST.json")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(files == set(current) | {"MANIFEST.json"}, "Unlisted or missing archive member")
    for name, expected in current.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe member: " + name)
        path = root / relative
        require(path.resolve().is_relative_to(root) and digest(path) == expected, "Bad member: " + name)
        if path.suffix.lower() in {".py", ".pyc", ".pyd", ".so"} and name.startswith(BASE_NAME + "/"):
            require(name in old, "Unpinned scientific code: " + name)
    for name, expected in old.items():
        if name == "ONCE_OKU.txt":
            require(digest(root / "provenance_20261117" / name) == expected, "Historical README changed")
        else:
            require(current.get(name) == expected, "Original scientific member differs: " + name)
    return len(current)


class Replay:
    def __init__(self, root, output):
        self.root = root
        self.output = output
        self.base = root / BASE_NAME
        self.run = self.base / "training/runs" / RUN_NAME
        self.shap_dir = self.base / "analysis" / SHAP_NAME
        receipt = read_json(self.run / "training_receipt.json")
        checkpoint = PureWindowsPath(receipt["runs"][0]["best_checkpoint_path"])
        self.old_base = next(p for p in checkpoint.parents if p.name == BASE_NAME)
        self.old_gradient = PureWindowsPath(read_json(self.shap_dir / "environment/environment_receipt.json")["shap_gradient_source_path"])

    def path_string(self, value):
        if not isinstance(value, str):
            return value
        original = PureWindowsPath(value)
        if original == self.old_gradient:
            distribution = importlib.metadata.distribution("shap")
            return str(Path(distribution.locate_file("shap/explainers/_gradient.py")).resolve())
        if original.is_relative_to(self.old_base):
            relative = original.relative_to(self.old_base)
            require(".." not in relative.parts, "Unsafe provenance path")
            local = self.base.joinpath(*relative.parts).resolve()
            require(local.is_relative_to(self.base), "Provenance path escaped extracted root")
            return str(local)
        return value

    def read_rebased(self, path):
        def translate(value):
            if isinstance(value, dict):
                result = {self.path_string(key): translate(item) for key, item in value.items()}
                require(len(result) == len(value), "Ambiguous rebased key")
                return result
            if isinstance(value, list):
                return [translate(item) for item in value]
            return self.path_string(value)
        return translate(read_json(path))

    def read_checkpoint_receipt(self, path):
        # A complete dataset_identity is also embedded inside the .pt file.
        # Keep that record in its original form during equality checks.
        obj = read_json(path)
        if Path(path).name in {"training_receipt.json", "run_manifest.json"}:
            for run in obj.get("runs", [obj]):
                run["best_checkpoint_path"] = self.path_string(run["best_checkpoint_path"])
        return obj

    def module(self, path, replacement=None):
        """Load an already authenticated script; redirect only a literal I/O line."""
        module = types.ModuleType(path.stem)
        module.__file__ = str(path)
        module.PORTABLE_OUTPUT = self.output
        source = path.read_text(encoding="utf-8-sig")
        if replacement:
            before, after = replacement
            require(source.count(before) == 1, "Archive I/O binding differs: " + str(path))
            source = source.replace(before, after)
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module

    def fixed(self):
        folder = self.base / "training/comparison"
        sys.path.insert(0, str(folder))
        compare = self.module(folder / "evaluate_fixed_runs.py")
        compare.read_json = self.read_checkpoint_receipt
        data = compare.source_preflight(self.run)
        return compare, data, compare.six_fixed_runs(data)

    def predict(self, device_name):
        import numpy as np
        import torch
        compare, data, fixed = self.fixed()
        from models import build_selected_model
        states = compare.load_fixed_checkpoint_states(fixed, torch)
        require(device_name != "cuda" or torch.cuda.is_available(), "CUDA was requested and is unavailable")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(device_name)
        x = compare.checked_array("X_test", (584, 10, 8), "float32", data["metadata"])
        truth = compare.checked_array("y_test_raw", (584, 8), "float64", data["metadata"])
        predictions, checks, metrics = {}, {}, {}
        for (arm, seed), entry in states.items():
            model = build_selected_model().to(device)
            model.load_state_dict(entry["state"], strict=True)
            predictions[f"{arm}_seed{seed}"] = compare.predict_raw(model, x, data["scalers"], torch, device)
        for arm in ("mse", "mae"):
            ensemble = compare.raw_seed_ensemble([predictions[f"{arm}_seed{s}"] for s in compare.SEEDS])
            predictions[arm + "_ensemble"] = ensemble
        for name, values in predictions.items():
            original = np.load(self.run / "comparison" / f"predictions_raw_test_{name}.npy", allow_pickle=False)
            delta = float(np.abs(values - original).max())
            same_bytes = (values.dtype == original.dtype and values.shape == original.shape
                          and values.tobytes() == original.tobytes())
            checks[name] = {"bytewise_values_equal": same_bytes, "max_absolute_delta": delta}
            require(np.isfinite(values).all() and (same_bytes if device_name == "cuda" else delta < 0.0002),
                    f"{name} exceeds replay tolerance ({device_name}): {delta}")
            np.save(self.output / f"predictions_raw_test_{name}.npy", values, allow_pickle=False)
            if name.endswith("_ensemble"):
                def primary(pred):
                    _, absolute, _, pinball = compare.errors_and_losses(truth, pred)
                    return [float(absolute[:, :4].mean(0).mean()), float(pinball[:, 4:].mean(0).mean())]
                values_metrics, reference_metrics = primary(values), primary(original)
                require(np.array_equal(np.round(values_metrics, 6), np.round(reference_metrics, 6)), "Published macro losses differ")
                metrics[name] = dict(zip(["return_MAE", "vol_pinball_tau05"], values_metrics))
        return {"device": device_name, "test_dates": len(x), "prediction_checks": checks, "macro": metrics}

    def shap_contract(self):
        sys.path.insert(0, str(self.shap_dir))
        import pilot_vectorized_expected_gradients as pilot
        native_load = pilot.load_module
        def load(name, path):
            module = native_load(name, path)
            if path == self.shap_dir / "run_shap_stability.py":
                module.readj = self.read_rebased
                original_load = module.module_at
                def load_helper(label, target):
                    helper = original_load(label, target)
                    if target == self.base / "training/comparison/evaluate_fixed_runs.py":
                        helper.read_json = self.read_checkpoint_receipt
                    return helper
                module.module_at = load_helper
            return module
        pilot.load_module = load
        payload = pilot.prepare_v3()
        runner = self.module(self.shap_dir / "run_shap_stability_vectorized.py", (
            "output=base.RUN/'shap_stability_20261113'", "output=PORTABLE_OUTPUT/'shap'"))
        decision, proof = runner.validated_decision(payload)
        pilot.check_sources(payload)
        return pilot, runner, payload, decision, proof

    def shap(self, mode):
        import numpy as np
        pilot, runner, payload, decision, proof = self.shap_contract()
        if mode == "check":
            return {"checkpoints": len(payload["base_payload"]["fixed"]["runs"]), "verified_sources": len(payload["sources"]),
                    "verified_v3_pilot_sha256": proof["pilot_result_sha256"], "batch_size": decision["batch_size"]}
        base = payload["base"]
        original = payload["base_payload"]
        if mode == "shap-all":
            runner.execute(payload, False)
            products = sorted((self.output / "shap").glob("*__*__mc*.npz"))
            require(len(products) == 64, "Expected 48 individual and 16 ensemble SHAP conditions")
        else:
            torch, _, states, train, x, _, _ = base.environment_and_data(original)
            metrics = pilot.load_module("replay_metrics", self.shap_dir / "stability_metrics.py")
            arm, seed, mc = "mse", 123, base.MC_SEEDS[0]
            model, initial = base.build_frozen_model(states[(arm, seed)]["state"], torch)
            background = original["plan"]["backgrounds"][0]
            indices = np.asarray(background["indices"], dtype=np.int64)
            values, _, _ = pilot.estimator_values(model, x, train[indices], mc, decision["batch_size"], torch, payload, metrics)
            from train_p1 import state_hash
            require(state_hash(model) == initial and all(p.grad is None and not p.requires_grad for p in model.parameters()),
                    "SHAP changed checkpoint weights or accumulated parameter gradients")
            path = self.output / f'{arm}_seed{seed}__{background["id"]}__mc{mc}.npz'
            base.save_npz(path, {**values, "explained_test_indices": np.asarray(original["plan"]["explained"]["indices"], dtype=np.int64),
                                 "background_train_indices": indices})
            products = [path]
        compared = 0
        for path in products:
            with np.load(path, allow_pickle=False) as current, np.load(self.run / SHAP_NAME / path.name, allow_pickle=False) as ref:
                require(set(current.files) == set(ref.files), "SHAP field names changed")
                for key in ref.files:
                    require(current[key].dtype == ref[key].dtype and np.array_equal(current[key], ref[key]), f"SHAP differs: {path.name}/{key}")
                    compared += 1
        return {"conditions_recomputed": len(products), "arrays_exact": compared, "rows": 128, "integration_draws": 256}

    def tables(self, mode):
        if mode == "subperiods":
            folder = self.base / "analysis/subperiods_20261113"
            sys.path.insert(0, str(folder))
            module = self.module(folder / "analyze_subperiods.py")
            module.OUT = self.output / mode
            original = self.run / "subperiods_20261113"
        else:
            sys.path.insert(0, str(self.shap_dir))
            module = self.module(self.shap_dir / "analyze_saved_shap_stability.py", (
                "RESULTS = args.results_dir.resolve(); OUT = RESULTS/'stability'",
                "RESULTS = args.results_dir.resolve(); OUT = PORTABLE_OUTPUT/'stability'"))
            original = self.run / SHAP_NAME / mode
        module.readj = self.read_rebased
        module.PREFLIGHT = self.output / "preflight.json"
        for arguments in ([], ["--execute-analysis"]):
            sys.argv = [module.__file__, *arguments]
            module.main()
        expected = {p.name: digest(p) for p in original.glob("*.csv")}
        produced = {p.name: digest(p) for p in (self.output / mode).glob("*.csv")}
        require(bool(expected) and expected == produced, "Recomputed tables differ")
        return {"identical_csv_count": len(expected), "csv_sha256": produced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "predict", "subperiods", "stability", "shap-one", "shap-all"])
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside the extracted archive")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Prediction mode only; default CPU")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    require(sys.flags.optimize == 0, "Run without -O to retain all historical validation checks")
    require(args.device is None or args.mode == "predict", "--device applies only to predict")
    root, output = args.archive_root.resolve(strict=True), args.output_dir.resolve()
    require(not root.is_relative_to(output) and not output.is_relative_to(root) and not output.exists(),
            "Choose a fresh output directory outside the input archive")
    files = verify_tree(root)
    output.mkdir(parents=True, exist_ok=False)
    replay = Replay(root, output)
    if args.mode == "predict":
        result = replay.predict(args.device or "cpu")
    elif args.mode in {"subperiods", "stability"}:
        result = replay.tables(args.mode)
    else:
        result = replay.shap(args.mode)
    require(verify_tree(root) == files, "Input archive changed during replay")
    result.update(mode=args.mode, verified_manifest_entries=files, training_performed=False)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "replay_receipt.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
