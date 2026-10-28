# ============================================================
# STAGE 09 — EXECUTE 42 DM / HARVEY / HOLM COMPARISONS
# CONTROLLED SINGLE RUN
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import norm, t

import hashlib
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Canonical paths
# ============================================================

ROOT = Path(
    "/content/drive/MyDrive/tez_transformer_v4_repro"
)

PROTOCOL_JSON = (
    ROOT
    / "config/09_dm_harvey_holm_protocol_lock_v4.json"
)

NAIVE_LOSS_PATH = (
    ROOT
    / "results/baselines/naive/"
    "naive_baseline_loss_series_v4.npz"
)

LEARNED_LOSS_PATH = (
    ROOT
    / "results/baselines/learned/"
    "learned_baseline_loss_series_v4.npz"
)

GARCH_DIAGNOSTICS = (
    ROOT
    / "results/baselines/garch/"
    "garch_baseline_diagnostics_long_v4.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results/statistical_comparison"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "09_dm_harvey_holm_results_v4.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "09_dm_harvey_holm_summary_v4.json"
)

REPORT_MD = (
    OUTPUT_DIR
    / "09_dm_harvey_holm_report_v4.md"
)

EXPECTED_PROTOCOL_SHA = (
    "703574062b981e5a4c5187b476c5ce5f"
    "7b1e01688818c58ec6d28b436ffbcec1"
)

EXPECTED_N = 584
ALPHA = 0.05
FORECAST_HORIZON = 1


# ============================================================
# 1. Helper functions
# ============================================================

def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    if path.exists():
        raise FileExistsError(
            "Output already exists. "
            "Do not overwrite it:\n"
            f"{path}"
        )

    temporary = Path(
        str(path) + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(text)

    os.replace(
        temporary,
        path,
    )


def atomic_write_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    if path.exists():
        raise FileExistsError(
            "Output already exists. "
            "Do not overwrite it:\n"
            f"{path}"
        )

    temporary = Path(
        str(path) + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )

    os.replace(
        temporary,
        path,
    )


def dm_hln_test(
    final_loss: np.ndarray,
    baseline_loss: np.ndarray,
    forecast_horizon: int = 1,
) -> dict:
    """
    Locked orientation:
        d_t = baseline_loss_t - final_loss_t

    Positive mean differential:
        final model has lower average loss.
    """

    final_loss = np.asarray(
        final_loss,
        dtype=np.float64,
    ).reshape(-1)

    baseline_loss = np.asarray(
        baseline_loss,
        dtype=np.float64,
    ).reshape(-1)

    require(
        final_loss.shape
        == baseline_loss.shape,
        "Loss-series shape mismatch.",
    )

    require(
        final_loss.shape
        == (EXPECTED_N,),
        (
            "Unexpected loss-series shape: "
            f"{final_loss.shape}"
        ),
    )

    require(
        np.isfinite(final_loss).all(),
        "Final loss contains non-finite values.",
    )

    require(
        np.isfinite(baseline_loss).all(),
        "Baseline loss contains non-finite values.",
    )

    differential = (
        baseline_loss
        - final_loss
    )

    n = differential.size
    mean_d = float(
        np.mean(differential)
    )

    centered = (
        differential
        - mean_d
    )

    # DM lag-0 autocovariance:
    # gamma_0 = (1/T) * sum((d_t - mean(d))^2)
    gamma_0 = float(
        np.mean(
            centered ** 2
        )
    )

    all_zero = bool(
        np.all(differential == 0.0)
    )

    constant_differential = bool(
        np.all(
            differential
            == differential[0]
        )
    )

    if all_zero:
        dm_raw = 0.0
        p_raw_normal = 1.0
        dm_hln = 0.0
        p_hln_t = 1.0
        standard_error = 0.0
        variance_policy_case = (
            "ALL_DIFFERENTIALS_ZERO"
        )

    elif (
        constant_differential
        or gamma_0 <= 0.0
        or not np.isfinite(gamma_0)
    ):
        raise RuntimeError(
            "Nonzero mean with zero or invalid "
            "loss-differential variance. "
            "Locked protocol forbids fabricating "
            "a p-value."
        )

    else:
        standard_error = float(
            np.sqrt(
                gamma_0 / n
            )
        )

        dm_raw = float(
            mean_d
            / standard_error
        )

        p_raw_normal = float(
            2.0
            * norm.sf(
                abs(dm_raw)
            )
        )

        h = forecast_horizon

        hln_factor = float(
            np.sqrt(
                (
                    n
                    + 1
                    - 2 * h
                    + (
                        h
                        * (h - 1)
                        / n
                    )
                )
                / n
            )
        )

        dm_hln = float(
            hln_factor
            * dm_raw
        )

        p_hln_t = float(
            2.0
            * t.sf(
                abs(dm_hln),
                df=n - 1,
            )
        )

        variance_policy_case = (
            "STANDARD_POSITIVE_VARIANCE"
        )

    h = forecast_horizon

    hln_factor = float(
        np.sqrt(
            (
                n
                + 1
                - 2 * h
                + (
                    h
                    * (h - 1)
                    / n
                )
            )
            / n
        )
    )

    return {
        "n": int(n),
        "mean_final_loss": float(
            np.mean(final_loss)
        ),
        "mean_baseline_loss": float(
            np.mean(baseline_loss)
        ),
        "mean_loss_differential": mean_d,
        "median_loss_differential": float(
            np.median(differential)
        ),
        "loss_differential_gamma0": gamma_0,
        "dm_standard_error": standard_error,
        "dm_raw_statistic": dm_raw,
        "dm_raw_p_normal_two_sided": (
            p_raw_normal
        ),
        "hln_correction_factor": hln_factor,
        "dm_hln_statistic": dm_hln,
        "dm_hln_p_t_two_sided": p_hln_t,
        "variance_policy_case": (
            variance_policy_case
        ),
    }


def holm_adjust(
    frame: pd.DataFrame,
    family_column: str,
    p_column: str,
    alpha: float,
) -> pd.DataFrame:
    adjusted = frame.copy()

    adjusted[
        "holm_rank"
    ] = np.nan

    adjusted[
        "holm_critical_value"
    ] = np.nan

    adjusted[
        "holm_adjusted_p"
    ] = np.nan

    for family in adjusted[
        family_column
    ].unique():
        family_mask = (
            adjusted[family_column]
            == family
        )

        family_indices = adjusted.index[
            family_mask
        ].to_numpy()

        p_values = adjusted.loc[
            family_indices,
            p_column,
        ].to_numpy(
            dtype=float
        )

        require(
            np.isfinite(
                p_values
            ).all(),
            (
                "Non-finite Harvey p-value "
                f"in Holm family {family}."
            ),
        )

        m = len(p_values)

        order = np.argsort(
            p_values,
            kind="mergesort",
        )

        ordered_indices = (
            family_indices[order]
        )

        ordered_p = (
            p_values[order]
        )

        ranks = np.arange(
            1,
            m + 1,
            dtype=int,
        )

        raw_adjusted = (
            (m - ranks + 1)
            * ordered_p
        )

        monotone_adjusted = (
            np.maximum.accumulate(
                raw_adjusted
            )
        )

        monotone_adjusted = (
            np.minimum(
                monotone_adjusted,
                1.0,
            )
        )

        critical_values = (
            alpha
            / (m - ranks + 1)
        )

        adjusted.loc[
            ordered_indices,
            "holm_rank",
        ] = ranks

        adjusted.loc[
            ordered_indices,
            "holm_critical_value",
        ] = critical_values

        adjusted.loc[
            ordered_indices,
            "holm_adjusted_p",
        ] = monotone_adjusted

    adjusted[
        "holm_rank"
    ] = adjusted[
        "holm_rank"
    ].astype(int)

    return adjusted


# ============================================================
# 2. Pre-execution integrity checks
# ============================================================

print("=" * 120)
print(
    "STAGE 09 — EXECUTE 42 DM / HARVEY / "
    "HOLM COMPARISONS"
)
print("=" * 120)


for path in (
    PROTOCOL_JSON,
    NAIVE_LOSS_PATH,
    LEARNED_LOSS_PATH,
    GARCH_DIAGNOSTICS,
):
    require_file(path)


require(
    sha256_file(PROTOCOL_JSON)
    == EXPECTED_PROTOCOL_SHA,
    "Locked protocol SHA mismatch.",
)


protocol = load_json(
    PROTOCOL_JSON
)


require(
    protocol["status"]
    == "LOCKED_BEFORE_TEST_EXECUTION",
    "Protocol status is not locked.",
)

require(
    protocol["comparison_counts"]["total"]
    == 42,
    "Locked comparison count is not 42.",
)

require(
    protocol["comparison_counts"]["return"]
    == 20,
    "Locked return count is not 20.",
)

require(
    protocol["comparison_counts"]["volatility"]
    == 22,
    "Locked volatility count is not 22.",
)


# Rehash every locked input again immediately before testing.
verified_inputs = 0

for identity in protocol[
    "locked_input_identities"
]:
    path = (
        ROOT
        / identity["path"]
    )

    require_file(path)

    require(
        int(path.stat().st_size)
        == int(identity["size_bytes"]),
        (
            "Locked input size mismatch:\n"
            f"{identity['path']}"
        ),
    )

    require(
        sha256_file(path)
        == identity["sha256"],
        (
            "Locked input SHA mismatch:\n"
            f"{identity['path']}"
        ),
    )

    verified_inputs += 1


# Prevent accidental overwrite.
existing_outputs = [
    path
    for path in (
        RESULTS_CSV,
        SUMMARY_JSON,
        REPORT_MD,
    )
    if path.exists()
]

if existing_outputs:
    raise FileExistsError(
        "One or more Stage 09 outputs already exist. "
        "Do not overwrite them:\n"
        + "\n".join(
            str(path)
            for path in existing_outputs
        )
    )


# ============================================================
# 3. Load locked loss series
# ============================================================

with np.load(
    NAIVE_LOSS_PATH,
    allow_pickle=False,
) as archive:
    naive_losses = {
        key: archive[key].astype(
            np.float64
        )
        for key in archive.files
    }


with np.load(
    LEARNED_LOSS_PATH,
    allow_pickle=False,
) as archive:
    learned_losses = {
        key: archive[key].astype(
            np.float64
        )
        for key in archive.files
    }


garch = pd.read_csv(
    GARCH_DIAGNOSTICS,
    low_memory=False,
)


def get_loss_pair(
    comparison: dict,
):
    task = comparison["task"]
    asset = comparison["asset"]
    family = comparison[
        "baseline_family"
    ]
    baseline_model = comparison[
        "baseline_model"
    ]

    if task == "return":
        final_key = (
            f"final_return_abs__{asset}"
        )
    elif task == "volatility":
        final_key = (
            f"final_vol_pinball__{asset}"
        )
    else:
        raise ValueError(
            f"Unexpected task: {task}"
        )

    require(
        final_key in naive_losses,
        (
            "Missing final loss key: "
            f"{final_key}"
        ),
    )

    final_loss = naive_losses[
        final_key
    ]

    if family == "naive":
        naive_key_map = {
            "ReturnZero": (
                f"return_zero_abs__{asset}"
            ),
            "ReturnPersistence": (
                f"return_persistence_abs__{asset}"
            ),
            "VolPersistence": (
                f"vol_persistence_pinball__{asset}"
            ),
        }

        require(
            baseline_model
            in naive_key_map,
            (
                "Unexpected naive model: "
                f"{baseline_model}"
            ),
        )

        baseline_key = naive_key_map[
            baseline_model
        ]

        require(
            baseline_key in naive_losses,
            (
                "Missing naive loss key: "
                f"{baseline_key}"
            ),
        )

        baseline_loss = naive_losses[
            baseline_key
        ]

    elif family == "learned":
        baseline_key = (
            f"{baseline_model}__"
            f"{task}__{asset}"
        )

        require(
            baseline_key
            in learned_losses,
            (
                "Missing learned loss key: "
                f"{baseline_key}"
            ),
        )

        baseline_loss = learned_losses[
            baseline_key
        ]

    elif family == "garch_family":
        require(
            task == "volatility",
            "GARCH comparison must be volatility.",
        )

        pair = garch[
            garch["model_id"]
            .astype(str)
            .eq(baseline_model)
            & garch["asset"]
            .astype(str)
            .eq(asset)
            & garch["status"]
            .astype(str)
            .eq("OK")
        ].copy()

        pair["anchor_index"] = (
            pd.to_numeric(
                pair["anchor_index"],
                errors="raise",
            ).astype(int)
        )

        pair = pair.sort_values(
            "anchor_index"
        ).reset_index(drop=True)

        require(
            len(pair) == EXPECTED_N,
            (
                "Incomplete GARCH pair: "
                f"{baseline_model} | {asset}"
            ),
        )

        require(
            np.array_equal(
                pair[
                    "anchor_index"
                ].to_numpy(dtype=int),
                np.arange(
                    EXPECTED_N,
                    dtype=int,
                ),
            ),
            (
                "GARCH anchor-index mismatch: "
                f"{baseline_model} | {asset}"
            ),
        )

        baseline_loss = (
            pd.to_numeric(
                pair[
                    "primary_pinball_loss"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

    else:
        raise ValueError(
            (
                "Unexpected baseline family: "
                f"{family}"
            )
        )

    require(
        final_loss.shape
        == (EXPECTED_N,),
        (
            "Final loss shape mismatch: "
            f"{final_key}"
        ),
    )

    require(
        baseline_loss.shape
        == (EXPECTED_N,),
        (
            "Baseline loss shape mismatch: "
            f"{baseline_model} | {asset}"
        ),
    )

    return (
        final_loss,
        baseline_loss,
    )


# ============================================================
# 4. Execute the 42 locked comparisons
# ============================================================

result_rows = []


for comparison in protocol[
    "comparison_inventory"
]:
    final_loss, baseline_loss = (
        get_loss_pair(
            comparison
        )
    )

    test_result = dm_hln_test(
        final_loss=final_loss,
        baseline_loss=baseline_loss,
        forecast_horizon=(
            FORECAST_HORIZON
        ),
    )

    mean_baseline_loss = (
        test_result[
            "mean_baseline_loss"
        ]
    )

    mean_differential = (
        test_result[
            "mean_loss_differential"
        ]
    )

    if mean_baseline_loss != 0.0:
        relative_reduction_pct = float(
            100.0
            * mean_differential
            / mean_baseline_loss
        )
    else:
        relative_reduction_pct = np.nan

    row = {
        "comparison_id": comparison[
            "comparison_id"
        ],
        "task": comparison["task"],
        "asset": comparison["asset"],
        "reference_model": comparison[
            "reference_model"
        ],
        "baseline_family": comparison[
            "baseline_family"
        ],
        "baseline_model": comparison[
            "baseline_model"
        ],
        "loss_function": comparison[
            "loss_function"
        ],
        "holm_family": comparison[
            "holm_family"
        ],
        **test_result,
        "relative_loss_reduction_pct": (
            relative_reduction_pct
        ),
    }

    result_rows.append(row)


results = pd.DataFrame(
    result_rows
)


require(
    len(results) == 42,
    "Execution did not produce 42 rows.",
)

require(
    results[
        "comparison_id"
    ].nunique()
    == 42,
    "Comparison IDs are not unique.",
)

require(
    int(
        (
            results["task"]
            == "return"
        ).sum()
    )
    == 20,
    "Return result count mismatch.",
)

require(
    int(
        (
            results["task"]
            == "volatility"
        ).sum()
    )
    == 22,
    "Volatility result count mismatch.",
)


# ============================================================
# 5. Apply locked Holm–Bonferroni correction
# ============================================================

results = holm_adjust(
    frame=results,
    family_column="holm_family",
    p_column=(
        "dm_hln_p_t_two_sided"
    ),
    alpha=ALPHA,
)


results[
    "null_rejected_holm"
] = (
    results[
        "holm_adjusted_p"
    ]
    < ALPHA
)


def classify_result(row) -> str:
    if not bool(
        row["null_rejected_holm"]
    ):
        return "NOT_SIGNIFICANT"

    if (
        row[
            "mean_loss_differential"
        ]
        > 0
    ):
        return (
            "SIGNIFICANT_FINAL_BETTER"
        )

    if (
        row[
            "mean_loss_differential"
        ]
        < 0
    ):
        return (
            "SIGNIFICANT_BASELINE_BETTER"
        )

    return "SIGNIFICANT_ZERO_DIRECTION"


results[
    "interpretation"
] = results.apply(
    classify_result,
    axis=1,
)


def mean_direction(value: float) -> str:
    if value > 0:
        return "FINAL_LOWER_MEAN_LOSS"

    if value < 0:
        return "BASELINE_LOWER_MEAN_LOSS"

    return "EQUAL_MEAN_LOSS"


results[
    "mean_loss_direction"
] = results[
    "mean_loss_differential"
].map(
    mean_direction
)


require(
    np.isfinite(
        results[
            "holm_adjusted_p"
        ].to_numpy(dtype=float)
    ).all(),
    "Holm adjustment produced non-finite values.",
)


# ============================================================
# 6. Locked interpretation counts
# ============================================================

interpretation_counts = {
    key: int(
        (
            results[
                "interpretation"
            ]
            == key
        ).sum()
    )
    for key in (
        "SIGNIFICANT_FINAL_BETTER",
        "SIGNIFICANT_BASELINE_BETTER",
        "NOT_SIGNIFICANT",
        "SIGNIFICANT_ZERO_DIRECTION",
    )
}


task_counts = {}

for task in (
    "return",
    "volatility",
):
    subset = results[
        results["task"]
        == task
    ]

    task_counts[task] = {
        "comparisons": int(
            len(subset)
        ),
        "significant_final_better": int(
            (
                subset[
                    "interpretation"
                ]
                == (
                    "SIGNIFICANT_FINAL_BETTER"
                )
            ).sum()
        ),
        "significant_baseline_better": int(
            (
                subset[
                    "interpretation"
                ]
                == (
                    "SIGNIFICANT_BASELINE_BETTER"
                )
            ).sum()
        ),
        "not_significant": int(
            (
                subset[
                    "interpretation"
                ]
                == "NOT_SIGNIFICANT"
            ).sum()
        ),
    }


# ============================================================
# 7. Prepare deterministic result outputs
# ============================================================

column_order = [
    "comparison_id",
    "task",
    "asset",
    "reference_model",
    "baseline_family",
    "baseline_model",
    "loss_function",
    "holm_family",
    "n",
    "mean_final_loss",
    "mean_baseline_loss",
    "mean_loss_differential",
    "median_loss_differential",
    "relative_loss_reduction_pct",
    "loss_differential_gamma0",
    "dm_standard_error",
    "dm_raw_statistic",
    "dm_raw_p_normal_two_sided",
    "hln_correction_factor",
    "dm_hln_statistic",
    "dm_hln_p_t_two_sided",
    "holm_rank",
    "holm_critical_value",
    "holm_adjusted_p",
    "null_rejected_holm",
    "mean_loss_direction",
    "interpretation",
    "variance_policy_case",
]

results = results[
    column_order
]


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


atomic_write_csv(
    RESULTS_CSV,
    results,
)

results_csv_sha = sha256_file(
    RESULTS_CSV
)


significant = results[
    results["null_rejected_holm"]
].copy()


report_lines = [
    "# Stage 09 — DM, Harvey ve Holm Sonuç Raporu",
    "",
    "## Çalıştırma Kimliği",
    "",
    f"- Protokol SHA256: `{EXPECTED_PROTOCOL_SHA}`",
    f"- Test gözlemi: `{EXPECTED_N}`",
    f"- Tahmin ufku: `h = {FORECAST_HORIZON}`",
    f"- Anlamlılık düzeyi: `{ALPHA}`",
    "- Kayıp farkı: "
    "`baseline loss - final model loss`",
    "",
    "## Karşılaştırma Sayıları",
    "",
    "- Getiri: 20",
    "- Volatilite: 22",
    "- Toplam: 42",
    "",
    "## Holm Sonuç Özeti",
    "",
    (
        "- Nihai model anlamlı derecede daha iyi: "
        f"{interpretation_counts['SIGNIFICANT_FINAL_BETTER']}"
    ),
    (
        "- Baseline anlamlı derecede daha iyi: "
        f"{interpretation_counts['SIGNIFICANT_BASELINE_BETTER']}"
    ),
    (
        "- Anlamlı fark saptanmadı: "
        f"{interpretation_counts['NOT_SIGNIFICANT']}"
    ),
    "",
    "## Görev Bazında Sonuçlar",
    "",
    (
        "- Getiri — nihai model daha iyi: "
        f"{task_counts['return']['significant_final_better']}; "
        "baseline daha iyi: "
        f"{task_counts['return']['significant_baseline_better']}; "
        "anlamlı değil: "
        f"{task_counts['return']['not_significant']}."
    ),
    (
        "- Volatilite — nihai model daha iyi: "
        f"{task_counts['volatility']['significant_final_better']}; "
        "baseline daha iyi: "
        f"{task_counts['volatility']['significant_baseline_better']}; "
        "anlamlı değil: "
        f"{task_counts['volatility']['not_significant']}."
    ),
    "",
    "## Holm Sonrası Anlamlı Karşılaştırmalar",
    "",
]


if significant.empty:
    report_lines.append(
        "Holm düzeltmesi sonrasında anlamlı "
        "karşılaştırma bulunmamıştır."
    )
else:
    for _, row in significant.iterrows():
        report_lines.append(
            (
                f"- `{row['comparison_id']}`: "
                f"{row['interpretation']}; "
                "ortalama kayıp farkı="
                f"{row['mean_loss_differential']:.12g}; "
                "HLN p="
                f"{row['dm_hln_p_t_two_sided']:.12g}; "
                "Holm p="
                f"{row['holm_adjusted_p']:.12g}."
            )
        )


report_lines.extend(
    [
        "",
        "## Yorumlama Sınırı",
        "",
        (
            "Anlamlı olmayan sonuç, iki modelin "
            "eşdeğer olduğunu kanıtlamaz."
        ),
        (
            "Yön yalnızca p-değerinden değil, "
            "ortalama kayıp farkının işaretinden "
            "belirlenmiştir."
        ),
        "",
        "## Çalıştırma Sınırları",
        "",
        "- Model yeniden eğitilmedi.",
        "- Yeni tahmin üretilmedi.",
        "- Hiperparametre değiştirilmedi.",
        "- Kayıp fonksiyonu değiştirilmedi.",
        "- Eksik USDTRY–GARCH çiftleri test edilmedi.",
        "",
    ]
)


report_text = (
    "\n".join(
        report_lines
    )
    + "\n"
)


atomic_write_text(
    REPORT_MD,
    report_text,
)

report_md_sha = sha256_file(
    REPORT_MD
)


summary = {
    "project_version": "v4",
    "stage": (
        "09_STATISTICAL_MODEL_COMPARISON"
    ),
    "status": "COMPLETED",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "protocol": {
        "path": str(
            PROTOCOL_JSON.relative_to(
                ROOT
            )
        ),
        "sha256": EXPECTED_PROTOCOL_SHA,
        "locked_inputs_verified": (
            verified_inputs
        ),
    },
    "method": {
        "loss_differential": (
            "baseline_loss - final_model_loss"
        ),
        "alternative": "two_sided",
        "forecast_horizon_h": 1,
        "dm_maximum_lag": 0,
        "dm_raw_reference": (
            "standard_normal"
        ),
        "harvey_correction": True,
        "harvey_p_reference": (
            "student_t_df_T_minus_1"
        ),
        "multiple_testing": (
            "Holm-Bonferroni"
        ),
        "alpha": ALPHA,
        "holm_families": {
            "RETURN_PRIMARY": 20,
            "VOLATILITY_PRIMARY": 22,
        },
    },
    "comparison_counts": {
        "return": 20,
        "volatility": 22,
        "total": 42,
    },
    "interpretation_counts": (
        interpretation_counts
    ),
    "task_counts": task_counts,
    "outputs": {
        "results_csv": {
            "path": str(
                RESULTS_CSV.relative_to(
                    ROOT
                )
            ),
            "sha256": results_csv_sha,
        },
        "report_md": {
            "path": str(
                REPORT_MD.relative_to(
                    ROOT
                )
            ),
            "sha256": report_md_sha,
        },
    },
    "scope": {
        "usdtry_removed_from_thesis": False,
        "usdtry_garch_pairs_tested": False,
        "partial_garch_pairs_tested": False,
        "models_refitted": False,
        "new_predictions_generated": False,
    },
}


summary_text = (
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


atomic_write_text(
    SUMMARY_JSON,
    summary_text,
)

summary_json_sha = sha256_file(
    SUMMARY_JSON
)


# ============================================================
# 8. Independent post-write verification
# ============================================================

reloaded_results = pd.read_csv(
    RESULTS_CSV
)

reloaded_summary = load_json(
    SUMMARY_JSON
)


require(
    len(reloaded_results) == 42,
    "Written CSV row count mismatch.",
)

require(
    reloaded_results[
        "comparison_id"
    ].nunique()
    == 42,
    "Written CSV comparison IDs are not unique.",
)

require(
    reloaded_summary["status"]
    == "COMPLETED",
    "Written summary status mismatch.",
)

require(
    reloaded_summary[
        "comparison_counts"
    ]["total"]
    == 42,
    "Written summary comparison count mismatch.",
)

require(
    sha256_file(RESULTS_CSV)
    == results_csv_sha,
    "Results CSV SHA changed after writing.",
)

require(
    sha256_file(REPORT_MD)
    == report_md_sha,
    "Report Markdown SHA changed after writing.",
)

require(
    sha256_file(SUMMARY_JSON)
    == summary_json_sha,
    "Summary JSON SHA changed after writing.",
)


# ============================================================
# 9. Console result
# ============================================================

console_columns = [
    "task",
    "asset",
    "baseline_model",
    "mean_loss_differential",
    "dm_hln_statistic",
    "dm_hln_p_t_two_sided",
    "holm_adjusted_p",
    "interpretation",
]


print("\nAll 42 comparisons:")
print("-" * 120)

with pd.option_context(
    "display.max_rows",
    100,
    "display.max_columns",
    None,
    "display.width",
    220,
):
    print(
        results[
            console_columns
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 120)
print(
    "STAGE 09 DM / HARVEY / HOLM EXECUTION "
    "— COMPLETED"
)
print("=" * 120)

print("\nInterpretation counts:")
print(
    "Significant final better    :",
    interpretation_counts[
        "SIGNIFICANT_FINAL_BETTER"
    ],
)
print(
    "Significant baseline better :",
    interpretation_counts[
        "SIGNIFICANT_BASELINE_BETTER"
    ],
)
print(
    "Not significant             :",
    interpretation_counts[
        "NOT_SIGNIFICANT"
    ],
)

print("\nTask counts:")
print(
    "Return:",
    task_counts["return"],
)
print(
    "Volatility:",
    task_counts["volatility"],
)

print("\nOutput files:")
print(RESULTS_CSV)
print("SHA256:", results_csv_sha)

print()
print(SUMMARY_JSON)
print("SHA256:", summary_json_sha)

print()
print(REPORT_MD)
print("SHA256:", report_md_sha)

print("\nLocked input files verified:", verified_inputs)
print("Models fitted              : NO")
print("Predictions generated      : NO")
print("DM tests performed         : YES — 42")
print("Harvey corrections applied : YES — 42")
print("Holm families corrected    : YES — 2")
print("Files written              : 3")

print("=" * 120)