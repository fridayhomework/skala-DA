from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

if __name__ == "__main__":
    matplotlib.use("Agg")

from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "archive"
OUT_DIR = ROOT / "output" / "day1"
FIG_DIR = OUT_DIR / "figures"

BATCH_FILES = {
    "Batch 1": DATA_DIR / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": DATA_DIR / "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": DATA_DIR / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

COLORS = {"Batch 1": "#1f77b4", "Batch 2": "#ff7f0e", "Batch 3": "#2ca02c"}

EARLY_FEATURES = [
    "qd_10", "qd_100", "qd_delta_100_10", "qd_slope_10_100",
    "ir_10", "ir_100", "ir_delta_100_10", "tavg_100", "tmax_100",
    "chargetime_100", "mean_qd_100", "std_qd_100", "mean_ir_100",
    "mean_tavg_100", "mean_tmax_100", "mean_chargetime_100",
    "first_c_rate", "dq_mean", "dq_min", "dq_max",
    "dq_log_var", "dq_range", "dq_skew", "dq_kurtosis",
]


def _read_ref(f: h5py.File, ref) -> np.ndarray:
    return np.asarray(f[ref]).squeeze()


def _read_string(f: h5py.File, ref) -> str:
    values = np.asarray(f[ref]).astype(np.uint32).ravel()
    return "".join(chr(int(v)) for v in values if v)


def _safe_at(values: np.ndarray, cycle_number: int) -> float:
    idx = cycle_number - 1
    if idx < 0 or idx >= len(values):
        return np.nan
    return float(values[idx])


def _slope(values: np.ndarray, start: int = 10, end: int = 100) -> float:
    stop = min(end, len(values))
    begin = min(start - 1, stop)
    y = np.asarray(values[begin:stop], dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    x = np.arange(begin + 1, stop + 1)[mask]
    return float(np.polyfit(x, y[mask], 1)[0])


def _knee_point(qd: np.ndarray) -> float:
    """Estimate the knee as the maximum downward deviation from a start-end chord."""
    qd = np.asarray(qd, dtype=float)
    valid = np.isfinite(qd)
    if valid.sum() < 180:
        return np.nan
    qd = pd.Series(qd).interpolate(limit_direction="both").rolling(21, center=True, min_periods=1).median().to_numpy()
    start = min(99, len(qd) // 5)
    end = len(qd) - 1
    if end - start < 100:
        return np.nan
    x = np.arange(start, end + 1, dtype=float)
    y = qd[start : end + 1]
    x_norm = (x - x[0]) / (x[-1] - x[0])
    chord = y[0] + (y[-1] - y[0]) * x_norm
    deviation = y - chord
    lo = int(0.15 * len(deviation))
    hi = int(0.92 * len(deviation))
    if hi <= lo:
        return np.nan
    return float(start + lo + np.argmax(deviation[lo:hi]) + 1)


def _parse_first_c_rate(policy: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*C", policy)
    return float(match.group(1)) if match else np.nan


def load_batches() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    rows: list[dict] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    voltage_grids: dict[str, np.ndarray] = {}

    for batch_name, path in BATCH_FILES.items():
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            n_cells = batch["cycle_life"].shape[0]
            for i in range(n_cells):
                cell_id = f"{batch_name.replace(' ', '')}_C{i + 1:02d}"
                cycle_life = float(_read_ref(f, batch["cycle_life"][i, 0]))
                policy = _read_string(f, batch["policy_readable"][i, 0])
                summary = f[batch["summary"][i, 0]]
                arrays = {name: np.asarray(summary[name]).squeeze().astype(float) for name in summary.keys()}
                qd = arrays["QDischarge"]
                ir = arrays["IR"]
                tavg = arrays["Tavg"]
                tmax = arrays["Tmax"]
                chargetime = arrays["chargetime"]

                cycles = f[batch["cycles"][i, 0]]
                qdlin_refs = cycles["Qdlin"]
                dq = np.full(1000, np.nan)
                if qdlin_refs.shape[0] >= 100:
                    q10 = _read_ref(f, qdlin_refs[9, 0]).astype(float).ravel()
                    q100 = _read_ref(f, qdlin_refs[99, 0]).astype(float).ravel()
                    m = min(len(q10), len(q100), 1000)
                    dq[:m] = q100[:m] - q10[:m]

                vdlin = _read_ref(f, batch["Vdlin"][i, 0]).astype(float).ravel()
                if batch_name not in voltage_grids:
                    voltage_grids[batch_name] = vdlin

                finite_dq = dq[np.isfinite(dq)]
                rows.append(
                    {
                        "batch": batch_name,
                        "cell_id": cell_id,
                        "cell_index": i + 1,
                        "cycle_life": cycle_life,
                        "log_cycle_life": np.log(cycle_life) if np.isfinite(cycle_life) else np.nan,
                        "policy": policy,
                        "first_c_rate": _parse_first_c_rate(policy),
                        "n_recorded_cycles": len(qd),
                        "knee_cycle": _knee_point(qd),
                        "qd_10": _safe_at(qd, 10),
                        "qd_100": _safe_at(qd, 100),
                        "qd_delta_100_10": _safe_at(qd, 100) - _safe_at(qd, 10),
                        "qd_slope_10_100": _slope(qd),
                        "ir_10": _safe_at(ir, 10),
                        "ir_100": _safe_at(ir, 100),
                        "ir_delta_100_10": _safe_at(ir, 100) - _safe_at(ir, 10),
                        "tavg_100": _safe_at(tavg, 100),
                        "tmax_100": _safe_at(tmax, 100),
                        "chargetime_100": _safe_at(chargetime, 100),
                        "mean_qd_100": float(np.nanmean(qd[:100])),
                        "std_qd_100": float(np.nanstd(qd[:100], ddof=1)),
                        "mean_ir_100": float(np.nanmean(ir[:100])),
                        "mean_tavg_100": float(np.nanmean(tavg[:100])),
                        "mean_tmax_100": float(np.nanmean(tmax[:100])),
                        "mean_chargetime_100": float(np.nanmean(chargetime[:100])),
                        "dq_mean": float(np.mean(finite_dq)) if finite_dq.size else np.nan,
                        "dq_min": float(np.min(finite_dq)) if finite_dq.size else np.nan,
                        "dq_max": float(np.max(finite_dq)) if finite_dq.size else np.nan,
                        "dq_var": float(np.var(finite_dq)) if finite_dq.size else np.nan,
                        "dq_log_var": float(np.log10(np.var(finite_dq))) if finite_dq.size and np.var(finite_dq) > 0 else np.nan,
                        "dq_range": float(np.ptp(finite_dq)) if finite_dq.size else np.nan,
                        "dq_skew": float(stats.skew(finite_dq, nan_policy="omit")) if finite_dq.size else np.nan,
                        "dq_kurtosis": float(stats.kurtosis(finite_dq, nan_policy="omit")) if finite_dq.size else np.nan,
                    }
                )
                # Keep the cycle-level arrays for data-quality checks and EDA.
                # These signals must share the same cycle axis before they are
                # compared or combined into per-cycle features.
                curves[cell_id] = {
                    "qd": qd,
                    "ir": ir,
                    "tavg": tavg,
                    "tmax": tmax,
                    "chargetime": chargetime,
                    "dq": dq,
                }

    df = pd.DataFrame(rows)
    df["knee_fraction"] = df["knee_cycle"] / df["cycle_life"]
    return df, curves, voltage_grids


def batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for batch, g in df.groupby("batch", sort=False):
        valid = g["cycle_life"].dropna()
        records.append(
            {
                "batch": batch,
                "cells_total": len(g),
                "target_valid": len(valid),
                "target_missing": int(g["cycle_life"].isna().sum()),
                "cycle_life_mean": valid.mean(),
                "cycle_life_median": valid.median(),
                "cycle_life_min": valid.min(),
                "cycle_life_max": valid.max(),
                "short_lt_500_n": int((valid < 500).sum()),
                "short_lt_500_pct": float((valid < 500).mean() * 100),
                "long_gt_1000_n": int((valid > 1000).sum()),
                "long_gt_1000_pct": float((valid > 1000).mean() * 100),
                "knee_median": g["knee_cycle"].median(),
                "knee_fraction_median": g["knee_fraction"].replace([np.inf, -np.inf], np.nan).median(),
            }
        )
    return pd.DataFrame(records)


def feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[np.isfinite(df["cycle_life"])].copy()
    for feature in EARLY_FEATURES:
        pair = valid[[feature, "cycle_life", "log_cycle_life"]].dropna()
        if len(pair) < 4 or pair[feature].nunique() < 2:
            continue
        pearson = stats.pearsonr(pair[feature], pair["log_cycle_life"])
        spearman = stats.spearmanr(pair[feature], pair["cycle_life"])
        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "pearson_vs_log_life": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_vs_life": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "abs_spearman": abs(spearman.statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["cycle_life"])
    result = (
        valid.groupby(["batch", "policy"], as_index=False)
        .agg(
            n=("cycle_life", "size"),
            mean_life=("cycle_life", "mean"),
            std_life=("cycle_life", "std"),
            median_life=("cycle_life", "median"),
            first_c_rate=("first_c_rate", "first"),
        )
        .sort_values(["batch", "mean_life"], ascending=[True, False])
    )
    return result


def within_batch_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch, group in df.groupby("batch", sort=False):
        for feature in EARLY_FEATURES:
            pair = group[[feature, "cycle_life"]].dropna()
            if len(pair) < 4 or pair[feature].nunique() < 2:
                continue
            result = stats.spearmanr(pair[feature], pair["cycle_life"])
            rows.append({"batch": batch, "feature": feature, "n": len(pair), "spearman": result.statistic, "p_value": result.pvalue})
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, curves: dict[str, dict[str, np.ndarray]], voltage_grids: dict[str, np.ndarray], corr: pd.DataFrame, policy: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 12, "axes.labelsize": 9})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    bins = np.arange(300, 2051, 100)
    for ax, batch in zip(axes, BATCH_FILES):
        values = df.loc[df["batch"] == batch, "cycle_life"].dropna()
        ax.hist(values, bins=bins, color=COLORS[batch], alpha=0.82, edgecolor="white")
        ax.axvline(values.median(), color="#202020", linestyle="--", linewidth=1.2, label=f"median {values.median():.0f}")
        ax.set_title(f"{batch} (n={len(values)})")
        ax.set_xlabel("Cycle life")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Cells")
    fig.suptitle("Cycle-life distribution differs strongly by batch", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_cycle_life_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, batch in zip(axes, BATCH_FILES):
        subset = df[df["batch"] == batch]
        for _, row in subset.iterrows():
            qd = curves[row.cell_id]["qd"]
            ax.plot(np.arange(1, len(qd) + 1), qd, color=COLORS[batch], alpha=0.18, linewidth=0.65)
        ax.set_title(batch)
        ax.set_xlabel("Cycle")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Discharge capacity Qd (Ah)")
    fig.suptitle("Capacity fade is nonlinear and cell-specific", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_degradation_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, batch in zip(axes, BATCH_FILES):
        subset = df[(df["batch"] == batch) & df["cycle_life"].notna()].sort_values("cycle_life")
        if subset.empty:
            continue
        picks = [subset.iloc[0], subset.iloc[len(subset) // 2], subset.iloc[-1]]
        for row, label in zip(picks, ["short", "median", "long"]):
            dq = curves[row.cell_id]["dq"]
            v = voltage_grids[batch][: len(dq)]
            m = np.isfinite(dq) & np.isfinite(v)
            ax.plot(v[m], dq[m], linewidth=1.2, label=f"{label}: {row.cycle_life:.0f}")
        ax.invert_xaxis()
        ax.set_title(batch)
        ax.set_xlabel("Voltage (V)")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Delta Q(V): cycle 100 - 10 (Ah)")
    fig.suptitle("Early-cycle voltage-capacity change separates lifetimes", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_delta_q_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for batch in BATCH_FILES:
        g = df[(df["batch"] == batch) & df["cycle_life"].notna() & df["first_c_rate"].notna()]
        jitter = np.linspace(-0.035, 0.035, len(g))
        ax.scatter(g["first_c_rate"] + jitter, g["cycle_life"], s=27, alpha=0.72, color=COLORS[batch], label=batch)
        rho = stats.spearmanr(g["first_c_rate"], g["cycle_life"]).statistic
        ax.text(0.02, 0.95 - 0.07 * list(BATCH_FILES).index(batch), f"{batch}: rho={rho:+.2f}", transform=ax.transAxes, color=COLORS[batch], fontsize=9)
    ax.set_xlabel("First-step charge rate (C)")
    ax.set_ylabel("Cycle life")
    ax.set_title("First C-rate alone does not explain cycle life consistently", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_policy_lifetime.png", bbox_inches="tight")
    plt.close(fig)

    top = corr.head(12).sort_values("spearman_vs_life")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#c44e52" if x < 0 else "#4c72b0" for x in top["spearman_vs_life"]]
    ax.barh(top["feature"], top["spearman_vs_life"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with cycle life")
    ax.set_title("Strongest early-cycle signals", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_feature_correlations.png", bbox_inches="tight")
    plt.close(fig)

    selected = corr.head(8)["feature"].tolist()
    matrix = df[selected].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
    ax.set_yticks(range(len(selected)), selected)
    for i in range(len(selected)):
        for j in range(len(selected)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Multicollinearity among top features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_multicollinearity.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, curves, voltage_grids = load_batches()
    summary = batch_summary(df)
    corr = feature_correlations(df)
    policy = policy_summary(df)
    within_corr = within_batch_correlations(df)
    make_plots(df, curves, voltage_grids, corr, policy)

    df.to_csv(OUT_DIR / "cell_features.csv", index=False)
    summary.to_csv(OUT_DIR / "batch_summary.csv", index=False)
    corr.to_csv(OUT_DIR / "feature_correlations.csv", index=False)
    within_corr.to_csv(OUT_DIR / "within_batch_correlations.csv", index=False)
    policy.to_csv(OUT_DIR / "policy_summary.csv", index=False)

    top_corr = corr.head(8)[["feature", "spearman_vs_life", "pearson_vs_log_life", "n"]].to_dict("records")
    payload = {
        "batch_summary": summary.round(4).to_dict("records"),
        "top_correlations": top_corr,
        "overall_first_c_rate_spearman": float(stats.spearmanr(df["first_c_rate"], df["cycle_life"], nan_policy="omit").statistic),
        "missing_targets": df[df["cycle_life"].isna()][["batch", "cell_id", "cell_index"]].to_dict("records"),
    }
    (OUT_DIR / "insights.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
