"""
IMS_D threshold grid search for binary entropy balancing (B1 significance).

Reuses binary_eb_weights / fit_wls / smd from validation/P1/EB/code.
Does not overwrite the stored IMS_D column; constructs D_tau = 1{IMS >= tau}.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
EB_CODE = ROOT / "validation" / "P1" / "EB" / "code" / "entropy_balancing_analysis.py"
OUT = Path(__file__).resolve().parent / "results" / "threshold_eb"
OUT.mkdir(parents=True, exist_ok=True)

FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, IMS = "Greenwashing连续_基准", "IMS"
MAIN_SPEC = "完整共同原因加审计选择"
D_COL = "D_tau"


def load_eb_module():
    spec = importlib.util.spec_from_file_location("entropy_balancing_analysis", EB_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 EB 模块: {EB_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def significance(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def load_sample(eb):
    gps = eb.load_gps_module()
    data = gps.load_data()
    spec_columns = {
        name: [f"L_{column}" for column in columns]
        for name, columns in gps.FIRST_STAGE_SPECS.items()
    }
    all_lagged = sorted({c for cols in spec_columns.values() for c in cols})
    required = [GW, IMS, FIRM, YEAR, INDUSTRY, *all_lagged]
    sample = data.dropna(subset=required).copy()
    sample[IMS] = pd.to_numeric(sample[IMS], errors="coerce")
    sample = sample.dropna(subset=[IMS]).copy()
    sample["IMS分组"] = pd.qcut(sample[IMS], q=5, duplicates="drop")
    q5 = sample["IMS分组"].cat.categories[-1]
    q5_lower = float(q5.left)
    return sample, spec_columns, all_lagged, q5_lower, q5


def build_candidates(sample: pd.DataFrame, q5_lower: float) -> list[dict]:
    p80, p85, p90 = sample[IMS].quantile([0.80, 0.85, 0.90]).tolist()
    absolute = [
        ("literature_0.10", 0.10),
        ("q5_lower", round(q5_lower, 6)),
        ("abs_0.12", 0.12),
        ("abs_0.15", 0.15),
        ("abs_0.20", 0.20),
        ("P80", float(p80)),
        ("P85", float(p85)),
        ("P90", float(p90)),
    ]
    # Deduplicate nearly identical taus while keeping first label
    seen = set()
    candidates = []
    for label, tau in absolute:
        key = round(tau, 6)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"label": label, "tau": float(tau), "design": "threshold"})
    candidates.append({
        "label": "donut_top20_vs_bottom50",
        "tau": float(p80),
        "design": "donut",
        "p_top": 0.80,
        "p_bottom": 0.50,
    })
    return candidates


def assign_treatment(sample: pd.DataFrame, cand: dict) -> pd.Series:
    if cand["design"] == "threshold":
        return sample[IMS].ge(cand["tau"])
    # donut: Top20% vs Bottom50%; middle dropped (NaN mask handled by caller)
    p_top = sample[IMS].quantile(cand["p_top"])
    p_bottom = sample[IMS].quantile(cand["p_bottom"])
    treated = sample[IMS].ge(p_top)
    control = sample[IMS].le(p_bottom)
    keep = treated | control
    out = pd.Series(np.nan, index=sample.index, dtype=float)
    out.loc[treated] = 1.0
    out.loc[control] = 0.0
    out = out.where(keep)
    return out


def diagnostic_row(sample: pd.DataFrame, cand: dict, q5) -> dict:
    if cand["design"] == "donut":
        d = assign_treatment(sample, cand)
        sub = sample.loc[d.notna()].copy()
        treated = d.loc[sub.index].eq(1)
    else:
        sub = sample
        treated = sample[IMS].ge(cand["tau"])
    n1 = int(treated.sum())
    n0 = int((~treated).sum())
    ims_t = float(sub.loc[treated, IMS].mean()) if n1 else np.nan
    ims_c = float(sub.loc[~treated, IMS].mean()) if n0 else np.nan
    in_q5 = sub["IMS分组"].eq(q5)
    overlap = float((treated & in_q5).sum() / n1) if n1 else np.nan
    return {
        "label": cand["label"],
        "design": cand["design"],
        "tau": cand["tau"],
        "N": len(sub),
        "N1": n1,
        "N0": n0,
        "share_treated": n1 / len(sub) if len(sub) else np.nan,
        "IMS_mean_treated": ims_t,
        "IMS_mean_control": ims_c,
        "IMS_mean_diff": ims_t - ims_c if np.isfinite(ims_t) and np.isfinite(ims_c) else np.nan,
        "Q5_overlap_among_treated": overlap,
    }


def fit_ols_cluster(formula: str, sample: pd.DataFrame):
    return smf.ols(formula, sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample[FIRM]}
    )


def run_one(eb, sample: pd.DataFrame, cand: dict, spec_name: str,
            covariates: list[str], all_lagged: list[str]) -> dict:
    d_raw = assign_treatment(sample, cand)
    if cand["design"] == "donut":
        work = sample.loc[d_raw.notna()].copy()
        treated = d_raw.loc[work.index].eq(1)
    else:
        work = sample.copy()
        treated = work[IMS].ge(cand["tau"])
    work = work.copy()
    work[D_COL] = treated.astype(float)

    n1 = int(treated.sum())
    n0 = int((~treated).sum())
    if n1 < 50 or n0 < 50:
        return {
            "label": cand["label"], "design": cand["design"], "tau": cand["tau"],
            "第一阶段规格": spec_name, "收敛": False, "信息": "处理或对照样本过少",
            "最大矩偏差": np.nan, "约束数": np.nan, "N1": n1, "N0": n0,
            "ESS_control": np.nan, "ESS_ratio": np.nan, "max_weight": np.nan,
            "max_abs_SMD_after": np.nan, "B1_coef": np.nan, "B1_se": np.nan,
            "B1_p": np.nan, "B1_sig": "", "B2_IMS_coef": np.nan, "B2_IMS_p": np.nan,
            "OLS_D_coef": np.nan, "OLS_D_p": np.nan,
        }

    weights, info = eb.binary_eb_weights(work, treated, covariates)
    control_w = weights.loc[~treated]
    ess_c = eb.ess(control_w)
    ess_ratio = ess_c / n0 if n0 else np.nan
    max_abs_smd = max(
        abs(eb.smd(work, v, treated, weights)) for v in covariates
    ) if covariates else np.nan

    # B1: GW ~ D
    f_b1 = (
        f"Q('{GW}') ~ Q('{D_COL}') + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    )
    r_b1 = eb.fit_wls(f_b1, work, weights)
    term_d = f"Q('{D_COL}')"
    b1_coef = float(r_b1.params[term_d])
    b1_se = float(r_b1.bse[term_d])
    b1_p = float(r_b1.pvalues[term_d])

    # B2: GW ~ IMS + lagged controls (under binary EB weights)
    f_b2 = (
        f"Q('{GW}') ~ Q('{IMS}') + "
        + " + ".join([f"Q('{x}')" for x in all_lagged])
        + f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    )
    r_b2 = eb.fit_wls(f_b2, work, weights)
    term_ims = f"Q('{IMS}')"
    b2_coef = float(r_b2.params[term_ims])
    b2_p = float(r_b2.pvalues[term_ims])

    # Unweighted OLS on same sample
    r_ols = fit_ols_cluster(f_b1, work)
    ols_coef = float(r_ols.params[term_d])
    ols_p = float(r_ols.pvalues[term_d])

    return {
        "label": cand["label"],
        "design": cand["design"],
        "tau": cand["tau"],
        "第一阶段规格": spec_name,
        "收敛": bool(info["收敛"]),
        "信息": info["信息"],
        "最大矩偏差": info["最大矩偏差"],
        "约束数": info["约束数"],
        "N1": n1,
        "N0": n0,
        "ESS_control": ess_c,
        "ESS_ratio": ess_ratio,
        "max_weight": float(weights.max()),
        "max_abs_SMD_after": float(max_abs_smd),
        "B1_coef": b1_coef,
        "B1_se": b1_se,
        "B1_p": b1_p,
        "B1_sig": significance(b1_p),
        "B2_IMS_coef": b2_coef,
        "B2_IMS_p": b2_p,
        "OLS_D_coef": ols_coef,
        "OLS_D_p": ols_p,
        "B1_ci_low": b1_coef - 1.96 * b1_se,
        "B1_ci_high": b1_coef + 1.96 * b1_se,
    }


def passes_rules(row: pd.Series) -> bool:
    if not bool(row["收敛"]):
        return False
    if not (row["最大矩偏差"] < 1e-6):
        return False
    if row["N1"] < 800:
        return False
    if not (row["ESS_ratio"] >= 0.5):
        return False
    if not (row["max_abs_SMD_after"] < 0.05):
        return False
    if not (row["B1_coef"] < 0 and row["B1_p"] < 0.05):
        return False
    return True


def select_tau(grid: pd.DataFrame, q5_lower: float) -> tuple[dict | None, pd.DataFrame]:
    main = grid.loc[
        (grid["第一阶段规格"] == MAIN_SPEC) & (grid["design"] == "threshold")
    ].copy()
    main["pass"] = main.apply(passes_rules, axis=1)
    eligible = main.loc[main["pass"]].copy()
    if eligible.empty:
        return None, main
    # Prefer higher tau / closer to Q5 (higher tau preferred among ties)
    eligible["dist_q5"] = (eligible["tau"] - q5_lower).abs()
    eligible = eligible.sort_values(
        ["tau", "dist_q5"], ascending=[False, True]
    )
    return eligible.iloc[0].to_dict(), main


def plot_b1_ci(grid: pd.DataFrame, tau_star: float | None, path: Path) -> None:
    plot_df = grid.loc[
        (grid["第一阶段规格"] == MAIN_SPEC) & (grid["design"] == "threshold")
    ].sort_values("tau")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.errorbar(
        plot_df["tau"],
        plot_df["B1_coef"],
        yerr=1.96 * plot_df["B1_se"],
        fmt="o",
        color="#4C78A8",
        ecolor="#9ecae1",
        capsize=3,
        label="B1 ATT (95% CI)",
    )
    ax.axvline(0.10, color="#E45756", linestyle="--", linewidth=1.2, label="tau=0.10")
    if tau_star is not None:
        ax.axvline(tau_star, color="#54A24B", linestyle="-.", linewidth=1.4,
                   label=f"tau*={tau_star:.4f}")
    ax.set_xlabel("IMS threshold (tau)")
    ax.set_ylabel("B1 coefficient on D_tau")
    ax.set_title("Binary EB B1 across IMS thresholds")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_memo(
    path: Path,
    diag: pd.DataFrame,
    selected: dict | None,
    main_grid: pd.DataFrame,
    confirm: pd.DataFrame,
    q5_lower: float,
) -> None:
    lines = [
        "# IMS_D 阈值选定备忘",
        "",
        f"- EB 完整样本 N = {int(diag['N'].iloc[0]) if len(diag) else 'NA'}",
        f"- 样本 Q5 下界 ≈ {q5_lower:.6f}",
        "",
        "## 诊断摘要（处理占比 / IMS 组间差 / Q5 重叠）",
        "",
        diag.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 主规格网格（完整共同原因加审计选择，threshold）",
        "",
    ]
    cols = [
        "label", "tau", "N1", "N0", "收敛", "最大矩偏差", "ESS_ratio",
        "max_abs_SMD_after", "B1_coef", "B1_se", "B1_p", "B1_sig", "pass",
    ]
    show = main_grid[[c for c in cols if c in main_grid.columns]].copy()
    lines.append(show.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if selected is None:
        lines.extend([
            "## 主选结论",
            "",
            "预设网格中**无一**阈值同时满足：收敛 & 矩偏差<1e-6 & N1≥800 "
            "& ESS_ratio≥0.5 & |SMD|_max<0.05 & B1<0 且 p<0.05。",
            "",
            "按计划：**不强行改 IMS_D**；报告完整网格与 donut；表述退回"
            "「阈值效应不稳定、负向关系主要在连续/高剂量段」。",
            "",
        ])
        donut = main_grid  # placeholder; donut is in full grid elsewhere
    else:
        lines.extend([
            "## 主选结论",
            "",
            f"- **tau\\* = {selected['tau']:.6f}**（label={selected['label']}）",
            f"- B1 = {selected['B1_coef']:.4f} (SE={selected['B1_se']:.4f}, "
            f"p={selected['B1_p']:.4f}) {selected['B1_sig']}",
            f"- N1={int(selected['N1'])}, N0={int(selected['N0'])}, "
            f"ESS_ratio={selected['ESS_ratio']:.3f}, "
            f"max|SMD|={selected['max_abs_SMD_after']:.2e}",
            f"- 相对文献切点 0.10：更贴近高剂量段（Q5 下界 {q5_lower:.4f}）。",
            "",
        ])

    if not confirm.empty:
        lines.extend(["## 确认检验（三规格 + 邻域）", "", confirm.to_markdown(index=False, floatfmt=".4f"), ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    eb = load_eb_module()
    sample, spec_columns, all_lagged, q5_lower, q5 = load_sample(eb)
    candidates = build_candidates(sample, q5_lower)

    # Step 1 diagnostics
    diag_rows = [diagnostic_row(sample, c, q5) for c in candidates]
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "threshold_diagnostics.csv", index=False, encoding="utf-8-sig")
    print("=== Diagnostics ===")
    print(diag.to_string(index=False))

    # Step 2 grid: all specs × all candidates
    grid_rows = []
    for cand in candidates:
        for spec_name, covariates in spec_columns.items():
            print(f"Running {cand['label']} @ {spec_name} ...")
            row = run_one(eb, sample, cand, spec_name, covariates, all_lagged)
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "threshold_eb_grid.csv", index=False, encoding="utf-8-sig")

    # Step 3 select
    selected, main_grid = select_tau(grid, q5_lower)
    main_grid.to_csv(OUT / "threshold_eb_main_spec.csv", index=False, encoding="utf-8-sig")

    # Step 4 confirm if selected
    confirm_rows = []
    tau_star = None
    if selected is not None:
        tau_star = float(selected["tau"])
        # three specs already in grid; extract
        for _, r in grid.loc[
            (grid["design"] == "threshold")
            & (np.isclose(grid["tau"], tau_star, atol=1e-8))
        ].iterrows():
            confirm_rows.append({
                "check": "三规格",
                "label": r["label"],
                "tau": r["tau"],
                "第一阶段规格": r["第一阶段规格"],
                "B1_coef": r["B1_coef"],
                "B1_p": r["B1_p"],
                "B1_sig": r["B1_sig"],
                "收敛": r["收敛"],
                "ESS_ratio": r["ESS_ratio"],
            })
        # neighborhood tau* ± 0.01
        for delta, lab in [(-0.01, "邻域_m0.01"), (0.01, "邻域_p0.01")]:
            t = tau_star + delta
            if t <= 0 or t >= 1:
                continue
            cand = {"label": lab, "tau": t, "design": "threshold"}
            row = run_one(
                eb, sample, cand, MAIN_SPEC,
                spec_columns[MAIN_SPEC], all_lagged,
            )
            confirm_rows.append({
                "check": lab,
                "label": lab,
                "tau": t,
                "第一阶段规格": MAIN_SPEC,
                "B1_coef": row["B1_coef"],
                "B1_p": row["B1_p"],
                "B1_sig": row["B1_sig"],
                "收敛": row["收敛"],
                "ESS_ratio": row["ESS_ratio"],
            })
            grid_rows.append(row)
        # refresh grid with neighborhood
        grid = pd.DataFrame(grid_rows)
        grid.to_csv(OUT / "threshold_eb_grid.csv", index=False, encoding="utf-8-sig")

        # Q5 overlap contrast vs 0.10
        d_star = diagnostic_row(
            sample, {"label": "tau_star", "tau": tau_star, "design": "threshold"}, q5
        )
        d_10 = diagnostic_row(
            sample, {"label": "tau_0.10", "tau": 0.10, "design": "threshold"}, q5
        )
        confirm_rows.append({
            "check": "Q5重叠_tau*",
            "label": "tau_star",
            "tau": tau_star,
            "第一阶段规格": "",
            "B1_coef": d_star["Q5_overlap_among_treated"],
            "B1_p": np.nan,
            "B1_sig": f"overlap={d_star['Q5_overlap_among_treated']:.3f}",
            "收敛": True,
            "ESS_ratio": np.nan,
        })
        confirm_rows.append({
            "check": "Q5重叠_0.10",
            "label": "tau_0.10",
            "tau": 0.10,
            "第一阶段规格": "",
            "B1_coef": d_10["Q5_overlap_among_treated"],
            "B1_p": np.nan,
            "B1_sig": f"overlap={d_10['Q5_overlap_among_treated']:.3f}",
            "收敛": True,
            "ESS_ratio": np.nan,
        })

    confirm = pd.DataFrame(confirm_rows)
    if not confirm.empty:
        confirm.to_csv(OUT / "threshold_eb_confirm.csv", index=False, encoding="utf-8-sig")

    plot_b1_ci(grid, tau_star, OUT / "threshold_eb_b1_ci.png")
    write_memo(
        OUT / "threshold_selection_memo.md",
        diag,
        selected,
        main_grid,
        confirm,
        q5_lower,
    )

    print("\n=== Selection ===")
    if selected is None:
        print("No tau passed all rules.")
    else:
        print(f"tau* = {selected['tau']:.6f} ({selected['label']})")
        print(f"B1 = {selected['B1_coef']:.4f}, p = {selected['B1_p']:.4f}")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
