"""Replace the Huazheng score with environmental administrative penalties."""

from __future__ import annotations

import pandas as pd

from common import (
    TABLES,
    TEX,
    coef_row,
    ensure_dirs,
    fit_full,
    industry_year_z,
    load_common,
    load_penalty,
)


def cell(coef: float, se: float, stars: str) -> str:
    return f"\\shortstack{{{coef:.4f}{stars} \\\\ ({se:.4f})}}"


def main() -> None:
    ensure_dirs()
    data = load_common()
    penalty = load_penalty()
    data = data.merge(penalty, on=["证券代码", "时间"], how="left")
    for column in ("Penalty_D", "Penalty_Count", "PenaltyAmt", "LnPenaltyAmt"):
        data[column] = data[column].fillna(0.0)
    peer_mean = data.groupby(["时间", "ExpertiseIndustryCode"])["GI"].transform("mean")
    data["GW_Penalty_D"] = ((data["GI"] > peer_mean) & data["Penalty_D"].eq(1)).astype(float)
    data["PenaltyCount_z"] = industry_year_z(data, "Penalty_Count", "PenaltyCount_z")
    data["GW_Penalty"] = data["GI_z_行业年度"] + data["PenaltyCount_z"]
    coverage = pd.DataFrame([{
        "共同样本": len(data),
        "Penalty_D均值": float(data["Penalty_D"].mean()),
        "Penalty_D为1": int(data["Penalty_D"].eq(1).sum()),
        "LnPenaltyAmt非零": int(data["LnPenaltyAmt"].gt(0).sum()),
        "连续处罚缺口非缺失": int(data["GW_Penalty"].notna().sum()),
        "二元处罚缺口均值": float(data["GW_Penalty_D"].mean()),
    }])
    coverage.to_csv(TABLES / "penalty_coverage.csv", index=False, encoding="utf-8-sig")

    rows = []
    outcomes = [("二元处罚缺口", "GW_Penalty_D"), ("连续处罚缺口", "GW_Penalty")]
    for core in ("IMS", "IMS_D"):
        for label, column in outcomes:
            result, _sample = fit_full(data, column, core)
            rows.append(coef_row(result, label, column, core, ""))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "penalty_gap_regression.csv", index=False, encoding="utf-8-sig")

    lookup = {(row["核心变量"], row["模型"]): row for row in rows}
    order = [("IMS", "二元处罚缺口"), ("IMS_D", "二元处罚缺口"), ("IMS", "连续处罚缺口"), ("IMS_D", "连续处罚缺口")]
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption{Action-claim gap using environmental penalties instead of the Huazheng score}",
        "    \\label{tab:penalty_gap}",
        "    \\begin{tabular}{lrrrr}",
        "        \\toprule",
        "        & \\multicolumn{2}{c}{High GI and penalty} & \\multicolumn{2}{c}{GI z-score plus penalty-count z-score} \\\\",
        "        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "        & (1) \\(IMS\\) & (2) \\(IMS_D\\) & (3) \\(IMS\\) & (4) \\(IMS_D\\) \\\\",
        "        \\midrule",
    ]
    coefs = [cell(lookup[key]["系数"], lookup[key]["标准误"], lookup[key]["显著性"]) for key in order]
    ns = [f"{lookup[key]['样本数']:,}" for key in order]
    r2 = [f"{lookup[key]['调整R方']:.4f}" for key in order]
    lines.append("        Specialization & " + " & ".join(coefs) + " \\\\")
    lines.append("        Controls & Yes & Yes & Yes & Yes \\\\")
    lines.append("        Observations & " + " & ".join(ns) + " \\\\")
    lines.append("        Adjusted $R^2$ & " + " & ".join(r2) + " \\\\")
    lines.append("        Year and industry fixed effects & Yes & Yes & Yes & Yes \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    (TEX / "penalty_gap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(coverage.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
