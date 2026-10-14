"""Re-estimate the gap after adverse substantive sentences leave the GI numerator."""

from __future__ import annotations

import pandas as pd

from common import (
    FIRM_YEAR_COUNTS,
    TABLES,
    TEX,
    coef_row,
    ensure_dirs,
    fit_full,
    industry_year_z,
    load_common,
    pad_code,
)


def cell(coef: float, se: float, stars: str) -> str:
    return f"\\shortstack{{{coef:.4f}{stars} \\\\ ({se:.4f})}}"


def main() -> None:
    ensure_dirs()
    if not FIRM_YEAR_COUNTS.exists():
        raise FileNotFoundError("Run 01_adverse_share.py before 02_gi_plus.py.")
    counts = pd.read_csv(FIRM_YEAR_COUNTS, dtype={"证券代码": str}, encoding="utf-8-sig")
    counts["证券代码"] = counts["证券代码"].map(pad_code)
    counts["时间"] = pd.to_numeric(counts["时间"], errors="coerce")
    data = load_common().merge(counts, on=["证券代码", "时间"], how="left")
    total = data["S"] + data["P"]
    data["GI_layerb"] = data["S"] / total
    data.loc[total.fillna(0).le(0), "GI_layerb"] = pd.NA
    data["GI_plus"] = (data["S"] - data["S_adv"]) / total
    data.loc[total.fillna(0).le(0), "GI_plus"] = pd.NA
    data["GI_plus_z"] = industry_year_z(data, "GI_plus", "GI_plus_z")
    data["Greenwashing_plus"] = data["GI_plus_z"] - data["ESGE_z_行业年度"]
    matched = data["GI_layerb"].notna() & data["GI"].notna()
    gap = (data.loc[matched, "GI_layerb"] - data.loc[matched, "GI"]).abs()
    diagnostic = pd.DataFrame([{
        "共同样本": len(data),
        "匹配到LayerB": int(data["S"].notna().sum()),
        "GI与LayerB相关系数": float(data.loc[matched, "GI"].corr(data.loc[matched, "GI_layerb"])),
        "GI绝对差中位数": float(gap.median()) if len(gap) else None,
        "绝对差大于0.01的比例": float(gap.gt(0.01).mean()) if len(gap) else None,
        "S_adv合计": float(data["S_adv"].sum()),
    }])
    diagnostic.to_csv(TABLES / "gi_plus_match.csv", index=False, encoding="utf-8-sig")

    rows = []
    outcomes = [("基准缺口", "Greenwashing连续_基准"), ("净正面缺口", "Greenwashing_plus")]
    for core in ("IMS", "IMS_D"):
        for label, column in outcomes:
            result, _sample = fit_full(data, column, core)
            rows.append(coef_row(result, label, column, core, ""))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "gi_plus_regression.csv", index=False, encoding="utf-8-sig")

    lookup = {(row["核心变量"], row["模型"]): row for row in rows}
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption{Action-claim gap after removing adverse substantive sentences}",
        "    \\label{tab:gi_plus}",
        "    \\begin{tabular}{lrrrr}",
        "        \\toprule",
        "        & \\multicolumn{2}{c}{Baseline gap} & \\multicolumn{2}{c}{Favorable-action gap} \\\\",
        "        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "        & (1) \\(IMS\\) & (2) \\(IMS_D\\) & (3) \\(IMS\\) & (4) \\(IMS_D\\) \\\\",
        "        \\midrule",
    ]
    order = [("IMS", "基准缺口"), ("IMS_D", "基准缺口"), ("IMS", "净正面缺口"), ("IMS_D", "净正面缺口")]
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
    (TEX / "gi_plus.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(diagnostic.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
