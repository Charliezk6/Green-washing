"""Separate the specialization association into the GI and Huazheng components."""

from __future__ import annotations

import pandas as pd

from common import TABLES, TEX, coef_row, ensure_dirs, fit_full, load_common


def cell(coef: float, se: float, stars: str) -> str:
    return f"\\shortstack{{{coef:.4f}{stars} \\\\ ({se:.4f})}}"


def main() -> None:
    ensure_dirs()
    data = load_common()
    specs = [
        ("声称侧", "GI_z_行业年度", []),
        ("声称侧控制绩效", "GI_z_行业年度", ["ESGE_z_行业年度"]),
        ("绩效侧", "ESGE_z_行业年度", []),
    ]
    rows = []
    for core in ("IMS", "IMS_D"):
        for label, column, extra in specs:
            result, _sample = fit_full(data, column, core, extra)
            rows.append(coef_row(result, label, column, core, "+".join(extra)))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "component_split.csv", index=False, encoding="utf-8-sig")

    lookup = {(row["核心变量"], row["模型"]): row for row in rows}
    headers = [
        ("IMS", "声称侧"),
        ("IMS", "声称侧控制绩效"),
        ("IMS", "绩效侧"),
        ("IMS_D", "声称侧"),
        ("IMS_D", "声称侧控制绩效"),
        ("IMS_D", "绩效侧"),
    ]
    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\caption{Specialization associations with the communication and performance components}",
        "    \\label{tab:component_split}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lrrrrrr}",
        "        \\toprule",
        "        & \\multicolumn{3}{c}{\\(IMS\\)} & \\multicolumn{3}{c}{\\(IMS_D\\)} \\\\",
        "        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "        & (1) \\(GI_z\\) & (2) \\(GI_z\\) controlling \\(ESGE_z\\) & (3) \\(ESGE_z\\) & (4) \\(GI_z\\) & (5) \\(GI_z\\) controlling \\(ESGE_z\\) & (6) \\(ESGE_z\\) \\\\",
        "        \\midrule",
    ]
    coefs = [cell(lookup[key]["系数"], lookup[key]["标准误"], lookup[key]["显著性"]) for key in headers]
    ns = [f"{lookup[key]['样本数']:,}" for key in headers]
    r2 = [f"{lookup[key]['调整R方']:.4f}" for key in headers]
    lines.append("        Specialization & " + " & ".join(coefs) + " \\\\")
    lines.append("        Controls & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    lines.append("        Observations & " + " & ".join(ns) + " \\\\")
    lines.append("        Adjusted $R^2$ & " + " & ".join(r2) + " \\\\")
    lines.append("        Year and industry fixed effects & Yes & Yes & Yes & Yes & Yes & Yes \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table}")
    (TEX / "component_split.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
