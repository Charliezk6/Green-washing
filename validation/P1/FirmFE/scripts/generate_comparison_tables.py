"""Compare firm-FE core coefficients against baseline industry+year FE results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = json.loads((ROOT / "config/firm_fe_config.json").read_text(encoding="utf-8"))
CORE = CONFIG["core_variables"]
SPECS = ["(1) 一元回归", "(2) 财务控制", "(3) 财务与治理控制", "(4) 全部控制"]


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(coef: float, se: float, p: float) -> str:
    return f"{coef:.4f}{stars(p)} ({se:.4f})"


def core_rows(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    subset = frame[frame["变量"].isin(CORE)].copy()
    subset["来源"] = source
    return subset


def main() -> None:
    firm = pd.read_csv(ROOT / "outputs/coefficients/firm_fe_coefficients.csv", encoding="utf-8-sig")
    baseline = pd.read_csv(
        (PROJECT_ROOT / CONFIG["baseline_coefficients"]).resolve(),
        encoding="utf-8-sig",
    )
    firm_core = core_rows(firm, "FirmFE")
    base_core = core_rows(baseline, "Baseline")

    rows = []
    for core in CORE:
        for spec in SPECS:
            model = f"{core} {spec}"
            b = base_core[(base_core["模型"] == model) & (base_core["变量"] == core)]
            f = firm_core[(firm_core["模型"] == model) & (firm_core["变量"] == core)]
            if b.empty or f.empty:
                raise ValueError(f"Missing core row for model={model}")
            b_row, f_row = b.iloc[0], f.iloc[0]
            rows.append({
                "核心变量": core,
                "规格": spec,
                "Baseline_系数": b_row["系数"],
                "Baseline_标准误": b_row["标准误"],
                "Baseline_t值": b_row["t值"],
                "Baseline_p值": b_row["p值"],
                "Baseline_显著性": b_row["显著性"],
                "Baseline_样本数": int(b_row["样本数"]),
                "Baseline_调整R方": b_row["调整R方"],
                "Baseline_FE": "行业+年份",
                "FirmFE_系数": f_row["系数"],
                "FirmFE_标准误": f_row["标准误"],
                "FirmFE_t值": f_row["t值"],
                "FirmFE_p值": f_row["p值"],
                "FirmFE_显著性": f_row["显著性"],
                "FirmFE_样本数": int(f_row["样本数"]),
                "FirmFE_调整R方": f_row["调整R方"],
                "FirmFE_FE": "企业+年份",
                "系数差_FirmFE减Baseline": f_row["系数"] - b_row["系数"],
                "Baseline_显示": cell(b_row["系数"], b_row["标准误"], b_row["p值"]),
                "FirmFE_显示": cell(f_row["系数"], f_row["标准误"], f_row["p值"]),
            })

    table = pd.DataFrame(rows)
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "firm_fe_vs_baseline_core.csv", index=False, encoding="utf-8-sig")

    display = table[
        ["核心变量", "规格", "Baseline_显示", "FirmFE_显示", "系数差_FirmFE减Baseline",
         "Baseline_样本数", "FirmFE_样本数", "Baseline_调整R方", "FirmFE_调整R方"]
    ].copy()
    display["系数差_FirmFE减Baseline"] = display["系数差_FirmFE减Baseline"].map(lambda x: f"{x:.4f}")
    display["Baseline_调整R方"] = display["Baseline_调整R方"].map(lambda x: f"{x:.4f}")
    display["FirmFE_调整R方"] = display["FirmFE_调整R方"].map(lambda x: f"{x:.4f}")
    display = display.rename(columns={
        "Baseline_显示": "Baseline（行业+年份FE）",
        "FirmFE_显示": "FirmFE（企业+年份FE）",
        "系数差_FirmFE减Baseline": "系数差",
        "Baseline_样本数": "N_Baseline",
        "FirmFE_样本数": "N_FirmFE",
        "Baseline_调整R方": "AdjR2_Baseline",
        "FirmFE_调整R方": "AdjR2_FirmFE",
    })
    latex = display.to_latex(
        index=False,
        escape=True,
        caption="Firm FE versus baseline industry FE: core coefficients",
        label="tab:firm-fe-vs-baseline",
        column_format="llrrrrrrr",
    )
    (out / "firm_fe_vs_baseline_core.tex").write_text(latex, encoding="utf-8")
    print(f"wrote {out / 'firm_fe_vs_baseline_core.csv'}")


if __name__ == "__main__":
    main()
