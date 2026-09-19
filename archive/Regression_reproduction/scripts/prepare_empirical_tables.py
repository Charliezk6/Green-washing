from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
BASELINE_SCRIPT = ROOT / "scripts" / "reproduce_regression.py"
OUTPUT = ROOT / "outputs" / "paper_tables"

spec = importlib.util.spec_from_file_location("reproduce_regression", BASELINE_SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

VARIABLES = [module.DEPENDENT, *module.CORE, *module.ALL_CONTROLS]
BINARY = {"IMS_D", "SOE", "Dual", "Big4", "IsChangeFirm"}
LABELS = {
    module.DEPENDENT: "Greenwashing",
    "IMS": "IMS", "IMS_D": "IMS_D", "Size": "Size", "Lev": "Lev",
    "ROA": "ROA", "CFO": "CFO", "Growth": "Growth", "FirmAgeLog": "FirmAge",
    "SOE": "SOE", "Dual": "Dual", "BoardSize": "BoardSize",
    "IndependentDirectorRatio": "IndDirector", "Big4": "Big4",
    "LnAuditFee": "AuditFee", "AuditorTenure": "Tenure", "IsChangeFirm": "AuditorChange",
}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> None:
    data = module.load_data()
    required = [module.DEPENDENT, *module.CORE, *module.ALL_CONTROLS, "证券代码", "时间", "ExpertiseIndustryCode"]
    common = data.dropna(subset=required).copy()
    if len(common) != 8129:
        raise ValueError(f"Expected 8,129 observations, found {len(common):,}")

    desc = common[VARIABLES].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc[["count", "mean", "std", "25%", "50%", "75%"]]
    desc.index = [LABELS[v] for v in VARIABLES]

    coefficients = pd.DataFrame(np.eye(len(VARIABLES)), index=VARIABLES, columns=VARIABLES)
    pvalues = pd.DataFrame(np.zeros((len(VARIABLES), len(VARIABLES))), index=VARIABLES, columns=VARIABLES)
    methods = pd.DataFrame("", index=VARIABLES, columns=VARIABLES)
    for i, left in enumerate(VARIABLES):
        for j, right in enumerate(VARIABLES[:i]):
            method = "Spearman" if left in BINARY or right in BINARY else "Pearson"
            result = spearmanr(common[left], common[right]) if method == "Spearman" else pearsonr(common[left], common[right])
            coefficients.loc[left, right] = coefficients.loc[right, left] = result.statistic
            pvalues.loc[left, right] = pvalues.loc[right, left] = result.pvalue
            methods.loc[left, right] = methods.loc[right, left] = method

    display = pd.DataFrame("", index=VARIABLES, columns=VARIABLES)
    for i, left in enumerate(VARIABLES):
        for j, right in enumerate(VARIABLES):
            if i == j:
                display.loc[left, right] = "1.000"
            elif i > j:
                display.loc[left, right] = f"{coefficients.loc[left, right]:.3f}{stars(pvalues.loc[left, right])}"
    display.index = [LABELS[v] for v in VARIABLES]
    display.columns = [LABELS[v] for v in VARIABLES]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    common.to_csv(OUTPUT / "common_sample_8129.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT / "descriptive_statistics_8129.csv", encoding="utf-8-sig")
    coefficients.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_coefficients.csv", encoding="utf-8-sig")
    pvalues.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_pvalues.csv", encoding="utf-8-sig")
    methods.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_methods.csv", encoding="utf-8-sig")
    display.to_csv(OUTPUT / "correlation_display.csv", encoding="utf-8-sig")

    print(f"common_sample={len(common)} firms={common['证券代码'].nunique()}")
    print(desc.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Greenwashing-IMS Pearson={coefficients.loc[module.DEPENDENT, 'IMS']:.6f}, p={pvalues.loc[module.DEPENDENT, 'IMS']:.6g}")
    print(f"Greenwashing-IMS_D Spearman={coefficients.loc[module.DEPENDENT, 'IMS_D']:.6f}, p={pvalues.loc[module.DEPENDENT, 'IMS_D']:.6g}")


if __name__ == "__main__":
    main()
