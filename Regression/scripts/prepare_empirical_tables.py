from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_regression as regression

CONFIG = json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs/paper_tables"
VARIABLES = [regression.module.DEPENDENT, *regression.module.CORE, *regression.module.ALL_CONTROLS]
BINARY = {"IMS_D", "SOE", "Dual", "Big4", "IsChangeFirm"}
LABELS = {regression.module.DEPENDENT: "Greenwashing", "IMS": "IMS", "IMS_D": "IMS_D", "Size": "Size", "Lev": "Lev", "ROA": "ROA", "CFO": "CFO", "Growth": "Growth", "FirmAge": "ListingAge", "SOE": "SOE", "Dual": "Dual", "BoardSize": "BoardSize", "IndependentDirectorRatio": "IndDirector", "Big4": "Big4", "LnAuditFee": "AuditFee", "AuditorTenure": "Tenure", "IsChangeFirm": "AuditorChange"}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> None:
    data = regression.module.load_data()
    required = [regression.module.DEPENDENT, *regression.module.CORE, *regression.module.ALL_CONTROLS, "证券代码", "时间", "ExpertiseIndustryCode"]
    common = data.dropna(subset=required).copy()
    common, _ = regression.module.winsorize_controls(common)
    expected = int(CONFIG["expected_common_sample"])
    if len(common) != expected:
        raise ValueError(f"Expected {expected:,} observations, found {len(common):,}")
    desc = common[VARIABLES].describe(percentiles=[.25, .5, .75]).T[["count", "mean", "std", "25%", "50%", "75%"]]
    desc.index = [LABELS[v] for v in VARIABLES]
    coeff = pd.DataFrame(np.eye(len(VARIABLES)), index=VARIABLES, columns=VARIABLES)
    pvals = pd.DataFrame(np.zeros((len(VARIABLES), len(VARIABLES))), index=VARIABLES, columns=VARIABLES)
    methods = pd.DataFrame("", index=VARIABLES, columns=VARIABLES)
    for i, left in enumerate(VARIABLES):
        for right in VARIABLES[:i]:
            method = "Spearman" if left in BINARY or right in BINARY else "Pearson"
            result = spearmanr(common[left], common[right]) if method == "Spearman" else pearsonr(common[left], common[right])
            coeff.loc[left, right] = coeff.loc[right, left] = result.statistic
            pvals.loc[left, right] = pvals.loc[right, left] = result.pvalue
            methods.loc[left, right] = methods.loc[right, left] = method
    display = pd.DataFrame("", index=VARIABLES, columns=VARIABLES)
    for i, left in enumerate(VARIABLES):
        for j, right in enumerate(VARIABLES):
            if i == j: display.loc[left, right] = "1.000"
            elif i > j: display.loc[left, right] = f"{coeff.loc[left, right]:.3f}{stars(pvals.loc[left, right])}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    common.to_csv(OUTPUT / f"common_sample_{expected}.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT / f"descriptive_statistics_{expected}.csv", encoding="utf-8-sig")
    coeff.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_coefficients.csv", encoding="utf-8-sig")
    pvals.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_pvalues.csv", encoding="utf-8-sig")
    methods.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_methods.csv", encoding="utf-8-sig")
    display.rename(index=LABELS, columns=LABELS).to_csv(OUTPUT / "correlation_display.csv", encoding="utf-8-sig")
    print(f"common_sample={len(common)} firms={common['证券代码'].nunique()}")


if __name__ == "__main__":
    main()
