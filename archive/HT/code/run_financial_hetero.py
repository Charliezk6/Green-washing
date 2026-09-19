"""Financial-side heterogeneity: capex, CIP, and provisions, reported in parallel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from hetero_common import load_analysis_panel, run_split, run_three_layer, TABLES


def main() -> None:
    data = load_analysis_panel()
    rows = []

    specs = [
        ("Capex_TA", None, "Capex intensity"),
        ("HighCapex", "HighCapex", "Capex high/low"),
        ("CIP_TA", None, "CIP intensity"),
        ("HighCIP", "HighCIP", "CIP high/low"),
        ("Provision_D", "Provision_D", "Provision dummy"),
        ("Provision_TA", None, "Provision intensity"),
        ("PPE_TA", None, "PPE robustness"),
    ]
    for core in ["IMS", "IMS_D"]:
        for moderator, splitter, label in specs:
            table, _, _ = run_three_layer(data, core, moderator, label)
            rows.append(table)
            if splitter is not None and core == "IMS":
                rows.append(run_split(data, core, splitter, label))
            if splitter is not None and core == "IMS_D" and moderator == splitter:
                rows.append(run_split(data, core, splitter, label))

        observed_capex = data[data["Capex_observed"].eq(1)]
        table, _, _ = run_three_layer(observed_capex, core, "Capex_TA", "Capex observed subsample")
        rows.append(table)
        observed_cip = data[data["CIP_observed"].eq(1)]
        table, _, _ = run_three_layer(observed_cip, core, "CIP_TA", "CIP observed subsample")
        rows.append(table)
        observed_prov = data[data["Provision_observed"].eq(1)]
        table, _, _ = run_three_layer(observed_prov, core, "Provision_D", "Provision observed subsample")
        rows.append(table)

        lagged = data.dropna(subset=["L_Capex_TA", "L_CIP_TA"])
        table, _, _ = run_three_layer(lagged, core, "L_Capex_TA", "Capex lag")
        rows.append(table)
        table, _, _ = run_three_layer(lagged, core, "L_CIP_TA", "CIP lag")
        rows.append(table)

    output = pd.concat(rows, ignore_index=True)
    output.to_csv(TABLES / "financial_hetero_coefficients.csv", index=False, encoding="utf-8-sig")
    focus = output[output["variable"].str.contains(r"IMS|:")]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
