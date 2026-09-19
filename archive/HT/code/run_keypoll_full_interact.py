"""KeyPoll constrained vs full interaction and subsample IMS differences."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from hetero_common import (
    TABLES,
    constrained_formula,
    fit_cluster,
    full_interact_formula,
    load_analysis_panel,
    run_split,
    run_three_layer,
    subsample_formula,
    tidy_focus,
)


def main() -> None:
    data = load_analysis_panel()
    rows = []

    for core in ["IMS", "IMS_D"]:
        table, sample, _ = run_three_layer(data, core, "KeyPollMonUnit", "KeyPoll")
        rows.append(table)
        rows.append(run_split(data, core, "KeyPollMonUnit", "KeyPoll"))

        extra = fit_cluster(sample, full_interact_formula(core, "KeyPollMonUnit", extra_fe=True))
        rows.append(tidy_focus(extra, f"KeyPoll full {core} x KeyPollMonUnit industry-year FE", [core, "KeyPollMonUnit", f"{core}:KeyPollMonUnit"], sample, "full + industry-year FE"))

        lagged = data.dropna(subset=["L_KeyPollMonUnit"])
        table_l, lagged_sample, _ = run_three_layer(lagged, core, "L_KeyPollMonUnit", "KeyPoll lag")
        rows.append(table_l)

        no_penalty = data[data["Penalty_D"].fillna(0).eq(0)]
        table_p, _, _ = run_three_layer(no_penalty, core, "KeyPollMonUnit", "KeyPoll drop penalty years")
        rows.append(table_p)

    for core in ["IMS", "IMS_D"]:
        table, _, _ = run_three_layer(data, core, "Penalty_D", "Penalty contrast")
        rows.append(table)
        rows.append(run_split(data, core, "Penalty_D", "Penalty contrast"))

    output = pd.concat(rows, ignore_index=True)
    output.to_csv(TABLES / "keypoll_full_interact_coefficients.csv", index=False, encoding="utf-8-sig")
    focus = output[output["variable"].str.contains(r"IMS|KeyPoll|Penalty", regex=True)]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
