"""Re-run baseline and P1 analyses on the SOE-imputed shadow panel.

Production input files and current paper outputs are left unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parents[1]
OUT = AUDIT / "outputs/soe_impute"
SHADOW_COMBINED = OUT / "combined_panel_soe_imputed.csv"
SHADOW_COMMON = OUT / "common_sample_15381_soe_imputed.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_baseline() -> None:
    sys.path.insert(0, str(ROOT / "Regression/scripts"))
    import reproduce_regression as regression
    import prepare_empirical_tables as prepare
    import generate_paper_tables as paper

    baseline_dir = ensure(OUT / "baseline")
    paper_dir = ensure(OUT / "paper_tables")
    regression.CONFIG["input_panel"] = str(SHADOW_COMBINED)
    regression.CONFIG["output_dir"] = str(baseline_dir)
    regression.main()

    prepare.OUTPUT = paper_dir
    prepare.regression = regression
    prepare.main()

    paper.OUT = paper_dir
    paper.COEF = baseline_dir / "基准回归完整系数.csv"
    paper.DESC = paper_dir / "descriptive_statistics_15381.csv"
    paper.CORR = paper_dir / "correlation_coefficients.csv"
    paper.PVAL = paper_dir / "correlation_pvalues.csv"
    paper.main()
    print("baseline done")


def run_twoway() -> None:
    module = load_module(
        "soe_impute_twoway",
        ROOT / "validation/P1/Two way cluster/scripts/run_robustness_checks.py",
    )
    module.INPUT = SHADOW_COMMON
    module.OUT = ensure(OUT / "Two way cluster")
    module.main()
    tables = load_module(
        "soe_impute_twoway_tables",
        ROOT / "validation/P1/Two way cluster/scripts/generate_robustness_tables.py",
    )
    tables.SOURCE = module.OUT / "coefficients/robustness_coefficients.csv"
    tables.OUT = ensure(module.OUT / "tables")
    tables.main()
    print("two-way cluster done")


def run_firmfe() -> None:
    module = load_module(
        "soe_impute_firmfe",
        ROOT / "validation/P1/FirmFE/scripts/run_firm_fe_regression.py",
    )
    module.ROOT = ensure(OUT / "FirmFE")
    module.CONFIG["input_panel"] = str(SHADOW_COMBINED)
    module.main()
    compare = load_module(
        "soe_impute_firmfe_compare",
        ROOT / "validation/P1/FirmFE/scripts/generate_comparison_tables.py",
    )
    compare.ROOT = module.ROOT
    compare.CONFIG["baseline_coefficients"] = str(OUT / "baseline/基准回归完整系数.csv")
    compare.main()
    print("firm FE done")


def run_loo() -> None:
    module = load_module("soe_impute_loo", ROOT / "validation/P1/LOO/run_loo_analysis.py")
    module.INPUT = SHADOW_COMMON
    module.OUTPUT = ensure(OUT / "LOO")
    module.DERIVED = ensure(OUT / "LOO/derived")
    module.main()
    print("LOO done")


def run_lagged() -> None:
    module = load_module(
        "soe_impute_lagged",
        ROOT / "validation/P1/lagged regression/code/lagged_regression.py",
    )
    module.INPUT = SHADOW_COMMON
    module.RESULTS_DIR = ensure(OUT / "lagged regression")
    module.REGRESSION_OUTPUT = module.RESULTS_DIR / "lagged_ims_regression.csv"
    module.DIAGNOSTICS_OUTPUT = module.RESULTS_DIR / "lagged_ims_sample_diagnostics.csv"
    module.main()
    print("lagged regression done")


def run_switching() -> None:
    module = load_module(
        "soe_impute_switching",
        ROOT / "validation/P1/auditor change/run_switching_analysis.py",
    )
    module.INPUT = SHADOW_COMMON
    module.OUTPUT = ensure(OUT / "auditor change")
    module.main()
    print("switching done")


def run_gps() -> None:
    module = load_module(
        "soe_impute_gps",
        ROOT / "validation/P1/GPS-IPW/code/gps_ipw_analysis.py",
    )
    module.INPUT = SHADOW_COMMON
    module.OUTPUT = ensure(OUT / "GPS-IPW")
    module.main()
    print("GPS-IPW done")
    return module


def run_eb(gps_module) -> None:
    module = load_module(
        "soe_impute_eb",
        ROOT / "validation/P1/EB/code/entropy_balancing_analysis.py",
    )
    module.OUTPUT_BINARY = ensure(OUT / "EB")
    module.run_binary_eb(
        gps_module,
        {name: [f"L_{column}" for column in columns] for name, columns in gps_module.FIRST_STAGE_SPECS.items()},
        sorted({f"L_{column}" for columns in gps_module.FIRST_STAGE_SPECS.values() for column in columns}),
    )
    print("EB done")


def run_psm() -> None:
    module = load_module(
        "soe_impute_psm",
        ROOT / "validation/P1/PSM/code/psm_binary_0.20.py",
    )
    module.OUT = ensure(OUT / "PSM")
    module.NOTES = ensure(OUT / "PSM/notes")
    gps = load_module("soe_impute_gps_for_psm", ROOT / "validation/P1/GPS-IPW/code/gps_ipw_analysis.py")
    gps.INPUT = SHADOW_COMMON
    gps.OUTPUT = ensure(OUT / "GPS-IPW")
    module.load_gps_module = lambda: gps
    original_load = module.load_analysis_sample

    def load_analysis_sample():
        data = gps.load_data()
        required = [module.GW, module.IMS, module.FIRM, module.YEAR, module.INDUSTRY, *module.FULL_LAGS]
        sample = data.dropna(subset=required).copy()
        sample[module.IMS] = pd.to_numeric(sample[module.IMS], errors="coerce")
        sample = sample.dropna(subset=[module.IMS]).copy()
        sample[module.TREAT] = sample[module.IMS].ge(module.BINARY_IMS_THRESHOLD).astype(int)
        if sample[module.TREAT].nunique() != 2:
            raise ValueError("IMS_D must contain both 0 and 1")
        sample = sample.reset_index(drop=True)
        sample["row_id"] = range(len(sample))
        return sample

    module.load_analysis_sample = load_analysis_sample
    module.main()
    print("PSM done")
    del original_load


def run_gps_table() -> None:
    module = load_module(
        "soe_impute_gps_table",
        ROOT / "validation/P1/GPS-IPW/scripts/generate_binary_matching_table.py",
    )
    module.EB_SOURCE = OUT / "EB/binary_eb_regressions.csv"
    module.PSM_SOURCE = OUT / "PSM/tables/outcome_regressions.csv"
    module.OUT = ensure(OUT / "GPS-IPW/tables")
    module.main()
    print("GPS matching table done")


def run_hetero() -> None:
    sys.path.insert(0, str(ROOT / "Regression/scripts"))
    import reproduce_regression as regression

    regression.CONFIG["input_panel"] = str(SHADOW_COMBINED)
    hetero_dir = ROOT / "validation/P1/mechanism_verifiability/code"
    sys.path.insert(0, str(hetero_dir))
    import hetero_common as common

    common.load_data = regression.load_data
    common.winsorize_controls = regression.winsorize_controls
    common.OUT = ensure(OUT / "mechanism_verifiability")
    common.DERIVED = ensure(common.OUT / "derived")
    common.TABLES = ensure(common.OUT / "tables")
    original_derived = ROOT / "validation/P1/mechanism_verifiability/results/derived"
    for name in ["mechanism_firm_year_panel.csv", "financial_crosscheck_panel.csv"]:
        source = original_derived / name
        target = common.DERIVED / name
        if not target.exists():
            target.write_bytes(source.read_bytes())

    lit = load_module("soe_impute_lit_hetero", hetero_dir / "run_lit_hetero.py")
    lit.DERIVED = common.DERIVED
    lit.TABLES = common.TABLES
    lit.load_analysis_panel = common.load_analysis_panel
    lit.run_split = common.run_split
    lit.run_three_layer = common.run_three_layer
    lit.main()
    print("lit hetero done")


def collect_key_results() -> None:
    rows = []

    def add(analysis: str, variable: str, coefficient, std_error, p_value, n, extra=""):
        rows.append(
            {
                "analysis": analysis,
                "variable": variable,
                "coefficient": coefficient,
                "std_error": std_error,
                "p_value": p_value,
                "n": n,
                "note": extra,
            }
        )

    baseline = pd.read_csv(OUT / "baseline/基准回归完整系数.csv", encoding="utf-8-sig")
    for model in ["IMS (4) 全部控制", "IMS_D (4) 全部控制"]:
        for variable in ["IMS", "IMS_D", "SOE"]:
            found = baseline[(baseline["模型"] == model) & (baseline["变量"] == variable)]
            if found.empty:
                continue
            row = found.iloc[0]
            add("baseline " + model, variable, row["系数"], row["标准误"], row["p值"], row["样本数"])

    desc = pd.read_csv(OUT / "paper_tables/descriptive_statistics_15381.csv", encoding="utf-8-sig")
    soe_row = desc[desc.iloc[:, 0].astype(str).eq("SOE")]
    if soe_row.empty and "SOE" in desc.index:
        add("descriptive", "SOE_mean", float(desc.loc["SOE", "mean"]), None, None, 15381)
    elif not soe_row.empty:
        add("descriptive", "SOE_mean", float(soe_row.iloc[0]["mean"]), None, None, 15381)

    twoway = pd.read_csv(OUT / "Two way cluster/coefficients/robustness_coefficients.csv", encoding="utf-8-sig")
    for _, row in twoway[twoway["variable"].isin(["IMS", "IMS_D", "SOE"])].iterrows():
        add("two-way " + row["cluster"], row["variable"], row["coefficient"], row["std_error"], row["p_value"], row["n"])

    firm = pd.read_csv(OUT / "FirmFE/outputs/coefficients/firm_fe_coefficients.csv", encoding="utf-8-sig")
    for model in ["IMS (4) 全部控制", "IMS_D (4) 全部控制"]:
        for variable in ["IMS", "IMS_D", "SOE"]:
            found = firm[(firm["模型"] == model) & (firm["变量"] == variable)]
            if found.empty:
                continue
            row = found.iloc[0]
            add("firm FE " + model, variable, row["系数"], row["标准误"], row["p值"], row["样本数"])

    loo = pd.read_csv(OUT / "LOO/coefficients/loo_regression_coefficients.csv", encoding="utf-8-sig")
    for _, row in loo[loo["variable"].isin(["IMS_LOO", "IMS_LOO_D", "SOE"])].iterrows():
        add("LOO", row["variable"], row["coefficient"], row["std_error"], row["p_value"], row["n"])

    lagged = pd.read_csv(OUT / "lagged regression/lagged_ims_regression.csv", encoding="utf-8-sig")
    for _, row in lagged[lagged["变量"].isin(["IMS", "L_SOE"])].iterrows():
        add("lagged", row["变量"], row["系数"], row["标准误"], row["p值"], row["样本数"])

    gps = pd.read_csv(OUT / "GPS-IPW/gps_ipw_regressions.csv", encoding="utf-8-sig")
    focus = gps[(gps["第一阶段规格"] == "完整共同原因加审计选择") & (gps["权重规格"] == "P1--P99截尾") & (gps["模型"] == "G2_增强GPS-IPW")]
    if not focus.empty:
        row = focus.iloc[0]
        add("GPS G2", "IMS", row["系数"], row["标准误"], row["p值"], row["样本数"])

    eb = pd.read_csv(OUT / "EB/binary_eb_regressions.csv", encoding="utf-8-sig")
    focus = eb[(eb["第一阶段规格"] == "完整共同原因加审计选择") & (eb["模型"] == "B2_二元EB_连续IMS_完整滞后控制")]
    if not focus.empty:
        row = focus.iloc[0]
        add("EB B2", row["变量"], row["系数"], row["标准误"], row["p值"], row["样本数"])

    psm = pd.read_csv(OUT / "PSM/tables/outcome_regressions.csv", encoding="utf-8-sig")
    focus = psm[(psm["规格"] == "PS-Full") & (psm["模型"] == "M2_完整滞后协变量+固定效应")]
    if not focus.empty:
        row = focus.iloc[0]
        add("PSM Full M2", row["解释变量"], row["系数"], row["标准误"], row["p值"], row["样本数"])

    switching = pd.read_csv(OUT / "auditor change/tables/switching_model_coefficients.csv", encoding="utf-8-sig")
    focus = switching[(switching["规格"] == "W1_精简控制") & (switching["估计方法"] == "LPM") & (switching["变量"].str.contains("L_SOE|L_Greenwashing|SwitchInExpert", regex=True))]
    for _, row in focus.iterrows():
        add("switching W1 LPM", row["变量"], row["系数"], row["标准误"], row["p值"], row["样本数"])

    hetero = pd.read_csv(OUT / "mechanism_verifiability/tables/lit_hetero_coefficients.csv", encoding="utf-8-sig")
    soe_models = hetero[hetero["model"].str.contains("State ownership", na=False)]
    for _, row in soe_models[soe_models["variable"].str.contains(r"IMS|SOE", regex=True)].iterrows():
        add("hetero " + row["model"], row["variable"], row["coefficient"], row["std_error"], row["p_value"], row["n"])

    pd.DataFrame(rows).to_csv(OUT / "rerun_key_coefficients.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    if not SHADOW_COMBINED.exists() or not SHADOW_COMMON.exists():
        raise FileNotFoundError("Shadow panels missing; run diagnose_soe_missing.py first")
    log = []
    steps = [
        ("baseline", run_baseline),
        ("twoway", run_twoway),
        ("firmfe", run_firmfe),
        ("loo", run_loo),
        ("lagged", run_lagged),
        ("switching", run_switching),
        ("gps", lambda: run_gps()),
        ("eb", lambda: run_eb(sys.modules["soe_impute_gps"])),
        ("psm", run_psm),
        ("gps_table", run_gps_table),
        ("hetero", run_hetero),
        ("collect", collect_key_results),
    ]
    for name, function in steps:
        print(f"\n=== {name} ===")
        try:
            function()
            log.append({"step": name, "status": "ok"})
        except Exception as exc:
            log.append({"step": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {name}: {type(exc).__name__}: {exc}")
            if name in {"baseline"}:
                raise
    (OUT / "rerun_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
