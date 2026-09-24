"""Standard binary entropy balancing (Hainmueller 2012 ATT).

Treatment: SPEC = 1 if the client's audit firm has the largest audit-fee
market share in the client's industry-year.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.linalg import qr

sys.dont_write_bytecode = True
TEST_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TEST_ROOT.parent
GPS_CODE = PROJECT_ROOT / "validation/P1/GPS-IPW/code/gps_ipw_analysis.py"
SPEC_PANEL = TEST_ROOT / "data/combined_regression_panel_15381_spec.csv"
OUTPUT_BINARY = Path(__file__).resolve().parents[1] / "results"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, IMS, TREAT = "Greenwashing连续_基准", "IMS", "SPEC"
TREAT_DEFINITION = "审计收费行业-年度份额第一"


def load_gps_module():
    spec = importlib.util.spec_from_file_location("gps_ipw_analysis", GPS_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载GPS代码: {GPS_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.INPUT = SPEC_PANEL
    return module


def significance(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def ess(weights: pd.Series | np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    return float(values.mean()) if weights is None else float(np.average(values, weights=weights))


def smd(sample: pd.DataFrame, variable: str, group: pd.Series,
        weights: pd.Series | None = None) -> float:
    treated = sample.loc[group, variable]
    comparison = sample.loc[~group, variable]
    wt = None if weights is None else weights.loc[group]
    wc = None if weights is None else weights.loc[~group]
    mt = weighted_mean(treated, wt)
    mc = weighted_mean(comparison, wc)
    vt = treated.var(ddof=1) if wt is None else np.average((treated - mt) ** 2, weights=wt)
    vc = comparison.var(ddof=1) if wc is None else np.average((comparison - mc) ** 2, weights=wc)
    pooled = np.sqrt((vt + vc) / 2)
    return float((mt - mc) / pooled) if pooled > 0 else np.nan


def prepare_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    x = frame.astype(float).to_numpy()
    names = list(frame.columns)
    finite = np.isfinite(x).all(axis=0)
    variable = np.nanstd(x, axis=0) > 1e-12
    keep = finite & variable
    x = x[:, keep]
    names = [name for name, use in zip(names, keep) if use]
    if x.shape[1]:
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        _, r, pivots = qr(x, mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(r)) > 1e-10))
        keep = np.sort(pivots[:rank])
        x = x[:, keep]
        names = [names[i] for i in keep]
    return x, names


def _newton_core(design: np.ndarray, target: np.ndarray, lam0: np.ndarray,
                 maxiter: int) -> tuple[np.ndarray, bool, int]:
    """Damped Newton iterations for the exponential-tilting dual."""
    n, k = design.shape

    def probabilities(lam: np.ndarray) -> np.ndarray:
        log_p = design @ lam
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        return p / p.sum()

    def objective(lam: np.ndarray) -> float:
        log_p = design @ lam
        return float(np.log(np.exp(log_p - np.max(log_p)).sum()) + np.max(log_p) - target @ lam - np.log(n))

    lam = lam0.copy()
    converged = False
    iteration = 0
    for iteration in range(maxiter):
        p = probabilities(lam)
        gradient = p @ design - target
        error = float(np.max(np.abs(gradient)))
        if error < 1e-8:
            converged = True
            break
        centered = design - p @ design
        hessian = (centered * p[:, None]).T @ centered + np.eye(k) * 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian, rcond=1e-10) @ gradient
        norm = np.linalg.norm(step)
        if not np.isfinite(norm):
            break
        if norm > 10:
            step *= 10 / norm
        current = objective(lam)
        accepted = False
        scale = 1.0
        while scale > 1e-8:
            candidate = lam - scale * step
            if np.isfinite(objective(candidate)) and objective(candidate) < current:
                lam = candidate
                accepted = True
                break
            scale *= .5
        if not accepted:
            break
    return lam, converged, iteration + 1


def _continuation_solve(design: np.ndarray, target: np.ndarray,
                        maxiter: int) -> tuple[np.ndarray, bool, int]:
    """Homotopy fallback: move the target from the base mean to the target in steps."""
    start = design.mean(axis=0)
    lam = np.zeros(design.shape[1])
    steps = 8
    while steps <= 64:
        ok = True
        for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
            path_target = (1.0 - t) * start + t * target
            lam, ok, _ = _newton_core(design, path_target, lam, maxiter=100)
            if not ok:
                break
        if ok:
            lam, ok, iterations = _newton_core(design, target, lam, maxiter=maxiter)
            if ok:
                return lam, True, iterations
        steps *= 2
    return lam, False, 0


def solve_entropy_weights(design: np.ndarray, target: np.ndarray | None = None,
                          maxiter: int = 500) -> tuple[np.ndarray, dict]:
    """Solve the KL-minimizing exponential-tilting problem by damped Newton steps."""
    n, k = design.shape
    target = np.zeros(k) if target is None else np.asarray(target, dtype=float)
    if k == 0:
        return np.ones(n), {"success": True, "message": "无约束", "max_moment_error": 0.0,
                            "iterations": 0, "求解方式": "无约束"}

    lam, converged, iterations = _newton_core(design, target, np.zeros(k), maxiter)
    method = "直接Newton迭代"
    if not converged:
        lam, converged, iterations = _continuation_solve(design, target, maxiter)
        method = "同伦延拓Newton迭代"

    log_p = design @ lam
    log_p -= np.max(log_p)
    p = np.exp(log_p)
    p /= p.sum()
    error = float(np.max(np.abs(p @ design - target)))
    weights = n * p
    success = bool(converged and np.isfinite(weights).all() and error < 1e-6)
    return weights, {"success": success,
                     "message": "Newton迭代收敛" if success else "未达到矩约束收敛阈值",
                     "max_moment_error": error, "iterations": iterations, "求解方式": method}


def binary_eb_weights(sample: pd.DataFrame, group: pd.Series,
                      covariates: list[str]) -> tuple[pd.Series, dict]:
    """Hainmueller (2012) EB: reweight SPEC=0 to match SPEC=1."""
    treatment = group.to_numpy(dtype=bool)
    control = ~treatment
    frame = sample[covariates].copy()
    for column in covariates:
        if sample[column].nunique() > 2:
            frame[f"{column}^2"] = sample[column] ** 2
    design, names = prepare_design(frame)
    target = design[treatment].mean(axis=0)
    weights_control, info = solve_entropy_weights(design[control], target=target)
    weights_control *= treatment.sum() / control.sum()
    weights = pd.Series(1.0, index=sample.index, dtype=float)
    weights.loc[control] = weights_control
    return weights, {"规格": "一阶矩+连续变量二阶矩", "收敛": info["success"],
                     "信息": info["message"], "最大矩偏差": info["max_moment_error"],
                     "约束数": len(names), "处理组权重和": float(weights.loc[treatment].sum()),
                     "控制组权重和": float(weights.loc[control].sum()),
                     "处理组样本数": int(treatment.sum()), "控制组样本数": int(control.sum())}


def fit_wls(formula: str, sample: pd.DataFrame, weights: pd.Series):
    valid = np.isfinite(weights.to_numpy()) & (weights.to_numpy() > 0)
    analysis = sample.loc[valid].copy()
    analysis_weights = weights.loc[valid]
    if analysis.empty:
        raise ValueError("熵平衡权重没有可用于回归的正值观测")
    return smf.wls(formula, analysis, weights=analysis_weights).fit(
        cov_type="cluster", cov_kwds={"groups": analysis[FIRM]})


def add_weight_diagnostic(rows: list[dict], method: str, spec_name: str,
                          weight_spec: str, sample: pd.DataFrame, weights: pd.Series,
                          info: dict | None = None) -> None:
    w = weights.dropna()
    rows.append({"方法": method, "第一阶段规格": spec_name, "权重规格": weight_spec,
                 "样本数": len(w), "企业数": sample.loc[w.index, FIRM].nunique(),
                 "均值": w.mean(), "标准差": w.std(), "最小值": w.min(),
                 "P01": w.quantile(.01), "P05": w.quantile(.05), "中位数": w.median(),
                 "P95": w.quantile(.95), "P99": w.quantile(.99), "最大值": w.max(),
                 "最大权重占比": w.max() / w.sum(), "ESS": ess(w), **(info or {})})


def run_binary_eb(gps, spec_columns: dict[str, list[str]], all_lagged: list[str]) -> None:
    data = gps.load_data()
    required = [GW, IMS, TREAT, FIRM, YEAR, INDUSTRY, *all_lagged]
    data[IMS] = pd.to_numeric(data[IMS], errors="coerce")
    data[TREAT] = pd.to_numeric(data[TREAT], errors="coerce")
    sample = data.dropna(subset=required).copy()
    sample[TREAT] = sample[TREAT].astype(int)
    if sample.empty or sample[TREAT].nunique() != 2:
        raise ValueError(f"{TREAT} 必须同时包含0和1")
    n1 = int(sample[TREAT].sum())
    n0 = int(len(sample) - n1)
    print(f"[binary EB] treatment={TREAT}: treated={n1}, control={n0}, N={len(sample)}")

    regressions, balances, diagnostics, audits = [], [], [], []
    treatment = sample[TREAT].eq(1)
    for spec_name, covariates in spec_columns.items():
        weights, info = binary_eb_weights(sample, treatment, covariates)
        audits.append({"方法": "标准二元熵平衡", "处理变量": TREAT,
                       "处理定义": TREAT_DEFINITION,
                       "第一阶段规格": spec_name, **info})
        for variable in covariates:
            balances.extend([
                {"方法": "标准二元熵平衡", "第一阶段规格": spec_name,
                 "处理变量": TREAT, "处理定义": TREAT_DEFINITION,
                 "协变量": variable, "调整方式": "未调整",
                 "SMD": smd(sample, variable, treatment)},
                {"方法": "标准二元熵平衡", "第一阶段规格": spec_name,
                 "处理变量": TREAT, "处理定义": TREAT_DEFINITION,
                 "协变量": variable, "调整方式": "熵平衡后",
                 "SMD": smd(sample, variable, treatment, weights)},
            ])
        add_weight_diagnostic(diagnostics, "标准二元熵平衡", spec_name, "主规格",
                              sample, weights, info)
        for model, regressors in [
            ("B1_二元EB_无结果方程控制", [TREAT]),
        ]:
            formula = f"Q('{GW}') ~ " + " + ".join(
                [f"Q('{x}')" for x in regressors] +
                [f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"])
            result = fit_wls(formula, sample, weights)
            term = f"Q('{regressors[0]}')"
            p_value = float(result.pvalues[term])
            regressions.append({"方法": "标准二元熵平衡", "处理变量": TREAT,
                                 "处理定义": TREAT_DEFINITION,
                                 "第一阶段规格": spec_name, "模型": model,
                                 "变量": regressors[0], "系数": result.params[term],
                                 "标准误": result.bse[term], "t值": result.tvalues[term],
                                 "p值": p_value, "显著性": significance(p_value),
                                 "样本数": int(result.nobs),
                                 "企业数": sample.loc[result.model.data.row_labels, FIRM].nunique(),
                                 "调整R方": result.rsquared_adj,
                                 "有效样本量ESS": ess(weights)})

    OUTPUT_BINARY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(regressions).to_csv(OUTPUT_BINARY / "binary_eb_regressions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balances).to_csv(OUTPUT_BINARY / "binary_eb_covariate_balance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT_BINARY / "binary_eb_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(OUTPUT_BINARY / "binary_eb_moment_audit.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(regressions)[["第一阶段规格", "模型", "系数", "标准误", "p值", "显著性", "样本数", "有效样本量ESS"]].to_string(index=False))
    print(pd.DataFrame(audits)[["第一阶段规格", "收敛", "最大矩偏差", "处理组样本数", "控制组样本数"]].to_string(index=False))


def main() -> None:
    gps = load_gps_module()
    spec_columns = {name: [f"L_{column}" for column in columns]
                    for name, columns in gps.FIRST_STAGE_SPECS.items()}
    all_lagged = sorted({column for columns in spec_columns.values() for column in columns})
    run_binary_eb(gps, spec_columns, all_lagged)


if __name__ == "__main__":
    main()
