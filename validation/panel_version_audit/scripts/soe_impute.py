"""SOE reconstruction: same-year HLD aggregation, firm carry, residual coded 0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parents[1]
OUT = AUDIT / "outputs/soe_impute"
HOLDER = ROOT / "appendix/上市公司控制人文件152909828(仅供南京审计大学使用)/HLD_Contrshr.csv"
PANEL = ROOT / "Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv"
COMMON = ROOT / "Regression/outputs/paper_tables/common_sample_15381.csv"
SOE_CODES = {1100, 2000, 2100, 2120}
MAX_CARRY_GAP = 2


def pad_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def parse_s0702b(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "nan", "None", "NaN"}:
        return np.nan
    try:
        return float(int(float(text)))
    except ValueError:
        return np.nan


def soe_from_code(code: object) -> float:
    if pd.isna(code):
        return 0.0
    return 1.0 if int(code) in SOE_CODES else 0.0


def token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"", "nan", "None", "NaN"}:
        return ""
    return text


def is_no_controller(s0701a: object, s0701b: object) -> bool:
    return token(s0701a) == "0" or token(s0701b) == "0"


def load_holder() -> pd.DataFrame:
    holder = pd.read_csv(HOLDER, dtype={"Stkcd": str}, encoding="utf-8-sig", low_memory=False)
    holder["证券代码"] = pad_code(holder["Stkcd"])
    holder["时间"] = pd.to_datetime(holder["Reptdt"], errors="coerce").dt.year.astype("Int64")
    holder["S0702b_code"] = holder["S0702b"].map(parse_s0702b)
    holder["no_controller"] = [
        is_no_controller(a, b) for a, b in zip(holder["S0701a"], holder["S0701b"])
    ]
    holder["file_order"] = np.arange(len(holder))
    return holder.dropna(subset=["证券代码", "时间"]).copy()


def last_write_wins(holder: pd.DataFrame) -> pd.DataFrame:
    last = holder.sort_values("file_order").groupby(["证券代码", "时间"], as_index=False, sort=False).tail(1)
    last = last.rename(columns={"S0702b": "S0702b_last", "S0702b_code": "S0702b_last_code"})
    return last[["证券代码", "时间", "S0702b_last", "S0702b_last_code", "S0701a", "S0701b", "no_controller"]]


def aggregate_firm_year(holder: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = holder.groupby(["证券代码", "时间"], sort=False)
    for (code, year), group in grouped:
        codes = group["S0702b_code"].dropna()
        n_rows = int(len(group))
        n_codes = int(len(codes))
        n_soe = int(codes.isin(list(SOE_CODES)).sum())
        n_nonsoe = n_codes - n_soe
        has_no_controller = bool(group["no_controller"].any())
        if n_soe > 0:
            soe = 1.0
            source = "same_year_code"
        elif n_nonsoe > 0:
            soe = 0.0
            source = "same_year_code"
        elif has_no_controller:
            soe = 0.0
            source = "no_controller"
        else:
            soe = np.nan
            source = "missing"
        rows.append(
            {
                "证券代码": code,
                "时间": int(year),
                "SOE_obs": soe,
                "source_obs": source,
                "n_hld_rows": n_rows,
                "n_hld_codes": int(len(codes)),
                "n_hld_soe_codes": n_soe,
                "has_no_controller": has_no_controller,
            }
        )
    return pd.DataFrame(rows)


def firm_carry(observed: pd.DataFrame, max_gap: int = MAX_CARRY_GAP) -> pd.DataFrame:
    frame = observed.sort_values(["证券代码", "时间"]).copy()
    carried = []
    sources = []
    for _, group in frame.groupby("证券代码", sort=False):
        soe = group["SOE_obs"].to_numpy(dtype=float)
        source = group["source_obs"].tolist()
        years = group["时间"].to_numpy()
        known = ~np.isnan(soe)
        if known.any():
            unique = np.unique(soe[known])
            if len(unique) == 1:
                constant = float(unique[0])
                known_years = years[known]
                for i, year in enumerate(years):
                    if known[i]:
                        continue
                    gap = np.abs(known_years - year).min()
                    if gap <= max_gap:
                        soe[i] = constant
                        source[i] = "firm_carry"
        carried.append(pd.Series(soe, index=group.index))
        sources.extend(source)
    frame["SOE_carried"] = pd.concat(carried)
    frame["source_carried"] = sources
    return frame


def attach_to_panel(panel: pd.DataFrame, carried: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["证券代码"] = pad_code(data["证券代码"])
    data["时间"] = pd.to_numeric(data["时间"], errors="coerce").astype("Int64")
    data["S0702b_code_panel"] = data["S0702b"].map(parse_s0702b)
    data["SOE_fill0"] = data["S0702b_code_panel"].map(soe_from_code)
    lookup = carried.set_index(["证券代码", "时间"])
    keys = pd.MultiIndex.from_frame(data[["证券代码", "时间"]])
    data["SOE_obs"] = keys.map(lookup["SOE_obs"])
    data["source_obs"] = keys.map(lookup["source_obs"]).fillna("missing")
    data["SOE_carried"] = keys.map(lookup["SOE_carried"])
    data["source_carried"] = keys.map(lookup["source_carried"])
    data["n_hld_rows"] = keys.map(lookup["n_hld_rows"])
    data["n_hld_codes"] = keys.map(lookup["n_hld_codes"])
    data["n_hld_soe_codes"] = keys.map(lookup["n_hld_soe_codes"])
    data["has_no_controller"] = keys.map(lookup["has_no_controller"])

    still_missing = data["SOE_carried"].isna()
    if still_missing.any():
        firm_known = (
            carried.dropna(subset=["SOE_carried"])
            .groupby("证券代码")
            .agg(
                soe_values=("SOE_carried", lambda s: tuple(sorted(set(s.tolist())))),
                years=("时间", lambda s: tuple(int(x) for x in s.tolist())),
            )
        )
        for index in data.index[still_missing]:
            firm = data.at[index, "证券代码"]
            year = data.at[index, "时间"]
            if firm not in firm_known.index or pd.isna(year):
                continue
            values = firm_known.at[firm, "soe_values"]
            if len(values) != 1:
                continue
            known_years = np.asarray(firm_known.at[firm, "years"], dtype=float)
            if np.abs(known_years - year).min() <= MAX_CARRY_GAP:
                data.at[index, "SOE_carried"] = float(values[0])
                data.at[index, "source_carried"] = "firm_carry"

    data["SOE_missing"] = data["SOE_carried"].isna().astype(int)
    data["SOE_imputed"] = data["SOE_carried"].fillna(0.0)
    data["SOE_source"] = np.where(
        data["SOE_carried"].notna(),
        data["source_carried"].fillna("same_year_code"),
        "residual_zero",
    )
    return data


def reconstruct_soe(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace SOE using same-year HLD aggregation, firm carry, residual 0."""
    holder = load_holder()
    carried = firm_carry(aggregate_firm_year(holder))
    attached = attach_to_panel(panel, carried)
    extra = [
        "S0702b_code_panel",
        "SOE_fill0",
        "SOE_obs",
        "source_obs",
        "SOE_carried",
        "source_carried",
        "n_hld_rows",
        "n_hld_codes",
        "n_hld_soe_codes",
        "has_no_controller",
        "SOE_imputed",
    ]
    attached["SOE"] = attached["SOE_imputed"]
    return attached.drop(columns=[column for column in extra if column in attached.columns])


def load_production_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)


def load_common_sample() -> pd.DataFrame:
    return pd.read_csv(COMMON, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)


def diagnose(panel: pd.DataFrame, holder: pd.DataFrame, last: pd.DataFrame, common_keys: pd.Series) -> dict:
    attached = panel.copy()
    attached["_key"] = pad_code(attached["证券代码"]) + "_" + pd.to_numeric(attached["时间"], errors="coerce").astype("Int64").astype(str)
    attached["in_common"] = attached["_key"].isin(set(common_keys))
    attached["panel_s0702b_missing"] = attached["S0702b_code_panel"].isna()
    last = last.copy()
    last["_key"] = last["证券代码"] + "_" + last["时间"].astype(int).astype(str)
    last_map = last.set_index("_key")
    attached["last_code"] = attached["_key"].map(last_map["S0702b_last_code"])
    attached["last_missing"] = attached["last_code"].isna()

    def summarize(frame: pd.DataFrame, scope: str) -> dict:
        missing = frame["panel_s0702b_missing"]
        recovered_same_year = missing & frame["source_obs"].eq("same_year_code")
        recovered_no_controller = missing & frame["source_obs"].eq("no_controller")
        last_empty_other_code = missing & frame["last_missing"] & frame["n_hld_codes"].fillna(0).gt(0)
        return {
            "scope": scope,
            "n": int(len(frame)),
            "S0702b_missing_panel": int(missing.sum()),
            "S0702b_missing_rate": float(missing.mean()),
            "missing_but_same_year_code": int(recovered_same_year.sum()),
            "missing_but_no_controller": int(recovered_no_controller.sum()),
            "last_write_empty_but_other_row_has_code": int(last_empty_other_code.sum()),
            "SOE_fill0_mean": float(frame["SOE_fill0"].mean()),
            "SOE_fill0_ones": int(frame["SOE_fill0"].eq(1).sum()),
            "SOE_imputed_mean": float(frame["SOE_imputed"].mean()),
            "SOE_imputed_ones": int(frame["SOE_imputed"].eq(1).sum()),
            "source_same_year_code": int(frame["SOE_source"].eq("same_year_code").sum()),
            "source_no_controller": int(frame["SOE_source"].eq("no_controller").sum()),
            "source_firm_carry": int(frame["SOE_source"].eq("firm_carry").sum()),
            "source_residual_zero": int(frame["SOE_source"].eq("residual_zero").sum()),
            "flipped_0_to_1": int(frame["SOE_fill0"].eq(0).mul(frame["SOE_imputed"].eq(1)).sum()),
            "flipped_1_to_0": int(frame["SOE_fill0"].eq(1).mul(frame["SOE_imputed"].eq(0)).sum()),
        }

    summary = pd.DataFrame(
        [
            summarize(attached, "input_18077"),
            summarize(attached.loc[attached["in_common"]], "common_15381"),
        ]
    )
    source_counts = (
        attached.loc[attached["in_common"], "SOE_source"]
        .value_counts(dropna=False)
        .rename_axis("SOE_source")
        .reset_index(name="n")
    )
    flipped = attached.loc[
        attached["in_common"] & attached["SOE_fill0"].ne(attached["SOE_imputed"]),
        [
            "证券代码",
            "时间",
            "S0702b",
            "SOE_fill0",
            "SOE_imputed",
            "SOE_source",
            "n_hld_rows",
            "n_hld_codes",
            "has_no_controller",
        ],
    ]
    return {
        "summary": summary,
        "source_counts": source_counts,
        "flipped": flipped,
        "payload": {
            "input_n": int(len(attached)),
            "common_n": int(attached["in_common"].sum()),
            "common_soe_fill0": float(attached.loc[attached["in_common"], "SOE_fill0"].mean()),
            "common_soe_imputed": float(attached.loc[attached["in_common"], "SOE_imputed"].mean()),
        },
    }


def write_shadow_panels(attached: pd.DataFrame, common: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    shadow = attached.copy()
    shadow["SOE"] = shadow["SOE_imputed"]
    combined_path = OUT / "combined_panel_soe_imputed.csv"
    shadow.to_csv(combined_path, index=False, encoding="utf-8-sig")

    common = common.copy()
    common["证券代码"] = pad_code(common["证券代码"])
    common["时间"] = pd.to_numeric(common["时间"], errors="coerce")
    keys = pad_code(attached["证券代码"]) + "_" + pd.to_numeric(attached["时间"], errors="coerce").astype("Int64").astype(str)
    lookup = attached.assign(_key=keys).drop_duplicates("_key").set_index("_key")
    common_key = pad_code(common["证券代码"]) + "_" + common["时间"].astype("Int64").astype(str)
    common["SOE_fill0"] = common_key.map(lookup["SOE_fill0"])
    common["SOE_imputed"] = common_key.map(lookup["SOE_imputed"])
    common["SOE_source"] = common_key.map(lookup["SOE_source"])
    common["SOE_missing"] = common_key.map(lookup["SOE_missing"])
    common["SOE"] = common["SOE_imputed"]
    common_path = OUT / "common_sample_15381_soe_imputed.csv"
    common.to_csv(common_path, index=False, encoding="utf-8-sig")
    if len(common) != 15381:
        raise ValueError(f"Shadow common sample has {len(common)} rows, expected 15,381")
    if common["SOE"].isna().any():
        raise ValueError("Shadow common sample still has missing SOE")
    return {"combined": combined_path, "common": common_path, "n_common": int(len(common))}


def build_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    holder = load_holder()
    last = last_write_wins(holder)
    aggregated = aggregate_firm_year(holder)
    carried = firm_carry(aggregated)
    panel = load_production_panel()
    attached = attach_to_panel(panel, carried)
    common = load_common_sample()
    common["证券代码"] = pad_code(common["证券代码"])
    common_keys = pad_code(common["证券代码"]) + "_" + pd.to_numeric(common["时间"], errors="coerce").astype("Int64").astype(str)
    diagnosis = diagnose(attached, holder, last, common_keys)
    diagnosis["summary"].to_csv(OUT / "soe_impute_summary.csv", index=False, encoding="utf-8-sig")
    diagnosis["source_counts"].to_csv(OUT / "soe_impute_source_counts.csv", index=False, encoding="utf-8-sig")
    diagnosis["flipped"].to_csv(OUT / "soe_impute_flipped_rows.csv", index=False, encoding="utf-8-sig")
    carried.to_csv(OUT / "hld_firm_year_soe.csv", index=False, encoding="utf-8-sig")
    paths = write_shadow_panels(attached, common)
    meta = {
        **diagnosis["payload"],
        "combined_panel": str(paths["combined"]),
        "common_sample": str(paths["common"]),
        "soe_codes": sorted(SOE_CODES),
        "max_carry_gap": MAX_CARRY_GAP,
        "production_panel_untouched": str(PANEL),
        "production_common_untouched": str(COMMON),
    }
    (OUT / "soe_impute_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(diagnosis["summary"].to_string(index=False))
    return meta
