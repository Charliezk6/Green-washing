"""Build unique-sentence AuditIntersect measures and validation samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from lexicon import SAMPLE_PER_FAMILY, SEED, classify_sentence, gold_auditable_anchor

ROOT = Path(__file__).resolve().parents[4]
LAYER_B_FILES = [
    ROOT / "Y_variable_reproduction/data/input/manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/non_manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/expansion/layer_b_results.jsonl",
]
PANEL = ROOT / "Regression/outputs/paper_tables/common_sample_15381.csv"
OUT = Path(__file__).resolve().parents[1]
DERIVED = OUT / "results/derived"
VALIDATION = OUT / "results/validation"
DERIVED.mkdir(parents=True, exist_ok=True)
VALIDATION.mkdir(parents=True, exist_ok=True)


def pad_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def load_unique_sentences() -> pd.DataFrame:
    rows = []
    seen = set()
    for path in LAYER_B_FILES:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sentence = (record.get("sentence") or "").strip()
                if not sentence:
                    continue
                code = pad_code(record.get("证券代码"))
                year = int(record["时间"])
                key = (code, year, sentence)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "证券代码": code,
                        "时间": year,
                        "sentence": sentence,
                        "word": record.get("word"),
                        "judgment": record.get("judgment"),
                    }
                )
    pairs = pd.DataFrame(rows)
    unique = (
        pairs.groupby(["证券代码", "时间", "sentence"], as_index=False)
        .agg(pair_count=("word", "size"), judgment_max=("judgment", "max"))
    )
    flags = unique["sentence"].map(classify_sentence)
    unique["equipment"] = flags.map(lambda item: int(item.equipment))
    unique["capex"] = flags.map(lambda item: int(item.capex))
    unique["penalty"] = flags.map(lambda item: int(item.penalty))
    unique["remediation"] = flags.map(lambda item: int(item.remediation))
    unique["quantity"] = flags.map(lambda item: int(item.quantity))
    unique["boundary"] = flags.map(lambda item: int(item.boundary))
    unique["intersect"] = flags.map(lambda item: int(item.intersect))
    unique["soft"] = flags.map(lambda item: int(item.soft))
    unique["template"] = flags.map(lambda item: int(item.template))
    return unique


def sample_family(data: pd.DataFrame, mask: pd.Series, family: str, rng: np.random.Generator) -> pd.DataFrame:
    eligible = data.loc[mask].drop_duplicates(subset=["sentence"])
    n_take = min(SAMPLE_PER_FAMILY, len(eligible))
    if n_take == 0:
        return pd.DataFrame()
    chosen = eligible.sample(n=n_take, random_state=int(rng.integers(0, 10**9)))
    chosen = chosen.copy()
    chosen["family"] = family
    chosen["gold_auditable"] = chosen["sentence"].map(gold_auditable_anchor)
    chosen["rule_positive"] = chosen["intersect"]
    chosen["match"] = (chosen["gold_auditable"] == chosen["rule_positive"]).astype(int)
    return chosen[
        [
            "family",
            "证券代码",
            "时间",
            "sentence",
            "equipment",
            "capex",
            "penalty",
            "remediation",
            "quantity",
            "boundary",
            "intersect",
            "soft",
            "template",
            "gold_auditable",
            "rule_positive",
            "match",
        ]
    ]


def validation_samples(sentences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    samples = pd.concat(
        [
            sample_family(sentences, sentences["intersect"].eq(1) & sentences["boundary"].eq(0), "intersect_positive", rng),
            sample_family(sentences, sentences["boundary"].eq(1), "intersect_boundary", rng),
            sample_family(sentences, sentences["soft"].eq(1), "soft", rng),
        ],
        ignore_index=True,
    )
    accuracy = (
        samples.groupby("family")
        .agg(
            n=("match", "size"),
            accuracy=("match", "mean"),
            gold_positive_rate=("gold_auditable", "mean"),
            rule_positive_rate=("rule_positive", "mean"),
        )
        .reset_index()
    )
    # Soft family: the rule is Intersect=0; accuracy is whether gold also says not auditable.
    return samples, accuracy


def firm_year_panel(sentences: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    grouped = sentences.groupby(["证券代码", "时间"], as_index=False).agg(
        EnvSentence_Count=("sentence", "size"),
        Intersect_Count=("intersect", "sum"),
        Soft_Count=("soft", "sum"),
        Equipment_Count=("equipment", "sum"),
        Capex_Count=("capex", "sum"),
        PenaltyClaim_Count=("penalty", "sum"),
        Remediation_Count=("remediation", "sum"),
        Quantity_Count=("quantity", "sum"),
        Boundary_Count=("boundary", "sum"),
    )
    grouped["AuditIntersect_Share"] = grouped["Intersect_Count"] / grouped["EnvSentence_Count"]
    grouped["Soft_Share"] = grouped["Soft_Count"] / grouped["EnvSentence_Count"]
    grouped["Equipment_Share"] = grouped["Equipment_Count"] / grouped["EnvSentence_Count"]
    grouped["Capex_Share"] = grouped["Capex_Count"] / grouped["EnvSentence_Count"]
    grouped["PenaltyClaim_Share"] = grouped["PenaltyClaim_Count"] / grouped["EnvSentence_Count"]
    grouped["Remediation_Share"] = grouped["Remediation_Count"] / grouped["EnvSentence_Count"]
    grouped["LnIntersect_Count"] = np.log1p(grouped["Intersect_Count"])

    keys = common[["证券代码", "时间", "ExpertiseIndustryCode", "GI"]].copy()
    keys["证券代码"] = keys["证券代码"].map(pad_code)
    keys["时间"] = pd.to_numeric(keys["时间"], errors="coerce")
    merged = keys.merge(grouped, on=["证券代码", "时间"], how="left", validate="one_to_one")
    # Do not recode missing shares as zero: no classified sentence is not "all soft".
    median = merged.groupby(["ExpertiseIndustryCode", "时间"])["AuditIntersect_Share"].transform("median")
    merged["HighIntersect"] = np.where(
        merged["AuditIntersect_Share"].notna(),
        (merged["AuditIntersect_Share"] > median).astype(float),
        np.nan,
    )
    return merged


def main() -> None:
    sentences = load_unique_sentences()
    samples, accuracy = validation_samples(sentences)
    samples.to_csv(VALIDATION / "lexicon_validation_sample.csv", index=False, encoding="utf-8-sig")
    accuracy.to_csv(VALIDATION / "lexicon_accuracy.csv", index=False, encoding="utf-8-sig")

    common = pd.read_csv(PANEL, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    panel = firm_year_panel(sentences, common)
    panel.to_csv(DERIVED / "audit_intersect_firm_year.csv", index=False, encoding="utf-8-sig")

    coverage = pd.DataFrame(
        [
            {
                "unique_sentences": len(sentences),
                "intersect_sentences": int(sentences["intersect"].sum()),
                "soft_sentences": int(sentences["soft"].sum()),
                "boundary_sentences": int(sentences["boundary"].sum()),
                "firm_years_in_common": len(panel),
                "firm_years_with_share": int(panel["AuditIntersect_Share"].notna().sum()),
                "mean_share": float(panel["AuditIntersect_Share"].mean()),
                "mean_soft_share": float(panel["Soft_Share"].mean()),
            }
        ]
    )
    coverage.to_csv(VALIDATION / "sentence_coverage.csv", index=False, encoding="utf-8-sig")
    print(coverage.to_string(index=False))
    print(accuracy.to_string(index=False))


if __name__ == "__main__":
    main()
