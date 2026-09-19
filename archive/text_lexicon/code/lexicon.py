"""Pre-registered dictionaries for auditable versus soft environmental claims.

The dictionaries are locked before any outcome regression. They do not reuse the
Layer B GI label (`judgment`). A sentence is auditable if it contains a factual
anchor that could be compared with audited information under CAS 1521.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEED = 20260915
SAMPLE_PER_FAMILY = 100

# Strong equipment/facility anchors (plan: 设备, 环保设施, 污染治理设施).
EQUIPMENT_STRONG = [
    "环保设施",
    "污染治理设施",
    "环保设备",
    "治污设施",
    "治污设备",
    "脱硫设备",
    "脱硝设备",
    "除尘设备",
    "除尘设施",
    "污水处理设施",
    "污水处理设备",
    "废气治理设施",
    "废水治理设施",
    "污染防治设施",
    "废水处理设施",
    "废气处理设施",
]

# Broad 设备 counts only with an environmental co-occurrence. First-pass
# validation showed bare 设备 hits production machinery, fire-safety kit, and
# product descriptions. The co-occurrence restriction was locked before any
# outcome regression.
EQUIPMENT_CONTEXT = [
    "环保",
    "污染",
    "废气",
    "废水",
    "污水",
    "除尘",
    "脱硫",
    "脱硝",
    "治污",
    "排污",
    "监测",
    "节能",
    "减排",
    "voc",
    "rto",
    "rco",
    "活性炭",
    "排放",
    "固废",
    "危废",
    "噪声",
]
EQUIPMENT_EXCLUDE = [
    "电子设备",
    "通信设备",
    "通讯设备",
    "医疗设备",
    "运输设备",
    "办公设备",
    "专用设备",
    "通用设备",
    "仪器设备采购",
    "激光",
    "消防设备",
]

# Capital investment / construction (plan: 环保投入, 环保投资, 技术改造, 在建工程, 技改).
# Bare 投资 / 工程 are not main-rule terms.
CAPEX_TERMS = [
    "环保投入",
    "环保投资",
    "环境投入",
    "环境投资",
    "污染治理投资",
    "技术改造",
    "在建工程",
    "技改",
]

# Penalty / rectification.
PENALTY_TERMS = [
    "处罚",
    "罚款",
    "整改",
    "限期治理",
]
PENALTY_EXCLUDE = [
    "不受处罚",
    "未受到处罚",
    "未受处罚",
    "不存在处罚",
    "无环保处罚",
    "未被处罚",
]

# Remediation. Bare 修复 is noisy (credit repair, recovery rates). Require an
# environmental co-occurrence, locked after the first-pass sample.
REMEDIATION_STRONG = ["土壤修复", "生态修复", "污染修复", "环境修复", "修复工程", "泄漏检测与修复"]
REMEDIATION_CONTEXT = ["土壤", "生态", "污染", "环境", "土地", "水体", "矿山", "ldar"]
REMEDIATION_EXCLUDE = ["修复率", "回收率", "修复完成率", "植被修复率", "信用修复", "信用系统修复"]

# Quantity units that can be checked against accounts or compliance records.
# Years such as 2021 are not treated as quantities.
UNIT_PATTERN = re.compile(
    r"(万元|亿元|万吨|千吨|千克|公斤|千瓦时|千瓦|兆瓦|万千瓦|立方米|万立方米|"
    r"毫克|公顷|亩|吨标煤|(?<![A-Za-z0-9])吨(?![A-Za-z0-9]))"
)

# Soft / symbolic talk. A sentence is soft only if it has these markers and does
# not contain an auditable anchor.
SOFT_TERMS = [
    "致力于",
    "倡导",
    "理念",
    "绿色发展",
    "生态文明",
    "碳中和愿景",
    "社会责任",
]

TEMPLATE_NOISE = [
    "是否属于环境保护部门公布的重点排污单位",
    "上市公司及其子公司是否属于",
    "环境信息的披露",
    "详见公司披露",
]


@dataclass(frozen=True)
class SentenceFlags:
    equipment: bool
    capex: bool
    penalty: bool
    remediation: bool
    quantity: bool
    boundary: bool
    intersect: bool
    soft: bool
    template: bool


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _equipment(text: str) -> tuple[bool, bool]:
    lowered = text.lower()
    strong = _contains_any(text, EQUIPMENT_STRONG)
    if strong:
        return True, False
    if _contains_any(text, EQUIPMENT_EXCLUDE):
        return False, False
    broad_raw = "设备" in text
    env_context = _contains_any(lowered, EQUIPMENT_CONTEXT)
    return False, broad_raw and env_context


def _remediation(text: str) -> tuple[bool, bool]:
    if _contains_any(text, REMEDIATION_EXCLUDE):
        return False, False
    strong = _contains_any(text, REMEDIATION_STRONG)
    if strong:
        return True, False
    broad = ("修复" in text) and _contains_any(text.lower(), REMEDIATION_CONTEXT)
    return False, broad


def classify_sentence(sentence: str) -> SentenceFlags:
    text = (sentence or "").strip()
    template = _contains_any(text, TEMPLATE_NOISE)
    equipment_strong, equipment_broad = _equipment(text)
    equipment = equipment_strong or equipment_broad
    capex = _contains_any(text, CAPEX_TERMS)
    penalty = _contains_any(text, PENALTY_TERMS) and not _contains_any(text, PENALTY_EXCLUDE)
    remediation_strong, remediation_broad = _remediation(text)
    remediation = remediation_strong or remediation_broad
    quantity = bool(UNIT_PATTERN.search(text))
    # Boundary for validation: 设备 or 修复 present, but not an auditable intersect.
    raw_equipment = "设备" in text
    raw_repair = "修复" in text
    intersect = (not template) and (
        equipment_strong or capex or penalty or remediation_strong or quantity or equipment_broad or remediation_broad
    )
    boundary = (raw_equipment or raw_repair) and not intersect
    soft = (not template) and (not intersect) and _contains_any(text, SOFT_TERMS)
    return SentenceFlags(
        equipment=equipment,
        capex=capex,
        penalty=penalty,
        remediation=remediation,
        quantity=quantity,
        boundary=boundary,
        intersect=intersect,
        soft=soft,
        template=template,
    )


# Independent validation criterion: a factual anchor that could be compared with
# financial statements, CIP, expenditures, provisions, or compliance records.
# This is not a reuse of the GI judgment and is applied only to the validation sample.
GOLD_POSITIVE_HINTS = [
    "环保设施",
    "污染治理设施",
    "环保设备",
    "环保投入",
    "环保投资",
    "在建工程",
    "技术改造",
    "技改",
    "处罚",
    "罚款",
    "整改",
    "限期治理",
    "土壤修复",
    "生态修复",
    "污染修复",
    "脱硫",
    "脱硝",
    "除尘",
    "污水处理",
    "污染防治设施",
    "废水处理设施",
    "废气治理设施",
    "废水治理设施",
]
GOLD_NEGATIVE_HINTS = [
    "致力于",
    "倡导",
    "理念",
    "绿色发展",
    "生态文明",
    "社会责任",
    "是否属于环境保护部门公布的重点排污单位",
    "修复率",
    "回收率",
]


def gold_auditable_anchor(sentence: str) -> int:
    """Return 1 if the sentence supplies a cross-checkable environmental fact.

    Used only for validation accuracy, not for constructing the moderator.
    """
    text = (sentence or "").strip()
    if not text:
        return 0
    if _contains_any(text, TEMPLATE_NOISE):
        return 0
    if _contains_any(text, GOLD_NEGATIVE_HINTS) and not _contains_any(text, GOLD_POSITIVE_HINTS) and not UNIT_PATTERN.search(text):
        return 0
    if _contains_any(text, PENALTY_EXCLUDE):
        penalty_hit = False
    else:
        penalty_hit = _contains_any(text, ["处罚", "罚款", "整改", "限期治理"])
    hard_action = _contains_any(text, GOLD_POSITIVE_HINTS) or penalty_hit
    has_unit = bool(UNIT_PATTERN.search(text))
    lowered = text.lower()
    equipment_with_context = ("设备" in text) and _contains_any(lowered, EQUIPMENT_CONTEXT) and not _contains_any(text, EQUIPMENT_EXCLUDE)
    remediation_with_context = ("修复" in text) and _contains_any(lowered, REMEDIATION_CONTEXT) and not _contains_any(text, REMEDIATION_EXCLUDE)
    if hard_action or has_unit or equipment_with_context or remediation_with_context:
        if _contains_any(text, ["修复率", "回收率", "信用修复"]) and not _contains_any(text, REMEDIATION_STRONG):
            return 0
        return 1
    return 0
