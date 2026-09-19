# Lexicon lock note

First-pass dictionary treated bare 「设备」 and 「修复」 as auditable. On a 100-sentence boundary sample, independent gold labels found that only about 9 percent of those sentences supplied a cross-checkable environmental fact (production machinery, fire-safety kit, credit repair, and recovery rates were common noise).

The locked dictionary therefore requires:
- 「设备」 plus an environmental co-occurrence (环保/污染/废气/废水/污水/除尘/脱硫/脱硝/治污/排污/监测/节能/减排/VOC/RTO/RCO/活性炭/排放/固废/危废/噪声), after excluding electronic, medical, transport, office, and laser-processing equipment;
- 「修复」 plus soil/ecology/pollution/land/water/LDAR context, after excluding 修复率, 回收率, and 信用修复.

Accuracy on the locked-lexicon validation sample (seed 20260915, 100 sentences per family):
- intersect_positive: 0.95
- intersect_boundary (now the excluded residual): 1.00
- soft: 1.00

No outcome regression was estimated before this lock.
