# 面板版本审计（只读）

对照 `archive/Regression_reproduction`（共同样本 8,129）与 `Regression`（共同样本 14,778）。不修改生产输入。

```text
python validation/panel_version_audit/scripts/run_panel_version_audit.py
```

结论见 `notes/version_change_audit_memo.md`。主要输出在 `outputs/`：`sample_flow.csv`、`soe_audit_summary.csv`、`soe_recode_fill0.csv`、`y_rebuild_vs_stored.csv`、`controls_x_audit.csv`、`composition_original_vs_expansion.csv`、`nested_regression_diagnostics.csv`。
