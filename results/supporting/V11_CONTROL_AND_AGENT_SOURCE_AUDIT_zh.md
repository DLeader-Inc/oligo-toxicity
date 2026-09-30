# V11 历史对照和证据流程来源审计

本记录只审计已存代码及制品，不新增训练或文献结论。

## 历史性能对照

- `scripts/analyze_expert_rule_paper.py` 从 `outputs/integrated_auto_system_v1/report/endpoint_stage_metrics.csv` 取得前两列历史差值。
- `scripts/build_integrated_auto_system_report.py` 的来源阶段明确筛选 `oligogym_published_xgb`，不是正文最佳 CNN／Transformer。项目 XGBoost 参数由 `src/benchmarking/oligogym_same_split.py` 的 `fit_predict_project_xgb_features` 和 `scripts/benchmark_aso_atlas2_project_representation.py` 的 `PROJECT_PARAMS` 定义。
- Atlas 来源阶段的 `outputs/aso_atlas2_four_stage_factorial_v1/full/run_manifest.json` 指定原尺度、不取对数；固定融合阶段的 `outputs/integrated_auto_system_v1/stages_v2/run_manifest.json` 指定 ALT 使用 log1p。因此旧表示／融合差值对 ALT 混入了尺度改变，不能用作纯描述符贡献。
- `scripts/benchmark_registry_nine_endpoints.py` 的历史阶段每划分、每分支拟合一个种子模型；最终系统平均五个种子。历史阶段之间的种子约定也不同。保留数值但不解释为完整因子实验。
- `rule_log1p` 来自 `outputs/expert_rule_scale_v1/analysis/policy_endpoint_metrics.csv` 中 `fixed_expert_identity` 与 `expert_rule_scale` 的同配置比较，才是单独尺度变化。

## 解释流程

`scripts/build_paper_interpretability_agent.py` 的 `render_story` 是预写双语模板，`audit_story` 执行十项文本存在性检查。`run_loop` 追加的修复指令未被叙述函数使用，不能证明动态修复。归档轨迹证明确定性证据汇集、输入哈希、模板生成与检查结果，不证明本案例进行了自主多智能体推理。论文的文献解释及科学修订属于作者审阅；新架构图据此区分自动步骤和作者工作。

## 图表验证

`scripts/plot_reader_controls_v11.py` 直接读取现有九终点 CSV，不修改任何数值，生成中英文性能／历史对照图和实际证据流程图，均保存 PNG 与 SVG。中文使用已安装的 Noto CJK 字体。脚本成功运行；已目视检查英文对照图和中文架构图，无文字裁切或方框缺字。图例明确历史表示／融合、ALT 尺度混杂以及不可相加。

修订限于 V11 两份 Supplement 的 S1/S13、标题及图文件。正文由主代理负责，S6/S19 的主体修订未覆盖。
