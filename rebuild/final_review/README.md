# 复赛审阅包

本目录是 Final Freeze 之前的审阅件，不是可发布的 `final_project/`。原始六个模型目录未移动或删除。

- `最终项目技术报告.md`：10 节统一技术报告。
- `24项结论.md`：对任务最后 24 项问题逐项回答。
- `问题一至四最终详解.md`：选定模型及证据边界。
- `outputs/tables/`：从完成后的选定链 CSV 复制的论文数值。
- `outputs/figures/`：选定链重新生成的图，分解和预测图均标为探索性。
- `verification/final_freeze_checklist.json`：明确记录尚未满足的 P4 证据门槛。
- `artifact_manifest.csv`：每个打包产物的来源、SHA256 与大小。

隔离的一键入口在 `../winner_chain/run_all.py`；25 个顶层阶段的原始日志仍保留于 `../winner_chain/one_click_*.log`，扩展后的内层审阅表通过断点续跑完成。复赛结果和详细模型选择记录保留于 `../problem1/` 至 `../problem4/`。新增历史实测目标后，应重新挑战 P4 并更新本包，不能直接改写验证状态。
