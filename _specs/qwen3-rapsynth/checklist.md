# Checklist

✅ = 已通过实装/运行验证；🔒 = 已就绪、但需合规数据或额外 GPU 富余后执行（框架阶段未伪造完成）。

- [x] 研究仓库 `07_说唱合成Qwen3RapSynth/` 目录结构已初始化，探针脚本语法校验通过并通过运行
- [x] 控制面可行性探查完成：**确认并修正**「Qwen3-TTS 支持 pitch_curve/duration」假设（实测不支持），成稿 `docs/01_控制面探查报告.md`
- [x] 显式注入不可用 → 已设计并采用间接控制面（x-vector 音色 + time-stretch/pitch 后处理 + 逐行拍网格编排）
- [x] 数据标注工具已实装（`prepare_prosody_data.py`，F0/时长/能量 → CSV/JSONL，UTF-8 无 BOM）
- [ ] 🔒 数据来源开源/合规并记录授权数量：需要用户提供说唱干声+歌词数据集（工具已就绪，架子未跑实际数据）
- [x] 规则先验基线 + 学习模型统一接口已实现（`rules.py` + `ProsodyPredictorBase`），16GB 约束已写入
- [x] LSTM 学习预测器已实装并烟测通过（`learned.py`，`train_prosody_predictor.py`，教师蒸馏 loss 10.0→2.06，generate_rap 自动选型 `learned`）；正式版需真实/MFA 标签重训
- [x] 逐音节 F0 轮廓注入已实装自检通过（`apply_f0_contour` + `syllable_f0_delta`，`test_f0_contour.py` 实测四窗 F0≈目标）——「音素≈音节级」注入接口可用（真实增益待 hi-fi 重训+全风格对拍）
- [x] 韵律参数已注入 Qwen3-TTS（间接层 `injector.py`），注入层独立可关、不影响既有外挂
- [x] 音节级 vs 行级 vs 无注入消融已实测（`bench_ablation.py`）：音节级 onset 误差 0.1468s 优于基线 0.1637s（docs/04）
- [x] `scripts/generate_rap.py` 打通 `歌词+BPM → 干声 wav`，CLI 可跑并产出真实干声
- [x] 客观指标脚本已实现并有结果（对拍/F0 RMSE/BPM 漂移/押韵/能量 × 风格）
- [x] 与基线（裸 Base+情感标签）对比表已实测（快嘴全表在 docs/04）：音节级 0.1468s 优于基线 0.1637s；行级≈基线（诚实标注）
- [x] CLI 一键生成可用（歌词 + 风格 + BPM）
- [x] ≥3 段不同风格干声已输出：`output/samples/快嘴_96bpm.wav` `旋律说唱_84bpm.wav` `硬核_72bpm.wav`
- [x] `docs/00_技术报告.md`（方法 / 实验 / 结果分析）已成稿
- [x] 阶段进展与问题已在最终回复向用户如实汇报（含未完成项与原因）