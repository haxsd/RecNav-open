# RecNav：面向无地图 ObjectNav 的预算感知恢复增强

[中文](README.md) · [English](README-EN.md)

> RecNav 是在匹配的 PixNav 风格基础导航器之上即插即用的恢复层：用姿态无关恢复门控、紧凑恢复记忆和有界宏动作恢复，减少无地图 ObjectNav 的迷路与卡死。

本仓库是论文方法的公开代码发布；Python 包仍名为 `ada_semnav`（与项目现有代码兼容），论文与文档中的方法名是 RecNav。

## 核心能力

- **姿态无关恢复门控**：不依赖位姿真值判断何时进入恢复，适配无地图设置
- **紧凑恢复记忆**：记录可恢复骨架（节点 / 边），供恢复阶段做有界搜索
- **有界宏动作恢复**：宏动作候选排序 + 闭环进度验证，限制调用预算与重试次数
- **预算与遥测**：统一记录 LLM 调用、探索步数、重试等预算事件，schema 固化
- **锁定轨迹协议**：基础导航器先记录 `planner_trace.jsonl`，恢复变体重放到触发前，隔离 VLM 采样与规划调用差异
- **可复现发布**：提供 E1 / E2 / E3 实验启动脚本、结果汇总、预算曲线与 McNemar 显著性检验工具

## 快速开始

本仓库只含代码，不含 HM3D / ObjectNav 数据、模型权重与论文草稿。Habitat 需按官方说明先安装 `habitat-sim` / `habitat-lab`。

```bash
conda create -n recnav python=3.10
conda activate recnav
pip install -r requirements.txt
cp .env.example .env          # 填 Habitat 路径、模型路径与 API Key
```

外部视觉模块装到 `pixnav/thirdparty/`（GroundingDINO、Segment Anything，用 `pip install -e` 安装），权重放到 `pixnav/checkpoints/`；HM3D 场景与 ObjectNav v2 episode 放到 `data/`。

环境自检：

```bash
./run.sh check-dataset   # 校验 HM3D / ObjectNav 目录结构
./run.sh env-check       # 检查依赖、路径与权重
./run.sh llm-smoke       # 测试 OpenAI 兼容模型连通性
./run.sh smoke-test --steps 10
```

## 复现实验

先跑 baseline 生成 planner trace，再用锁定轨迹跑 RecNav：

```bash
bash experiments/run_experiment.sh f0 42 100 200
export PLANNER_TRACE_SOURCE="$PWD/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"
bash experiments/run_experiment.sh adarec_locked 42 100 200
```

论文批次（E2 需先跑，它为 E1 / E3 生成 baseline 轨迹）：

```bash
bash experiments/launch_e2_budget_curve.sh
bash experiments/launch_e1_table1.sh
bash experiments/launch_e3_ablation.sh
```

结果汇总与统计：

```bash
./run.sh analyze-results --base artifacts/runs/experiments
./run.sh plot-budget --base artifacts/runs/experiments --out figures/budget_curve.pdf
./run.sh mcnemar --baseline <baseline_csv> --treatment <treatment_csv>
```

## 文档

- [docs/reproduce.md](docs/reproduce.md)：完整复现流程与实验变体
- [docs/planner_trace_format.md](docs/planner_trace_format.md)：planner trace 格式
- [docs/artifact_policy.md](docs/artifact_policy.md)：产物与数据管理约定
- [scripts/README.md](scripts/README.md)：脚本入口清单

## 致谢与许可证

基础导航策略来自 PixNav / Pixel Navigator，开放词汇感知使用 GroundingDINO 与 Segment Anything。许可证：MIT（见 [LICENSE](LICENSE)）。
