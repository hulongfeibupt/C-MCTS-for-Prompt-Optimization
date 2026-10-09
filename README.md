## 项目作用

针对已通过 GRPO 对齐的小语言模型（如 Qwen3-14B），用 蒙特卡洛树搜索（MCTS）在"提示词空间"中自动搜索最优任务提示词，提升下游分类任务的 F1、宏平均 Precision、宏平均 Recall。

## 目录结构

```text
mcts_prompt_gen_v4_major/
├── main.py                            # CLI 入口 (--target_label / --iterations …)
├── mcts_core.py                       # PUCT 选择 / Q 回溯 / 树结构
├── evaluator.py                       # 真实推理评估 + 软约束奖励
├── action_generator.py                # 候选动作生成（含 softmax 先验）
├── deepseek_api.py                    # DeepSeek API 客户端（带重试）
├── confusion_analyzer.py              # 混淆矩阵解析 → confused_labels
├── shared_config.py                   # 全局配置（API / 数据 / MCTS / 容忍度）
├── label_prompts_config.py            # 业务标签的 prompt 字典（被读写）
├── fastapi_model_server.py            # 推理服务（vLLM + Qwen3-14B）
├── batch_evaluate_labels.py           # 多标签批量评估脚本
├── diagnose_vllm_init.py              # 推理服务诊断工具
├── test_evaluator.py                  # 评估器单元测试
├── restart_server.sh                  # 推理服务重启脚本
├── run_batch_optimization_full.sh     # 多标签批处理包装
├── confusion_matrix/                  # 混淆矩阵生成（含 V2 等价匹配）
├── datasets/                          # 验证数据集
├── execute/                           # 历史执行脚本
├── mcts_cache/                        # Semantic Result Memoization 缓存
├── mcts_error_logs/                   # 误判样本导出
├── baseline_pipeline/                 # 基线管理（独立模块）
├── optimized_prompt_<label>_<ts>.txt  # MCTS 产出的优化 prompt
├── summary_<label>.json               # 每次任务输出：基线 / 提升 / 约束违反
├── mcts_log.txt                       # 评估器主日志
└── PromptTree：…pdf                    # 论文 PDF
```
## 快速开始

### 1. 启动推理服务

```bash
conda activate q3_1
sh restart_server.sh
```
或显式启动：

```bash
python -u fastapi_model_server.py
```

### 2. 运行 MCTS 优化

```bash
python main.py \
  --target_label "XXXXX" \
  --data_path /ai/.../datasets/major_02.xlsx \
  --model_path /ai/pretrain/.../0316_B_grpo_v4_1000steps \
  --confusion_file /ai/.../confusion_matrix/xxx.xlsx \
  --iterations 50 \
  --gpu_ids 4,5,6,7 \
  --api_url http://localhost:8082 \
  --use_v2_evaluation True
```

### 3. 批量多标签

```bash
sh run_batch_optimization_full.sh
```
或用 Mac 本地控制台 `mcts-ui`（推荐）：


## 关键模块

### `mcts_core.py`
- `TreeNode.best_child()`：PUCT 选择。
- `MCTS.search()`：四阶段循环。
- `_backpropagate()`：增量均值更新 Q。
- `get_best_path()`：取 `argmax Q` 的叶节点。

### `evaluator.py`
- 两阶段评估：Level A（1000 条粗筛） + Level B（聚焦目标 + 混淆）。
- `DROP_TOLERANCE = 0.01`：混淆标签硬约束容差。
- `dead_epsilon = 1e-4`：死节点阈值。
- `mcts_cache/<label>_cache.pkl`：SRM 哈希缓存。
- `summary_<label>.json`：基线 / 提升 / 违规记录。

### `action_generator.py`
- 调用 DeepSeek 返回结构化 `{action, confidence}`。
- `_softmax(confidence)` 输出 MCTS 先验 P。
- 失败重试：指数退避。

### `fastapi_model_server.py`
- `/infer`：vLLM 推理。
- `/workspace/files` `/workspace/upload`：UI 远程数据桥接。
- `/workspace/mcts/{run,status,stop}`：批量任务队列。
- `loop.run_in_executor`：避免推理阻塞事件循环。

### `confusion_matrix/generate_confusion_analysis_v2.py`
- 读预测结果 Excel。
- 写三张 sheet：混淆对 / 标签统计 / 错误样本。
- V2 等价匹配：`check_match(true, pred) ∈ {✅, ❌}`。

### `shared_config.py`
- `API_URL` `MODEL_PATH`：推理服务配置。
- `MCTSConfig.C_PUCT = 1.4`、`MAX_ITERATIONS = 15`。
- `MCTSConfig.DROP_TOLERANCE = 0.01`。

## 配置开关

| 配置 | 默认 | 用途 |
|---|---|---|
| `use_v2_evaluation` | True | V2 语义匹配 vs V1 字符串相等 |
| `auto_generate_confusion` | False | 自动调用混淆分析 |
| `write_back` | False（UI 端） | 是否覆盖 `label_prompts_config.py` |
| `level_a_sample_size` | 1000 | Level A 粗筛样本量 |
| `c_puct` | 1.4（shared）/ 2.0（main） | 探索常数 |
| `max_iterations` | 15（shared）/ 默认 100 | MCTS 搜索轮数 |

## 相关项目

- `mcts-ui/`：Mac 本地控制台（`http://127.0.0.1:18120`）。
- `paper/`：论文相关材料。
- `agent_evolver/`：自演化训练流程。

## 许可与说明

本仓库为论文配套实现，主要面向研究复现与业务验证。请勿在生产环境直接使用默认 API Key。
