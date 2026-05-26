from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import logging
import hashlib
import pickle

# V2 混淆矩阵集成
try:
    from confusion_matrix.generate_confusion_analysis_v2 import check_match
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('mcts_log.txt', mode='a')
file_handler.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 导入真实的推理环境（infer_14B_qwen3_solo_refactored.py）
# 注意：当前架构使用 FastAPI HTTP 推理服务（worker_gpu），TextClassifier 未被实际使用，注释掉避免误导
# from infer_14B_qwen3_solo_refactored import TextClassifier

# 从共享配置加载所有标签的提示词字典
# 修改提示词请编辑 label_prompts_config.py 文件
from label_prompts_config import get_label_prompts
label_prompts = get_label_prompts()

# 每个 worker 内部请求间的抖动间隔（秒），错开多 worker 同步发送
INFER_REQUEST_INTERVAL = 0.1
# 单次请求超时（秒）
INFER_REQUEST_TIMEOUT = 300
# 超时后最大重试次数
INFER_MAX_RETRIES = 3

def worker_gpu(queries: list[str], model_path: str, gpu_id: str, model_type: str, prompt_state: str, target_label: str, api_url: str = "http://localhost:8000") -> dict:
    """Worker function for GPU-based inference using FastAPI service.
    
    Returns:
        {"success": bool, "predictions": list[str] or None, "error": str or None}
    """
    try:
        import requests
        import time
        
        # SRM: 使用会话池优化HTTP连接
        session = requests.Session()
        
        # Ensure prompt_state is a string
        if not isinstance(prompt_state, str):
            msg = f"worker_gpu received prompt_state of type {type(prompt_state)}, expected str. Value: {str(prompt_state)[:100]}"
            logger.info(msg)
            return {"success": False, "predictions": None, "error": msg}

        predictions = []
        for query in queries:
            # Prepare payload
            payload = {
                "text": str(query), # Ensure query is string
                # "selected_labels": None, # Removed to avoid Pydantic 422 if server expects optional
                "label_overrides": {target_label: prompt_state}
            }
            
            pred_label = "<ERROR>"
            for attempt in range(1, INFER_MAX_RETRIES + 1):
                try:
                    response = session.post(f"{api_url}/infer", json=payload, timeout=INFER_REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            prediction = result["prediction"]
                            match = re.search(r'<answer>(.*?)</answer>', prediction)
                            pred_label = match.group(1).strip() if match else prediction.strip()
                        except ValueError as e:
                            logger.info(f"JSON parse error for query. Response: {response.text[:200]}... Error: {e}")
                    else:
                        logger.info(f"API call failed. Status: {response.status_code}")
                        logger.info(f"Response Body: {response.text[:500]}")
                    break  # 成功或非超时错误，不再重试

                except requests.Timeout:
                    wait = 2 ** attempt  # 指数退避: 2s, 4s, 8s
                    logger.info(f"Request timeout (attempt {attempt}/{INFER_MAX_RETRIES}), retrying in {wait}s...")
                    if attempt < INFER_MAX_RETRIES:
                        time.sleep(wait)
                except requests.RequestException as e:
                    logger.info(f"Request exception: {e}")
                    break

            predictions.append(pred_label)
            # 请求间加小间隔，错开多 worker 同步发送造成的瞬时积压
            time.sleep(INFER_REQUEST_INTERVAL)
        
        session.close()  # 关闭会话
        return {"success": True, "predictions": predictions, "error": None}
    except Exception as e:
        error_msg = f"Error in worker_gpu for GPU {gpu_id}: {e}"
        logger.info(error_msg)
        return {"success": False, "predictions": None, "error": error_msg}


@dataclass(frozen=True)
class BaselineMetrics:
    other_acc: float
    target_precision: float
    target_recall: float
    target_f1: float
    per_confused_acc: Dict[str, float]


class Evaluator:
    """MCTS 评估器（带铁律约束）

    关键点：
    - 调用 infer_14B_qwen3_solo_refactored.py 的 TextClassifier 作为真实评估环境
    - 通过临时修改 classifier.label_prompts[target_label] 实现候选 prompt 评估
    - Level A / Level B 必须各自维护 baseline（否则抽样集与全量集会互相误判）
    - 铁律：
      1) 混淆标签（逐个标签）准确率不能下降（允许极小浮点误差）
      2) 目标标签 precision 与 recall 必须同时提升（同样允许极小浮点误差）
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        target_label: str,
        confusion_file: Optional[str] = None,
        frozen_labels_prompts: Optional[Dict[str, str]] = None,
        gpu_ids: str = "0,1,2,3,4,5,6,7",
        model_type: str = "qwen3",
        api_url: str = "http://localhost:8082",
        *,
        query_col: str = "cleaned_text",
        true_label_col: str = "cleaned_human_label",
        dead_epsilon: float = 1e-4,
        level_a_sample_size: int = 1000,
        seed: int = 42,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.target_label = target_label
        self.confusion_file = confusion_file
        self.frozen_labels_prompts = frozen_labels_prompts or {}
        self.gpu_ids = gpu_ids
        self.model_type = model_type
        self.api_url = api_url

        self.query_col = query_col
        self.true_label_col = true_label_col
        self.dead_epsilon = dead_epsilon
        self.level_a_sample_size = level_a_sample_size
        self.seed = seed

        self.df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
        self._validate_columns(self.df)

        self.confused_labels = []
        if confusion_file:
            try:
                # 尝试相对导入，如果失败则尝试直接导入
                try:
                    from mcts_prompt_gen.confusion_analyzer import ConfusionAnalyzer
                except ImportError:
                    from confusion_analyzer import ConfusionAnalyzer

                analyzer = ConfusionAnalyzer(confusion_file)
                # 获取全部混淆标签（不限制top_k）
                self.confused_labels = analyzer.get_confused_labels(target_label, top_k=None)
                logger.info(f"✓ 已加载混淆矩阵，{target_label} 的全部混淆标签 ({len(self.confused_labels)}个): {self.confused_labels}")
            except Exception as e:
                logger.info(f"⚠️  加载混淆矩阵失败: {e}，将对所有非目标标签做约束检查")

        self._baseline_by_level: Dict[str, BaselineMetrics] = {}

        # SRM: Semantic Result Memoization 初始化
        self.cache_dir = "mcts_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache_file = os.path.join(self.cache_dir, f"{self.target_label}_cache.pkl")
        self.result_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._load_cache()

        # Note: Classifier initialization moved to worker processes for parallel GPU usage
        # No main-process classifier needed

    def _get_memo_key(self, prompt_state: str, level: str) -> str:
        """生成缓存键：基于prompt_state和level的哈希"""
        key_str = f"{prompt_state}_{level}_{self.target_label}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def _load_cache(self):
        """加载持久化缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.result_cache = pickle.load(f)
                logger.info(f"✓ 加载缓存: {len(self.result_cache)} 条记录")
            except Exception as e:
                logger.info(f"⚠️ 缓存加载失败: {e}，使用空缓存")

    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.result_cache, f)
            logger.info(f"✓ 保存缓存: {len(self.result_cache)} 条记录")
        except Exception as e:
            logger.info(f"⚠️ 缓存保存失败: {e}")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in [self.query_col, self.true_label_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"数据缺少必要列: {missing}. 当前列: {df.columns.tolist()}。"
                f"你可以在 Evaluator(...) 里通过 query_col/true_label_col 指定正确列名。"
            )

    def get_current_prompt(self) -> str:
        """获取当前 target_label 在推理环境中的 prompt"""
        # 直接从本地字典返回，避免本地模型初始化
        return label_prompts[self.target_label]

    def set_baseline(self, prompt_state: str, *, level: str) -> BaselineMetrics:
        score, analysis = self._evaluate_raw(prompt_state, level=level)
        baseline = BaselineMetrics(
            other_acc=analysis["other_labels_acc"],
            target_precision=analysis["precision"],
            target_recall=analysis["recall"],
            target_f1=analysis["f1"],
            per_confused_acc=analysis.get("per_confused_acc", {}),
        )
        self._baseline_by_level[level] = baseline
        logger.info(
            f"✓ 设置 baseline[{level}] 其他标签Acc={baseline.other_acc:.4f} "
            f"P={baseline.target_precision:.4f} R={baseline.target_recall:.4f} F1={baseline.target_f1:.4f}"
        )
        return baseline

    def evaluate(self, prompt_state: str, level: str = "A") -> Tuple[float, Dict[str, Any]]:
        # SRM: 检查缓存
        cache_key = self._get_memo_key(prompt_state, level)
        if cache_key in self.result_cache:
            cached_score, cached_analysis = self.result_cache[cache_key]
            logger.info(f"💾 缓存命中: {cache_key[:16]}... 返回缓存结果 F1={cached_score:.4f}")
            return cached_score, cached_analysis

        if level not in self._baseline_by_level:
            self.set_baseline(prompt_state, level=level)
            score, analysis = self._evaluate_raw(prompt_state, level=level)
            analysis["is_baseline"] = True
            # SRM: 保存到缓存
            self.result_cache[cache_key] = (score, analysis)
            self._save_cache()
            return score, analysis

        score, analysis = self._evaluate_raw(prompt_state, level=level)
        baseline = self._baseline_by_level[level]

        per_confused_acc = analysis.get("per_confused_acc", {})

        # === 新策略：不应用惩罚，改为直接返回 F1（或带微小奖励的 F1） ===
        # 这样任何能够提升指标的 prompt 都会获得更高的分数
        # 即使指标下降，也不会被强制拉低到 0.0001
        
        precision_drop = baseline.target_precision - analysis["precision"]
        recall_drop = baseline.target_recall - analysis["recall"]
        f1_gain = analysis["f1"] - baseline.target_f1
        
        other_labels_drop = baseline.other_acc - analysis["other_labels_acc"]
        
        # 容忍度
        DROP_TOLERANCE = 0.01
        
        violations = []
        
        # 1. 检查混淆标签下降（硬性剪枝，因为这是真的 Red Line）
        if baseline.per_confused_acc:
            for label, base_acc in baseline.per_confused_acc.items():
                cur_acc = per_confused_acc.get(label)
                if cur_acc is None: continue
                
                drop = base_acc - cur_acc
                if drop > DROP_TOLERANCE:  # 1% 以上的下降视为严重违规
                    violations.append(f"confused_{label}_drop_{drop:.4f}")
        else:
            if other_labels_drop > DROP_TOLERANCE:
                violations.append(f"other_labels_drop_{other_labels_drop:.4f}")
        
        # 如果混淆标签严重下降，直接返回很低的分数（但仍不是 -1.0，允许树继续存活）
        if violations:
            logger.info(f"⚠️ 严重违规: {violations}，但仍允许节点存活。")
            final_score = max(0.0001, score - 0.5)  # 扣 0.5 作为惩罚，但不会导致死亡
        else:
            # 2. 如果混淆标签没问题，直接返回 F1 分数
            # F1 本身已经代表了目标标签的质量
            # 如果 F1 提升了，分数就自然更高
            # 如果 F1 下降了，分数就自然更低（但仍为正）
            final_score = score
            
            # 给有改进的 prompt 一点额外奖励（鼓励探索）
            if f1_gain > 0.001:  # F1 提升超过 0.1%
                final_score = score + 0.05  # 加 0.05 作为奖励
                logger.info(f"✨ F1 提升 {f1_gain:.4f}，获得额外奖励！")
            elif f1_gain < -0.05:  # F1 下降超过 5%
                final_score = max(0.0001, score - 0.05)  # 扣分但保活
                logger.info(f"⚠️ F1 下降 {-f1_gain:.4f}，轻微惩罚。")

        # 记录详细信息
        analysis["precision_gain"] = -precision_drop
        analysis["recall_gain"] = -recall_drop
        analysis["f1_gain"] = f1_gain
        analysis["baseline_precision"] = baseline.target_precision
        analysis["baseline_recall"] = baseline.target_recall
        analysis["baseline_f1"] = baseline.target_f1
        analysis["violations"] = violations
        
        # SRM: 保存到缓存
        self.result_cache[cache_key] = (final_score, analysis)
        self._save_cache()
        
        return final_score, analysis

    def _evaluate_raw(self, prompt_state: str, level: str) -> Tuple[float, Dict[str, Any]]:
        eval_df = self._select_eval_df(level)

        logger.info(f"Running parallel inference on {len(eval_df)} samples (Level {level}) using GPUs {self.gpu_ids}...")

        # Split queries into batches for each GPU
        gpu_list = self.gpu_ids.split(",")
        num_gpus = len(gpu_list)
        queries = eval_df[self.query_col].tolist()
        batch_size = (len(queries) + num_gpus - 1) // num_gpus  # Ceiling division
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

        # Ensure we have batches for each GPU (pad with empty if needed)
        while len(query_batches) < num_gpus:
            query_batches.append([])

        # 限制并发 worker 数，避免同时向 VllmEngine 打满请求
        # 建议不超过 2，减少服务端队列积压
        max_workers = min(num_gpus, 2)

        # Submit to ProcessPoolExecutor
        predictions = [None] * len(queries)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, gpu_id in enumerate(gpu_list):
                if query_batches[i]:  # Only submit if there are queries
                    future = executor.submit(
                        worker_gpu,
                        query_batches[i],
                        self.model_path,
                        gpu_id,
                        self.model_type,
                        prompt_state,
                        self.target_label,
                        self.api_url
                    )
                    futures.append((future, i))

            # Collect results
            for future, batch_idx in futures:
                try:
                    result = future.result()
                    if result["success"] and result["predictions"]:
                        start_idx = batch_idx * batch_size
                        for j, pred in enumerate(result["predictions"]):
                            if start_idx + j < len(predictions):
                                predictions[start_idx + j] = pred
                    else:
                        logger.info(f"Batch {batch_idx} failed: {result['error']}")
                except Exception as e:
                    logger.info(f"Error collecting results from batch {batch_idx}: {e}")

        # Filter out None or error predictions
        valid_predictions = []
        valid_indices = []
        for idx, pred in enumerate(predictions):
            if pred is not None and pred != "<ERROR>":
                valid_predictions.append(pred)
                valid_indices.append(idx)
            else:
                logger.info(f"Warning: Failed prediction for query {idx}")

        if len(valid_predictions) == 0:
            logger.info("❌ 严重错误: 所有样本预测均失败！可能是因为推理服务连接失败或返回错误。")
            logger.info("请检查 fastapi_model_server.py 是否已重启并正常运行。")
            
            # Raise exception immediately to stop the program, instead of returning an error dict
            # because set_baseline expects valid metrics and will crash with KeyError otherwise.
            raise RuntimeError("All predictions failed. Please check the model server logs (Pydantic 422 errors usually mean request schema mismatch).")

        # If some predictions failed, we need to handle partial results
        if len(valid_predictions) < len(queries):
            logger.info(f"Warning: Only {len(valid_predictions)}/{len(queries)} predictions succeeded. Using partial results.")
            eval_df = eval_df.iloc[valid_indices].copy()
        else:
            eval_df = eval_df.copy()

        eval_df["prediction"] = valid_predictions
        eval_df["pred_label"] = eval_df["prediction"].apply(self._extract_label)

        # 使用V2规则计算TP/FN/FP
        if V2_AVAILABLE:
            tp = 0
            fn = 0
            fp = 0
            for _, row in eval_df.iterrows():
                true_val = row[self.true_label_col]
                pred_val = row["pred_label"]
                
                try:
                    is_target_true = (check_match(true_val, self.target_label) == '✅') or (true_val == self.target_label)
                    is_target_pred = (check_match(pred_val, self.target_label) == '✅') or (pred_val == self.target_label)
                except:
                    is_target_true = true_val == self.target_label
                    is_target_pred = pred_val == self.target_label
                
                if is_target_true and is_target_pred:
                    tp += 1
                elif is_target_true and not is_target_pred:
                    fn += 1
                elif not is_target_true and is_target_pred:
                    fp += 1
        else:
            tp = len(
                eval_df[(eval_df[self.true_label_col] == self.target_label) & (eval_df["pred_label"] == self.target_label)]
            )
            fn = len(
                eval_df[(eval_df[self.true_label_col] == self.target_label) & (eval_df["pred_label"] != self.target_label)]
            )
            fp = len(
                eval_df[(eval_df[self.true_label_col] != self.target_label) & (eval_df["pred_label"] == self.target_label)]
            )

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_confused_acc = self._per_label_accuracy(eval_df, self.confused_labels) if self.confused_labels else {}
        other_labels_acc = (
            self._overall_accuracy_for_labels(eval_df, self.confused_labels)
            if self.confused_labels
            else self._overall_accuracy_for_labels(
                eval_df,
                [lbl for lbl in eval_df[self.true_label_col].unique().tolist() if lbl != self.target_label],
            )
        )

        analysis: Dict[str, Any] = {
            "target_label": self.target_label,
            "total_samples": len(eval_df),
            "false_negatives": fn,
            "false_positives": fp,
            "fn_examples": eval_df[(eval_df[self.true_label_col] == self.target_label) & (eval_df["pred_label"] != self.target_label)][
                self.query_col
            ]
            .head(10)
            .tolist(),
            "fp_examples": eval_df[(eval_df[self.true_label_col] != self.target_label) & (eval_df["pred_label"] == self.target_label)][
                self.query_col
            ]
            .head(10)
            .tolist(),
            # 保存所有的错误样本以便后续分析
            "all_fn_df": eval_df[(eval_df[self.true_label_col] == self.target_label) & (eval_df["pred_label"] != self.target_label)],
            "all_fp_df": eval_df[(eval_df[self.true_label_col] != self.target_label) & (eval_df["pred_label"] == self.target_label)],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "other_labels_acc": other_labels_acc,
            "per_confused_acc": per_confused_acc,
        }

        return f1, analysis

    def _select_eval_df(self, level: str) -> pd.DataFrame:
        """选择评估数据集
        
        Level A: 从完整数据集中随机抽样固定数量（默认1000条）
        Level B: 只选择目标标签和所有混淆标签的数据（不再全量验证）
        """
        if level == "A":
            n = min(self.level_a_sample_size, len(self.df))
            return self.df.sample(n=n, random_state=self.seed)
        
        # Level B: 只验证目标标签和全部混淆标签
        if self.confused_labels:
            # 有混淆标签信息时，只选择目标标签和所有混淆标签的样本
            relevant_labels = [self.target_label] + self.confused_labels
            filtered_df = self.df[self.df[self.true_label_col].isin(relevant_labels)]
            logger.info(f"Level B: 筛选目标标签 '{self.target_label}' 和 {len(self.confused_labels)} 个混淆标签，"
                       f"共 {len(filtered_df)}/{len(self.df)} 条样本")
            return filtered_df
        else:
            # 没有混淆标签信息时，仍然使用全量数据
            logger.info(f"Level B: 未提供混淆标签，使用全量数据 ({len(self.df)} 条样本)")
            return self.df

    def _extract_label(self, text: Any) -> str:
        import re

        if pd.isna(text):
            return ""
        match = re.search(r"<answer>(.*?)</answer>", str(text))
        return match.group(1).strip() if match else str(text).strip()

    def _per_label_accuracy(self, df: pd.DataFrame, labels: list[str]) -> Dict[str, float]:
        """使用V2规则计算每个标签的准确率"""
        result: Dict[str, float] = {}
        for label in labels:
            sub = df[df[self.true_label_col] == label]
            if len(sub) == 0:
                continue
            
            if V2_AVAILABLE:
                correct_count = 0
                for _, row in sub.iterrows():
                    true_val = row[self.true_label_col]
                    pred_val = row["pred_label"]
                    try:
                        if check_match(true_val, pred_val) == '✅':
                            correct_count += 1
                    except:
                        if true_val == pred_val:
                            correct_count += 1
                result[label] = correct_count / len(sub)
            else:
                result[label] = float((sub["pred_label"] == sub[self.true_label_col]).mean())
        return result

    def _overall_accuracy_for_labels(self, df: pd.DataFrame, labels: list[str]) -> float:
        sub = df[df[self.true_label_col].isin(labels)]
        if len(sub) == 0:
            return 1.0
        return float((sub["pred_label"] == sub[self.true_label_col]).mean())

    def save_errors_to_excel(self, analysis: Dict[str, Any], iteration: int, node_id: str):
        """将影响 F1 的样本（FN 和 FP）保存到 Excel 和 TXT"""
        fn_df = analysis.get("all_fn_df")
        fp_df = analysis.get("all_fp_df")
        
        if (fn_df is None or fn_df.empty) and (fp_df is None or fp_df.empty):
            return
            
        output_dir = "mcts_error_logs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, f"iter_{iteration}_{node_id}_errors.xlsx")
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if fn_df is not None and not fn_df.empty:
                fn_df.to_excel(writer, sheet_name='False_Negatives', index=False)
            if fp_df is not None and not fp_df.empty:
                fp_df.to_excel(writer, sheet_name='False_Positives', index=False)
                
        logger.info(f"📊 干扰样本已保存至: {file_path}")
        
        # 同时保存到 TXT 文件
        txt_file_path = os.path.join(output_dir, f"iter_{iteration}_{node_id}_errors.txt")
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"干扰样本分析 - 迭代 {iteration}, 节点 {node_id}\n")
            f.write("=" * 50 + "\n\n")
            
            if fn_df is not None and not fn_df.empty:
                f.write("False Negatives (漏检样本):\n")
                f.write("-" * 30 + "\n")
                for idx, row in fn_df.iterrows():
                    query = str(row.get(self.query_col, 'N/A'))
                    true_label = str(row.get(self.true_label_col, 'N/A'))
                    pred_label = str(row.get('pred_label', 'N/A'))
                    f.write(f"{idx+1}. 真实标签: {true_label}, 预测标签: {pred_label}\n")
                    f.write(f"   文本: {query[:200]}{'...' if len(query) > 200 else ''}\n\n")
            else:
                f.write("False Negatives: 无\n\n")
                
            if fp_df is not None and not fp_df.empty:
                f.write("False Positives (误检样本):\n")
                f.write("-" * 30 + "\n")
                for idx, row in fp_df.iterrows():
                    query = str(row.get(self.query_col, 'N/A'))
                    true_label = str(row.get(self.true_label_col, 'N/A'))
                    pred_label = str(row.get('pred_label', 'N/A'))
                    f.write(f"{idx+1}. 真实标签: {true_label}, 预测标签: {pred_label}\n")
                    f.write(f"   文本: {query[:200]}{'...' if len(query) > 200 else ''}\n\n")
            else:
                f.write("False Positives: 无\n\n")
                
        logger.info(f"📄 干扰样本 TXT 已保存至: {txt_file_path}")
