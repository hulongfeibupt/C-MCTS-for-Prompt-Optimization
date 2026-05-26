#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试评估器 - 批量评估所有标签的性能
只运行一次推理，将结果添加到原始Excel的最后一列，然后调用 calculate_sft_accuracy.py 计算准确率
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
import subprocess
from tqdm import tqdm

# 将当前目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入必要的模块
from evaluator import Evaluator, worker_gpu
from concurrent.futures import ProcessPoolExecutor


def run_inference(data_path: str, model_path: str, gpu_ids: str, 
                 model_type: str, api_url: str) -> Tuple[pd.DataFrame, List[Optional[str]]]:
    """
    运行一次完整推理
    
    Returns:
        (原始DataFrame, 预测结果列表)
    """
    print(f"\n{'='*80}")
    print("开始全量数据推理（仅运行一次）")
    print(f"{'='*80}")
    
    # 读取原始数据
    df = pd.read_excel(data_path)
    print(f"数据集总样本数: {len(df)}")
    
    # 获取列名配置
    evaluator = Evaluator(
        model_path=model_path,
        data_path=data_path,
        target_label="无风险",  # 随便用一个标签获取配置
        confusion_file=None,
        gpu_ids=gpu_ids,
        model_type=model_type,
        api_url=api_url
    )
    
    queries = df[evaluator.query_col].tolist()
    
    print(f"开始并行推理 {len(queries)} 个样本，使用 GPU: {gpu_ids}")
    
    # 并行推理
    gpu_list = gpu_ids.split(",")
    num_gpus = len(gpu_list)
    batch_size = (len(queries) + num_gpus - 1) // num_gpus
    query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    while len(query_batches) < num_gpus:
        query_batches.append([])
    
    predictions: List[Optional[str]] = [None] * len(queries)
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, gpu_id in enumerate(gpu_list):
            if query_batches[i]:
                future = executor.submit(
                    worker_gpu,
                    query_batches[i],
                    model_path,
                    gpu_id,
                    model_type,
                    "",  # 不覆盖提示词
                    "",  # 不指定目标标签
                    api_url
                )
                futures.append((future, i))
        
        # 收集结果（带进度条）
        with tqdm(total=len(futures), desc="推理进度", unit="批次") as pbar:
            for future, batch_idx in futures:
                try:
                    result = future.result()
                    if result["success"] and result["predictions"]:
                        start_idx = batch_idx * batch_size
                        for j, pred in enumerate(result["predictions"]):
                            if start_idx + j < len(predictions):
                                predictions[start_idx + j] = pred
                    pbar.update(1)
                except Exception as e:
                    print(f"批次 {batch_idx} 推理失败: {e}")
                    pbar.update(1)
    
    print(f"✓ 推理完成")
    
    return df, predictions


def main():
    parser = argparse.ArgumentParser(description="批量评估所有标签的性能")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_type", type=str, default="qwen3", help="模型类型")
    parser.add_argument("--gpu_ids", type=str, default="4,5,6,7", help="GPU ID列表")
    parser.add_argument("--api_url", type=str, default="http://localhost:8082", help="FastAPI服务地址")
    parser.add_argument("--prediction_col", type=str, default="qwen3-14b预测", help="预测结果列名")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"批量标签评估工具")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {args.data_path}")
    print(f"模型: {args.model_path}")
    print(f"GPU: {args.gpu_ids}")
    
    # 步骤1: 运行推理
    df, predictions = run_inference(
        args.data_path,
        args.model_path,
        args.gpu_ids,
        args.model_type,
        args.api_url
    )
    
    # 步骤2: 将预测结果添加到最后一列
    df[args.prediction_col] = predictions
    
    # 步骤3: 保存到新Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"evaluation_result_{timestamp}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✓ 结果已保存到: {output_path}")
    
    # 步骤4: 调用 calculate_sft_accuracy.py 计算准确率
    print(f"\n{'='*80}")
    print("调用 calculate_sft_accuracy.py 计算准确率")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            ["python", "calculate_sft_accuracy.py", output_path, 
             "--prediction-col", args.prediction_col],
            cwd=current_dir,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✓ 准确率计算完成")
        else:
            print(f"\n⚠️ 准确率计算返回非零退出码: {result.returncode}")
    except Exception as e:
        print(f"\n❌ 调用 calculate_sft_accuracy.py 失败: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果文件: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


"""
使用示例:

conda activate q3_1 && cd /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/

python test_evaluator.py \
  --data_path "/ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/datasets/major_02.xlsx" \
  --model_path "/ai/pretrain/ai-model-train-work-order-complaint-major-v3/0316_B_grpo_v4_1000steps" \
  --model_type "qwen3" \
  --gpu_ids "4,5,6,7" \
  --api_url "http://localhost:8082"

  
"""
