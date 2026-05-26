#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估所有标签的精确率和召回率
读取Excel文件，使用FastAPI服务进行推理，然后调用calculate_sft_accuracy.py计算精确率和召回率
"""

import os
import sys
import pandas as pd
import requests
from datetime import datetime
import re
from typing import Dict, List

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入准确率计算模块
from calculate_sft_accuracy import calculate_accuracy_with_file


def read_excel_file(file_path: str) -> pd.DataFrame:
    """读取Excel文件"""
    print(f"正在读取文件: {file_path}")
    df = pd.read_excel(file_path)
    print(f"成功读取 {len(df)} 条数据")
    return df


def call_api_for_prediction(text: str, api_url: str = "http://localhost:8000", timeout: int = 120) -> str:
    """
    调用FastAPI接口获取预测结果
    
    Args:
        text: 输入文本
        api_url: API服务地址
        timeout: 超时时间（秒）
        
    Returns:
        预测结果（提取<answer>标签内的内容）
    """
    try:
        payload = {
            "text": text
        }
        response = requests.post(f"{api_url}/infer", json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "")
            
            # 提取<answer>标签内的内容
            match = re.search(r'<answer>(.*?)</answer>', prediction)
            if match:
                return match.group(1).strip()
            else:
                return prediction.strip()
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            return f"<ERROR: {response.status_code}>"
            
    except requests.RequestException as e:
        print(f"请求异常: {e}")
        return f"<ERROR: {str(e)}>"
    except Exception as e:
        print(f"预测失败: {e}")
        return f"<ERROR: {str(e)}>"


def batch_inference(df: pd.DataFrame, query_col: str = "query", api_url: str = "http://localhost:8000") -> List[str]:
    """
    批量推理
    
    Args:
        df: 数据DataFrame
        query_col: 包含查询文本的列名
        api_url: API服务地址
        
    Returns:
        预测结果列表
    """
    print(f"\n开始批量推理 {len(df)} 条数据...")
    predictions = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"进度: {idx}/{len(df)}")
        
        query = row[query_col]
        prediction = call_api_for_prediction(query, api_url)
        predictions.append(prediction)
    
    print(f"批量推理完成！")
    return predictions


def save_predictions_to_excel(df: pd.DataFrame, predictions: List[str], output_path: str, prediction_col_name: str = "qwen3-14b预测"):
    """
    将预测结果保存到新的Excel文件
    
    Args:
        df: 原始DataFrame
        predictions: 预测结果列表
        output_path: 输出文件路径
        prediction_col_name: 预测结果列名
    """
    # 创建新的DataFrame，包含预测结果
    df_output = df.copy()
    df_output[prediction_col_name] = predictions
    
    # 保存到Excel
    df_output.to_excel(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")


def main(excel_file_path: str, api_url: str = "http://localhost:8000", output_dir: str = "output"):
    """
    主函数：读取Excel，调用API推理，计算精确率和召回率
    
    Args:
        excel_file_path: 输入的Excel文件路径
        api_url: FastAPI服务地址
        output_dir: 输出目录
    """
    print("=" * 80)
    print("批量评估所有标签的精确率和召回率")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. 读取Excel文件
    df = read_excel_file(excel_file_path)
    
    # 检查必需的列
    if "query" not in df.columns:
        raise ValueError("Excel文件中缺少'query'列")
    if "人工标注标签" not in df.columns:
        raise ValueError("Excel文件中缺少'人工标注标签'列")
    
    # 2. 批量推理
    predictions = batch_inference(df, query_col="query", api_url=api_url)
    
    # 3. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 4. 保存预测结果到临时Excel文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_output_path = os.path.join(output_dir, f"predictions_temp_{timestamp}.xlsx")
    save_predictions_to_excel(df, predictions, temp_output_path, prediction_col_name="qwen3-14b预测")
    
    # 5. 调用calculate_sft_accuracy.py计算精确率和召回率
    print("\n" + "=" * 80)
    print("开始计算精确率和召回率...")
    print("=" * 80)
    
    accuracy, accuracy_report_path = calculate_accuracy_with_file(
        filename=temp_output_path,
        prediction_col="qwen3-14b预测",
        human_label_col="人工标注标签"
    )
    
    # 6. 重命名准确率报告文件
    final_report_path = os.path.join(output_dir, f"accuracy_report_{timestamp}.xlsx")
    if os.path.exists(accuracy_report_path):
        import shutil
        shutil.move(accuracy_report_path, final_report_path)
        print(f"\n最终报告已保存到: {final_report_path}")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"预测结果文件: {temp_output_path}")
    print(f"精确率召回率报告: {final_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量评估所有标签的精确率和召回率")
    parser.add_argument("excel_file", help="输入的Excel文件路径")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    
    args = parser.parse_args()
    
    # API URL 作为初始变量，默认使用本地8000端口
    API_URL = "http://localhost:8000"
    
    main(
        excel_file_path=args.excel_file,
        api_url=API_URL,
        output_dir=args.output_dir
    )

# 使用方法：

# python batch_evaluate_labels.py "/ai/ltx/zhongda/datasets/预测校正-1206.xlsx" --output-dir output