#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享配置文件

集中管理所有项目级别的配置参数，确保多个文件使用一致的配置。

配置类别：
1. 模型配置 - 模型路径、类型等
2. 硬件配置 - GPU设置
3. 推理配置 - temperature、max_tokens等
4. 服务配置 - API地址、端口等
5. 数据配置 - 文件路径、列名等

使用方法：
    from shared_config import ModelConfig, InferenceConfig, ServiceConfig
    
    # 使用默认配置
    model_path = ModelConfig.MODEL_PATH
    
    # 或创建配置实例
    config = ModelConfig()
    model_path = config.MODEL_PATH

更新时间: 2026年1月25日
"""

import os
from typing import List, Dict


class ModelConfig:
    """模型相关配置"""
    
    # 模型路径（默认值）
    MODEL_PATH = '/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922'
    
    # 模型类型
    MODEL_TYPE = 'qwen3'  # 支持: qwen3, llama, etc.
    
    # 模型最小显存要求（GB）
    MIN_MEMORY_GB = 15
    
    @classmethod
    def get_model_path(cls) -> str:
        """获取模型路径"""
        return cls.MODEL_PATH
    
    @classmethod
    def set_model_path(cls, path: str):
        """设置模型路径"""
        cls.MODEL_PATH = path


class GPUConfig:
    """GPU硬件配置"""
    
    # FastAPI 服务使用的 GPU（用于多卡推理）
    FASTAPI_GPU_IDS = ['0', '1', '2', '3']
    
    # MCTS 评估使用的 GPU（可以和FastAPI不同）
    MCTS_GPU_IDS = ['0', '1', '2', '3', '4', '5', '6', '7']
    
    # 单机推理脚本默认GPU
    DEFAULT_GPU_IDS = ['0', '1', '2', '3']
    
    # 张量并行度（通常等于GPU数量）
    TENSOR_PARALLEL_SIZE = len(FASTAPI_GPU_IDS)
    
    @classmethod
    def get_fastapi_gpus(cls) -> List[str]:
        """获取FastAPI服务使用的GPU"""
        return cls.FASTAPI_GPU_IDS
    
    @classmethod
    def get_mcts_gpus(cls) -> str:
        """获取MCTS使用的GPU（逗号分隔字符串）"""
        return ','.join(cls.MCTS_GPU_IDS)
    
    @classmethod
    def get_default_gpus(cls) -> List[str]:
        """获取默认GPU列表"""
        return cls.DEFAULT_GPU_IDS


class InferenceConfig:
    """推理参数配置"""
    
    # FastAPI 分类推理参数（低温度确保稳定）
    FASTAPI_MAX_TOKENS = 500
    FASTAPI_TEMPERATURE = 0.01
    
    # 标准推理参数（用于单机脚本）
    STANDARD_MAX_TOKENS = 500
    STANDARD_TEMPERATURE = 0.7
    
    # 请求超时时间（秒）
    REQUEST_TIMEOUT = 120
    
    # VllmEngine 配置
    GPU_MEMORY_UTILIZATION = 0.9
    ENABLE_PREFIX_CACHING = True  # KV 缓存复用
    
    @classmethod
    def get_fastapi_config(cls) -> Dict:
        """获取FastAPI推理配置"""
        return {
            'max_tokens': cls.FASTAPI_MAX_TOKENS,
            'temperature': cls.FASTAPI_TEMPERATURE
        }


class V2ConfusionConfig:
    """V2混淆矩阵系统配置"""
    
    # 启用V2系统
    USE_V2_CONFUSION_ANALYSIS = True
    
    # 混淆分析配置
    CONFUSION_ANALYSIS_CONFIG = {
        "use_primary_labels": True,  # 使用主标签分析
        "use_secondary_labels": False,  # 是否分析次标签
        "use_risk_labels": False,  # 是否分析风险标签
        "max_error_samples": 10,  # 每个混淆对最多保存10个错误样本
        "output_sheets": ["mixed_confusion", "per_label_stats", "error_samples", "complete_analysis"]  # 输出Excel表单
    }
    
    # 评估器V2配置
    EVALUATOR_CONFIG_V2 = {
        "use_rule_based_matching": True,  # 使用规则基匹配而非字符串相等
        "tolerance_threshold": 0.01,  # 1%容差
        "equivalence_pair_weight": 1.0,  # 等价对权重
        "enable_caching": True  # 启用缓存
    }
    
    # 混淆矩阵文件路径模板
    CONFUSION_MATRIX_OUTPUT_TEMPLATE = "confusion_matrix_v2_{target_label}.xlsx"
    
    @classmethod
    def get_standard_config(cls) -> Dict:
        """获取标准推理配置"""
        return {
            'max_tokens': cls.STANDARD_MAX_TOKENS,
            'temperature': cls.STANDARD_TEMPERATURE
        }


class ServiceConfig:
    """服务相关配置"""
    
    # FastAPI 服务地址
    API_HOST = "0.0.0.0"
    API_URL = f"http://localhost:8000"
    
    # API 端点
    INFER_ENDPOINT = "/infer"
    HEALTH_ENDPOINT = "/health"
    
    @classmethod
    def get_api_url(cls) -> str:
        """获取API完整地址"""
        return cls.API_URL
    
    @classmethod
    def get_infer_url(cls) -> str:
        """获取推理端点完整URL"""
        return f"{cls.API_URL}{cls.INFER_ENDPOINT}"


class DataConfig:
    """数据相关配置"""
    
    # 默认列名
    QUERY_COLUMN = "query"
    LABEL_COLUMN = "label"
    TRUE_LABEL_COLUMN = "true_label"
    
    # 数据集路径（可选，根据需要设置）
    DATASET_DIR = "./datasets"
    OUTPUT_DIR = "./output"
    CACHE_DIR = "./mcts_cache"
    
    # Level A/B 抽样比例
    LEVEL_A_SAMPLE_RATIO = 0.3  # Level A 使用30%数据
    
    @classmethod
    def get_query_col(cls) -> str:
        """获取查询列名"""
        return cls.QUERY_COLUMN
    
    @classmethod
    def get_label_col(cls) -> str:
        """获取标签列名"""
        return cls.LABEL_COLUMN


class MCTSConfig:
    """MCTS算法配置"""
    
    # MCTS 核心参数
    MAX_ITERATIONS = 15
    C_PUCT = 1.4  # 探索常数
    
    # 评估参数
    DROP_TOLERANCE = 0.01  # 1% 容忍度
    
    # 缓存设置
    ENABLE_CACHE = True
    CACHE_FILE_PREFIX = "mcts_cache"
    
    @classmethod
    def get_max_iterations(cls) -> int:
        """获取最大迭代次数"""
        return cls.MAX_ITERATIONS
    
    @classmethod
    def get_c_puct(cls) -> float:
        """获取探索常数"""
        return cls.C_PUCT


# ==================== 便捷访问函数 ====================

def get_model_path() -> str:
    """获取模型路径"""
    return ModelConfig.get_model_path()


def get_api_url() -> str:
    """获取API服务地址"""
    return ServiceConfig.get_api_url()


def get_fastapi_gpus() -> List[str]:
    """获取FastAPI使用的GPU"""
    return GPUConfig.get_fastapi_gpus()


def get_mcts_gpus() -> str:
    """获取MCTS使用的GPU（逗号分隔）"""
    return GPUConfig.get_mcts_gpus()


def get_inference_config(service_type: str = 'fastapi') -> Dict:
    """
    获取推理配置
    
    Args:
        service_type: 'fastapi' 或 'standard'
    """
    if service_type == 'fastapi':
        return InferenceConfig.get_fastapi_config()
    else:
        return InferenceConfig.get_standard_config()


# ==================== 配置打印函数 ====================

def print_config():
    """打印当前所有配置"""
    print("=" * 60)
    print("当前共享配置")
    print("=" * 60)
    
    print("\n【模型配置】")
    print(f"  模型路径: {ModelConfig.MODEL_PATH}")
    print(f"  模型类型: {ModelConfig.MODEL_TYPE}")
    print(f"  最小显存: {ModelConfig.MIN_MEMORY_GB} GB")
    
    print("\n【GPU配置】")
    print(f"  FastAPI GPU: {GPUConfig.FASTAPI_GPU_IDS}")
    print(f"  MCTS GPU: {GPUConfig.MCTS_GPU_IDS}")
    print(f"  默认 GPU: {GPUConfig.DEFAULT_GPU_IDS}")
    print(f"  张量并行: {GPUConfig.TENSOR_PARALLEL_SIZE}")
    
    print("\n【推理配置】")
    print(f"  FastAPI: max_tokens={InferenceConfig.FASTAPI_MAX_TOKENS}, "
          f"temperature={InferenceConfig.FASTAPI_TEMPERATURE}")
    print(f"  标准: max_tokens={InferenceConfig.STANDARD_MAX_TOKENS}, "
          f"temperature={InferenceConfig.STANDARD_TEMPERATURE}")
    print(f"  超时时间: {InferenceConfig.REQUEST_TIMEOUT}秒")
    
    print("\n【服务配置】")
    print(f"  API地址: {ServiceConfig.API_URL}")
    print(f"  推理端点: {ServiceConfig.INFER_ENDPOINT}")
    print(f"  健康检查: {ServiceConfig.HEALTH_ENDPOINT}")
    
    print("\n【数据配置】")
    print(f"  查询列: {DataConfig.QUERY_COLUMN}")
    print(f"  标签列: {DataConfig.LABEL_COLUMN}")
    print(f"  数据集目录: {DataConfig.DATASET_DIR}")
    print(f"  输出目录: {DataConfig.OUTPUT_DIR}")
    
    print("\n【MCTS配置】")
    print(f"  最大迭代: {MCTSConfig.MAX_ITERATIONS}")
    print(f"  探索常数: {MCTSConfig.C_PUCT}")
    print(f"  下降容忍: {MCTSConfig.DROP_TOLERANCE}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 测试：打印所有配置
    print_config()
