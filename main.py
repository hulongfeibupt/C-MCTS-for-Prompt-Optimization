import os
import sys
import argparse
import re
import json
from datetime import datetime

# 将当前目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 将父目录添加到 sys.path 以便导入其他模块 (如果需要)
sys.path.append(os.path.dirname(current_dir))

from mcts_core import MCTS
from action_generator import ActionGenerator
from evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="MCTS Prompt Generator")
    parser.add_argument("--target_label", type=str, required=True, help="需要优化的目标标签名称")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据集路径 (Excel/CSV)")
    parser.add_argument("--model_path", type=str, required=True, help="用于评估的模型路径")
    parser.add_argument("--model_type", type=str, default="qwen3", help="模型类型，如 qwen3, qwen2_5 等")
    parser.add_argument("--confusion_file", type=str, default=None, help="混淆矩阵文件路径（强烈推荐）")
    parser.add_argument("--auto_generate_confusion", type=bool, default=False, help="是否自动生成混淆矩阵（需要提供data_path和target_label）")
    parser.add_argument("--use_v2_evaluation", type=bool, default=True, help="是否使用V2规则基评估（默认True）")
    parser.add_argument("--iterations", type=int, default=100, help="MCTS 迭代次数（推荐50-200次，机器运算成本低）")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="使用的 GPU ID 列表")
    parser.add_argument("--api_url", type=str, default="http://localhost:8082", help="FastAPI 推理服务地址")
    return parser.parse_args()

def get_config_file_path():
    """获取 label_prompts_config.py 的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "label_prompts_config.py")

def extract_initial_prompt_from_config(target_label):
    """
    从 label_prompts_config.py 中的 LABEL_PROMPTS 字典提取初始 prompt
    """
    config_path = get_config_file_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 直接从文件读取 LABEL_PROMPTS 定义
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 LABEL_PROMPTS 字典中的目标标签
    pattern = rf'"{re.escape(target_label)}"\s*:\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"未在 label_prompts_config.py 的 LABEL_PROMPTS 中找到 '{target_label}'")

def write_back_optimized_prompt(target_label, optimized_prompt):
    """
    将优化后的 prompt 写回 label_prompts_config.py 的 LABEL_PROMPTS 字典
    """
    config_path = get_config_file_path()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换对应标签的 prompt（保持三引号格式）
    pattern = rf'("{re.escape(target_label)}"\s*:\s*""").*?(""")'
    
    def repl_func(match):
        return match.group(1) + optimized_prompt + match.group(2)
    
    new_content = re.sub(pattern, repl_func, content, flags=re.DOTALL)
    
    # 备份原文件
    backup_path = config_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份原文件到: {backup_path}")
    
    # 写回修改
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"✓ 已更新 label_prompts_config.py 中 '{target_label}' 的 prompt")

def main():
    args = parse_args()
    
    print(f"=== Starting MCTS Prompt Optimization for Label: {args.target_label} ===")
    print(f"Time: {datetime.now()}")
    print(f"Config File: {get_config_file_path()}")
    print(f"V2 Evaluation Enabled: {args.use_v2_evaluation}")
    
    # 自动生成混淆矩阵（如果需要）
    if args.auto_generate_confusion and not args.confusion_file:
        try:
            from confusion_matrix.generate_confusion_analysis_v2 import generate_confusion_analysis
            print("\n🔄 Generating confusion matrix using V2 system...")
            confusion_file = generate_confusion_analysis(
                data_path=args.data_path,
                output_path="confusion_matrix_v2.xlsx",
                target_labels=[args.target_label]
            )
            args.confusion_file = confusion_file
            print(f"✅ Confusion matrix generated: {confusion_file}")
        except Exception as e:
            print(f"⚠️ Failed to auto-generate confusion matrix: {e}")
    
    # 1. 从 label_prompts_config.py 读取初始 prompt
    print("\nLoading initial prompt from label_prompts_config.py...")
    initial_prompt = extract_initial_prompt_from_config(args.target_label)
    print(f"Initial Prompt Length: {len(initial_prompt)} chars")
    print(f"Preview: {initial_prompt[:200]}...")

    # 2. 初始化组件
    print("Initializing components...")
    
    # Action Generator (DeepSeek) - 使用固定的 API Key
    DEEPSEEK_API_KEY = "sk-gzhFbRu8rikjl4N606EcB46974264f098246D1EeE59eCc20"
    action_gen = ActionGenerator(api_key=DEEPSEEK_API_KEY)
    
    # Evaluator (推理模型)
    # 注意: Evaluator 内部会初始化模型，这可能需要一些时间
    evaluator = Evaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        target_label=args.target_label,
        confusion_file=args.confusion_file,
        gpu_ids=args.gpu_ids,
        model_type=args.model_type,
        api_url=args.api_url
    )
    
    # 打印混淆标签信息
    if evaluator.confused_labels:
        print(f"\n📊 混淆标签分析:")
        print(f"   容易与 '{args.target_label}' 混淆的标签: {', '.join(evaluator.confused_labels)}")
        print(f"   优化时将重点关注这些标签的准确率\n")
    else:
        print(f"\n⚠️  未提供混淆矩阵文件，将监控所有标签的准确率\n")
    
    # MCTS Engine
    mcts = MCTS(
        root_prompt=initial_prompt,
        evaluator=evaluator,
        action_generator=action_gen,
        c_puct=2.0 # 探索系数提高，鼓励更多探索（机器成本低，不怕迭代多）
    )
    
    # 2.5. 设置 baseline（Level A 和 Level B 分别设置）
    print("\n=== Setting Baselines ===")
    print("Setting Level A baseline...")
    evaluator.set_baseline(initial_prompt, level="A")
    print("Setting Level B baseline...")
    evaluator.set_baseline(initial_prompt, level="B")
    
    # 3. 运行 MCTS
    print(f"Starting MCTS search for {args.iterations} iterations...")
    best_path = mcts.search(iterations=args.iterations)
    
    # 4. 输出结果
    print("\n=== Optimization Finished ===")
    if not best_path:
        print("No valid path found.")
        return

    best_node = best_path[-1]
    print(f"Best Node Found:")
    print(f"  - Visits (N): {best_node.N}")
    print(f"  - Value (Q): {best_node.Q:.4f}")
    print(f"  - Action Taken: {best_node.action_taken}")
    
    # 保存混淆矩阵摘要 JSON（供 shell 脚本读取）
    ea = getattr(best_node, 'error_analysis', {}) or {}
    bl = evaluator._baseline_by_level.get("A")
    summary = {
        "baseline_f1": bl.target_f1 if bl else None,
        "final_f1": ea.get("f1", best_node.Q),
        "f1_gain": (ea.get("f1", best_node.Q) - bl.target_f1) if bl and ea else None,
        "false_negatives": ea.get("false_negatives"),
        "false_positives": ea.get("false_positives"),
        "total_samples": ea.get("total_samples"),
    }
    with open(f"summary_{args.target_label}.json", "w") as sf:
        json.dump(summary, sf)
    
    # 4.1 保存到独立文本文件（记录）
    output_file = f"optimized_prompt_{args.target_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Target Label: {args.target_label}\n")
        f.write(f"Final Score: {best_node.Q:.4f}\n")
        f.write("="*50 + "\n")
        f.write(best_node.prompt_state)
    print(f"✓ Optimized prompt saved to: {output_file}")
    
    # 4.2 写回 label_prompts_config.py 的 LABEL_PROMPTS
    print("\n=== Writing Back to label_prompts_config.py ===")
    write_back_optimized_prompt(args.target_label, best_node.prompt_state)
    print("\n✅ 优化完成！新 prompt 已更新到 label_prompts_config.py，所有引用文件会自动使用更新后的提示词")

if __name__ == "__main__":
    main()


"""

conda activate q3_1 && cd /ai/ltx/zhongda/mcts_prompt_gen_v4

python main.py \
  --target_label "伤害他人" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"


# 意外受伤


python main.py \
  --target_label "意外受伤" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"


# 行程延误

python main.py \
  --target_label "行程延误" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"

# 限制人身自由

python main.py \
  --target_label "限制人身自由" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"


# 性骚扰
python main.py \
  --target_label "性骚扰" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"

# 无风险
python main.py \
  --target_label "无风险" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 10 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"



conda activate q3_1 && cd /ai/ltx/zhongda/mcts_prompt_gen_v4

# 交通事故
python main.py \
  --target_label "交通事故" \
  --data_path "/ai/ltx/zhongda/datasets/与训练集同分布测试集-正负样本-已质检-纠正-干净版本-0105.xlsx" \
  --model_path "/ai/ltx/zhongda/weight_file/merged_global_step_1900-8922" \
  --model_type "qwen3" \
  --confusion_file "/ai/ltx/zhongda/datasets/label_confusion_analysis.xlsx" \
  --iterations 20 \
  --gpu_ids "0,1,2,3" \
  --api_url "http://localhost:8000"

# 当一个完成之后，看前部标签的运行。然后。看看 交通事故 的F1 是多少

"""

