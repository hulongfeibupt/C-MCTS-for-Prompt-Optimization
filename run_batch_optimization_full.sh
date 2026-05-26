#!/bin/bash

# MCTS 批量标签优化脚本 - 【全量版】
# 使用方法: bash run_batch_optimization_full.sh
# 覆盖率: 30 个标签（按错误率从高到低排序）

cd /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major || exit 1

echo "======================================================================"
echo "🚀 开始批量优化多标签提示词 【全量版】"
echo "======================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "预计耗时: ~4.5 小时"
echo ""

# ============================================================================
# 【配置区】全量优化 - 30 个标签（按错误率从高到低排序）
# ============================================================================


MCTS_LABELS="暴力犯罪-敲诈1千元以内 交通事故-轻微伤-门诊治疗-无需就医 盗窃-物品价值1000元以内 疑似迷药-疑似迷药 已自残自杀未死亡 \
             性骚扰-触碰隐私部位 性骚扰-语言骚扰-淫秽语言 限制人身自由-司乘已分离 交通事故-轻伤-住院治疗-观察 敏感身份进线-特殊人群 \
             群体事件-上访闹访 无风险 信息泄露-将用户信息发布到网上 酒驾-疑似酒驾 肢体冲突-打架-轻微拉扯-推搡-无需就医 \
             盗窃-物品价值1000元以上 跟踪尾随围堵 限制人身自由-司乘未分离 涉毒-疑似车内吸毒 扬言伤害他人 \
             肢体冲突-打架-轻微伤-门诊治疗-无需就医 失联-女性失联 延误行程乘客索赔 损毁财物-打砸车辆 意外受伤-轻微伤-门诊治疗-无需就医"

MCTS_ITERATIONS="5 10 3 3 3 \
                 10 10 10 10 5 \
                 5 20 5 10 10 \
                 3 10 5 5 10 \
                 8 8 8 8 8"


# 配置参数
DATA_PATH="/ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/datasets/major_02.xlsx"
MODEL_PATH="/ai/pretrain/ai-model-train-work-order-complaint-major-v3/0316_B_grpo_v4_1000steps"
#MODEL_PATH="/ai/pretrain/ai-model-train-work-order-complaint-major-v3/checkpoint-400-online"
MODEL_TYPE="qwen3"
CONFUSION_FILE="/ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/confusion_matrix/confusion_grpo-1000-22-37.xlsx"
GPU_IDS="4,5,6,7" #目前只能是4个，因为很多都写死了。
API_URL="http://localhost:8082"
PYTHON_BIN="python3"

# ============================================================================
# 执行批量优化
# ============================================================================

total_labels=0
for label in $MCTS_LABELS; do
    total_labels=$((total_labels + 1))
done

total_iterations=0
for iter in $MCTS_ITERATIONS; do
    total_iterations=$((total_iterations + iter))
done

# 逐个优化标签
label_idx=0
echo ""
echo "======================================================================"
echo "📊 混淆矩阵汇总（实时更新）"
echo "======================================================================"
printf "%-35s %10s %10s %10s %8s %8s %8s\n" "标签" "基线F1" "优化后F1" "ΔF1" "FN" "FP" "样本数"
echo "----------------------------------------------------------------------------------------------------"

for label in $MCTS_LABELS; do
    label_idx=$((label_idx + 1))
    
    # 获取对应的迭代次数
    iteration_idx=0
    iterations=""
    for iter in $MCTS_ITERATIONS; do
        iteration_idx=$((iteration_idx + 1))
        if [ $iteration_idx -eq $label_idx ]; then
            iterations=$iter
            break
        fi
    done
    
    echo ""
    echo "======================================================================"
    echo "📝 [$label_idx/$total_labels] 正在优化标签: $label (迭代: $iterations 次)"
    echo "======================================================================"
    $PYTHON_BIN main.py \
      --target_label "$label" \
      --data_path "$DATA_PATH" \
      --model_path "$MODEL_PATH" \
      --model_type "$MODEL_TYPE" \
      --confusion_file "$CONFUSION_FILE" \
      --iterations $iterations \
      --gpu_ids "$GPU_IDS" \
      --api_url "$API_URL"
    
    echo ""
    echo "✓ [$label_idx/$total_labels] $label - 已完成"
    
    # 读取混淆矩阵摘要并展示
    SUMMARY_FILE="summary_${label}.json"
    if [ -f "$SUMMARY_FILE" ]; then
        $PYTHON_BIN -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    s = json.load(f)
gain = s.get('f1_gain', 0)
marker = '✅' if gain >= 0 else '⚠️'
print(f\"{'$label':35s} {s.get('baseline_f1','?'):>10} {s.get('final_f1','?'):>10} {gain:>+9.4f} {marker} {s.get('false_negatives','?'):>7} {s.get('false_positives','?'):>7} {s.get('total_samples','?'):>7}\")
"
    else
        printf "%-35s %10s %10s %10s %8s %8s %8s\n" "$label" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "======================================================================"
echo "🎉 全量优化完成！"
echo "======================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 优化统计:"
echo "  • 优化标签数: $total_labels"
echo "  • 总迭代数: $total_iterations"
echo "  • 覆盖率: 100%"
echo "  • 预期性能提升: +10-15% 宏召回率"
echo ""
echo "✨ 优化结果已保存到 label_prompts_config.py"
echo "📁 日志文件位置: ./mcts_logs/"
echo ""


# 第零步先清洗：
# conda activate q3_1 && sh /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/restart_server.sh

# 第一步先启动：
# cd /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major && conda activate q3_1 && python fastapi_model_server.py

# 第二步才运行这个；
# conda activate q3_1 && cd /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major && sh run_batch_optimization_full.sh

# 在这个上继续迭代。