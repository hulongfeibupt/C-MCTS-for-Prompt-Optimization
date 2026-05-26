# fastapi_model_server.py
# 独立的 FastAPI 服务，启动时加载模型权重，支持多卡推理

import os
import re
import json
import torch
import copy
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swift.llm import VllmEngine, InferRequest, RequestConfig
from typing import List, Dict, Any, Optional

# 导入共享的标签提示词配置
from label_prompts_config import LABEL_PROMPTS

# 配置参数
#MODEL_CKPT_DIR = "/ai/pretrain/ai-model-train-work-order-complaint-major-v3/0316_B_grpo_v4_1000steps"
MODEL_CKPT_DIR ="/ai/pretrain/ai-model-train-work-order-complaint-major-v3/checkpoint-400-online"
# GPU 配置
GPU_IDS = ['4', '5', '6', '7']
TENSOR_PARALLEL_SIZE = len(GPU_IDS)


def postprocess_label(init_label):
    """
    对模型输出的标签进行后处理
    
    Args:
        init_label: 模型原始输出（含 <think>...</think><answer>...</answer>）
        
    Returns:
        处理后的纯标签字符串，不含任何 XML 标签
    """
    valid_labels = list(LABEL_PROMPTS.keys())
    raw = str(init_label)

    # === 第一步：精准从 <answer>...</answer> 提取，彻底隔离 <think> 内容 ===
    match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
    else:
        # 无 <answer> 标签时，去除所有 XML 标签后取剩余文本
        final_answer = re.sub(r'<[^>]+>', '', raw).strip()

    # === 第二步：外投标签互斥规则（仅在 answer 内容中判断）===
    try:
        public_labels = [
            '已发生政府渠道外投',
            '已发生非政府渠道外投',
            '扬言政府渠道外投',
            '扬言非政府渠道外投'
        ]
        present = [lbl for lbl in public_labels if lbl in final_answer]
        if len(present) >= 2:
            preserved = None
            if '已发生政府渠道外投' in present and '已发生非政府渠道外投' in present:
                preserved = '已发生政府渠道外投'
            elif '已发生非政府渠道外投' in present and '扬言政府渠道外投' in present:
                preserved = '已发生非政府渠道外投'
            elif '扬言政府渠道外投' in present and '扬言非政府渠道外投' in present:
                preserved = '扬言政府渠道外投'
            else:
                for lbl in public_labels:
                    if lbl in present:
                        preserved = lbl
                        break
            if preserved is not None:
                return preserved
    except Exception:
        pass

    # === 第三步：白名单校验 ===
    if final_answer in valid_labels:
        return final_answer

    # 特殊情况兜底
    if "性骚扰" in final_answer and "骚扰乘客" in final_answer:
        return "骚扰-司机骚扰乘客"

    # 最长子串匹配（优先长标签，避免短标签误匹配）
    for valid_l in sorted(valid_labels, key=lambda x: -len(x)):
        if valid_l in final_answer:
            return valid_l

    return "无风险"


def build_system_prompt(selected_labels: Optional[List[str]] = None, label_overrides: Optional[Dict[str, str]] = None) -> str:
    """构建系统提示词"""
    current_prompts = LABEL_PROMPTS.copy()
    
    if label_overrides:
        current_prompts.update(label_overrides)
        
    if selected_labels is None:
        selected_labels = list(current_prompts.keys())

    labels_content = "\n            ".join([
        current_prompts[label] 
        for label in selected_labels 
        if label in current_prompts
    ])

    system_prompt = f"""
            首先你是一个人，同时也是一名优秀的网约车客服人员，主要负责乘客端重大风险业务，你能够依据如下规则,判断当前乘客输入工单属于哪一种重大风险类别。
            
            {labels_content}
            
            # 输出要求
            1. 所有输出标签必须属于数组 {list(current_prompts.keys())} 标签。
            2. 必须且只能在开头使用 <think> </think> 标签包裹你的思考过程。
            3. 输出预测的结果，必须放到 <answer> </answer> 之中，绝不能遗漏。

            
            # 强化示例 (你必须遵循此格式)
            输入信息: < 刚上车司机就锁门了，说不给加50块钱就不让我下车！ >
            <think>用户明确表示正在车上，且司机由于要求加钱锁住车门不让下车，司乘尚未分离，符合"限制人身自由-司乘未分离"标签定义。</think>
            <answer>限制人身自由-司乘未分离</answer>
            """
    return system_prompt


# FastAPI 应用
app = FastAPI(title="文本分类推理服务", description="基于 Qwen3-14B 的网约车风险分类服务")

model_engine = None


class InferenceRequest(BaseModel):
    text: str
    selected_labels: Optional[List[str]] = None
    label_overrides: Optional[Dict[str, str]] = None


class InferenceResponse(BaseModel):
    prediction: str
    raw_response: str = ""


@app.on_event("startup")
async def load_model():
    """启动时加载模型权重"""
    global model_engine
    try:
        print(f"正在加载模型权重到 GPU: {GPU_IDS} ...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_IDS)

        model_engine = VllmEngine(
            model_id_or_path=MODEL_CKPT_DIR,
            model_type='qwen3',
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        )
        print("✅ 模型加载完成！")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """推理接口"""
    if model_engine is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        query = f"""输入信息: < {request.text} > """
        
        system_prompt = build_system_prompt(request.selected_labels, request.label_overrides)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]

        request_config = RequestConfig(
            max_tokens=2000,
            temperature=0.1,
            top_p=0.1,
            repetition_penalty=1.05
        )
        infer_req = InferRequest(messages=messages)

        resp_list = model_engine.infer([infer_req], request_config)
        raw_response = resp_list[0].choices[0].message.content
        
        # 后处理
        label = postprocess_label(raw_response)

        return InferenceResponse(
            prediction=label,
            raw_response=raw_response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/")
async def root():
    return {"message": "FastAPI Model Server is Running", "endpoints": {"/infer": "POST", "/health": "GET"}}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_engine is not None}


@app.post("/reload_config")
async def reload_config():
    """重新加载 label_prompts_config.py 配置（接受提示词更新后调用）"""
    global LABEL_PROMPTS
    import importlib
    import label_prompts_config as _lpc_module
    importlib.reload(_lpc_module)
    LABEL_PROMPTS = _lpc_module.LABEL_PROMPTS
    print(f"✓ 配置已重载，标签数: {len(LABEL_PROMPTS)}")
    return {"status": "reloaded", "label_count": len(LABEL_PROMPTS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)


# 输出长度，依然可能是对最终指标产生影响的因素之一，因为之前的识别就搞错很多事情，况且识别等等以后一定要进行日志分析，这点一定要做。
"""
Step 1: 全量推理建立基线
======================================================================
✓ 基线配置已备份: /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/label_prompts_config.py.baseline
正在读取数据集: /ai/ltx/zhongda3_0/mcts_prompt_gen_v4_major/datasets/major_02.xlsx
数据量: 2579 条
开始并发推理 (并发数: 15) ...
  推理进度: 100/2579 (3.9%)
  推理进度: 200/2579 (7.8%)
  推理进度: 300/2579 (11.6%)
  推理进度: 400/2579 (15.5%)
  推理进度: 500/2579 (19.4%)
  推理进度: 600/2579 (23.3%)
  推理进度: 700/2579 (27.1%)
  推理进度: 800/2579 (31.0%)
  推理进度: 900/2579 (34.9%)
  推理进度: 1000/2579 (38.8%)
  推理进度: 1100/2579 (42.7%)
  推理进度: 1200/2579 (46.5%)
  推理进度: 1300/2579 (50.4%)
  推理进度: 1400/2579 (54.3%)
  推理进度: 1500/2579 (58.2%)
  推理进度: 1600/2579 (62.0%)
  推理进度: 1700/2579 (65.9%)
  推理进度: 1800/2579 (69.8%)
  推理进度: 1900/2579 (73.7%)
  推理进度: 2000/2579 (77.5%)
  推理进度: 2100/2579 (81.4%)
  推理进度: 2200/2579 (85.3%)
  推理进度: 2300/2579 (89.2%)
  推理进度: 2400/2579 (93.1%)
  推理进度: 2500/2579 (96.9%)


=== 整体指标 ===
总样本数:        2579
预测正确数:      2434
预测错误数:      145
整体准确率:      0.9438  (94.38%)
宏平均精确率:    0.8929  (89.29%)
宏平均召回率:    0.8626  (86.26%)
宏平均F1:        0.8700  (87.00%)
加权平均精确率:  0.9450  (94.50%)
加权平均召回率:  0.9411  (94.11%)
加权平均F1:      0.9414  (94.14%)

=== 各标签指标 ===
                          样本数(support)    精确率%    召回率%     F1%
标签                                                            
无风险                               1434   98.12   98.26   98.19
敏感身份进线-警察来电调取信息-非遗失物品               70   94.20   92.86   93.53
肢体冲突-打架-轻微拉扯-推搡-无需就医                61   86.67   85.25   85.95
交通事故-纯车损                            55   91.07   92.73   91.89
意外受伤-轻微伤-门诊治疗-无需就医                  54   88.89   88.89   88.89
限制人身自由-司乘已分离                        49   86.27   89.80   88.00
失联-未成年人失联                           43   93.48  100.00   96.63
敏感身份进线-警察来电调取信息-遗失物品                42   95.35   97.62   96.47
性骚扰-触碰隐私部位                          37   96.55   75.68   84.85
扬言伤害他人                              37   94.29   89.19   91.67
损毁财物-其他财物-除车辆外                      36   97.06   91.67   94.29
酒驾-疑似酒驾                             35   93.94   88.57   91.18
交通事故-轻伤-住院治疗-观察                     35   87.50   80.00   83.58
失联-女性失联                             34  100.00   97.06   98.51
骚扰-司机骚扰乘客                           34   82.05   94.12   87.67
扬言自残自杀                              33  100.00   96.97   98.46
调取流水-司机来电调取流水                       33   96.97   96.97   96.97
合规问题-司机不合规                          30  100.00   93.33   96.55
性骚扰-语言骚扰-淫秽语言                       29   92.59   86.21   89.29
跟踪尾随围堵                              28   82.76   85.71   84.21
延误行程乘客索赔                            28   85.71   85.71   85.71
交通事故-轻微伤-门诊治疗-无需就医                  27   76.00   70.37   73.08
损毁财物-打砸车辆                           27   78.12   92.59   84.75
敏感身份进线-特殊人群                         25   95.24   80.00   86.96
信息泄露-将用户信息发布到网上                     24   86.36   79.17   82.61
肢体冲突-打架-轻微伤-门诊治疗-无需就医               20   68.00   85.00   75.56
失联-其他失联                             20  100.00  100.00  100.00
一键报警                                18  100.00   94.44   97.14
性骚扰-触碰身体非隐私部位                       17   58.62  100.00   73.91
持危险物品-刀-棒-砖头等威胁-持危险物品未伤人            17  100.00   94.12   96.97
限制人身自由-司乘未分离                        15   72.22   86.67   78.79
意外受伤-轻伤-住院治疗-观察                     15   82.35   93.33   87.50
肢体冲突-打架-轻伤-住院治疗-观察                  13   85.71   92.31   88.89
群体事件-上访闹访                           10   81.82   90.00   85.71
涉毒-疑似车内吸毒                            9  100.00  100.00  100.00
偷拍-没有将信息发布到网上                        9   66.67   44.44   53.33
暴力犯罪-敲诈1千元以内                         8   71.43   62.50   66.67
性骚扰-暴露隐私部位或手淫                        7   87.50  100.00   93.33
暴力犯罪-抢劫财物                            7  100.00  100.00  100.00
盗窃-物品价值1000元以内                       6  100.00   66.67   80.00
交通事故-重伤-ICU-致残                       6  100.00   83.33   90.91
盗窃-物品价值1000元以上                       6   71.43   83.33   76.92
诈骗-诈骗3万元以内                           5  100.00   60.00   75.00
交通事故-死亡                              5  100.00   60.00   75.00
酒驾-酒驾被抓                              4   80.00  100.00   88.89
群体事件-线上罢工                            4  100.00   50.00   66.67
性骚扰-发送黄色信息或图片                        3  100.00  100.00  100.00
疑似迷药-疑似迷药                            3   66.67   66.67   66.67
性骚扰-强奸或强奸未遂                          3  100.00  100.00  100.00
已自残自杀未死亡                             3  100.00   66.67   80.00
暴力犯罪-绑架                              3  100.00  100.00  100.00
行程中-疑似存在人身安全                         2   50.00   50.00   50.00
意外受伤-重伤-ICU-致残                       1  100.00  100.00  100.00
暴力犯罪-敲诈1千元以上                         1  100.00  100.00  100.00
酒驾-酒驾乘客有证据                           0    0.00    0.00    0.00

=== 错误分析 (共 145 条) ===
  '无风险': 错误 25/1434 (1.74%)
  '性骚扰-触碰隐私部位': 错误 9/37 (24.32%)
  '肢体冲突-打架-轻微拉扯-推搡-无需就医': 错误 9/60 (15.00%)
  '交通事故-轻微伤-门诊治疗-无需就医': 错误 8/27 (29.63%)
  '交通事故-轻伤-住院治疗-观察': 错误 7/35 (20.00%)
  '意外受伤-轻微伤-门诊治疗-无需就医': 错误 6/54 (11.11%)
  '偷拍-没有将信息发布到网上': 错误 5/9 (55.56%)
  '敏感身份进线-警察来电调取信息-非遗失物品': 错误 5/70 (7.14%)
  '敏感身份进线-特殊人群': 错误 5/25 (20.00%)
  '酒驾-疑似酒驾': 错误 4/35 (11.43%)
  '跟踪尾随围堵': 错误 4/28 (14.29%)
  '信息泄露-将用户信息发布到网上': 错误 4/23 (17.39%)
  '延误行程乘客索赔': 错误 4/28 (14.29%)
  '交通事故-纯车损': 错误 4/55 (7.27%)
  '性骚扰-语言骚扰-淫秽语言': 错误 4/29 (13.79%)

详细结果已保存到: accuracy_analysis_result_20260429_172315.xlsx


"""