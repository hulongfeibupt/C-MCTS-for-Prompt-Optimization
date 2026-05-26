import math
import copy
import numpy as np
import traceback
import logging

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

class TreeNode:
    def __init__(self, prompt_state, parent=None, prior_prob=0.0, action_taken=None):
        """
        MCTS 节点
        :param prompt_state: 当前节点的提示词状态 (List[str] 或 str)
        :param parent: 父节点
        :param prior_prob: P(s, a) - DeepSeek 给出的置信度
        :param action_taken: 到达此节点所采取的动作 (描述字符串)
        """
        self.prompt_state = prompt_state
        self.parent = parent
        self.children = []
        self.action_taken = action_taken
        
        # MCTS 统计量
        self.N = 0  # 访问次数
        self.Q = 0.0  # 平均价值 (Reward)
        self.P = prior_prob  # 先验概率
        
        self.is_dead = False  # 如果 Level A 评估失败，标记为死亡
        self.error_analysis = None  # 存储该节点的错误分析报告，供子节点扩展使用
        self.expansion_failed = False  # 扩展失败标记：API调用失败时设置，防止重复无效重试

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c_puct=1.0):
        """
        根据 PUCT 公式选择最佳子节点
        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        best_score = -float('inf')
        best_node = None
        
        # 父节点访问次数的平方根
        sqrt_n_parent = math.sqrt(self.N)
        
        for child in self.children:
            if child.is_dead:
                continue
                
            # PUCT 公式
            exploit = child.Q
            explore = c_puct * child.P * sqrt_n_parent / (1 + child.N)
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_node = child
                
        return best_node

class MCTS:
    def __init__(self, root_prompt, evaluator, action_generator, c_puct=1.0):
        """
        MCTS 搜索引擎
        :param root_prompt: 初始提示词
        :param evaluator: 评估器实例 (Evaluator)
        :param action_generator: 动作生成器实例 (ActionGenerator)
        :param c_puct: 探索系数
        """
        self.root = TreeNode(prompt_state=root_prompt, prior_prob=1.0)
        self.evaluator = evaluator
        self.action_generator = action_generator
        self.c_puct = c_puct
        
    def search(self, iterations=50):
        """
        执行 MCTS 搜索
        :param iterations: 迭代次数
        """
        dead_count = 0
        successful_count = 0
        
        for i in range(iterations):
            logger.info(f"\n" + "="*80)
            logger.info(f"🚀 MCTS Iteration {i+1}/{iterations} 开始")
            logger.info("="*80)
            
            # 1. Selection
            logger.info(f"🔍 [Selection] 正在从根节点开始选择最优路径...")
            node = self._select(self.root)
            logger.info(f"📍 [Selection] 选中节点: {node.action_taken if node.action_taken else 'Root'}")
            
            # 2. Expansion
            # 如果节点未被评估过且不是死节点，则扩展
            if not node.is_dead:
                # 如果是叶子节点且扩展未失败过，尝试扩展
                # expansion_failed=True 时跳过，防止 API 故障导致每轮都重试同一个失败节点
                if node.is_leaf() and not node.expansion_failed:
                    self._expand(node, iteration=i+1)
                
                # 如果扩展后仍是叶子（无法扩展），则回溯当前值
                # 否则选择一个新子节点进行模拟
                if not node.is_leaf():
                    node = self._select(node) # 移动到新扩展的子节点
                    logger.info(f"🌿 [Expansion] 进入新扩展的子节点: {node.action_taken}")
            
            # 3. Simulation (Evaluation)
            logger.info(f"🎲 [Simulation] 开始评估节点效用...")
            reward = self._simulate(node, iteration=i+1)
            
            # 4. Backpropagation
            logger.info(f"⬆️ [Backpropagation] 正在回溯更新节点统计信息 (Reward: {reward:.4f})...")
            self._backpropagate(node, reward)
            
            if reward < 0:
                dead_count += 1
            else:
                successful_count += 1
            
            logger.info(f"\n✨ Iteration {i+1} 结束. 当前路径奖励: {reward:.4f}")
            logger.info(f"📊 实时统计: {successful_count} 成功 / {dead_count} 失败 / {i+1} 总计 (失败率: {dead_count/(i+1)*100:.1f}%)")
            # 记录每次迭代的混淆矩阵指标（写入 mcts_log.txt）
            if hasattr(node, 'error_analysis') and node.error_analysis:
                ea = node.error_analysis
                logger.info(f"📈 [Confusion] Iter {i+1}: F1={ea.get('f1',0):.4f} P={ea.get('precision',0):.4f} R={ea.get('recall',0):.4f} FN={ea.get('false_negatives',0)} FP={ea.get('false_positives',0)} OA={ea.get('other_labels_acc',0):.4f}")

        logger.info(f"\n" + "#"*80)
        logger.info(f"🏁 MCTS 搜索完成总结")
        logger.info(f"总迭代次数: {iterations}")
        logger.info(f"成功节点: {successful_count}")
        logger.info(f"失败节点: {dead_count}")
        logger.info(f"成功率: {successful_count/iterations*100:.1f}%")
        
        return self.get_best_path()

    def _select(self, node):
        """
        选择阶段：一直下潜直到叶子节点
        """
        while not node.is_leaf():
            next_node = node.best_child(self.c_puct)
            if next_node is None:
                # 所有子节点都 Dead 了，返回当前节点并标记
                logger.info(f"⚠️ All children are dead at this node. Marking as dead.")
                node.is_dead = True
                return node
            node = next_node
        return node

    def _expand(self, node, iteration=0, retry_count=0):
        """
        扩展阶段：调用 DeepSeek 生成候选动作
        :param retry_count: 重试次数（当生成的动作质量不高时）
        """
        print(f"🌱 [Expansion] 正在扩展节点 (N={node.N}, Q={node.Q:.4f}, retry={retry_count})...")
        
        # 1. 获取错误分析（必须有才能生成动作）
        parent_analysis = node.error_analysis
        if parent_analysis is None:
            logger.info("🧐 [Expansion] 正在生成父节点的初始分析报告...")
            try:
                _, parent_analysis = self.evaluator.evaluate(node.prompt_state, level='B')
                node.error_analysis = parent_analysis
                # 打印详细的思考过程
                self._print_analysis_details(parent_analysis, "父节点")
                
                # 保存父节点干扰样本到 Excel
                node_name = "Root" if node.parent is None else "Parent"
                self.evaluator.save_errors_to_excel(parent_analysis, iteration, node_name)
            except Exception as e:
                logger.info(f"❌ [Expansion] 生成分析报告失败: {e}")
                node.is_dead = True
                return

        # 2. 调用 ActionGenerator 生成候选动作
        confused_labels = getattr(self.evaluator, 'confused_labels', [])
        
        try:
            actions = self.action_generator.generate_actions(
                current_prompt=node.prompt_state,
                error_analysis=parent_analysis,
                confused_labels=confused_labels,
                k=6
            )
        except Exception as e:
            logger.info(f"❌ Failed to generate actions: {e}")
            actions = None
            
        if not actions:
            logger.info("⚠️ No actions generated. Marking expansion as failed to prevent repeated retries.")
            node.expansion_failed = True
            return

        # 3. 创建子节点
        logger.info(f"📝 Creating {len(actions)} child nodes...")
        for i, action in enumerate(actions):
            child = TreeNode(
                prompt_state=action.get('new_prompt', ''),
                parent=node,
                prior_prob=action.get('confidence', 0.5),
                action_taken=action.get('description', f'Action_{i}')
            )
            node.children.append(child)
        
        # FBPS: Frontier-Batch Parallel Simulation - 批次并行评估所有新子节点
        logger.info(f"🚀 [FBPS] 批次并行评估 {len(node.children)} 个新子节点...")
        for child in node.children:
            try:
                # 使用 Level B 快速评估（避免硬性剪枝）
                reward, analysis = self.evaluator.evaluate(child.prompt_state, level='B')
                child.Q = reward  # 直接设置 Q 值作为预评估奖励
                child.N = 1  # 设置访问次数为1，表示已评估
                child.error_analysis = analysis  # 保存分析供后续使用
                
                logger.info(f"  ✓ 子节点评估完成: {child.action_taken[:30]}... Q={child.Q:.4f}")
            except Exception as e:
                logger.info(f"  ❌ 子节点评估失败: {child.action_taken[:30]}... {e}")
                child.Q = 0.0001  # 失败时设置低奖励
                child.N = 1
                child.is_dead = True  # 标记为死亡
        
        logger.info(f"✓ [FBPS] 批次评估完成，所有子节点已预设 Q 值。")

    def _simulate(self, node, iteration=0):
        """
        模拟阶段：使用 Evaluator 评估节点价值（只用 Level B，避免硬性剪枝）
        FBPS: 如果节点已预评估，直接返回预设 Q 值
        """
        node_display = node.action_taken if node.action_taken else 'Root'
        logger.info(f"🧪 [Simulation] 评估目标: {node_display}")
        
        # FBPS: 如果节点已预评估（N > 0），直接返回预设 Q 值
        if node.N > 0:
            logger.info(f"💡 [FBPS] 节点已预评估，直接返回 Q={node.Q:.4f}")
            return node.Q
        
        try:
            # 直接用 Level B 全量评估
            score_b, analysis_b = self.evaluator.evaluate(node.prompt_state, level='B')
            
            # 保存错误分析供下次扩展使用
            node.error_analysis = analysis_b
            
            # 打印详细的分类思考/判别信息
            self._print_analysis_details(analysis_b, node_display)
            
            # 保存干扰样本到 Excel
            node_id = "".join([c for c in str(node_display)[:20] if c.isalnum() or c==' ']).replace(' ', '_')
            self.evaluator.save_errors_to_excel(analysis_b, iteration, node_id)
            
            logger.info(f"✅ [Simulation] 分值: {score_b:.4f} (F1: {analysis_b.get('f1', 0):.4f})")
            return score_b
            
        except Exception as e:
            logger.info(f"❌ [Simulation] 评估过程报错: {e}")
            traceback.print_exc()
            return 0.0001

    def _backpropagate(self, node, reward):
        """
        回溯阶段：更新路径上的 N 和 Q
        """
        while node is not None:
            node.N += 1
            # Q 值更新公式: Q_new = (Q_old * (N-1) + Reward) / N
            node.Q = (node.Q * (node.N - 1) + reward) / node.N
            node = node.parent

    def get_best_path(self):
        """
        获取最佳路径：选择 Q 值最高的路径（这代表最优的提示词）
        不再使用访问次数 N 作为判断标准，因为我们已保证所有存活的节点都会被评估。
        """
        node = self.root
        path = [node]
        
        logger.info(f"\n🎯 Tracing best path from root...")
        
        while not node.is_leaf():
            # 选择 Q 值最高的子节点（这代表最优的提示词）
            if not node.children:
                logger.info(f"  Leaf reached (no children)")
                break
                
            # 按 Q 值排序，选择最好的
            best_child = max(node.children, key=lambda x: x.Q)
            
            node = best_child
            path.append(node)
            logger.info(f"  Level {len(path)-1}: Q={node.Q:.4f}, N={node.N}, Action: {node.action_taken[:50] if node.action_taken else 'Root'}")
        
        logger.info(f"\n✅ Best path found (depth={len(path)-1}):")
        logger.info(f"  Root: Q={path[0].Q:.4f}")
        if len(path) > 1:
            logger.info(f"  Best child: Q={path[-1].Q:.4f}, Action: {path[-1].action_taken[:100] if path[-1].action_taken else 'N/A'}")
        else:
            logger.info(f"  ⚠️ No children found! Returning root node.")
        
        return path

    def _print_analysis_details(self, analysis, node_name):
        """打印详细的评估分析过程（思考过程）"""
        logger.info(f"\n--- 🧠 {node_name} 详细判别信息 ---")
        logger.info(f"| 目标标签: {analysis.get('target_label')}")
        logger.info(f"| 样本总量: {analysis.get('total_samples')}")
        logger.info(f"| 判别性能: P={analysis.get('precision', 0):.4f}, R={analysis.get('recall', 0):.4f}, F1={analysis.get('f1', 0):.4f}")
        logger.info(f"| 错误统计: 假阴性(漏检)={analysis.get('false_negatives', 0)}, 假阳性(误检)={analysis.get('false_positives', 0)}")
        
        fn_examples = analysis.get('fn_examples', [])
        if fn_examples:
            logger.info(f"| 典型漏检案例 (FN):")
            for i, ex in enumerate(fn_examples[:3]):
                logger.info(f"|   [{i+1}] {str(ex)[:100]}...")
                
        fp_examples = analysis.get('fp_examples', [])
        if fp_examples:
            logger.info(f"| 典型误检案例 (FP):")
            for i, ex in enumerate(fp_examples[:3]):
                logger.info(f"|   [{i+1}] {str(ex)[:100]}...")
                
        violations = analysis.get('violations', [])
        if violations:
            logger.info(f"| ⚠️ 铁律违反: {', '.join(violations)}")
        else:
            logger.info(f"| ✓ 铁律检查: 通过 (未造成其他标签准确率明显下降)")
        logger.info("-" * (len(node_name) + 26))
