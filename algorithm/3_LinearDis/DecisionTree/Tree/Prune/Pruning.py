import numpy as np
import copy
from ..DecisionTreeBase import DecisionTreeBase


def pessimistic_prune(tree):
    """
    对分类决策树执行悲观剪枝（Pessimistic Error Pruning）。

    基于训练误差的悲观估计（假设误差服从二项分布），
    当将某内部节点剪为叶节点后误差不增时，
    删除其子树以简化模型、减少过拟合。

    Parameters
    ----------
    tree : DecisionTreeBase
        已训练好的分类决策树（CART 或连续特征的 C4.5）。

    Returns
    -------
    tree : DecisionTreeBase
        剪枝后的树（原地修改并返回）。
    """
    # === 1. 检查是否为 DecisionTreeBase 子类 ===
    if not isinstance(tree, DecisionTreeBase):
        raise TypeError(f"输入对象类型 {type(tree)} 不是 DecisionTreeBase 的子类。")

    # === 2. 检查树是否已训练 ===
    if not hasattr(tree, "root") or tree.root is None:
        raise ValueError("该树尚未训练（root 节点为空）。")

    # === 3. 检查是否支持剪枝 ===
    tree_type = type(tree).__name__.lower()
    if "id3" in tree_type:
        raise TypeError("ID3Tree 不支持悲观剪枝（仅适用于二叉划分树）。")

    if "c45" in tree_type and tree.feature_types:
        # 检查是否包含离散特征
        discrete_count = sum(1 for t in tree.feature_types.values() if t == "discrete")
        if discrete_count > 0:
            raise TypeError("C4.5Tree 中包含离散特征，不建议使用悲观剪枝。")

    if "cart" not in tree_type:
        print(f"[警告] {type(tree).__name__} 未显式标识为 CART 树，自动尝试执行剪枝。")

    # === 4. 检查任务类型 ===
    if getattr(tree, "task", None) != "classification":
        raise TypeError("悲观剪枝仅适用于分类树。")

    # === 5. DFS 剪枝实现 ===
    def _prune(node):
        if node.is_leaf:
            return node

        if node.left:
            node.left = _prune(node.left)
        if node.right:
            node.right = _prune(node.right)

        # 悲观误差估计
        E_sub = (node.left.error + node.right.error)
        E_leaf = (node.error + 0.5) / node.n_samples

        if E_leaf <= E_sub:
            node.left = node.right = None
            node.is_leaf = True
            node.split_type = "leaf"
        return node

    tree.root = _prune(tree.root)

    return tree


def cost_complexity_prune(tree, alpha, X_val, y_val):
    """
    代价复杂度剪枝（Cost-Complexity Pruning）
    适用于 CART 回归树。

    剪枝思想：
        在每个内部节点 t，定义局部复杂度参数：
            alpha_t = (R(t) - R(T_t)) / (|T_t| - 1)
        其中 R(T_t) 为该子树的误差（残差平方和），|T_t| 为子树叶节点数。
        每次剪掉 alpha_t 最小的节点子树，生成一系列子树 T0 ⊃ T1 ⊃ T2 ...
        再通过验证集选择泛化误差最小的那棵树。

    Parameters
    ----------
    tree : CartRegressionTree
        已训练完成的 CART 回归树。
    alpha : float
        剪枝惩罚系数，控制模型复杂度（alpha 越大，剪枝越强）。
    X_val : np.ndarray, shape = (n_samples, n_features)
        验证集特征矩阵。
    y_val : np.ndarray, shape = (n_samples,)
        验证集标签向量。

    Returns
    -------
    best_tree : CartRegressionTree
        剪枝后在验证集上表现最优的回归树。
    """
    # ======================
    # 辅助函数定义
    # ======================
    def is_leaf(node):
        return node is None or node.is_leaf

    def count_leaves(node):
        """统计子树的叶节点数"""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    def subtree_error(node):
        """计算子树的残差平方和（error 已在节点中保存）"""
        if node.is_leaf:
            return node.error
        return subtree_error(node.left) + subtree_error(node.right)

    def compute_alpha(node):
        """计算节点的 α_t（若是叶节点则返回 ∞）"""
        if node.is_leaf:
            return float("inf")
        R_t = node.error
        R_Tt = subtree_error(node)
        n_leaves = count_leaves(node)
        if n_leaves <= 1:
            return float("inf")
        return (R_t - R_Tt) / (n_leaves - 1)

    def prune_node(node, target):
        """递归剪掉 α_t == target 的节点"""
        if node.is_leaf:
            return node
        alpha_t = compute_alpha(node)
        if np.isclose(alpha_t, target):
            # 剪枝：将该子树替换为叶节点
            node.left = None
            node.right = None
            node.is_leaf = True
            node.split_type = "leaf"
            node.value = node.prediction
            return node
        else:
            if node.left:
                node.left = prune_node(node.left, target)
            if node.right:
                node.right = prune_node(node.right, target)
            return node

    # ======================
    # 主循环：生成剪枝路径
    # ======================
    trees = [copy.deepcopy(tree)]
    alphas = [0.0]
    val_errors = [np.mean((tree.predict(X_val) - y_val) ** 2)]

    current_tree = copy.deepcopy(tree)

    while True:
        # 1️⃣ 计算所有内部节点的 α_t
        alpha_list = []

        def collect_alphas(node):
            if node is None or node.is_leaf:
                return
            alpha_list.append(compute_alpha(node))
            collect_alphas(node.left)
            collect_alphas(node.right)

        collect_alphas(current_tree.root)
        if not alpha_list:
            break

        min_alpha = min(alpha_list)

        # 若当前 α 已超过设定阈值，则停止剪枝
        if min_alpha > alpha:
            break

        # 2️⃣ 剪去所有 α_t == min_alpha 的子树
        prune_node(current_tree.root, min_alpha)

        # 3️⃣ 保存该阶段剪枝后的树及验证集误差
        trees.append(copy.deepcopy(current_tree))
        alphas.append(min_alpha)
        val_pred = current_tree.predict(X_val)
        val_mse = np.mean((val_pred - y_val) ** 2)
        val_errors.append(val_mse)

        # 如果树已经全剪为叶节点，停止
        if is_leaf(current_tree.root):
            break

    # ======================
    # 选择最优子树
    # ======================
    best_idx = np.argmin(val_errors)
    best_tree = trees[best_idx]

    print(f"✅ 代价复杂度剪枝完成：生成 {len(trees)} 棵子树")
    print(f"→ 最优 α = {alphas[best_idx]:.4f}, 验证 MSE = {val_errors[best_idx]:.4f}")

    return best_tree

