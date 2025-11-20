import numpy as np


def _majority_or_mean(y, task="classification"):
    """
    根据任务类型返回节点预测值
    分类任务返回众数，回归任务返回均值
    :param y: np.ndarray，样本标签
    :param task: str, "classification" 或 "regression"
    :return: float 或 类别标签
    """
    if task == "regression":
        return np.mean(y)
    else:
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]


class Node:
    def __init__(self,
                 feature=None,
                 threshold=None,
                 children=None,
                 left=None, right=None,
                 value=None,
                 is_leaf=False,
                 operator=None,
                 feature_type=None,
                 split_type=None,
                 n_samples=0,
                 error=0.0,
                 prediction=None,
                 depth=0):
        """
        结点初始化
        :param feature: 当前节点划分特征索引
        :param threshold: 连续特征的划分阈值（仅C4.5/CART使用）
        :param children: 多叉分裂的子节点字典（ID3/C4.5离散特征）
        :param left: 二叉分裂左子节点（C4.5连续 / CART）
        :param right: 二叉分裂右子节点（C4.5连续 / CART）
        :param value: 叶子节点输出值
        :param is_leaf: 是否叶子节点
        :param operator: 运算符（'<=', '=='）
        :param feature_type: 特征类型（'discrete' / 'continuous'）
        :param split_type: 分裂类型（'multiway' / 'binary' / 'leaf'）
        :param n_samples: 当前节点样本数（用于剪枝）
        :param error: 当前节点误差（分类：误分数；回归：MSE * N）
        :param prediction: 当前节点预测输出（叶节点时 = value）
        :param depth: 当前节点深度（可选）
        """
        self.feature = feature
        self.threshold = threshold
        self.children = children if children is not None else {}
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
        self.operator = operator
        self.feature_type = feature_type
        self.split_type = split_type or ("leaf" if is_leaf else None)

        # === 剪枝统计相关 ===
        self.n_samples = n_samples
        self.error = error
        self.prediction = prediction if prediction is not None else value
        self.depth = depth

    def __repr__(self):
        """
        打印树节点内容，便于调试与结构可视化
        """
        if self.split_type == "leaf":
            return (f"<Leaf value={self.value}, n={self.n_samples}, "
                    f"error={self.error:.3f}>")

        if self.split_type == "binary":
            return (f"<Node feature={self.feature} ({self.feature_type}) "
                    f"{self.operator or '<='} {self.threshold if self.threshold is not None else ''}, "
                    f"n={self.n_samples}, error={self.error:.3f}>")

        if self.split_type == "multiway":
            child_keys = list(self.children.keys())
            if len(child_keys) > 5:
                keys_str = ", ".join(map(str, child_keys[:5])) + ", ..."
            else:
                keys_str = ", ".join(map(str, child_keys))
            return (f"<Node feature={self.feature} ({self.feature_type}) "
                    f"{self.operator or '=='} {{{keys_str}}}, "
                    f"n={self.n_samples}, error={self.error:.3f}>")

        return f"<Node feature={self.feature}, n={self.n_samples}, error={self.error:.3f}>"


class DecisionTreeBase:
    """
    决策树基类（NumPy版本）
    只接受 np.ndarray 类型的 X、y
    """

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_gain=1e-6):
        """
        :param max_depth: 最大树深
        :param min_samples_split: 节点最小分裂样本数
        :param min_gain: 最小增益阈值
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
        self.feature_types = None  # {feature_index: 'continuous' / 'discrete'}
        self.feature_names = None
        self.task = None

    def fit(self, X, y, feature_types_dict=None):
        """
        拟合模型（仅接受 NumPy 格式数据）
        :param X: np.ndarray, shape = (n_samples, n_features)
        :param y: np.ndarray, shape = (n_samples,)
        :param feature_types_dict: dict，{特征索引或名称: 'discrete'/'continuous'}
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X 必须为 numpy.ndarray，但检测到 {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y 必须为 numpy.ndarray，但检测到 {type(y)}")

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X 应为二维数组，y 应为一维数组。")

        if feature_types_dict is None:
            raise ValueError("必须提供 feature_types_dict 参数以区分特征类型。")

        # 如果 key 是字符串 → 保存真实列名并建立索引映射
        if all(isinstance(k, str) for k in feature_types_dict.keys()):
            self.feature_names = list(feature_types_dict.keys())
            self.feature_types = {i: feature_types_dict[name] for i, name in enumerate(self.feature_names)}
        else:
            # 若 key 已为索引
            self.feature_types = feature_types_dict
            self.feature_names = [str(i) for i in range(X.shape[1])]

        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        对输入样本进行预测（只接受 np.ndarray）
        :param X: np.ndarray, shape = (n_samples, n_features)
        :return: np.ndarray, shape = (n_samples,)
        """
        if self.root is None:
            raise ValueError("模型尚未训练，请先调用 fit()。")

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X 必须为 numpy.ndarray，但检测到 {type(X)}")

        preds = [self._traverse(self.root, sample) for sample in X]
        return np.array(preds)

    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        # === 1. 统计当前节点信息 ===
        n_samples = len(y)

        if self.task == "classification":
            # 多数类预测
            prediction = _majority_or_mean(y, self.task)
            # 当前节点误分数（用于悲观剪枝）
            error = np.sum(y != prediction)
        else:
            # 回归任务
            prediction = np.mean(y)
            # 当前节点残差平方和（用于代价复杂度剪枝）
            error = np.sum((y - prediction) ** 2)

        # 1. 停止条件
        if len(np.unique(y)) == 1:
            # 只有一类，结束
            return Node(value=prediction, is_leaf=True, split_type="leaf",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        if self.max_depth is not None and depth >= self.max_depth:
            # 达到最大深度，结束
            return Node(value=prediction, is_leaf=True, split_type="leaf",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        if n_samples < self.min_samples_split:
            # 当样本数太少时，直接停止分裂，把当前节点当成叶节点
            return Node(value=prediction, is_leaf=True, split_type="leaf",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        # 2. 寻找最佳划分
        best_feature, best_threshold, splits, operator, gain, split_type = self._choose_best_split(X, y)

        if best_feature is None or gain < self.min_gain:
            # 增益不够，终止
            return Node(value=prediction, is_leaf=True, split_type="leaf",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        feature_type = self.feature_types.get(best_feature, 'continuous')

        # 3. 根据分裂方式生成子节点
        if split_type == "multiway":
            children = {}
            for val, (X_sub, y_sub) in splits.items():
                children[val] = self._build_tree(X_sub, y_sub, depth + 1)
            return Node(feature=best_feature, children=children,
                        operator="==", feature_type=feature_type, split_type="multiway",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        elif split_type == "binary":
            (X_left, y_left), (X_right, y_right) = splits
            left_child = self._build_tree(X_left, y_left, depth + 1)
            right_child = self._build_tree(X_right, y_right, depth + 1)
            return Node(feature=best_feature, threshold=best_threshold,
                        left=left_child, right=right_child,
                        operator=operator, feature_type=feature_type, split_type="binary",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

        else:
            return Node(value=prediction, is_leaf=True, split_type="leaf",
                        n_samples=n_samples, error=error, prediction=prediction, depth=depth)

    def _choose_best_split(self, X, y):
        """
        寻找当前节点的最优划分特征及阈值（由子类实现）

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features)
            当前节点的特征矩阵。
        y : np.ndarray, shape = (n_samples,)
            当前节点对应的标签向量。

        Returns
        -------
        best_feature : int
            最优划分特征的索引。
        best_threshold : float or None
            对于二叉分裂，返回最优划分阈值；
            对于多叉分裂，返回 None。
        splits : dict or tuple
            - 若为多叉分裂：返回字典 {特征值: (X_sub, y_sub)}。
            - 若为二叉分裂：返回 ((X_left, y_left), (X_right, y_right))。
        operator : str
            分裂操作符，'==' 表示离散特征划分，'<=' 表示连续特征划分。
        gain : float
            当前最优划分对应的信息增益 / 增益率 / 基尼指数减少量。
        """
        raise NotImplementedError("子类必须实现特征选择逻辑 (_choose_best_split)。")

    def _traverse(self, node, x):
        """递归遍历决策树"""
        if node.split_type == "leaf":
            return node.value

        if node.split_type == "binary":
            if node.feature_type == "continuous":
                if x[node.feature] <= node.threshold:
                    return self._traverse(node.left, x)
                else:
                    return self._traverse(node.right, x)
            else:  # 离散型二叉划分
                if x[node.feature] in node.threshold:  # threshold 可以是一个集合
                    return self._traverse(node.left, x)
                else:
                    return self._traverse(node.right, x)

        if node.split_type == "multiway":
            val = x[node.feature]
            if val in node.children:
                return self._traverse(node.children[val], x)
            else:
                return node.value  # 未见过的取值 → 返回当前节点预测值

    def __repr__(self):
        """打印树结构"""
        lines = []
        self._print_tree(self.root, 0, lines)
        return "\n".join(lines)

    def _print_tree(self, node, depth, lines):
        """递归打印树结构"""
        indent = "  " * depth
        if node.split_type == "leaf":
            lines.append(f"{indent}Leaf: {node.value}")
        elif node.split_type == "binary":
            feature_name = self.feature_names[node.feature] if self.feature_names else f"X[{node.feature}]"
            lines.append(f"{indent}{feature_name} {node.operator} {node.threshold}")
            self._print_tree(node.left, depth + 1, lines)
            self._print_tree(node.right, depth + 1, lines)
        elif node.split_type == "multiway":
            feature_name = self.feature_names[node.feature] if self.feature_names else f"X[{node.feature}]"
            for val, child in node.children.items():
                lines.append(f"{indent}{feature_name} == {val}")
                self._print_tree(child, depth + 1, lines)
