from .DecisionTreeBase import DecisionTreeBase
from .Criterion import _cart_discrete_split
from .Criterion import _cart_continuous_split


class CartClassificationTree(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-6):
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_gain=min_gain)
        self.task = "classification"

    def _choose_best_split(self, X, y):
        """
        寻找当前节点的最优划分特征（CART 分类树）

        Returns
        -------
        best_feature : int
            最优划分特征索引
        best_threshold : float 或 set
            连续特征为阈值，离散特征为取值集合
        best_splits : tuple
            ((X_left, y_left), (X_right, y_right))
        best_operator : str
            分裂操作符（'==' 或 '<='）
        best_gain : float
            增益值（1 - 最优 Gini）
        split_type : str
            固定为 "binary"
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_splits = None
        best_operator = None
        best_gini = float("inf")  # Gini 越小越好
        split_type = "binary"

        # 遍历所有特征
        for feature_idx in range(n_features):
            feature_type = self.feature_types.get(feature_idx, 'continuous')
            if feature_type == 'discrete':
                gini_value, splits, threshold = _cart_discrete_split(X, y, feature_idx)
                operator = '=='

            elif feature_type == 'continuous':
                gini_value, splits, threshold = _cart_continuous_split(X, y, feature_idx)
                operator = '<='

            else:
                continue

            # 更新最优划分
            if gini_value < best_gini:
                best_gini = gini_value
                best_feature = feature_idx
                best_threshold = threshold
                best_splits = splits
                best_operator = operator

        # 若无可划分特征
        if best_feature is None or best_splits is None:
            return None, None, None, '==', 0.0, split_type

        # 计算 gain（1 - gini），越大越好
        best_gain = 1 - best_gini

        return best_feature, best_threshold, best_splits, best_operator, best_gain, split_type

