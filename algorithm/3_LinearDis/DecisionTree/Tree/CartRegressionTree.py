import numpy as np
from .DecisionTreeBase import DecisionTreeBase
from .Criterion import _cart_regression_split


class CartRegressionTree(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-6):
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_gain=min_gain)
        self.task = "regression"

    def _choose_best_split(self, X, y):
        """
        寻找当前节点的最优划分特征及阈值

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features)
            当前节点的特征矩阵。
        y : np.ndarray, shape = (n_samples,)
            当前节点对应的标签向量。

        Returns
        -------
        best_feature : int
            最优划分特征索引
        best_threshold : float
            连续特征为阈值，回归算法中默认全是连续值
        best_splits : tuple
            ((X_left, y_left), (X_right, y_right))
        best_operator : str
            分裂操作符（'<='）
        best_gain : float
            -min(MSE),MSE越小越好，为了符合外部比较逻辑，取负号
        split_type : str
            固定为 "binary"
        """
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_splits = None
        best_operator = "<="
        best_gain = 0.0
        split_type = "binary"
        best_mse = float('inf')

        parent_mse = np.var(y)  # 父节点总误差

        for feature_idx in range(n_features):
            mse, splits, threshold = _cart_regression_split(X, y, feature_idx)
            if mse < best_mse:
                best_feature = feature_idx
                best_threshold = threshold
                best_splits = splits
                best_mse = mse

        # 计算真实增益（误差下降量）
        best_gain = parent_mse - best_mse

        return best_feature, best_threshold, best_splits, best_operator, best_gain, split_type










