import numpy as np
from .Criterion import get_entropy
from .DecisionTreeBase import DecisionTreeBase
from .Criterion import _discrete_split


class ID3Tree(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-6):
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_gain=min_gain)
        self.task = "classification"

    def _choose_best_split(self, X, y):
        """
        寻找当前节点的最优划分特征（ID3 算法）

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
        best_threshold : None
            ID3 仅用于离散特征，无阈值。
        splits : dict
            返回字典 {特征值: (X_sub, y_sub)}。
        operator : str
            分裂操作符，固定为 '=='。
        gain : float
            当前最优划分对应的信息增益。
        split_type: str
            分裂方式，ID3算法只有 "multiway"
        """
        n_samples, n_features = X.shape

        # 当前节点的熵（划分前）
        base_entropy = get_entropy(y)

        best_gain = -1
        best_feature = None
        best_splits = None

        # 遍历所有特征
        for feature_idx in range(n_features):
            # 检查该特征是否为离散特征
            feature_type = self.feature_types.get(feature_idx, 'discrete')
            if feature_type != 'discrete':
                continue  # ID3只处理离散特征

            new_entropy, splits = _discrete_split(X, y, feature_idx)

            # 计算信息增益
            gain = base_entropy - new_entropy

            # 更新最优划分
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_splits = splits

        # 若所有特征信息增益均为 0 或未找到可划分特征
        if best_feature is None:
            return None, None, None, '==', 0.0, "multiway"

        return best_feature, None, best_splits, '==', best_gain, "multiway"

