from .DecisionTreeBase import DecisionTreeBase
from .Criterion import get_binary_gain_ratio
from .Criterion import get_multiway_gain_ratio


class C45Tree(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-6):
        super().__init__(max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_gain=min_gain)
        self.task = "classification"

    def _choose_best_split(self, X, y):
        """
        选择当前节点的最优划分特征（基于信息增益率）
        ------------------------------------------------
        Returns
        -------
        best_feature : int
            最优特征索引
        best_threshold : float or None
            连续特征的划分阈值
        best_splits : dict 或 tuple
            离散特征: {value: (X_sub, y_sub)}；
            连续特征: ((X_left, y_left), (X_right, y_right))
        best_operator : str
            划分操作符（'==' 或 '<='）
        best_gain_ratio : float
            对应的信息增益率
        best_split_type : str
            'multiway'（离散）或 'binary'（连续）
        """
        n_samples, n_features = X.shape
        best_gain_ratio = -1
        best_feature, best_threshold = None, None
        best_splits, best_operator, best_split_type = None, None, None

        # 遍历所有特征，分别计算信息增益率
        for feature_idx in range(n_features):
            feature_type = self.feature_types.get(feature_idx, 'discrete')

            if feature_type == 'discrete':
                gain_ratio, _, _, splits = get_multiway_gain_ratio(X, y, feature_idx)
                operator, threshold, split_type = '==', None, "multiway"

            elif feature_type == 'continuous':
                gain_ratio, _, _, splits, threshold = get_binary_gain_ratio(X, y, feature_idx)
                operator, split_type = '<=', "binary"

            else:
                continue

            # 更新最优划分
            if gain_ratio > best_gain_ratio:
                best_feature = feature_idx
                best_gain_ratio = gain_ratio
                best_threshold = threshold
                best_splits = splits
                best_operator = operator
                best_split_type = split_type

        # 无有效划分时返回叶节点
        if best_feature is None or best_gain_ratio <= 0:
            return None, None, None, '==', 0.0, "multiway"

        return best_feature, best_threshold, best_splits, best_operator, best_gain_ratio, best_split_type
