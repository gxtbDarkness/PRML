import numpy as np
import itertools


def get_entropy(y):
    """
    计算熵（加入极小平滑项以提升数值稳定性）
    :param y: 标签集合 np.ndarray, shape = (n_samples,)
    :return: 信息熵（float）
    """
    # 类型转换
    y = np.array(y)

    # 参数校验
    if y.ndim != 1:
        raise ValueError('y must be 1-dim')

    # 类别数小于 2 或空集，熵为 0
    if y.size == 0 or np.unique(y).size < 2:
        return 0.0

    # 统计各类别出现次数
    _, counts = np.unique(y, return_counts=True)

    # 计算概率分布
    probs = counts / y.size

    # ---- 数值平滑：避免 log(0) ----
    epsilon = 1e-12
    probs = np.clip(probs, epsilon, 1.0)

    # 计算信息熵
    entropy = -np.sum(probs * np.log2(probs))

    return entropy


def get_multiway_gain_ratio(X, y, feature_idx):
    """
    计算离散特征的信息增益比
    :param X: np.ndarray, shape = (n_samples, n_features)
    :param y: np.ndarray, shape = (n_samples,)
    :param feature_idx: 当前特征索引
    :return: gain_ratio, info_gain, split_info, splits
    """
    n_samples = len(y)
    base_entropy = get_entropy(y)
    cond_entropy, splits = _discrete_split(X, y, feature_idx)

    info_gain = base_entropy - cond_entropy

    # 计算 SplitInfo
    split_info = 0.0
    for _, (_, y_sub) in splits.items():
        p = len(y_sub) / n_samples
        if p > 0:
            split_info -= p * np.log2(p)

    gain_ratio = info_gain / split_info if split_info > 0 else 0.0
    return gain_ratio, info_gain, split_info, splits


def get_binary_gain_ratio(X, y, feature_idx):
    """
    计算连续特征（二叉划分）的信息增益比
    :param X: np.ndarray, shape = (n_samples, n_features)
    :param y: np.ndarray, shape = (n_samples,)
    :param feature_idx: 当前特征索引
    :return: gain_ratio, info_gain, split_info, splits, threshold
    """
    n_samples = len(y)
    base_entropy = get_entropy(y)

    # 获取最优二叉划分
    cond_entropy, splits, threshold = _continuous_split(X, y, feature_idx)
    if splits is None:
        return 0.0, 0.0, 0.0, None, None

    info_gain = base_entropy - cond_entropy

    # 计算 SplitInfo
    split_info = 0.0
    for _, y_sub in splits:
        p = len(y_sub) / n_samples
        if p > 0:
            split_info -= p * np.log2(p)

    gain_ratio = info_gain / split_info if split_info > 0 else 0.0
    return gain_ratio, info_gain, split_info, splits, threshold


def get_gini(y):
    """
    计算基尼指数
    :param y: 标签集合 np.ndarray, shape = (n_samples,)
    :return: 基尼指数（gini）
    """
    # 类型转换
    y = np.array(y)

    # 参数校验
    if y.ndim != 1:
        raise ValueError('y must be 1-dim')

    # 类别数小于 2 或为空集，熵为 0
    if y.size == 0 or np.unique(y).size < 2:
        return 0.0

    # 统计各类别出现次数
    _, counts = np.unique(y, return_counts=True)

    # 计算概率分布
    probs = counts / y.size

    epsilon = 1e-12
    probs = np.clip(probs, epsilon, 1.0)

    # 计算gini指数
    gini = 1 - np.sum(probs * probs)

    return gini


def _discrete_split(X, y, feature_idx):
    """
    计算离散型变量的条件熵并返回划分结果
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵。
    :param y: np.ndarray, shape = (n_samples,)
              标签向量。
    :param feature_idx: 划分参考的特征 index
    :return: 条件熵 cond_entropy, 划分集合 splits {特征值: (X_sub, y_sub)}
    """
    n_samples, _ = X.shape
    feature_values = X[:, feature_idx]
    # 取出该特征下的所有非重复值
    unique_values = np.unique(feature_values)

    splits = {}
    cond_entropy = 0.0

    # 按特征值划分子集, 并计算条件熵
    for value in unique_values:
        mask = (feature_values == value)
        X_sub, y_sub = X[mask], y[mask]
        splits[value] = (X_sub, y_sub)
        prob = len(y_sub) / n_samples
        cond_entropy += prob * get_entropy(y_sub)

    return cond_entropy, splits


def _continuous_split_with_threshold(X, y, feature_idx, threshold):
    """
    根据给定的阈值计算连续型随机变量的最优划分的条件熵并返回划分结果与条件熵
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵。
    :param y: np.ndarray, shape = (n_samples,)
              标签向量。
    :param feature_idx: 划分参考的特征 index
    :param threshold: float 阈值
    :return: 条件熵 cond_entropy, 划分 splits ((X_left, y_left), (X_right, y_right))
    """
    n_samples, _ = X.shape
    feature_values = X[:, feature_idx]
    mask = (feature_values <= threshold)
    X_left, y_left = X[mask], y[mask]
    X_right, y_right = X[~mask], y[~mask]
    # 跳过空集
    if len(y_left) == 0 or len(y_right) == 0:
        # 返回极大条件熵与空划分，表示该阈值无效
        return float("inf"), None

    cond_entropy = len(y_left) / n_samples * get_entropy(y_left) + len(y_right) / n_samples * get_entropy(y_right)

    return cond_entropy, ((X_left, y_left), (X_right, y_right))


def _continuous_split(X, y, feature_idx):
    """
    计算连续型随机变量的最优划分的条件熵并返回划分结果与条件熵
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵。
    :param y: np.ndarray, shape = (n_samples,)
              标签向量。
    :param feature_idx: 划分参考的特征 index
    :return: 最优条件熵 best_cond_entropy, 最佳划分 best_splits ((X_left, y_left), (X_right, y_right)), 最佳阈值
    """
    feature_values = X[:, feature_idx]
    unique_values = np.sort(np.unique(feature_values))

    # 若所有值相同，则无法划分
    if len(unique_values) == 1:
        return float("inf"), None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

    best_splits = None
    best_threshold = None
    best_cond_entropy = float("inf")

    for threshold in thresholds:
        cond_entropy, splits = _continuous_split_with_threshold(X, y, feature_idx, threshold)
        if cond_entropy < best_cond_entropy:
            best_cond_entropy = cond_entropy
            best_splits = splits
            best_threshold = threshold

    return best_cond_entropy, best_splits, best_threshold


def _cart_discrete_split(X, y, feature_idx):
    """
    CART 算法的离散特征二叉划分（使用外部 get_gini 计算基尼指数）
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵
    :param y: np.ndarray, shape = (n_samples,)
              标签向量
    :param feature_idx: int
              当前特征索引
    :return:
        best_gini : float
            最小加权 Gini 指数（即最优划分后的基尼）
        best_splits : tuple
            ((X_left, y_left), (X_right, y_right))
        best_threshold : set
            左分支取值集合（用于二叉划分）
    """
    n_samples, _ = X.shape
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values)

    # 若该特征只有一个取值，无需划分
    if len(unique_values) == 1:
        return float("inf"), None, None

    best_gini = float("inf")
    best_splits = None
    best_threshold = None

    # 枚举所有非空真子集（只取一半避免重复）
    for i in range(1, len(unique_values) // 2 + 1):
        for subset in itertools.combinations(unique_values, i):
            subset = set(subset)

            # 二叉划分
            mask = np.isin(feature_values, list(subset))
            X_left, y_left = X[mask], y[mask]
            X_right, y_right = X[~mask], y[~mask]

            # 跳过空集
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # 计算加权 Gini
            gini_split = (len(y_left) / n_samples) * get_gini(y_left) + (len(y_right) / n_samples) * get_gini(y_right)

            # 更新最优划分
            if gini_split < best_gini:
                best_gini = gini_split
                best_splits = ((X_left, y_left), (X_right, y_right))
                best_threshold = subset

    return best_gini, best_splits, best_threshold


def _cart_continuous_split(X, y, feature_idx):
    """
    CART 算法的连续特征二叉划分（使用外部 get_gini 计算基尼指数）
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵
    :param y: np.ndarray, shape = (n_samples,)
              标签向量
    :param feature_idx: int
              当前特征索引
    :return:
        best_gini : float
            最小加权 Gini 指数
        best_splits : tuple
            ((X_left, y_left), (X_right, y_right))
        best_threshold : float
            最优划分阈值
    """
    n_samples, _ = X.shape
    feature_values = X[:, feature_idx]

    # 若特征值全相同，则无法划分
    unique_values = np.unique(feature_values)
    if len(unique_values) == 1:
        return float("inf"), None, None

    # 候选阈值：相邻特征值的中点
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

    best_gini = float("inf")
    best_splits = None
    best_threshold = None

    for threshold in thresholds:
        # 二叉划分
        mask = (feature_values <= threshold)
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        # 跳过空集
        if len(y_left) == 0 or len(y_right) == 0:
            continue

        # 计算加权 Gini（使用外部函数 get_gini）
        gini_split = (len(y_left) / n_samples) * get_gini(y_left) + (len(y_right) / n_samples) * get_gini(y_right)

        # 更新最优划分
        if gini_split < best_gini:
            best_gini = gini_split
            best_splits = ((X_left, y_left), (X_right, y_right))
            best_threshold = threshold

    return best_gini, best_splits, best_threshold


def _cart_regression_split(X, y, feature_idx):
    """
    Cart 回归算法二叉划分（使用 MSE 进行评估）
    :param X: np.ndarray, shape = (n_samples, n_features)
              特征矩阵
    :param y: np.ndarray, shape = (n_samples,)
              标签向量
    :param feature_idx: int
              当前特征索引
    :return:
        best_mse : float
            最小 MSE
        best_splits : tuple
            ((X_left, y_left), (X_right, y_right))
        best_threshold : float
            最优划分阈值
    """
    n_samples, _ = X.shape
    feature_values = X[:, feature_idx]
    unique_values = np.sort(np.unique(feature_values))

    # 若该特征无可划分点
    if len(unique_values) == 1:
        return float("inf"), None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    best_mse = float("inf")
    best_splits = None
    best_threshold = None

    for threshold in thresholds:
        mask = feature_values <= threshold
        y_left, y_right = y[mask], y[~mask]
        X_left, X_right = X[mask], X[~mask]

        # 跳过无效划分
        if len(y_left) == 0 or len(y_right) == 0:
            continue

        # 加权平均 MSE（CART目标函数）
        mse = (len(y_left) * np.var(y_left) + len(y_right) * np.var(y_right)) / n_samples

        if mse < best_mse:
            best_mse = mse
            best_splits = ((X_left, y_left), (X_right, y_right))
            best_threshold = threshold

    return best_mse, best_splits, best_threshold
