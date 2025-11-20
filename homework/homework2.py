import numpy as np


def augmentation_for_two(X, y):
    """
    对 X 进行增广处理（在最后一列添加常数 1），以将偏置项 b 统一并入参数向量 W 中。
    :param X: 输入样本 [n_samples, p]
    :param y: 样本分类 [n_samples] y 属于 {-1, 1}
    :return: X: 增广样本 [n_samples, p + 1]
    """
    X_aug = np.hstack((X, np.ones((X.shape[0], 1)))) * y.reshape(-1, 1)
    return X_aug


def fit_for_two(X_aug, y, W, epoches=100, learning_rate=0.001):
    """
    二分类感知机训练函数
    :param learning_rate: 学习率
    :param X_aug: 增广样本 [n_samples, p + 1]
    :param y: 样本实际类别 [n_samples]
    :param W: 权重矩阵 [1, p + 1]
    :param epoches: 迭代次数
    :return: 训练过程 W 列表 [W1, W2, ..., Wk]
    """
    W_list = [W.copy()]
    n_samples, p_plus = X_aug.shape
    n_classes = W.shape[0]

    for epoch in range(epoches):
        error = 0
        for j in range(len(y)):
            x = X_aug[j: j + 1, :]  # X_aug[j : j + 1, :]这样子是二维切片，现在这个切片方法是一维
            pred = x @ W.T
            if pred <= 0:
                W += learning_rate * x
                error += 1
                W_list.append(W.copy())
        if error == 0:
            break

    return W_list


if __name__ == '__main__':
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    X_aug = augmentation_for_two(X, y)

    W = np.zeros((1, X_aug.shape[1]))
    W_list = fit_for_two(X_aug, y, W, epoches=100, learning_rate=1)

    print(W_list)
