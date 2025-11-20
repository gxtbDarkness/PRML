# utils.py 工具类，用于实现数据集的加载与相关预处理
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_car_evaluation_dataset():
    """
    加载 Car Evaluation（汽车评估）数据集，全部为离散变量，用于多分类任务。
    数据来源：UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data

    :return:
        X (pandas DataFrame): 包含特征数据及目标列 'target'
        feature_types_dict (dict): 特征类型字典，所有特征均为 'discrete'
    """
    df = pd.read_csv('./Datasets/car.csv', header=None)

    df.columns = [
        "buying", "maint", "doors", "persons",
        "lug_boot", "safety", "target"
    ]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_types_dict = {col: 'discrete' for col in X.columns}

    X['target'] = y

    return X, feature_types_dict


def load_iris_dataset():
    """
    加载鸢尾花数据集，均为连续变量，用于分类任务
    :return: X (pandas DataFrame)，包含特征数据；以及字典 feature_types_dict，表示特征类型
    """
    # 加载鸢尾花数据集
    iris = pd.read_csv('./Datasets/iris_dataset.csv')
    # 特征数据
    X = iris.iloc[:, :-1]  # 除了最后一列是标签，其他是特征
    y = iris.iloc[:, -1]
    # 特征类型标识: 所有特征均为连续变量
    feature_types_dict = {col: 'continuous' for col in X.columns}
    X['target'] = y
    # 返回特征数据和特征类型字典
    return X, feature_types_dict


def load_titanic_dataset():
    """
    加载泰坦尼克数据集，包含连续型变量和离散型变量，用于分类任务
    :return: X (pandas DataFrame)，包含特征数据；以及字典 feature_types_dict，表示特征类型
    """
    # 加载泰坦尼克数据集
    titanic = pd.read_csv('./Datasets/titanic_cleaned.csv')
    X = titanic.drop(columns=['Survived'])
    y = titanic['Survived']
    # 特征类型标识: 连续变量和离散变量混合
    feature_types_dict = {col: 'continuous' if X[col].dtype in ['float64', 'int64'] else 'discrete' for col in X.columns}
    X['Target'] = y
    return X, feature_types_dict


def load_boston_dataset():
    """
    加载波士顿房价数据集，包含连续型变量和离散型变量，用于回归任务
    :return: X (pandas DataFrame)，包含特征数据；以及字典 feature_types_dict，表示特征类型
    """
    # 加载波士顿房价数据集
    boston = pd.read_csv('./Datasets/boston_housing.csv')
    X = boston.iloc[:, :-1]  # 除了最后一列是标签，其他是特征
    y = boston.iloc[:, -1]
    # 特征类型标识: 连续变量和离散变量混合
    feature_types_dict = {col: 'continuous' if X[col].dtype in ['float64', 'int64'] else 'discrete' for col in X.columns}
    X['Target'] = y
    return X, feature_types_dict


def partition_dataset(X, ratio):
    """
    将特征数据按照比例划分为训练集与测试集（纯 NumPy 实现）
    :param X: 特征数据（numpy.ndarray 或 pandas.DataFrame）
              假定最后一列为目标标签
    :param ratio: 训练集占比，范围 [0, 1]
    :return: X_train, X_test, y_train, y_test （全部为 numpy.ndarray）
    """
    # 若输入为 DataFrame，则转为 NumPy 数组
    if hasattr(X, "values"):
        X = X.values

    # 分离特征与标签
    X_data = X[:, :-1]  # 除最后一列
    y_data = X[:, -1]   # 最后一列为目标

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=1 - ratio, random_state=42
    )

    return X_train, X_test, y_train, y_test


def partition_dataset_with_val(X, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集划分为训练集、验证集和测试集（纯 NumPy 实现）

    Parameters
    ----------
    X : numpy.ndarray 或 pandas.DataFrame
        输入数据，假定最后一列为目标标签。
    train_ratio : float, default=0.7
        训练集比例，范围 [0,1]。
    val_ratio : float, default=0.15
        验证集比例，范围 [0,1]。
    test_ratio : float, default=0.15
        测试集比例，范围 [0,1]。
        三者之和应为 1。

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : numpy.ndarray
        按比例划分后的特征与标签。
    """
    # 检查比例合法性
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train+val+test 比例之和必须为 1，但当前为 {total:.2f}")

    # 若输入为 DataFrame，则转为 NumPy 数组
    if hasattr(X, "values"):
        X = X.values

    # 分离特征与标签
    X_data = X[:, :-1]
    y_data = X[:, -1]

    # 先划出训练集与临时集（验证+测试）
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_data, y_data, test_size=(1 - train_ratio), random_state=42
    )

    # 按比例在临时集中再划分验证集与测试集
    val_portion = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_portion), random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

