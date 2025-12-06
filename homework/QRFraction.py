import numpy as np


def sgn(x: float) -> float:
    """
    sgn 函数实现
    :param x: 浮点数
    :return: sgn(x)
    """
    return 1 if x > 0 else -1


def norm(A: np.ndarray) -> float:
    """
    求矩阵的 F 范数
    :param A: 输入矩阵 形状为 [m,n]
    :return: ||A||_F
    """
    m, n = A.shape
    res = 0.0
    for i in range(m):
        for j in range(n):
            res += A[i, j] ** 2

    return res ** 0.5


def rotation_matrix(row1: int, row2: int, a: float, b: float, m: int) -> np.ndarray:
    """

    :param row1: 元素 1 对应的行
    :param row2: 元素 2 对应的行
    :param a: 元素 1
    :param b: 元素 2
    :param m: 旋转矩阵规模
    :return: G 旋转矩阵 形状 [m, m]
    """
    c = (a ** 2 + b ** 2) ** 0.5
    G = np.eye(m)

    G[row1, row1] = a / c
    G[row2, row2] = a / c
    G[row1, row2] = b / c
    G[row2, row1] = - b / c

    return G


def embed(A: np.ndarray, m: int) -> np.ndarray:
    """
    将 n 维矩阵扩展为 m 维矩阵 形状为[[I, 0], [0,A]]
    :param A: 输入矩阵 形状为 [n, n]
    :param m: 扩展后大小
    :return: 扩展矩阵 B
    """
    n = A.shape[0]
    assert A.shape == (n, n)
    assert m >= n
    k = m - n
    B = np.zeros((m, m), dtype=A.dtype)
    # 左上角单位阵
    B[:k, :k] = np.eye(k, dtype=A.dtype)
    # 右下角放 A
    B[k:, k:] = A
    return B


def Modified_Gram_Schmidt(A: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Modified Gram-Schmidt (MGS) 实现 QR 分解
    :param A: 输入矩阵 形状为 [m, n], 且要求 rank(A) = n
    :return: 正交分解 Q, R
    """
    m, n = A.shape
    # 将 A 按列拆分
    u = [A[:, i:i + 1] for i in range(n)]
    # 生成 R 矩阵
    R = np.zeros((n, n))

    for i in range(n):
        # 遍历每一列
        # 归一化
        v = norm(u[i])
        if v < 1e-16:
            print("数值异常")
            exit(0)
        u[i] = u[i] / v

        # 消除后续向量的对应分量，并将系数写入 R 矩阵
        R[i, i] = v
        for j in range(i + 1, n):
            r = u[i].T @ u[j]
            u[j] = u[j] - r * u[i]
            R[i, j] = r

    # 将正交向量堆叠为 Q
    Q = np.hstack(u)
    return Q, R


def Householder_Reflection(A: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Householder Reflection 实现 QR 分解
    :param A: 输入矩阵 形状为 [m, n], 且要求 rank(A) = n
    :return: 正交分解 Q, R
    """
    m, n = A.shape
    # 生成 P 矩阵，生成 R 矩阵
    P = np.eye(m)
    T = A.copy()

    for i in range(n):
        # 遍历前 n 列
        # 截取 u 向量，形状 [m - i, 1]
        u = np.array(T[i:m, i:i + 1])
        e1 = np.zeros((m - i, 1))
        e1[0, 0] = 1
        u = u - sgn(u[0, 0]) * norm(u) * e1

        # 计算反射矩阵
        v = (u.T @ u).item()
        if v < 1e-6:
            print("数值异常")
            exit(0)

        H = np.eye(m - i) - 2 * (u @ u.T) / v
        H = embed(H, m)

        # 记录结果
        P = H @ P
        T = H @ T

    Q = P.T[:, :n]
    R = T[:n, :]
    return Q, R


def Givens_Reflection(A: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Givens Reflection 实现 QR 分解
    :param A: 输入矩阵 形状为 [m, n], 且要求 rank(A) = n
    :return: 正交分解 Q, R
    """
    m, n = A.shape
    T = A.copy()
    P = np.eye(m)
    for j in range(n):
        # 遍历所有行
        for i in range(j + 1, m):
            # 遍历所有下三角元素
            a = T[j, j]  # 对角线元素
            b = T[i, j]  # 消去元素
            if b < 1e-16:
                continue
            # 获取旋转矩阵
            G = rotation_matrix(j, i, a, b, m)
            T = G @ T
            P = G @ P

    Q = P.T[:, :n]
    R = T[:n, :]
    return Q, R


def solve(A: np.ndarray, b: np.ndarray, type: int) -> np.ndarray:
    """
    求解方程
    :param A: 输入矩阵 形状为 [m, n], 且要求 rank(A) = n
    :param b: 输入矩阵 形状为 [m, 1]
    :param type: 使用的求解方法 0:MGS, 1:Householder, 2:Givens
    :return: 解或近似解 x 形状为
    """
    m, n = A.shape

    assert m >= n
    assert b.shape[0] == m
    assert b.shape[1] == 1

    Q = None
    R = None
    if type == 0:
        Q, R = Modified_Gram_Schmidt(A)
    elif type == 1:
        Q, R = Householder_Reflection(A)
    elif type == 2:
        Q, R = Givens_Reflection(A)
    else:
        print("没有输入正确的求解方法")
        exit(0)

    print("\n==============================")
    print("          QR 分解结果")
    print("==============================")
    print("Q =\n", Q)
    print("R =\n", R)

    print("\n==============================")
    print("     方程 Ax = b 的解/最小二乘解")
    print("==============================")

    b_hat = Q.T @ b
    x = np.zeros((n, 1))
    for i in reversed(range(n)):
        s = R[i, i+1:] @ x[i + 1:]
        x[i] = (b_hat[i] - s) / R[i, i]

    print(x)
    return x


def input_matrix(m, n):
    """
    矩阵输入
    :param m: 行数
    :param n: 列数
    :return: 输入矩阵
    """
    print("\n请输入矩阵 A（每行输入 {} 个数字，用空格分隔）".format(n))
    A = []
    for i in range(m):
        while True:
            try:
                row = input(f"A 第 {i+1} 行：")
                nums = list(map(float, row.split()))
                if len(nums) != n:
                    raise ValueError
                A.append(nums)
                break
            except:
                print("输入格式错误，请重新输入。")
    return np.array(A)


def input_vector(m):
    """
    向量输入
    :param m: 行数
    :return: 输入的列向量
    """
    print("\n请输入向量 b（每行 1 个数字）")
    b = []
    for i in range(m):
        while True:
            try:
                val = float(input(f"b 第 {i+1} 个元素："))
                b.append([val])
                break
            except:
                print("输入格式错误，请重新输入。")
    return np.array(b)


def main():
    print("===================================")
    print("   QR 分解程序（MGS / Householder / Givens）")
    print("===================================\n")

    # 选择方法
    while True:
        try:
            t = int(input("请选择方法（0-MGS，1-Householder，2-Givens）："))
            if t in [0, 1, 2]:
                break
            else:
                print("请输入 0、1 或 2。")
        except:
            print("输入错误，请输入数字。")

    # 输入矩阵大小
    while True:
        try:
            m = int(input("\n请输入矩阵 A 的行数 m："))
            n = int(input("请输入矩阵 A 的列数 n（要求 m ≥ n）："))
            if m >= n > 0 and m > 0:
                break
            else:
                print("必须满足 m ≥ n 且均为正整数。")
        except:
            print("输入错误，请输入整数。")

    A = input_matrix(m, n)
    b = input_vector(m)

    solve(A, b, t)


if __name__ == '__main__':
    main()
