import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# ---------- 统计参数 ----------
def get_parameter(X):
    """
    X.shape = (d, n)  列为样本
    返回: mean(d,1), C(d,d)
    """
    mean = np.mean(X, axis=1, keepdims=True)
    C = ((X - mean) @ (X - mean).T) / X.shape[1]
    return mean, C


# ---------- 判别函数（符号） ----------
def discriminant(mean, C, P_wi):
    """
    根据 mean, C, P_wi 生成符号判别函数 d(x)
    自动根据维度生成 (x1,...,xd)
    返回: d_expr (sympy expr), x_symbols(tuple)
    """
    dim = mean.shape[0]
    x_symbols = sp.symbols(f'x1:{dim + 1}')  # (x1,...,xd)
    x = sp.Matrix(x_symbols)

    m = sp.Matrix(mean)
    C = sp.Matrix(C)
    C_inv = C.inv()
    det_C = C.det()

    quad_form = ((x - m).T * C_inv * (x - m))[0]  # 标量
    d = sp.log(P_wi) - sp.Rational(1, 2) * sp.log(det_C) - sp.Rational(1, 2) * quad_form
    return sp.simplify(d), x_symbols


# ---------- 边界方程 ----------
def decision_boundary_expr(mean1, C1, P1, mean2, C2, P2):
    """
    生成 d1(x)-d2(x)=0 的符号表达式（任意维）
    若为二维（d=2），额外返回 A,b,c 使得 x^T A x + b^T x + c = 0
    返回: boundary_expr, x_symbols, (A,b,c 或 None)
    """
    d1, xs = discriminant(mean1, C1, P1)
    d2, _ = discriminant(mean2, C2, P2)
    expr = sp.simplify(sp.expand(d1 - d2))  # = 0 就是边界

    # 二维提取二次曲线系数
    if len(xs) == 2:
        x1, x2 = xs
        poly = sp.Poly(expr, x1, x2)  # 注意这里是 expr = 0
        # 系数（缺项默认0）
        a11 = poly.coeffs()[poly.monoms().index((2, 0))] if (2, 0) in poly.monoms() else 0
        a22 = poly.coeffs()[poly.monoms().index((0, 2))] if (0, 2) in poly.monoms() else 0
        a12 = poly.coeffs()[poly.monoms().index((1, 1))] if (1, 1) in poly.monoms() else 0
        b1 = poly.coeffs()[poly.monoms().index((1, 0))] if (1, 0) in poly.monoms() else 0
        b2 = poly.coeffs()[poly.monoms().index((0, 1))] if (0, 1) in poly.monoms() else 0
        c = poly.coeffs()[poly.monoms().index((0, 0))] if (0, 0) in poly.monoms() else 0

        # 注意 x^T A x = a11*x1^2 + 2*(a12/2)*x1*x2 + a22*x2^2
        # 这里 poly 对 x1*x2 的系数是 a12（全系数），故令 A12=A21=a12/2
        A = sp.Matrix([[a11, a12 / 2],
                       [a12 / 2, a22]])
        b = sp.Matrix([b1, b2])
        return sp.simplify(expr), xs, (sp.simplify(A), sp.simplify(b), sp.simplify(c))
    else:
        return sp.simplify(expr), xs, None


# ---------- 绘图并输出方程 ----------
def plot_and_print_boundary(X1, X2, P1=0.5, P2=0.5,
                            xlim=(-1, 8), ylim=(-1, 8), grid_size=400,
                            show_eq=True):
    """
    二维可视化：画出 d1-d2=0 的边界，并打印方程式
    对高维：仅打印边界式，不绘图
    """
    mean1, C1 = get_parameter(X1)
    mean2, C2 = get_parameter(X2)

    boundary_expr, xs, quad_pack = decision_boundary_expr(mean1, C1, P1, mean2, C2, P2)

    # 打印方程式
    if show_eq:
        print("决策边界方程（符号形式）: ")
        print("  ", sp.Eq(boundary_expr, 0))
        if quad_pack is not None:
            A, b, c = quad_pack
            print("\n二维二次曲线标准形式： x^T A x + b^T x + c = 0")
            print("A =");
            sp.pprint(A)
            print("b =", b.T)
            print("c =", c)

    # 仅在二维下绘图
    if len(xs) == 2:
        x1_sym, x2_sym = xs
        # 数值化 d1,d2 再做网格
        d1_expr, _ = discriminant(mean1, C1, P1)
        d2_expr, _ = discriminant(mean2, C2, P2)
        f1 = sp.lambdify((x1_sym, x2_sym), d1_expr, "numpy")
        f2 = sp.lambdify((x1_sym, x2_sym), d2_expr, "numpy")

        x1 = np.linspace(xlim[0], xlim[1], grid_size)
        x2 = np.linspace(ylim[0], ylim[1], grid_size)
        Xg, Yg = np.meshgrid(x1, x2)
        Z = f1(Xg, Yg) - f2(Xg, Yg)

        plt.figure(figsize=(6, 6))
        # 决策边界
        plt.contour(Xg, Yg, Z, levels=[0])
        # 区域填色
        plt.contourf(Xg, Yg, Z, levels=50, alpha=0.6)
        # 样本点
        plt.scatter(X1[0], X1[1], marker='o', label='ω1')
        plt.scatter(X2[0], X2[1], marker='s', label='ω2')

        plt.xlim(*xlim);
        plt.ylim(*ylim)
        plt.xlabel("x1");
        plt.ylabel("x2")
        plt.title("Bayes Discriminant Boundary (Gaussian Model)")
        plt.legend();
        plt.grid(True);
        plt.show()
    else:
        print("\n（非二维数据：已输出边界方程；不进行绘图。）")


# ===== 示例 =====
if __name__ == "__main__":
    X1 = np.array([[0, 2, 2, 0],
                   [0, 0, 2, 2]])
    X2 = np.array([[4, 6, 6, 4],
                   [4, 4, 6, 6]])
    plot_and_print_boundary(X1, X2, P1=0.5, P2=0.5)
