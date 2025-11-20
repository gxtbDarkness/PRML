import numpy as np


class DecisionTreeVisualizer:
    """
    决策树可视化工具类
    ------------------------------------------------------------
    支持：
        - Graphviz 渲染（推荐，结构清晰、美观）
        - NetworkX + Matplotlib 绘制（无需外部依赖）
    自动处理：
        - 连续特征阈值（float 格式）
        - 离散特征集合（set 格式）
        - 多叉分裂与二叉分裂
    ------------------------------------------------------------
    用法：
        viz = DecisionTreeVisualizer()
        viz.show(tree, filename="my_tree")
    """

    def __init__(self):
        """自动检测可视化后端"""
        try:
            import graphviz
            self.backend = "graphviz"
        except ImportError:
            try:
                import networkx
                import matplotlib.pyplot as plt
                self.backend = "networkx"
            except ImportError:
                raise ImportError("请安装 graphviz 或 networkx+matplotlib 任意一个可视化库。")

    # ============================================================
    # 主接口
    # ============================================================
    def show(self, tree, filename="decision_tree", view=True):
        """
        主接口：可视化决策树
        :param tree: 训练好的决策树实例（需有 tree.root）
        :param filename: 输出文件名（不含扩展名）
        :param view: 是否自动打开结果（Graphviz 模式）
        """
        if not hasattr(tree, "root"):
            raise ValueError("tree 对象无 root 属性，请确认传入的为训练后的决策树。")

        if self.backend == "graphviz":
            self._show_graphviz(tree, filename, view)
        else:
            self._show_networkx(tree)

    # ============================================================
    # Graphviz 可视化
    # ============================================================
    def _show_graphviz(self, tree, filename, view):
        """使用 Graphviz 可视化决策树"""
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("未检测到 graphviz，请安装：pip install graphviz")

        dot = Digraph(comment="Decision Tree", format="png")

        # ✅ 设置字体（推荐 DejaVu Sans，支持 ≤ ∈ 中文）
        dot.attr(fontname="DejaVu Sans")
        dot.node_attr.update(fontname="DejaVu Sans")
        dot.edge_attr.update(fontname="DejaVu Sans")

        self._add_nodes_graphviz(dot, tree.root, tree.feature_names)
        out_path = dot.render(filename=filename, view=view)
        print(f"✅ Graphviz 渲染完成：{out_path}")

    def _add_nodes_graphviz(self, dot, node, feature_names, parent_id=None, edge_label=""):
        """递归添加节点到 Graphviz 图"""
        node_id = str(id(node))

        # ---------- 叶节点 ----------
        if node.is_leaf:
            label = f"Leaf: {node.value}"
            dot.node(node_id, label, shape="box", style="filled", color="lightgrey")

        # ---------- 非叶节点 ----------
        else:
            feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"

            # 根据 threshold 类型生成标签
            if node.threshold is not None:
                if isinstance(node.threshold, (int, float, np.number)):
                    label = f"{feature_name} {node.operator} {node.threshold:.3f}"
                elif isinstance(node.threshold, set):
                    vals_str = ", ".join(map(str, sorted(list(node.threshold))))
                    label = f"{feature_name} ∈ {{{vals_str}}}"
                else:
                    label = f"{feature_name} {node.operator} {node.threshold}"
            else:
                label = f"{feature_name} {node.operator or ''}"

            dot.node(node_id, label, shape="ellipse", color="skyblue")

        # ---------- 连接边 ----------
        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        # ---------- 递归子节点 ----------
        if not node.is_leaf:
            if node.split_type == "binary":  # 二叉树
                if isinstance(node.threshold, (int, float, np.number)):
                    left_label = f"≤ {node.threshold:.3f}"
                    right_label = f"> {node.threshold:.3f}"
                else:
                    left_label = "∈ 左子集"
                    right_label = "∈ 右子集"
                self._add_nodes_graphviz(dot, node.left, feature_names, node_id, left_label)
                self._add_nodes_graphviz(dot, node.right, feature_names, node_id, right_label)
            elif node.split_type == "multiway":  # 多叉树（ID3/C4.5）
                for val, child in node.children.items():
                    self._add_nodes_graphviz(dot, child, feature_names, node_id, str(val))

    # ============================================================
    # NetworkX + Matplotlib 可视化（备用方案）
    # ============================================================
    def _show_networkx(self, tree):
        """使用 NetworkX + Matplotlib 绘制简易决策树"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("请安装 networkx 和 matplotlib。")

        def add_edges(G, node, parent=None, edge_label=""):
            node_id = str(id(node))

            # 生成标签
            if node.is_leaf:
                label = f"Leaf: {node.value}"
            else:
                feature_name = tree.feature_names[node.feature] if tree.feature_names else f"X[{node.feature}]"
                if node.threshold is not None:
                    if isinstance(node.threshold, (int, float, np.number)):
                        label = f"{feature_name}\n({node.operator} {node.threshold:.2f})"
                    elif isinstance(node.threshold, set):
                        vals_str = ", ".join(map(str, sorted(list(node.threshold))))
                        label = f"{feature_name}\n∈ {{{vals_str}}}"
                    else:
                        label = f"{feature_name}\n({node.operator} {node.threshold})"
                else:
                    label = f"{feature_name}\n({node.operator or ''})"

            G.add_node(node_id, label=label)
            if parent:
                G.add_edge(parent, node_id, label=edge_label)

            # 递归添加子节点
            if not node.is_leaf:
                if node.split_type == "binary":
                    add_edges(G, node.left, node_id, "Left")
                    add_edges(G, node.right, node_id, "Right")
                elif node.split_type == "multiway":
                    for val, child in node.children.items():
                        add_edges(G, child, node_id, str(val))

        G = nx.DiGraph()
        add_edges(G, tree.root)

        pos = nx.spring_layout(G, seed=42)
        labels = nx.get_node_attributes(G, "label")
        nx.draw(G, pos, labels=labels, node_color="lightblue", node_size=2500,
                font_size=8, font_weight="bold", with_labels=True)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.title("Decision Tree Visualization (NetworkX)")
        plt.axis("off")
        plt.show()
