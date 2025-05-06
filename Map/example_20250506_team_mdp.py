#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
from MDP_TG.mdp import Motion_MDP
from User.mdp3 import MDP3
from User.utils import print_c

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# 保留原始观测和控制逻辑
observation_dict = {
    'p': ['0', '5'],
    'q': ['1'],
    'u': ['2', '3', '4', '6'],
}

control_observable_dict = None

def generate_grid_graph(x, y, d=1, diagonal=False):
    """生成x*y的栅格图（可选对角线移动）"""
    G = nx.grid_2d_graph(x, y, create_using=nx.DiGraph)

    if diagonal:  # 添加对角线边
        for u in list(G.nodes()):
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                v = (u[0] + dx, u[1] + dy)
                if v in G.nodes():
                    G.add_edge(u, v)

    # 将节点重命名为字符串ID（行优先顺序）
    nodes = {f"{row * y + col}": {"pos": (col * d, (x - 1 - row) * d)}
             for row, col in sorted(G.nodes())}

    # 构建带权重的边（欧氏距离）
    edges = {}
    for (u_row, u_col), (v_row, v_col) in G.edges():
        u = str(u_row * y + u_col)
        v = str(v_row * y + v_col)
        distance = math.hypot(v_col - u_col, v_row - u_row) * d
        edges[(u, v)] = {"weight": distance}

    return nodes, edges


def build_mdp_with_grid(x, y, d=1):
    """构建结合栅格图和MDP逻辑的完整MDP"""
    grid_nodes, grid_edges = generate_grid_graph(x, y, d)

    robot_nodes_w_aps = defaultdict(dict)
    robot_edges = {}
    U = ['b']  # 不可观察动作

    # 基于位置分配APs
    for node_id, attrs in grid_nodes.items():
        x_pos, y_pos = attrs["pos"]
        if x_pos == 0 and y_pos == 0:
            robot_nodes_w_aps[node_id] = {frozenset({'upload'}): 1.0}  # 起点
        elif x_pos == (y - 1) * d and y_pos == (x - 1) * d:  # 修正终点位置计算
            robot_nodes_w_aps[node_id] = {frozenset({'recharge'}): 1.0}  # 终点
        else:
            robot_nodes_w_aps[node_id] = {frozenset({''}): 1.0}  # 普通节点

    # 为边分配概率和成本
    for (u, v), attrs in grid_edges.items():
        robot_edges[(u, 'a', v)] = (1.0, attrs["weight"])  # 可观察动作
        robot_edges[(u, 'b', v)] = (0.8, attrs["weight"])  # 不可观察动作

    return robot_nodes_w_aps, robot_edges, U, grid_nodes  # 返回grid_nodes用于可视化

def construct_team_mdp():
    """构建完整团队MDP"""
    return build_mdp_with_grid(5, 5)