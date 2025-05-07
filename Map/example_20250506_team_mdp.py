#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
from MDP_TG.mdp import Motion_MDP
from Map.simple_example_20250401 import initial_label
from User.mdp3 import MDP3
from User.utils import print_c

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

'''

(0, 0)                                                        (0, 5)
upload    空    recharge          recharge          空         inaccessible
空        空    空                 空                空         inaccessible
空        空    空                 gather            gather    空
空        空    空                 空                空         空
upload    空    investigate       空                空         upload
upload    空    inaccessible      inaccessible      空         upload
(5, 0)                                                        (5, 5)

入侵者只知道行信息，动作和观测无关
三个出发点，左上(0, 0)，左下（4, 0），右下(4, 5)

'''

# ----------------------------
# 保留原始观测和控制逻辑


special_grids_in_map = {
    (0, 0): {frozenset({'upload'}): 1.0},
    (0, 2): {frozenset({'recharge'}): 1.0},
    (0, 3): {frozenset({'recharge'}): 1.0},
    (2, 3): {frozenset({'gather'}): 1.0},
    (2, 4): {frozenset({'gather'}): 1.0},
    (4, 0): {frozenset({'upload'}): 1.0},
    (4, 1): {frozenset({'investigate'}): 1.0},
    (4, 5): {frozenset({'upload'}): 1.0},
    (5, 0): {frozenset({'upload'}): 1.0},
    (5, 5): {frozenset({'upload'}): 1.0},
}

# 不可达节点（将被删除）
inaccessible_grids_in_map = [
    (0, 5), (1, 5), (5, 2), (5, 3),
]

# 起点列表
start_positions = [(0, 0), (4, 0), (4, 5)]
U0_dict = {
    (0, 0) : 'R',
    (4, 0) : 'R',
    (4, 5) : 'L'
}

def generate_grid_graph(x, y, d=1, diagonal=False):
    """生成x*y的栅格图（可选对角线移动）"""
    G = nx.grid_2d_graph(x, y, create_using=nx.DiGraph)

    if diagonal:  # 添加对角线边
        for u in list(G.nodes()):
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                v = (u[0] + dx, u[1] + dy)
                if v in G.nodes():
                    G.add_edge(u, v)

    return G


def build_mdp_with_grid(x, y, start_position=None, d=1):
    global special_grids_in_map, inaccessible_grids_in_map, U0_dict
    """构建结合自定义坐标/命题/不可达点的MDP"""
    G = generate_grid_graph(x, y, d, True)
    G.remove_nodes_from(inaccessible_grids_in_map)

    robot_nodes_w_aps = defaultdict(dict)
    grid_nodes = {}
    robot_edges = {}

    for (row, col) in sorted(G.nodes()):
        node_id = f"{row * y + col}"
        pos = (col * d, (x - 1 - row) * d)
        grid_nodes[node_id] = {"pos": pos}

        if (row, col) in special_grids_in_map:
            robot_nodes_w_aps[node_id] = special_grids_in_map[(row, col)]
        else:
            robot_nodes_w_aps[node_id] = {frozenset({''}): 1.0}

    # 定义动作及其扰动方向与概率
    action_defs = {
        'u': [((-1, -1), 0.05), ((-1, 0), 0.9), ((-1, 1), 0.05)],
        'd': [((1, -1), 0.05), ((1, 0), 0.9), ((1, 1), 0.05)],
        'l': [((-1, -1), 0.05), ((0, -1), 0.9), ((1, -1), 0.05)],
        'r': [((-1, 1), 0.05), ((0, 1), 0.9), ((1, 1), 0.05)],
    }

    U = list(action_defs.keys())  # ['u', 'd', 'l', 'r']

    for (u_row, u_col) in G.nodes():
        u = str(u_row * y + u_col)
        for act, transitions in action_defs.items():
            for (dr, dc), prob in transitions:
                v_row, v_col = u_row + dr, u_col + dc
                v = str(v_row * y + v_col)
                if (v_row, v_col) in G:
                    dist = math.hypot(dc, dr) * d
                    robot_edges[(u, act, v)] = (prob, dist)


    # 计算起点 ID
    if start_position != None:
        start_ids = f"{start_position[0] * y + start_position[1]}"
    else:
        start_ids = None
    initial_label = tuple([k for k in robot_nodes_w_aps[start_ids].keys()])

    if start_position in U0_dict.keys():
        U0 = U0_dict[start_position]
    else:
        U0 = ['r']  # 不可观察动作

    return robot_nodes_w_aps, robot_edges, U0, grid_nodes, start_ids, initial_label


def observation_func(state_id, y_len):
    """入侵者的观测函数：只观测行信息"""
    row = int(state_id) // y_len
    return str(row)

def remove_specific_states_4_team_mdp(team_mdp:MDP3):
    state_list_to_remove = []

    # TODO
    team_mdp.remove_state_sequence(state_list_to_remove)

def visualize_grids_in_networkx():
    #TODO
    pass

def construct_team_mdp():
    """构建完整团队MDP"""
    # Added
    global start_positions

    robot_nodes_w_aps_1, robot_edges_1, U_1, grid_nodes_1, start_ids_1, initial_label_1 = build_mdp_with_grid(6, 6, start_position=start_positions[0])
    robot_nodes_w_aps_2, robot_edges_2, U_2, grid_nodes_2, start_ids_2, initial_label_2 = build_mdp_with_grid(6, 6, start_position=start_positions[1])

    #
    # 这里有一些斜角边因为共用动作冲突会只被保留一条
    mdp_r1 = Motion_MDP(robot_nodes_w_aps_1, robot_edges_1, U_1, start_ids_1, initial_label_1)
    mdp_r2 = Motion_MDP(robot_nodes_w_aps_2, robot_edges_2, U_2, start_ids_2, initial_label_2)
    #
    initial_node_list  = [start_ids_1, start_ids_2]
    initial_label_list = [initial_label_1, initial_label_2]
    team_mdp = MDP3()
    team_mdp.contruct_from_individual_mdps([mdp_r1, mdp_r2], initial_node_list, initial_label_list)

    # TODO
    # 1 影响系统安全性的remove, 除去开始点外处于同一点的状态
    team_mdp.remove_unsafe_nodes()
    #
    # 2 根据地图推导出来的特殊状态
    remove_specific_states_4_team_mdp(team_mdp)
    #
    # 去掉点以后剩下的出边得unify
    team_mdp.normalize_transition_probabilities()

    # Added
    initial_node = tuple(initial_node_list)
    initial_label = tuple([initial_label_1, initial_label_2])

    return team_mdp, initial_node, initial_label, grid_nodes_1

