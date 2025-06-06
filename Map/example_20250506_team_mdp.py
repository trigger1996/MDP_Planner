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

o->y
|
V
x

(0, 0)                                                        (0, 4)
upload    recharge          recharge          空         inaccessible
空        空                 空                空         inaccessible
空        空                 gather            gather    空
upload    investigate       空                空         upload
upload    inaccessible      inaccessible      空         upload
(4, 0)                                                        (4, 4)

入侵者只知道列信息，动作和观测无关
三个出发点，左上(0, 0)，左下（4, 0），右下(4, 4)

'''

# ----------------------------
# 保留原始观测和控制逻辑

x_len = 5
y_len = 5
robot_nodes_w_aps    = {}
special_grids_in_map = {
    (0, 0): {frozenset({'upload'}): 1.0},
    (0, 1): {frozenset({'recharge'}): 1.0},
    (0, 2): {frozenset({'recharge'}): 1.0},
    (2, 2): {frozenset({'gather'}): 1.0},
    (2, 3): {frozenset({'gather'}): 1.0},
    (3, 0): {frozenset({'upload'}): 1.0},
    (3, 1): {frozenset({'investigate'}): 1.0},
    (3, 4): {frozenset({'upload'}): 1.0},
    (4, 0): {frozenset({'upload'}): 1.0},
    (4, 4): {frozenset({'upload'}): 1.0},
}

# 不可达节点（将被删除）
inaccessible_grids_in_map = [
    (0, 4), (1, 4), (4, 2), (4, 3),
]

# 起点列表
start_positions = [(0, 0), (3, 0), (3, 4)]
U0_dict = {
    (0, 0) : 'r',
    (3, 0) : 'r',
    (3, 4) : 'l'
}

observation_dict = {}

def build_observation_dict_all_states(x_len, y_len):
    """
    构建观测字典：将每个观测值（列号）映射到该列上的所有状态 ID。
    """
    obs_dict = defaultdict(list)
    for row in range(x_len):
        for col in range(y_len):
            state_id = str(row * y_len + col)
            obs_val = observation_func(state_id, y_len)

            if (row, col) in inaccessible_grids_in_map:
                continue

            obs_dict[obs_val].append(state_id)
    return dict(obs_dict)

def obs_to_hashable(obs):
    return frozenset(Counter(obs).items())


def observation_func(state_id, y_len):
    """入侵者的观测函数：只观测列信息"""
    row = int(state_id) % y_len // 2               # TODO
    return str(row)

def team_observation_func(x, u=None):
    y = []
    for x_t in x:
        y_t = observation_func(x_t, y_len=6)
        y.append(y_t)
    y = Counter(y)              # MUST BE set or Counter here, for the observer cannot distinguish the sequence of the control
    #y = obs_to_hashable(y)
    #y = set(y)
    return y

def team_observation_inv_func(y):
    """
    根据输入的 Counter 返回对应的状态组合。
    规则：
      1. 对每个 key 的计数（如 {'p': 2}），重复其对应的状态元组多次（如 ('0', '5') 重复 2 次）。
      2. 输出为一个元组，其中每个元素是状态的元组。

    其他可能的输出形式（需修改代码）：
      - 返回 set: 使用 set.update() 合并所有状态，忽略计数。
      - 返回 Counter: 统计每个状态的累计出现次数。
    """
    result = []
    for key, count in y.items():
        states_tuple = tuple(observation_dict[key])  # 将列表转为元组，如 ('0', '5')
        result.extend([states_tuple] * count)       # 重复元组多次

    # Return the corresponding states as a tuple
    return tuple(result)  # 转为元组输出

    # Alternatively, to return as a set:
    # return set(observation_dict[observation_key])

    # Or as a Counter (counting each state once):
    # return Counter(observation_dict[observation_key])


def generate_grid_graph(x, y, d=1, diagonal=False):
    """生成x*y的栅格图（可选对角线移动）"""
    G = nx.grid_2d_graph(x, y, create_using=nx.MultiDiGraph)

    if diagonal:  # 添加对角线边
        for u in list(G.nodes()):
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                v = (u[0] + dx, u[1] + dy)
                if v in G.nodes():
                    G.add_edge(u, v)

    return G

def normalize_robot_edges(robot_edges):
    """
    对 robot_edges 中的每个 (state, action) 组，归一化其出边概率，使概率和为 1。
    输入:
        robot_edges: dict, 形如 {(u, act, v): (prob, cost), ...}
    输出:
        修改后的 robot_edges，所有同一 (u, act) 对应的 prob 总和为 1
    """
    # 收集每个 (u, act) 的所有目标节点和概率
    edge_group = defaultdict(list)
    for (u, act, v), (prob, cost) in robot_edges.items():
        edge_group[(u, act)].append((v, prob, cost))

    # 归一化每组 (u, act) 的 prob
    normalized_edges = {}
    for (u, act), transitions in edge_group.items():
        total_prob = sum(prob for _, prob, _ in transitions)
        if total_prob == 0:
            continue  # 避免除以 0
        for v, prob, cost in transitions:
            normalized_prob = prob / total_prob
            normalized_edges[(u, act, v)] = (normalized_prob, cost)

    return normalized_edges

def build_mdp_with_grid(x, y, start_position=None, d=1):
    global special_grids_in_map, inaccessible_grids_in_map, U0_dict

    G = generate_grid_graph(x, y, d, True)
    G.remove_nodes_from(inaccessible_grids_in_map)

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


    action_defs = {
        'u': [((-1, 0), 1.0)],
        'd': [((1, 0),  1.0)],
        'l': [((0, -1), 1.0)],
        'r': [((0, 1),  1.0)],
    }

    for (u_row, u_col) in G.nodes():
        u = str(u_row * y + u_col)

        for act in action_defs.keys():
            # 默认扰动定义
            transitions = action_defs[act]

            # 添加额外的“坑扰动”边（区域+动作触发）
            extra_transition = None
            extra_prob = 0.1  # 额外边的概率

            mid_x = x // 2
            mid_y = y // 2

            if u_row < mid_x and u_col < mid_y and act == 'r':
                extra_transition = (1, 1)  # 右下, element 1: 上下, element 2: 左右
            elif u_row < mid_x and u_col > mid_y and act == 'l':
                extra_transition = (1, -1)  # 左下
            elif u_row > mid_x and u_col < mid_y and act == 'r':
                extra_transition = (-1, 1)  # 右上
            elif u_row > mid_x and u_col > mid_y and act == 'l':
                extra_transition = (-1, -1)  # 左上

            # 添加基础边
            for (dr, dc), prob in transitions:                  # transitions <- action_defs[act]
                v_row, v_col = u_row + dr, u_col + dc
                v = str(v_row * y + v_col)
                if (v_row, v_col) in G:
                    dist = math.hypot(dc, dr) * d
                    robot_edges[(u, act, v)] = (prob, dist)

            # 添加额外扰动边
            if extra_transition is not None:
                dr, dc = extra_transition
                v_row, v_col = u_row + dr, u_col + dc
                v = str(v_row * y + v_col)
                if (v_row, v_col) in G:
                    dist = math.hypot(dc, dr) * d
                    # 注意：如果该边已经存在，会被覆盖，建议后续 normalize 再统一归一
                    robot_edges[(u, act, v)] = (extra_prob, dist)

    # 起点状态
    if start_position is not None:
        start_ids = f"{start_position[0] * y + start_position[1]}"
    else:
        start_ids = None
    initial_label = tuple([k for k in robot_nodes_w_aps[start_ids].keys()])

    if start_position in U0_dict.keys():
        U0 = U0_dict[start_position]
    else:
        U0 = ['r']

    # To normalize
    robot_edges = normalize_robot_edges(robot_edges)

    return robot_nodes_w_aps, robot_edges, U0, grid_nodes, start_ids, initial_label

def run_2_observations_seqs(x_u_seqs):
    y_seq = []
    for i in range(0, x_u_seqs.__len__() - 1, 2):
        x_t = x_u_seqs[i]
        u_t = x_u_seqs[i + 1]
        y_t = team_observation_func(x_t, u_t)
        y_seq.append(y_t)
        #y_seq.append(u_t)           # u is for display and NOT in actual sequences
    return y_seq

def observation_seq_2_inference(y_seq, robot_nodes_w_ap_t = None):
    global robot_nodes_w_aps
    global observation_dict, x_len, y_len
    if robot_nodes_w_ap_t != None and not robot_nodes_w_aps.__len__():
        robot_nodes_w_aps = robot_nodes_w_ap_t
    if robot_nodes_w_ap_t == None:
        robot_nodes_w_ap_t, *_ = build_mdp_with_grid(x_len, y_len, start_position=(0, 0))

    if not observation_dict.__len__():
        observation_dict = build_observation_dict_all_states(x_len, y_len)          # Added for single agent system

    x_inv_set_seq = []
    ap_inv_seq = []
    for i in range(0, y_seq.__len__()):
        x_inv_t = team_observation_inv_func(y_seq[i])
        #
        ap_inv_t = []
        for team_state_t in x_inv_t:
            ap_i = []
            for state_t in team_state_t:
                # ap_inv_t = ap_inv_t + list(robot_nodes_w_aps[state_t].keys())
                try:
                    ap_list_t = list(robot_nodes_w_aps[state_t].keys())
                except KeyError as e:
                    print_c(str(e), color='yellow', style='bold')

                for ap_t in ap_list_t:
                    ap_i = ap_i + list(ap_t)
            ap_i = tuple(set(ap_i))
            ap_inv_t.append(ap_i)

        ap_inv_t = list(set(ap_inv_t))
        x_inv_set_seq.append(x_inv_t)
        ap_inv_seq.append(ap_inv_t)
    return x_inv_set_seq, ap_inv_seq


def remove_specific_states_4_team_mdp(team_mdp:MDP3):
    state_list_to_remove = []

    # TODO
    team_mdp.remove_state_sequence(state_list_to_remove)

def visualize_grids_in_networkx(robot_nodes_w_aps, robot_edges, grid_nodes, start_node):
    """可视化 MDP 图，标出特殊节点和起点"""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.MultiDiGraph()

    # 添加节点和位置
    for node, attrs in grid_nodes.items():
        G.add_node(node, pos=attrs["pos"])

    # 添加带动作和权重的边
    for (u, a, v), (prob, cost) in robot_edges.items():
        G.add_edge(u, v, weight=cost, action=a, prob=prob)

    # 获取节点位置
    pos = nx.get_node_attributes(G, 'pos')

    # 绘图初始化
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightyellow', font_size=9, edge_color='gray')

    # 边标签：显示动作和距离代价
    edge_labels = {
        (u, v): f"{data['action']}\n{data['weight']:.1f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # 特殊节点：非空标签节点用绿色高亮
    special_nodes = [nid for nid in robot_nodes_w_aps if frozenset({''}) not in robot_nodes_w_aps[nid]]
    nx.draw_networkx_nodes(G, pos, nodelist=special_nodes, node_color='lightgreen', node_size=750)

    # 特殊节点标签：标出其命题集合
    special_labels = {
        nid: ','.join(list(list(label_dict.keys())[0]))  # 取 frozenset({'prop'}) 中的字符串
        for nid, label_dict in robot_nodes_w_aps.items()
        if frozenset({''}) not in label_dict
    }
    nx.draw_networkx_labels(G, pos, labels=special_labels, font_color='black', font_size=8, font_weight='bold')

    # 高亮起点节点（红色边框）
    if start_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='none', edgecolors='red', linewidths=2.5, node_size=750)

    plt.title("Grid MDP Visualization")
    plt.axis("off")
    plt.tight_layout()
    # plt.show()

def calculate_cost_from_runs(product_mdp, x, l, u, opt_prop, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []                      # [[value, current step, path_length], [value, current step, path_length], ...]
    cost_cycle = 0.
    #
    x_i_last = x[0]
    for i in range(0, x.__len__()):
        x_i = x[i]
        x_p = None
        l_i = list(l[i])
        if i < x.__len__() - 1:
            u_i = u[i]

        #
        if l_i.__len__() and opt_prop in l_i:
            if cost_list.__len__():
                path_length = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length = i
            #
            cost_current_step = [cost_cycle, i, path_length]
            cost_list.append(cost_current_step)
            cost_cycle = 0.
        #
        #
        for edge_t in list(product_mdp.graph['mdp'].edges(x_i_last, data=True)):
            if x_i == edge_t[1]:
                #
                event_t = list(edge_t[2]['prop'])[0]           # event_t = i_i???
                #
                cost_t = edge_t[2]['prop'][event_t][1]
                cost_cycle += cost_t

    if is_remove_zeros:
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass

    return cost_list

def plot_cost_hist(cost_list, bins=25, color='g', is_average=True, title="Cost Distribution", xlabel="Cost", ylabel="Probability"):
    plt.figure()
    cost_list.sort(key=lambda x: x[0])
    data_list = []
    for i in range(0, cost_list.__len__()):
        if is_average:
            data_list.append(cost_list[i][0] / cost_list[i][2])
        else:
            data_list.append(cost_list[i][0])

    sns.histplot(data_list, bins=bins, kde=True, color=color, stat="probability")    # stat="density" "probability"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()

def construct_team_mdp(is_visualize=False):
    """构建完整团队MDP"""
    # Added
    global start_positions, observation_dict
    global x_len, y_len

    observation_dict = build_observation_dict_all_states(x_len=5, y_len=5)

    robot_nodes_w_aps_1, robot_edges_1, U_1, grid_nodes_1, start_ids_1, initial_label_1 = build_mdp_with_grid(x_len, y_len, start_position=start_positions[1])
    robot_nodes_w_aps_2, robot_edges_2, U_2, grid_nodes_2, start_ids_2, initial_label_2 = build_mdp_with_grid(x_len, y_len, start_position=start_positions[2])

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
    # team_mdp.remove_unsafe_nodes()                        # Added
    #
    # 2 根据地图推导出来的特殊状态
    # remove_specific_states_4_team_mdp(team_mdp)           # Added
    #
    # 去掉点以后剩下的出边得unify
    team_mdp.normalize_transition_probabilities()

    # Added
    initial_node = tuple(initial_node_list)
    initial_label = tuple([initial_label_1, initial_label_2])

    if is_visualize:
        visualize_grids_in_networkx(robot_nodes_w_aps_1, robot_edges_1, grid_nodes_1, start_ids_1)

    return team_mdp, initial_node, initial_label, grid_nodes_1

