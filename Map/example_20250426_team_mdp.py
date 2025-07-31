#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MDP_Planner'))

import math
from collections import Counter
from MDP_TG.mdp import Motion_MDP
from User.mdp3 import MDP3
from User.utils import print_c

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



robot_nodes_w_aps = dict()
robot_edges = dict()
U = []
initial_node  = None
initial_label = None

'''
           /---a--> q_1          q_6 <-b--\   
          /                       |        \
         /                        |         \
        a                         a          b
        |                         |           \
        v                         v            \
    -> q_0 ---b---> q_2 <--b---> q_4 <--a---> q_5 <-  
        ^            |            ^
        |            |            |
        b            a            b
        \            |            /
         \           v           /
          \---b---> q_3 <---c---/

    q_0 -a-> q_0
    q_0 -b-> q_0
    q_6 -b-> q_6

    q_0 : upload        p
    q_1 : gather        q
    q_2 : gather        u
    q_3 : recharge      u
    q_4 : \emptyset     u
    q_5 : upload        p
    q_6 : recharge      u

'''

#
# in simulations, we can let those states with identical APs carry identical observations, which is to simulate the APs are observed satisfied
observation_dict = {
    'p': ['0', '5'],
    'q': ['1'],
    'u': ['2', '3', '4', '6'],
}

# TODO, 注意这里输入的ctrl_obs_dict是一个tuple, 所以需要单独生成一个team_ctrl_obs_dict
# control_observable_dict = {
#     ('a', 'a') : False,
#     ('a', 'b') : False,
#     ('b', 'a') : False,
#     ('b', 'b') : False
#     ...
#
# }
control_observable_dict = None

def build_individual_mdp(initial_node_t=None):

    global robot_nodes_w_aps, robot_edges, U, initial_node, initial_label

    # robot nodes
    # the lower satisfaction probability may result in conflict/failure in opacity constraint in user/lp.py
    robot_nodes_w_aps['0'] = { frozenset({'upload'})   : 1.0 }
    #
    robot_nodes_w_aps['1'] = { frozenset({'gather'})   : 1.  }
    robot_nodes_w_aps['2'] = { frozenset({'gather'})   : 1.  }
    robot_nodes_w_aps['3'] = { frozenset({'recharge'}) : 1.  }
    #
    robot_nodes_w_aps['4'] = { frozenset({''})         : 1.0 }
    robot_nodes_w_aps['5'] = { frozenset({'upload'})   : 0.9 }
    robot_nodes_w_aps['6'] = { frozenset({'recharge'}) : 1.  }
    #
    #
    robot_edges = {
        # x,   a,   x'  : prob, cost
        ('0', 'a', '1') : (1, 1),            # gather
        ('1', 'a', '0') : (1, 1),            #
        ('0', 'b', '0') : (0.05, 2),
        #
        ('0', 'b', '2') : (0.5, 3),
        ('2', 'a', '3') : (1, 2),
        #
        # ('2', 'b', '4') : (1, 1),
        #
        ('0', 'b', '3') : (0.45, 3),
        ('3', 'b', '0') : (0.8,  2),
        ('3', 'b', '4') : (0.2,  3),
        #
        #
        ('5', 'a', '4') : (1,   1),
        ('5', 'b', '6') : (1,   2),
        #
        ('6', 'b', '6') : (1,   0.5),
        ('6', 'a', '4') : (1,   2),
        #
        ('4', 'b', '2') : (1,   3),
        ('4', 'a', '3') : (0.5,  2),
        ('4', 'a', '5') : (0.5,  2),
    }

    #
    U = [ 'b' ]

    #
    if initial_node_t == None:
        initial_node  = '0'
    else:
        initial_node = initial_node_t
    initial_label = list(robot_nodes_w_aps[initial_node].keys())[0]


    return (robot_nodes_w_aps, robot_edges, U, initial_node, initial_label)

def construct_team_mdp():
    # Added
    global initial_node, initial_label

    initial_node_list = [ '0', '5' ]

    (robot_nodes_w_aps_1, robot_edges_1, U_1, initial_node_1, initial_label_1) = build_individual_mdp(initial_node_t=initial_node_list[0])
    (robot_nodes_w_aps_2, robot_edges_2, U_2, initial_node_2, initial_label_2) = build_individual_mdp(initial_node_t=initial_node_list[1])
    mdp_r1 = Motion_MDP(robot_nodes_w_aps_1, robot_edges_1, U_1, initial_node_1, initial_label_1)
    mdp_r2 = Motion_MDP(robot_nodes_w_aps_2, robot_edges_2, U_2, initial_node_2, initial_label_2)

    team_mdp = MDP3()
    team_mdp.contruct_from_individual_mdps([mdp_r1, mdp_r2], initial_node_list, [initial_label_1, initial_label_2])

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
    initial_node  = tuple(initial_node_list)
    initial_label = tuple([initial_label_1, initial_label_2])

    return team_mdp, initial_node, initial_label

def remove_specific_states_4_team_mdp(team_mdp:MDP3):
    state_list_to_remove = []

    # # TODO
    # for team_node_t in team_mdp.nodes():
    #     if team_node_t[0] == '4':
    #         state_list_to_remove.append(team_node_t)
    #     if team_node_t[0] == '5':
    #         state_list_to_remove.append(team_node_t)
    #     if team_node_t[0] == '6':
    #         state_list_to_remove.append(team_node_t)
    #
    #     if team_node_t[1] == '0':
    #         state_list_to_remove.append(team_node_t)
    #     if team_node_t[1] == '1':
    #         state_list_to_remove.append(team_node_t)

    team_mdp.remove_state_sequence(state_list_to_remove)

def obs_to_hashable(obs):
    return frozenset(Counter(obs).items())

def observation_func_0426(x, u=None):
    global observation_dict

    for y in observation_dict.keys():
        if x in observation_dict[y]:
            return y

    print("[observation_func_0426] Please check input x !")
    raise TypeError

    return None

def team_observation_func_0426(x, u=None):
    y = []
    for x_t in x:
        y_t = observation_func_0426(x_t)
        y.append(y_t)
    y = Counter(y)              # MUST BE set or Counter here, for the observer cannot distinguish the sequence of the control
    #y = obs_to_hashable(y)
    #y = set(y)
    return y

def observation_inv_func_0426(y):
    return observation_dict[y]

def team_observation_inv_func_0426(y):
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

def run_2_observations_seqs(x_u_seqs):
    y_seq = []
    for i in range(0, x_u_seqs.__len__() - 1, 2):
        x_t = x_u_seqs[i]
        u_t = x_u_seqs[i + 1]
        y_t = team_observation_func_0426(x_t, u_t)
        y_seq.append(y_t)
        #y_seq.append(u_t)           # u is for display and NOT in actual sequences
    return y_seq

def observation_seq_2_inference(y_seq):
    global robot_nodes_w_aps
    x_inv_set_seq = []
    ap_inv_seq = []
    for i in range(0, y_seq.__len__()):
        x_inv_t = team_observation_inv_func_0426(y_seq[i])
        #
        ap_inv_t = []
        for team_state_t in x_inv_t:
            ap_i = []
            for state_t in team_state_t:
                # ap_inv_t = ap_inv_t + list(robot_nodes_w_aps[state_t].keys())
                ap_list_t = list(robot_nodes_w_aps[state_t].keys())
                for ap_t in ap_list_t:
                    ap_i = ap_i + list(ap_t)
            ap_i = tuple(set(ap_i))
            ap_inv_t.append(ap_i)

        ap_inv_t = list(set(ap_inv_t))
        x_inv_set_seq.append(x_inv_t)
        ap_inv_seq.append(ap_inv_t)
    return x_inv_set_seq, ap_inv_seq

def calculate_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []                      # [[value, current step, path_length], [value, current step, path_length], ...]
    cost_cycle = 0.
    #
    x_i_last = x[0]
    for i in range(1, x.__len__()):
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

def calculate_sync_observed_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, ap_gamma, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []              # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    cost_list_gamma = []        # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    diff_exp_list = []          # [[value, current step pi, current step gamma, path_length], [value, current step, current step gamma, path_length], ...]
    #
    cost_cycle_pi = 0.
    cost_cycle_gamma = 0.
    #
    x_i_last = x[0]
    for i in range(1, x.__len__()):
        x_i_last = x[i - 1]
        o_i_last = o[i - 1]
        x_i = x[i]
        o_i = o[i]
        ol_i = list(ol_set[i])  # l_i = list(l[i])
        if i < x.__len__() - 1:
            # TODO
            u_i = u[i]

        #
        is_observed_ap_pi_gamma_found = False
        for o_t in ol_i:
            if ol_i.__len__() and opt_prop in o_t and ap_gamma in o_t:
                is_observed_ap_pi_gamma_found = True
                #break                                      # break不能加, 加了就出去了影响前后判断


        if is_observed_ap_pi_gamma_found:
            #
            # Add pi
            if cost_list.__len__():
                path_length_pi = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length_pi = i
            #
            cost_current_step = [cost_cycle_pi, i, path_length_pi]
            cost_list.append(cost_current_step)
            cost_cycle_pi = 0.
            #
            # Add gamma
            if cost_list_gamma.__len__():
                path_length_gamma = i - cost_list_gamma[cost_list_gamma.__len__() - 1][1]
            else:
                path_length_gamma = i
            #
            cost_current_step = [cost_cycle_gamma, i, path_length_gamma]
            cost_list_gamma.append(cost_current_step)
            cost_cycle_gamma = 0.
        #
        #
        available_event_cost_dict_pi = {}
        available_event_cost_dict_gamma = {}
        for o_last_t in o_i_last:
            for o_t in o_i:
                for edge_t in list(product_mdp.graph['mdp'].edges(o_last_t, data=True)):
                    if o_t == edge_t[1]:
                        label_o_t = list(product_mdp.graph['mdp'].nodes[o_t]['label'].keys())
                        label_o_t = [list(fs)[0] for fs in label_o_t if isinstance(fs, frozenset)]
                        #
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]
                        #
                        if opt_prop in label_o_t:
                            available_event_cost_dict_pi[event_t] = cost_t
                        if ap_gamma in label_o_t:
                            available_event_cost_dict_gamma[event_t] = cost_t
        if available_event_cost_dict_pi.__len__():
            min_event = min(available_event_cost_dict_pi, key=lambda e: available_event_cost_dict_pi[e])
            min_cost = available_event_cost_dict_pi[min_event]

            cost_cycle_pi += min_cost
        if available_event_cost_dict_gamma.__len__():
            min_event = min(available_event_cost_dict_gamma, key=lambda e: available_event_cost_dict_gamma[e])
            min_cost = available_event_cost_dict_gamma[min_event]

            cost_cycle_gamma += min_cost

        else:
            #
            # TODO
            # 用x代替?
            print_c("warning ...")

    used_gamma_indices = set()

    last_pi_step = 0
    last_gamma_step = 0

    for i in range(len(cost_list)):
        step_i = cost_list[i][1]

        # 在 gamma 列表中找与 step_i 最接近的 step_j，避免重复匹配
        min_j_index = -1
        min_step_diff = float('inf')
        for j in range(len(cost_list_gamma)):
            if j in used_gamma_indices:
                continue  # 防止重复匹配

            step_j = cost_list_gamma[j][1]
            diff = abs(step_i - step_j)
            if diff < min_step_diff:
                min_step_diff = diff
                min_j_index = j

        # 如果找不到匹配的 gamma（极端情况），跳过
        if min_j_index == -1:
            continue

        used_gamma_indices.add(min_j_index)

        cost_pi = cost_list[i]
        cost_gamma = cost_list_gamma[min_j_index]

        diff_cost = abs(cost_pi[0] - cost_gamma[0])
        path_length = max(cost_pi[1] - last_pi_step, cost_gamma[1] - last_gamma_step)

        diff_exp_list.append([
            diff_cost,
            cost_pi[1],
            cost_gamma[1],
            path_length
        ])

        last_pi_step = cost_pi[1]
        last_gamma_step = cost_gamma[1]

    if is_remove_zeros:
        #
        # pi
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass
        #
        # gamma
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list_gamma:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list_gamma.remove(cost_tuple_t)
            except:
                pass


    return cost_list, cost_list_gamma, diff_exp_list

def calculate_observed_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, ap_gamma, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []              # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    cost_list_gamma = []        # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    diff_exp_list = []          # [[value, current step pi, current step gamma, path_length], [value, current step, current step gamma, path_length], ...]
    #
    cost_cycle_pi = 0.
    cost_cycle_gamma = 0.
    for i in range(1, o.__len__()):
        x_i_last = x[i - 1]
        o_i_last = o[i - 1]
        x_i = x[i]
        o_i = o[i]
        ol_i = list(ol_set[i])  # l_i = list(l[i])
        if i < x.__len__() - 1:
            # TODO
            u_i = u[i]

        #
        is_observed_ap_pi_found = False
        is_observed_ap_gamma_found = False
        for o_t in ol_i:
            if ol_i.__len__() and opt_prop in o_t:
                is_observed_ap_pi_found = True
                #break                                      # break不能加, 加了就出去了影响前后判断
            #
            if ol_i.__len__() and ap_gamma in o_t:
                is_observed_ap_gamma_found = True
                #break

        if is_observed_ap_pi_found:
            if cost_list.__len__():
                path_length_pi = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length_pi = i
            #
            cost_current_step = [cost_cycle_pi, i, path_length_pi]
            cost_list.append(cost_current_step)
            cost_cycle_pi = 0.
        if is_observed_ap_gamma_found:
            if cost_list_gamma.__len__():
                path_length_gamma = i - cost_list_gamma[cost_list_gamma.__len__() - 1][1]
            else:
                path_length_gamma = i
            #
            cost_current_step = [cost_cycle_gamma, i, path_length_gamma]
            cost_list_gamma.append(cost_current_step)
            cost_cycle_gamma = 0.
        #
        #
        available_event_cost_dict_pi = {}
        available_event_cost_dict_gamma = {}
        for o_last_t in o_i_last:
            for o_t in o_i:
                for edge_t in list(product_mdp.graph['mdp'].edges(o_last_t, data=True)):
                    if o_t == edge_t[1]:
                        label_o_t = list(product_mdp.graph['mdp'].nodes[o_t]['label'].keys())
                        label_o_t = [list(fs)[0] for fs in label_o_t if isinstance(fs, frozenset)]
                        #
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]
                        #
                        if opt_prop in label_o_t:
                            available_event_cost_dict_pi[event_t] = cost_t
                        if ap_gamma in label_o_t:
                            available_event_cost_dict_gamma[event_t] = cost_t
        if available_event_cost_dict_pi.__len__():
            min_event = min(available_event_cost_dict_pi, key=lambda e: available_event_cost_dict_pi[e])
            min_cost = available_event_cost_dict_pi[min_event]

            cost_cycle_pi    += min_cost
        if available_event_cost_dict_gamma.__len__():
            min_event = min(available_event_cost_dict_gamma, key=lambda e: available_event_cost_dict_gamma[e])
            min_cost = available_event_cost_dict_gamma[min_event]

            cost_cycle_gamma += min_cost

        else:
            #
            # TODO
            # 用x代替?
            print_c("warning ...")


    #
    # 先全部算
    # 这个不同步的算法不准确, 但是能用
    used_gamma_indices = set()

    last_pi_step = 0
    last_gamma_step = 0

    for i in range(len(cost_list)):
        step_i = cost_list[i][1]

        # 在 gamma 列表中找与 step_i 最接近的 step_j，避免重复匹配
        min_j_index = -1
        min_step_diff = float('inf')
        for j in range(len(cost_list_gamma)):
            if j in used_gamma_indices:
                continue  # 防止重复匹配

            step_j = cost_list_gamma[j][1]
            diff = abs(step_i - step_j)
            if diff < min_step_diff:
                min_step_diff = diff
                min_j_index = j

        # 如果找不到匹配的 gamma（极端情况），跳过
        if min_j_index == -1:
            continue

        used_gamma_indices.add(min_j_index)

        cost_pi = cost_list[i]
        cost_gamma = cost_list_gamma[min_j_index]

        diff_cost = abs(cost_pi[0] - cost_gamma[0])
        path_length = max(cost_pi[1] - last_pi_step, cost_gamma[1] - last_gamma_step)

        diff_exp_list.append([
            diff_cost,
            cost_pi[1],
            cost_gamma[1],
            path_length
        ])

        last_pi_step = cost_pi[1]
        last_gamma_step = cost_gamma[1]

    if is_remove_zeros:
        #
        # pi
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass
        #
        # gamma
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list_gamma:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list_gamma.remove(cost_tuple_t)
            except:
                pass


    return cost_list, cost_list_gamma, diff_exp_list

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