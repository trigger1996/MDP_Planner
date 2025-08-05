#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MDP_Planner'))

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
from Map.example_20250506_team_mdp import start_positions, observation_dict
from Map.example_20250506_team_mdp import build_observation_dict_all_states, build_mdp_with_grid, observation_func, visualize_grids_in_networkx

control_observable_dict = None

def observation_func_0506(state_id, y_len=5):
    #return observation_func(state_id, y_len)
    #row = int(state_id) % y_len // 2               # TODO Modifiied, different observation function
    row = int(state_id) % y_len
    return str(row)

def observation_inv_function_0506(obs_val, observation_dict_t=observation_dict, x_len=5, y_len=5):
    """
    根据个体的观测值（列号）返回该观测值可能对应的所有状态ID列表。

    参数:
        obs_val: str, 观测值（如 '0', '1', '2' 等），表示列号
        observation_dict: 可选，如果已存在观测字典就用已有的；否则会重新构建。

    返回:
        List[str] 或 Set[str]，可能的状态ID（如 ['0', '5', '10', '15', '20']）
    """
    #global x_len, y_len
    #global inaccessible_grids_in_map

    # 如果没有传入观测字典，就构建一个
    if observation_dict_t is None or len(observation_dict_t) == 0:
        observation_dict_t = build_observation_dict_all_states(x_len, y_len)

    # 返回对应观测值的所有状态
    return observation_dict_t.get(str(obs_val), [])


def construct_single_agent_mdp(is_visualize=False):
    """构建完整团队MDP"""
    # Added
    global start_positions, observation_dict

    observation_dict = build_observation_dict_all_states(x_len=5, y_len=5)

    robot_nodes_w_aps_1, robot_edges_1, U_1, grid_nodes_1, start_ids_1, initial_label_1 = build_mdp_with_grid(5, 5, start_position=start_positions[1])

    #
    # 这里有一些斜角边因为共用动作冲突会只被保留一条
    mdp_r1 = Motion_MDP(robot_nodes_w_aps_1, robot_edges_1, U_1, start_ids_1, initial_label_1)

    if is_visualize:
        visualize_grids_in_networkx(robot_nodes_w_aps_1, robot_edges_1, grid_nodes_1, start_ids_1)

    return mdp_r1, start_ids_1, initial_label_1, grid_nodes_1