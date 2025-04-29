#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx

from itertools import product
from MDP_TG.mdp import Motion_MDP
from networkx import DiGraph
from User.utils import print_c

class MDP3(Motion_MDP):
    # ----construct probabilistic-labeled MDP----
    def __init__(self, node_dict=None, edge_dict=None, U=None, initial_node=None, initial_label=None):
        if node_dict != None:
            Motion_MDP.__init__(self, node_dict, edge_dict, U, initial_node, initial_label)
        else:
            DiGraph.__init__(self)

    def contruct_from_individual_mdps(self, mdp_list, initial_state_list, initial_label):
        assert len(mdp_list) == len(initial_state_list)
        self.clear()

        # 初始化状态（乘积状态元组）
        initial_state = tuple(initial_state_list)
        self.add_node(initial_state, label=None, act=set())
        self.graph['init_state'] = initial_state            # 设置初始状态（乘积状态）
        self.graph['init_label'] = initial_label            # 设置初始标签（外部传入）

        # 初始化队列与访问记录
        stack = [initial_state]
        visited = set()

        while stack:
            current_state = stack.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            # 获取每个 MDP 中当前状态的出边
            edge_lists = [
                list(mdp.out_edges(current_state[i], data=True))
                for i, mdp in enumerate(mdp_list)
            ]

            # 枚举每组出边的组合（乘积）
            for edge_comb in product(*edge_lists):
                valid = True
                next_state = []
                actions = []
                probs = []
                costs = []

                for i, (src, tgt, attr) in enumerate(edge_comb):
                    try:
                        action_i = next(iter(attr['prop'].keys()))
                        prob_i, cost_i = attr['prop'][action_i]
                    except (StopIteration, KeyError):
                        valid = False
                        break

                    next_state.append(tgt)
                    actions.append(action_i)
                    probs.append(prob_i)
                    costs.append(cost_i)

                # 要求所有动作完全一致（可以扩展为可观性约束）
                if not valid or not all(a == actions[0] for a in actions):
                    continue

                joint_action = actions[0]
                joint_prob = 1.0
                for p in probs:
                    joint_prob *= p
                max_cost = max(costs)

                next_state_tuple = tuple(next_state)
                if next_state_tuple not in self.nodes:
                    self.add_node(next_state_tuple)

                self.add_edge(
                    current_state,
                    next_state_tuple,
                    prop={joint_action: [joint_prob, max_cost]}
                )

                # 加入遍历队列
                if next_state_tuple not in visited:
                    stack.append(next_state_tuple)

    def remove_edge_sequence(mdp: DiGraph, edge_sequence: list):
        """
        从 MDP 中删除指定的一组边。

        参数：
            mdp: 一个 networkx.DiGraph 类型的 MDP 图。
            edge_sequence: 要删除的边列表，每个元素是一个三元组 (src, tgt, action)。
                           其中 action 是边上 'prop' 属性中的键。
        """
        for src, tgt, action in edge_sequence:
            if mdp.has_edge(src, tgt):
                prop_dict = mdp[src][tgt].get('prop', {})
                if action in prop_dict:
                    del prop_dict[action]
                    # 如果该边的所有动作都删光了，可以直接删边
                    if not prop_dict:
                        mdp.remove_edge(src, tgt)
                        print_c(f"[MDP3] Removed full edge: ({src}) -> ({tgt})")
                    else:
                        print_c(f"[MDP3] Removed action '{action}' from edge ({src}) -> ({tgt})")
                else:
                    print_c(f"[MDP3] Action '{action}' not found in edge ({src}) -> ({tgt})")
            else:
                print_c(f"[MDP3] Edge ({src}) -> ({tgt}) does not exist")

    def remove_state_sequence(self, state_sequence: list):
        for state in state_sequence:
            if self.has_node(state):
                self.remove_node(state)
                print_c(f"[MDP3] Removed state:   {state}")
            else:
                print_c(f"[MDP3] State not found: {state}")

    def remove_unsafe_nodes(self):
        nodes_to_remove = []
        #
        # 1 nodes with unsafe states
        for team_node_t in self.nodes():
            is_all_state_equal = True
            for individual_node_t in team_node_t:
                if individual_node_t != team_node_t[0]:
                    is_all_state_equal = False

            if is_all_state_equal:
                nodes_to_remove.append(team_node_t)

        if self.graph['init_state'] in nodes_to_remove:
            nodes_to_remove.remove(self.graph['init_state'])
        self.remove_state_sequence(nodes_to_remove)

    def normalize_transition_probabilities(mdp: DiGraph):
        """
        对于 MDP 中每个状态，归一化其所有出边中相同行为的转移概率之和为 1。
        修改 inplace。
        """
        for state in mdp.nodes:
            # 收集该状态的所有出边（按动作分组）
            action_edge_map = {}

            for _, tgt, attr in mdp.out_edges(state, data=True):
                for action, (prob, cost) in attr['prop'].items():
                    if action not in action_edge_map:
                        action_edge_map[action] = []
                    action_edge_map[action].append((tgt, prob, cost))

            # 对每个动作进行归一化
            for action, edge_list in action_edge_map.items():
                total_prob = sum(prob for _, prob, _ in edge_list)
                if total_prob == 0:
                    continue  # 避免除以 0

                for tgt, orig_prob, cost in edge_list:
                    normalized_prob = orig_prob / total_prob
                    mdp[state][tgt]['prop'][action][0] = normalized_prob  # 直接替换概率

