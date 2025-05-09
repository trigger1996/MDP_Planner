#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import networkx as nx

from collections import defaultdict
from itertools import product
from MDP_TG.mdp import Motion_MDP
from networkx import MultiDiGraph
from User.utils import print_c

class MDP3(Motion_MDP):
    # ----construct probabilistic-labeled MDP----
    def __init__(self, node_dict=None, edge_dict=None, U=None, initial_node=None, initial_label=None):
        if node_dict != None:
            Motion_MDP.__init__(self, node_dict, edge_dict, U, initial_node, initial_label)
        else:
            MultiDiGraph.__init__(self)

    def contruct_from_individual_mdps(self, mdp_list, initial_state_list, initial_label):
        assert len(mdp_list) == len(initial_state_list)
        self.clear()

        initial_state = tuple(initial_state_list)

        # 初始 label 是一个元组（每个个体的命题集合）
        initial_state = tuple(initial_state_list)
        initial_label_dicts = [mdp.nodes[s]['label'] for mdp, s in zip(mdp_list, initial_state)]
        merged_initial_label = self.merge_label_dicts(initial_label_dicts)

        # 初始 act 是所有动作组合的集合，如 {('a', 'a'), ('b', 'd'), ...}
        initial_action_sets = [set(mdp.nodes[s]['act']) for mdp, s in zip(mdp_list, initial_state)]
        initial_act_combinations = set(product(*initial_action_sets))

        self.add_node(initial_state, label=merged_initial_label, act=initial_act_combinations)
        self.graph['init_state'] = initial_state
        self.graph['init_label'] = list(merged_initial_label.keys())
        self.graph['U'] = set(product(*[mdp.graph['U'] for mdp in mdp_list]))

        stack = [initial_state]
        visited = set()

        while stack:
            current_state = stack.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            # 当前状态每个个体的出边
            edge_lists = [
                list(mdp.out_edges(current_state[i], data=True))
                for i, mdp in enumerate(mdp_list)
            ]

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

                if not valid:
                    continue

                joint_action = tuple(actions)  # ✅ 多个动作组成的联合动作
                joint_prob = 1.0
                for p in probs:
                    joint_prob *= p
                joint_cost = max(costs)

                next_state_tuple = tuple(next_state)
                if next_state_tuple not in self.nodes:
                    next_action_sets = [set(mdp.nodes[s]['act']) for mdp, s in zip(mdp_list, next_state)]
                    next_act_combinations = set(product(*next_action_sets))

                    label_dicts = [mdp.nodes[s]['label'] for mdp, s in zip(mdp_list, next_state)]
                    merged_label = self.merge_label_dicts(label_dicts)

                    self.add_node(next_state_tuple, label=merged_label, act=next_act_combinations)

                self.add_edge(
                    current_state,
                    next_state_tuple,
                    prop={joint_action: [joint_prob, joint_cost]}
                )

                if next_state_tuple not in visited:
                    stack.append(next_state_tuple)

    def remove_edge_sequence(mdp: MultiDiGraph, edge_sequence: list):
        """
        从 MDP 中删除指定的一组边。

        参数：
            mdp: 一个 networkx.MultiDiGraph 类型的 MDP 图。
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

    def merge_label_dicts(self, label_dicts):
        combined_probs = defaultdict(list)
        for label_dict in label_dicts:
            for prop, prob in label_dict.items():
                combined_probs[prop].append(prob)
        merged = {}
        for prop, prob_list in combined_probs.items():
            merged[prop] = 1 - math.prod(1 - p for p in prob_list)
        return merged

    def remove_state_sequence(self, state_sequence: list):
        for state in state_sequence:
            if self.has_node(state):
                self.remove_node(state)
                print_c(f"[MDP3] Removed state:   {state}", color='yellow')
            else:
                print_c(f"[MDP3] State not found: {state}", color='yellow')

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
            print_c(f"[MDP3] Restored state:   {self.graph['init_state']}", color='cyan')
        self.remove_state_sequence(nodes_to_remove)

    def normalize_transition_probabilities(mdp: MultiDiGraph):
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

