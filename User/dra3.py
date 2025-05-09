#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import fabs, sqrt

import networkx as nx
from networkx import MultiDiGraph, is_connected
from MDP_TG.mdp import find_MECs, find_SCCs
from MDP_TG.dra import Product_Dra
from User.vis2 import print_c

import random

def is_state_satisfy_ap(state, ap):
    if list(state[1]).__len__() == 0:
        return False
    else:
        if ap in set(state[1]):
            return True
        else:
            return False

def is_ap_satisfy_opacity(state_pi, state_gamma, ap_pi, ap_gamma):
    if list(state_pi[1]).__len__() == 0 and list(state_gamma[1]).__len__() == 0:
        return True
    elif ap_pi in set(state_pi[1]):
        if ap_gamma in set(state_gamma[1]):
            return True
        else:
            return False
    return True                 # 如果原状态的AP不是Pi

def obtain_differential_expected_cost(current_action, edge_pi, edge_gamma):
    #
    pr_pi   = edge_pi[2]['prop'][current_action][0]
    cost_pi = edge_pi[2]['prop'][current_action][1]
    expected_cost_pi = pr_pi * cost_pi
    #
    try:
        pr_gamma   = edge_gamma[2]['prop'][current_action][0]
        cost_gamma = edge_gamma[2]['prop'][current_action][1]
        expected_cost_gamma = pr_gamma * cost_gamma
        #
        different_expected_cost = fabs(expected_cost_pi - expected_cost_gamma)
    except KeyError:
        #
        # TODO
        # 看来两个必须action相同
        # 不然根据MDP的定义, transition cost是X \times U \times -> R, U不同整个transition也不同, 最后秋出来的cost也不同
        # print_c("[synthesize_w_opacity] WARNING no corresponding actions  %s // %s" % (str(edge_pi), str(edge_gamma),), color=36)
        different_expected_cost = fabs(expected_cost_pi)

    return different_expected_cost

def project_observer_state_2_sync_state(sync_mec_3_1, observer_state_set, is_required_in_sync_mec_3_1=True):
    sync_state_set = []
    for observer_state_t in observer_state_set:
        state_pi_t = observer_state_t[0]
        for state_gamma_t in set(observer_state_t[1]).union(observer_state_t[2]):
            sync_state_set.append((state_pi_t, state_gamma_t))

    sync_state_set.sort()
    sync_state_set = list(set(sync_state_set))

    if is_required_in_sync_mec_3_1:
        state_to_remove = []
        for sync_state_t in sync_state_set:
            if sync_state_t not in sync_mec_3_1:
                state_to_remove.append(sync_state_t)

        for sync_state_t in state_to_remove:
            try:
                sync_state_set.remove(sync_state_t)
            except:
                pass

    return sync_state_set

def project_sync_states_2_observer_states(observer_graph:nx.MultiDiGraph, sync_state_set):
    observer_state_set = []
    for observer_state_t in observer_graph.nodes:
        for sync_state_t in sync_state_set:
            state_pi_t = sync_state_t[0]
            state_gamma_t = sync_state_t[1]
            if state_pi_t == observer_state_t[0]:
                if state_gamma_t in observer_state_t[1]:
                    observer_state_set.append(observer_state_t)
    observer_state_set.sort()
    observer_state_set = list(set(observer_state_set))
    return observer_state_set

def project_sync_mec_3_2_observer_mec_3(observer_graph:nx.MultiDiGraph, sync_amec_3):
    mec_states = sync_amec_3[0]
    ip_states  = list(sync_amec_3[1])
    act_list   = {}
    #
    observer_state_1 = project_sync_states_2_observer_states(observer_graph, mec_states)
    observer_ip = project_sync_states_2_observer_states(observer_graph, ip_states)
    act_list         = {}   # TODO
    #
    return [observer_state_1, observer_ip, act_list]

class product_mdp3(Product_Dra):
    def __init__(self, mdp=None, dra=None):
        if mdp == None and dra == None:
            print_c("[product mdp 2] Initializing subgraph for product mdp ...", color='yellow')
        else:
            Product_Dra.__init__(self, mdp, dra)
            self.compute_S_f()
            self.sync_amec_set = list()
            self.current_sync_amec_index = 0
            #
            self.best_all_plan = dict()
            #

    def compute_S_f(self):
        # ----find all accepting End components----
        S = set(self.nodes())
        acc_pairs = self.graph['accept']
        S_f = []
        k = 1
        for pair in acc_pairs:
            # ---for each accepting pair
            print("+++++++++++++++++++++++++++++++++++++")
            print("++++++++++++ acc_pair %s ++++++++++++" % k)
            print("+++++++++++++++++++++++++++++++++++++")
            S_fi = []
            Ip = pair[0]
            Hp = pair[1]
            print("Ip size: %s" % len(Ip))
            print("Hp size: %s" % len(Hp))
            # ---find all MECs
            MEC, Act = find_MECs(self, S.difference(Hp))
            # ---find accepting ones
            for T in MEC:
                common = set(T.intersection(Ip))
                if common:
                    if len(T) > 1:
                        S_fi.append([T, common, Act])
                        print('S_fii added to S_fi!!, size: %s' % len(T))
                    if len(T) == 1:  # self-loop
                        common_cp = common.copy()
                        s = common_cp.pop()
                        if self.has_edge(s, s):                             # Added
                            loop_act_set = set(self[s][s]['prop'].keys())
                            loop_act = dict()
                            loop_act[s] = loop_act_set
                            S_fi.append([T, common, loop_act])
                            print('S_fii added to S_fi!!, size: %s' % len(T))
            if len(S_fi) > 0:
                S_f.append(S_fi)
                print("****S_fi added to S_f!!!, size: %s******" % len(S_fi))
            k += 1
        self.Sf = S_f
        if S_f:
            print("-------Accepting MEC for Prod DRA Computed-------")
            print("acc_pair number: %s" % str(k-1))
            print("Sf AMEC number: %s" % len(S_f))
        else:
            print("No accepting ECs found!")
            print("Check your MDP and Task formulation")
            print("Or try the relaxed plan")

    def composition(self, mdp_node, mdp_label, dra_node):
        prod_node = (mdp_node, mdp_label, dra_node)
        if not self.has_node(prod_node):
            Us = self.graph['mdp'].nodes[mdp_node]['act'].copy()
            self.add_node(prod_node, mdp=mdp_node,
                          label=mdp_label, dra=dra_node, act=Us)
            if ((mdp_node == self.graph['mdp'].graph['init_state']) and
                (mdp_label == self.graph['mdp'].graph['init_label'] or mdp_label in self.graph['mdp'].graph['init_label']) and      # CRITICAL
                    (dra_node in self.graph['dra'].graph['initial'])):
                self.graph['initial'].add(prod_node)
        return prod_node

    def construct_opaque_subgraph_2_amec(self, product_mdp_gamma:Product_Dra, sync_amec_3, sync_amec_graph, mec_pi_3, mec_gamma_3, ap_pi, ap_gamma, observation_func, ctrl_obs_dict):
        #
        # 目标是生成从初始状态到达sync_amec的通路
        ip = sync_amec_3[1]
        ip_pi = [ state_t[0] for state_t in ip ]
        # (current_pi_state, [observed_state_list], [])
        # if current sync state is a pi state
        # then element 2 will only be states that satisfies ap gamma, and element 3 will be the rest of observed state
        # otherwise, element 2 will be full set of observed states, and element 3 wil be empty
        initial_sync_state = []
        #
        if ap_gamma == 'recharge':
            debug_var = 2001
        #
        for state_pi in self.graph['initial']:
            observed_state_list = []
            for state_gamma in product_mdp_gamma.graph['initial']:
                observed_state_list.append(state_gamma)
            initial_sync_state.append((state_pi, tuple(observed_state_list), tuple()))
        stack_t = [state_t for state_t in initial_sync_state]       # make a copy
        visited = set()
        subgraph_2_amec_t = MultiDiGraph()
        subgraph_2_amec_t.graph['initial'] = initial_sync_state

        while stack_t:
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            # 获取当前状态的出边
            current_state_pi, current_state_gamma, state_list_3 = current_state
            next_state_list_pi = list(self.out_edges(current_state_pi, data=True))
            next_state_list_gamma = []
            for state_gamma_t in set(current_state_gamma).union(set(state_list_3)):
                next_state_list_gamma += list(product_mdp_gamma.out_edges(state_gamma_t, data=True))

            for edge_t_pi in next_state_list_pi:
                next_state_pi        = edge_t_pi[1]
                next_observed_states = []
                next_states_3        = []
                #
                is_current_pi_state = is_state_satisfy_ap(next_state_pi, ap_pi)         # whether current state satisfy ap pi
                #
                trans_pr_cost_list = {}
                diff_expected_cost_list = {}
                #
                # for debugging
                if next_state_pi in ip_pi:
                    debug_var = 1
                if next_state_pi[2] == 4:
                    debug_var = 1.5
                #
                for edge_t_gamma in next_state_list_gamma:
                    # 获取下一状态
                    next_state_gamma = edge_t_gamma[1]

                    # 1. 检查控制动作同步
                    try:
                        u_pi = next(iter(edge_t_pi[2]['prop'].keys()))
                        u_gamma = next(iter(edge_t_gamma[2]['prop'].keys()))
                        if isinstance(u_pi, tuple):
                            u_pi = u_pi[0]
                        if isinstance(u_gamma, tuple):
                            u_gamma = u_gamma[0]

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        elif u_pi != u_gamma and ctrl_obs_dict[u_pi] == True and ctrl_obs_dict[u_gamma] == True:
                            continue
                    except (StopIteration, KeyError):
                        continue  # 如果没有动作则跳过

                    # 2. 检查观测同步
                    if observation_func(next_state_pi[0]) != observation_func(next_state_gamma[0]):
                        continue

                    # 3. opacity requirement
                    # if not is_ap_satisfy_opacity(next_state_pi, next_state_gamma, ap_pi, ap_gamma):
                    #     continue

                    # 4. 构建转移属性
                    # if is_ap_identical(next_state_pi, next_state_gamma):
                    if is_current_pi_state:
                        if is_state_satisfy_ap(next_state_gamma, ap_gamma) or next_state_gamma == next_state_pi:
                            # for all state in mec_pi satisfying AP \pi, the corresponding state in mec_gamma satisfies AP gamma
                            next_observed_states.append(next_state_gamma)
                        else:
                            next_states_3.append(next_state_gamma)
                    else:
                        next_observed_states.append(next_state_gamma)

                    for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                        diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                        #
                        if action not in diff_expected_cost_list.keys():
                            diff_expected_cost_list[action] = {next_state_gamma : diff_cost}
                        else:
                            diff_expected_cost_list[action][next_state_gamma] = diff_cost

                next_observed_states.sort()
                next_states_3.sort()

                for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                    trans_pr_cost_list[action] = (prob, cost)                                                 # it is evident that this only corresponds to state_pi

                # for debugging
                if next_state_pi == ('0', frozenset({'upload'}), 1):
                    debug_var = 2

                # 添加转移
                next_observed_states = tuple(list(set(next_observed_states)))
                next_states_3        = tuple(list(set(next_states_3)))
                next_sync_state = (next_state_pi, next_observed_states, next_states_3, )
                #
                mapping_t = {}
                for observed_state_t in subgraph_2_amec_t.nodes():
                    if observed_state_t[0] == next_state_pi:
                        next_observed_states = list(next_observed_states)
                        next_observed_states = next_observed_states + list(observed_state_t[1])
                        #
                        next_states_3 = list()
                        next_states_3 = next_states_3 + list(observed_state_t[2])
                        #
                        next_observed_states = tuple(list(set(next_observed_states)))
                        next_states_3 = tuple(list(set(next_states_3)))
                        next_sync_state = (next_state_pi, next_observed_states, next_states_3,)
                        #
                        mapping_t[observed_state_t] = next_sync_state
                        break
                if mapping_t.__len__():
                    #
                    # 更新点
                    nx.relabel_nodes(subgraph_2_amec_t, mapping_t, copy=False)
                    #
                    # 更新exp_cost
                    for u, v, attr in subgraph_2_amec_t.edges(data=True):
                        diff_exp_dict_old = attr['diff_exp']
                        diff_exp_dict_new = {}

                        for action, cost_dict in diff_exp_dict_old.items():
                            new_cost_dict = {}
                            for next_state_gamma, diff_cost in cost_dict.items():
                                # 替换成新的 sync state（如果有被 relabel）
                                if next_state_gamma in mapping_t:
                                    updated_state = mapping_t[next_state_gamma]
                                else:
                                    updated_state = next_state_gamma
                                new_cost_dict[updated_state] = diff_cost
                            diff_exp_dict_new[action] = new_cost_dict

                        # 更新属性
                        attr['diff_exp'] = diff_exp_dict_new
                    #
                    # 更新当前点
                    if current_state in mapping_t.keys():
                        current_state = mapping_t[current_state]
                    #
                    # 更新未加入的点
                    stack_t = self.replace_list_items(stack_t, mapping_t)

                if current_state == (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1),), ()) or next_sync_state == (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1),), ()):
                    debug_var = 3

                if (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1),), ()) in stack_t:
                    debug_var = 4


                is_next_state_in_ip = next_state_pi in ip_pi
                #
                is_opacity = self.is_state_opacity_in_observer_graph(next_sync_state, mec_pi_3, mec_gamma_3, ap_pi, ap_gamma)

                # for debugging
                if is_next_state_in_ip:
                    debug_var = 2

                # 如果边已经存在，需要合并 diff_expected_cost_list
                if subgraph_2_amec_t.has_edge(current_state, next_sync_state):
                    for key_t in subgraph_2_amec_t[current_state][next_sync_state]:
                        old_attr = subgraph_2_amec_t[current_state][next_sync_state][key_t]
                        old_diff_exp = old_attr.get('diff_exp', {})

                        # 合并新的 diff_expected_cost_list 到旧的 diff_exp
                        for action, new_cost_dict in diff_expected_cost_list.items():
                            if action not in old_diff_exp:
                                old_diff_exp[action] = dict(new_cost_dict)
                            else:
                                for next_state_gamma, new_diff_cost in new_cost_dict.items():
                                    old_diff_exp[action][next_state_gamma] = new_diff_cost

                        # 更新该边属性
                        #subgraph_2_amec_t[current_state][next_sync_state]['diff_exp'] = old_diff_exp
                        subgraph_2_amec_t.edges[current_state, next_sync_state, key_t]['diff_exp'] = old_diff_exp
                else:
                    # 否则添加新边
                    subgraph_2_amec_t.add_edge(
                        current_state,
                        next_sync_state,
                        prop=trans_pr_cost_list,
                        diff_exp=diff_expected_cost_list,
                        is_opacity=is_opacity
                    )
                #
                # 其实这里解出来以后因为ip点毕竟是少数, 而且MEC强连通
                # 所以这不是一个巧合：即限制到达后ip的点以后, 其实已经能遍历大部分的点
                # 结论:
                # 当存在一个强连通组分, 其中所有状态均为ip(即sync_amec_3[1])的子集, 且初始状态必须经过该SCC才能到达其他状态的时候
                # 才会出现initial_subgraph的状态显著比其他状态少
                # if not is_next_state_in_mec_observer:            # append
                #     stack_t.append(next_sync_state)
                if not is_next_state_in_ip:
                    stack_t.append(next_sync_state)
                    stack_t = list(set(stack_t))
                    stack_t.sort()

        return subgraph_2_amec_t, initial_sync_state

    def construct_fullgraph_4_amec(self, initial_subgraph, product_mdp_gamma: Product_Dra, sync_amec_graph, mec_pi_3, mec_gamma_3, ap_pi, ap_gamma, observation_func, ctrl_obs_dict):
        #
        # 目标是生成从初始状态到达sync_amec的通路
        stack_t = [state_t for state_t in initial_subgraph.nodes()]  # make a copy
        visited = set()
        fullgraph_t = MultiDiGraph(initial_subgraph)

        while stack_t:
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            # 获取当前状态的出边
            current_state_pi, current_state_gamma, state_list_3 = current_state
            next_state_list_pi = list(self.out_edges(current_state_pi, data=True))
            next_state_list_gamma = []
            for state_gamma_t in set(current_state_gamma).union(set(state_list_3)):
                next_state_list_gamma += list(product_mdp_gamma.out_edges(state_gamma_t, data=True))

            for edge_t_pi in next_state_list_pi:
                next_state_pi = edge_t_pi[1]
                next_observed_states = []
                next_states_3 = []
                #
                is_current_pi_state = is_state_satisfy_ap(next_state_pi,
                                                          ap_pi)  # whether current state satisfy ap pi
                #
                trans_pr_cost_list = {}
                diff_expected_cost_list = {}
                #
                for edge_t_gamma in next_state_list_gamma:
                    # 获取下一状态
                    next_state_gamma = edge_t_gamma[1]

                    # 1. 检查控制动作同步
                    try:
                        u_pi = next(iter(edge_t_pi[2]['prop'].keys()))
                        u_gamma = next(iter(edge_t_gamma[2]['prop'].keys()))
                        if isinstance(u_pi, tuple):
                            u_pi = u_pi[0]
                        if isinstance(u_gamma, tuple):
                            u_gamma = u_gamma[0]

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        elif u_pi != u_gamma and ctrl_obs_dict[u_pi] == True and ctrl_obs_dict[u_gamma] == True:
                            continue
                    except (StopIteration, KeyError):
                        continue  # 如果没有动作则跳过

                    # 2. 检查观测同步
                    if observation_func(next_state_pi[0]) != observation_func(next_state_gamma[0]):
                        continue

                    # 3. opacity requirement
                    # if not is_ap_satisfy_opacity(next_state_pi, next_state_gamma, ap_pi, ap_gamma):
                    #     continue

                    # 4. 构建转移属性
                    # if is_ap_identical(next_state_pi, next_state_gamma):
                    if is_current_pi_state:
                        if is_state_satisfy_ap(next_state_gamma, ap_gamma) or next_state_gamma == next_state_pi:
                            # for all state in mec_pi satisfying AP \pi, the corresponding state in mec_gamma satisfies AP gamma
                            next_observed_states.append(next_state_gamma)
                        else:
                            next_states_3.append(next_state_gamma)
                    else:
                        next_observed_states.append(next_state_gamma)

                    for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                        diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                        #
                        if action not in diff_expected_cost_list.keys():
                            diff_expected_cost_list[action] = {next_state_gamma: diff_cost}
                        else:
                            diff_expected_cost_list[action][next_state_gamma] = diff_cost

                next_observed_states.sort()
                next_states_3.sort()

                for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                    trans_pr_cost_list[action] = (prob, cost)  # it is evident that this only corresponds to state_pi

                # 添加转移
                next_observed_states = tuple(list(set(next_observed_states)))
                next_states_3 = tuple(list(set(next_states_3)))
                next_sync_state = (next_state_pi, next_observed_states, next_states_3,)
                #
                mapping_t = {}
                for observed_state_t in fullgraph_t.nodes():
                    if observed_state_t[0] == next_state_pi:
                        next_observed_states = list(next_observed_states)
                        next_observed_states = next_observed_states + list(observed_state_t[1])
                        #
                        next_states_3 = list()
                        next_states_3 = next_states_3 + list(observed_state_t[2])
                        #
                        next_observed_states = tuple(list(set(next_observed_states)))
                        next_states_3 = tuple(list(set(next_states_3)))
                        next_sync_state = (next_state_pi, next_observed_states, next_states_3,)
                        #
                        mapping_t[observed_state_t] = next_sync_state
                        break
                if mapping_t.__len__():
                    nx.relabel_nodes(fullgraph_t, mapping_t, copy=False)
                    #
                    #
                    # 更新exp_cost
                    for u, v, attr in fullgraph_t.edges(data=True):
                        diff_exp_dict_old = attr['diff_exp']
                        diff_exp_dict_new = {}

                        for action, cost_dict in diff_exp_dict_old.items():
                            new_cost_dict = {}
                            for next_state_gamma, diff_cost in cost_dict.items():
                                # 替换成新的 sync state（如果有被 relabel）
                                if next_state_gamma in mapping_t:
                                    updated_state = mapping_t[next_state_gamma]
                                else:
                                    updated_state = next_state_gamma
                                new_cost_dict[updated_state] = diff_cost
                            diff_exp_dict_new[action] = new_cost_dict

                        # 更新属性
                        attr['diff_exp'] = diff_exp_dict_new
                    #
                    if current_state in mapping_t.keys():
                        current_state = mapping_t[current_state]
                    #
                    stack_t = self.replace_list_items(stack_t, mapping_t)

                is_opacity = self.is_state_opacity_in_observer_graph(next_sync_state, mec_pi_3, mec_gamma_3, ap_pi, ap_gamma)

                #
                # 有些时候点都有但边没有
                # if next_sync_state in initial_subgraph.nodes():
                #     #visited.add(next_sync_state)
                #     continue
                #
                # 如果边已经存在，需要合并 diff_expected_cost_list
                if fullgraph_t.has_edge(current_state, next_sync_state):
                    for key_t in fullgraph_t[current_state][next_sync_state]:
                        old_attr = fullgraph_t[current_state][next_sync_state][key_t]
                        old_diff_exp = old_attr.get('diff_exp', {})

                        # 合并新的 diff_expected_cost_list 到旧的 diff_exp
                        for action, new_cost_dict in diff_expected_cost_list.items():
                            if action not in old_diff_exp:
                                old_diff_exp[action] = dict(new_cost_dict)
                            else:
                                for next_state_gamma, new_diff_cost in new_cost_dict.items():
                                    old_diff_exp[action][next_state_gamma] = new_diff_cost

                        # 更新该边属性
                        fullgraph_t[current_state][next_sync_state][key_t]['diff_exp'] = old_diff_exp
                else:
                    fullgraph_t.add_edge(current_state, next_sync_state,
                                               prop=trans_pr_cost_list,
                                               diff_exp=diff_expected_cost_list,
                                               is_opacity=is_opacity)

                #
                # 其实这里解出来以后因为ip点毕竟是少数, 而且MEC强连通
                # 所以这不是一个巧合：即限制到达后ip的点以后, 其实已经能遍历大部分的点
                # 结论:
                # 当存在一个强连通组分, 其中所有状态均为ip(即sync_amec_3[1])的子集, 且初始状态必须经过该SCC才能到达其他状态的时候
                # 才会出现initial_subgraph的状态显著比其他状态少
                # if not is_next_state_in_mec_observer:            # append
                #     stack_t.append(next_sync_state)
                stack_t.append(next_sync_state)
                stack_t = list(set(stack_t))
                stack_t.sort()

        return fullgraph_t

    def re_synthesize_sync_amec(self, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        #
        # 想了一下这个还是沿用旧的
        # sync_amec的最大问题在于它的初始状态只凭借两个集合是很难确定的
        mec_state_set_pi = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = []
        for state_pi_t in mec_state_set_pi:
            for state_gamma_t in mec_state_set_gamma:
                ap_state_pi = set(state_pi_t[1])
                ap_state_gamma = set(state_gamma_t[1])
                if ap_pi in ap_state_pi:
                    if ap_gamma in ap_state_gamma:
                        stack_t.append((state_pi_t, state_gamma_t,))

        #
        stack_t = list(set(stack_t))
        visited = set()
        sync_mec_t = MultiDiGraph()
        #
        # Added
        for node_t in stack_t:
            sync_mec_t.add_node(node_t)


        if ap_gamma == 'recharge':
            debug_var = 2001

        while stack_t:
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            # 获取当前状态的出边
            current_state_pi, current_state_gamma = current_state
            next_state_list_pi = list(self.out_edges(current_state_pi, data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state_gamma, data=True))

            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    # 获取下一状态
                    next_state_pi = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)

                    # 1. 检查控制动作同步
                    try:
                        u_pi = next(iter(edge_t_pi[2]['prop'].keys()))
                        u_gamma = next(iter(edge_t_gamma[2]['prop'].keys()))
                        if isinstance(u_pi, tuple):
                            u_pi = u_pi[0]
                        if isinstance(u_gamma, tuple):
                            u_gamma = u_gamma[0]

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        elif u_pi != u_gamma and ctrl_obs_dict[u_pi] == True and ctrl_obs_dict[u_gamma] == True:
                            continue
                    except (StopIteration, KeyError):
                        continue  # 如果没有动作则跳过

                    if ap_gamma == 'recharge':
                        debug_var = 2002

                    # 2. 检查状态是否在AMEC中
                    if (next_state_pi not in mec_state_set_pi or
                            next_state_gamma not in mec_state_set_gamma):
                        continue

                    if ap_gamma == 'recharge':
                        debug_var = 2003
                        if '12' in next_state_pi:
                            if '2' in next_state_gamma:
                                debug_var = 2003.1                  # for breakpoints

                    # 3. 检查观测同步
                    if observation_func(next_state_pi[0]) != observation_func(next_state_gamma[0]):
                        continue

                    # 4. opacity requirement
                    is_opacity_satisfied = True
                    if not is_ap_satisfy_opacity(next_state_pi, next_state_gamma, ap_pi, ap_gamma):
                        is_opacity_satisfied = False

                    # 5. 构建转移属性
                    # if is_ap_identical(next_state_pi, next_state_gamma):
                    if True:
                        trans_pr_cost_list = {}
                        diff_expected_cost_list = {}
                        for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                            diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                            trans_pr_cost_list[action] = (prob, cost)
                            diff_expected_cost_list[action] = diff_cost

                        #
                        if is_opacity_satisfied:
                            sync_mec_t.add_edge(current_state, next_sync_state,
                                                prop=trans_pr_cost_list,
                                                diff_exp=diff_expected_cost_list)
                            stack_t.append(next_sync_state)
                            stack_t = list(set(stack_t))
                            stack_t.sort()                            

        # 完成后处理
        print_c(
            f"[synthesize_w_opacity] DFS completed, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}",
            color=33)

        if ap_gamma == 'recharge':
            debug_var = 2004

        if sync_mec_t.edges:
            # 检查强连通分量并保留最大的SCC
            scc_list = list(nx.strongly_connected_components(sync_mec_t))
            scc_list.sort(key=lambda x: len(x), reverse=True)

            if not scc_list:
                print_c("[synthesize_w_opacity] NO SCCs found...", color=33)
            else:
                # 移除非最大SCC的节点
                largest_scc = scc_list[0]
                nodes_to_remove = [node for scc in scc_list[1:] for node in scc]
                num_removed = 0

                for node in nodes_to_remove:
                    if node in sync_mec_t:
                        sync_mec_t.remove_node(node)
                        num_removed += 1
                        print_c(f"[synthesize_w_opacity] removing node: {node}", color=35)

                print_c(f"[synthesize_w_opacity] number of states removed: {num_removed}", color=33)

            # 保存结果
            self.sync_amec_set.append(sync_mec_t)
            self.current_sync_amec_index = len(self.sync_amec_set) - 1

        print_c(f"[synthesize_w_opacity] Generated sync_amec, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}")

    def project_sync_amec_back_to_mec_pi(self, sync_amec, original_mec_pi):

        mec_pi_subset_0 = []
        mec_pi_subset_1 = []
        mec_pi_subset_2 = {}

        for sync_state_t in sync_amec.nodes():
            state_pi = sync_state_t[0]

            if state_pi in original_mec_pi[0]:
                mec_pi_subset_0.append(sync_state_t)

            if state_pi in original_mec_pi[1]:
                mec_pi_subset_1.append(sync_state_t)

            if state_pi in original_mec_pi[2].keys():
                mec_pi_subset_2[sync_state_t] = original_mec_pi[2][state_pi]        # TODO to check, 似乎空集全没了

        mec_pi_subset_0 = list(set(mec_pi_subset_0))
        mec_pi_subset_1 = set(mec_pi_subset_1)

        return [mec_pi_subset_0, mec_pi_subset_1, mec_pi_subset_2]

    def is_state_opacity_in_observer_graph(self, state, mec_pi, mec_gamma, ap_pi, ap_gamma):
        current_state  = state[0]
        observed_state = list(state[1]) + list(state[2])

        is_observation_identical_state = not (observed_state.__len__() == 1 and current_state in observed_state)   # default: True, 这个很好理解, 没有打掩护的状态必然不满足Opacity
        is_ip_state = ap_pi in current_state[1]

        is_state_gamma_in_mec = True
        for state_gamma_t in observed_state:
            if state_gamma_t == current_state:
                continue
            if state_gamma_t not in mec_gamma[0]:
                is_state_gamma_in_mec = False
        is_state_pi_in_mec = current_state in mec_pi[0]
        is_mec_satisfied = is_state_gamma_in_mec and is_state_pi_in_mec or (not is_state_pi_in_mec and is_state_gamma_in_mec)

        if is_ip_state:
            is_pi_state_satisfy_ap = is_state_satisfy_ap(current_state, ap_pi)
            is_gamma_state_satisfy_ap = True
            for state_gamma_t in observed_state:
                if state_gamma_t == current_state:
                    continue
                is_gamma_state_satisfy_ap = is_state_satisfy_ap(state_gamma_t, ap_gamma)
                if not is_gamma_state_satisfy_ap:
                    break

        if is_ip_state:
            is_opacity = is_observation_identical_state and is_mec_satisfied and (is_pi_state_satisfy_ap and is_gamma_state_satisfy_ap)
        else:
            is_opacity = is_observation_identical_state and is_mec_satisfied

        if not is_opacity:
            debug_var = 6
            if is_observation_identical_state:
                debug_var = 6.1
            if is_mec_satisfied:
                debug_var = 6.2
        else:
            debug_var = 7

        return is_opacity

    def update_best_all_plan(self, best_all_plan, is_print_policy=True):
        # TODO
        # plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync],
        #              [plan_suffix, suffix_cost, suffix_risk],
        #              [MEC_pi[0], MEC_pi[1], Sr, Sd],
        #              [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index, MEC_gamma],
        #              [initial_subgraph, initial_sync_state, opaque_full_graph],
        #              [sync_mec_t, observer_mec_3]
        #              ])
        self.best_all_plan['plan_prefix']  = best_all_plan[0][0]
        self.best_all_plan['plan_suffix']  = best_all_plan[1][0]
        self.best_all_plan['ap_4_opacity'] = best_all_plan[3][0]
        self.best_all_plan['cost']         = [ best_all_plan[0][1], best_all_plan[1][1] ]
        self.best_all_plan['risk']         = [ best_all_plan[0][2], best_all_plan[1][2] ]
        self.best_all_plan['sync_amec_graph']        = self.sync_amec_set[best_all_plan[3][2]]
        self.best_all_plan['initial_subgraph']       = best_all_plan[4][0]
        self.best_all_plan['initial_observer_state'] = best_all_plan[4][1]                          # list of possible state, not a single state
        self.best_all_plan['opaque_full_graph']      = best_all_plan[4][2]
        self.best_all_plan['mec'] = {}
        self.best_all_plan['mec']['pi']       = [best_all_plan[2][0],    best_all_plan[2][1]]
        self.best_all_plan['mec']['gamma']    = [best_all_plan[3][3][0], best_all_plan[3][3][1]]
        self.best_all_plan['mec']['sync']     = [best_all_plan[5][0][0], best_all_plan[5][0][1]]
        self.best_all_plan['mec']['observer'] = [best_all_plan[5][1][0], best_all_plan[5][1][1]]

        if is_print_policy:
            print_c("policy for AP: %s" % str(self.best_all_plan['ap_4_opacity']))
            print_c("state action: probabilities")
            #
            self.print_policy(self.best_all_plan['plan_prefix'], self.best_all_plan['plan_suffix'])

    def print_policy(self, plan_prefix, plan_suffix):
        def format_policy_entry(state, actions, probs):
            # 对齐三个部分：STATE、ACTIONS、PROBS
            state_str = str(state)
            actions_str = str(actions)
            probs_str = str(probs)
            return "{:<180} {:<30}: {}".format(state_str, actions_str, probs_str)

        # 打印 Prefix 部分
        print_c("\nPrefix", color='bg_magenta', style='bold')
        header = "{:<180} {:<30} {}".format("STATE", "ACTIONS", ": PROBS")
        print_c(header, color='magenta', style='bold')
        for state_t in plan_prefix:
            actions, probs = plan_prefix[state_t]
            line = format_policy_entry(state_t, actions, probs)
            print_c(line, color='magenta')

        # 打印 Suffix 部分
        print_c("\nSuffix", color='bg_cyan', style='bold')
        header = "{:<180} {:<30} {}".format("STATE", "ACTIONS", ": PROBS")
        print_c(header, color='blue', style='bold')
        for state_t in plan_suffix:
            actions, probs = plan_suffix[state_t]
            line = format_policy_entry(state_t, actions, probs)
            print_c(line, color='blue')

    def execution_in_observer_graph(self, total_T, best_all_plan=None, initial_state_index=0):
        # ----plan execution with or without given observation----
        if best_all_plan == None:
            best_all_plan = self.best_all_plan
        # ----
        current_state, prod_state, mdp_state, observed_state, opacity_state, current_label, observed_label = self.get_individual_state_from_observers(self.best_all_plan['initial_observer_state'][initial_state_index])
        #
        X     = [ mdp_state ]                      # current state list
        OX    = [ current_state ]                  # observer state list
        O     = [ observed_state ]                 # observed state
        X_OPA = [ opacity_state ]
        L     = [ current_label ]
        OL    = [ observed_label ]
        U     = []
        M     = []
        #
        observer_t = self.best_all_plan['opaque_full_graph']
        is_now_in_suffix_cycle = False
        #
        for t in range(1, total_T):
            u, m = self.act_by_plan_in_observer(current_state, self.best_all_plan['plan_prefix'], self.best_all_plan['plan_suffix'], is_now_in_suffix_cycle)

            S = []
            P = []
            for state_t in observer_t.successors(current_state):
                for key_t in observer_t[current_state][state_t]:
                    prop = observer_t[current_state][state_t][key_t]['prop']
                    if (u in list(prop.keys())):
                        S.append(state_t)
                        P.append(prop[u][0])

            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            next_state = tuple(S[k])

            #
            # added
            state, prod_state, mdp_state, observed_state, opacity_state, current_label, observed_label = self.get_individual_state_from_observers(next_state)

            X.append(mdp_state)
            OX.append(state)
            O.append(observed_state)
            X_OPA.append(opacity_state)
            L.append(current_label)
            OL.append(observed_label)
            U.append(u)
            M.append(m)

            current_state = next_state
            is_now_in_suffix_cycle = next_state in self.best_all_plan['mec']['observer'][0]

        return X, OX, O, X_OPA, L, OL, U, M

    def execution(self, best_all_plan, total_T, state_seq, label_seq):
        # ----plan execution with or without given observation----
        t = 0
        X = []
        L = []
        U = []
        M = []
        PX = []
        m = 0
        # ----
        while (t <= total_T):
            if (t == 0):
                # print '---initial run----'
                mdp_state = state_seq[0]
                label = label_seq[0]
                initial_set = self.graph['initial'].copy()
                current_state = initial_set.pop()
            elif (t >= 1) and (len(state_seq) > t):
                # print '---observation given---'
                mdp_state = state_seq[t]
                label = label_seq[t]
                prev_state = tuple(current_state)
                error = True
                for next_state in self.successors(prev_state):
                    if((self.nodes[next_state]['mdp'] == mdp_state) and (self.nodes[next_state]['label'] == label) and (u in list(self[prev_state][next_state]['prop'].keys()))):
                        current_state = tuple(next_state)
                        error = False
                        break
                if error:
                    print(
                        'Error: The provided state and label sequences do NOT match the mdp structure!')
                    break
            else:
                # print '---random observation---'
                prev_state = tuple(current_state)
                S = []
                P = []
                if m < 2 or m == 10:                                    # in prefix or suffix, added, it is admissible for states to get in Ip again (m == 10)
                    for next_state in self.successors(prev_state):
                        prop = self[prev_state][next_state]['prop']
                        if (u in list(prop.keys())):
                            S.append(next_state)
                            P.append(prop[u][0])
                if m == 2:  # in bad states
                    # print 'in bad states'
                    Sd = best_all_plan[2][3]
                    Sf = best_all_plan[2][0]
                    Sr = best_all_plan[2][2]
                    (xf, lf, qf) = prev_state
                    postqf = self.graph['dra'].successors(qf)
                    for xt in self.graph['mdp'].successors(xf):
                        if xt != xf:
                            prop = self.graph['mdp'][xf][xt]['prop']
                            if u in list(prop.keys()):
                                prob_edge = prop[u][0]
                                label = self.graph['mdp'].nodes[xt]['label']
                                for lt in label.keys():
                                    prob_label = label[lt]
                                    dist = dict()
                                    for qt in postqf:
                                        if (xt, lt, qt) in Sf.union(Sr):
                                            dist[qt] = self.graph['dra'].check_distance_for_dra_edge(
                                                lf, qf, qt)
                                    if list(dist.keys()):
                                        qt = min(list(dist.keys()),
                                                 key=lambda q: dist[q])
                                        S.append((xt, lt, qt))
                                        P.append(prob_edge*prob_label)
                rdn = random.random()
                pc = 0
                for k, p in enumerate(P):
                    pc += p
                    if pc > rdn:
                        break
                current_state = tuple(S[k])
                mdp_state = self.nodes[current_state]['mdp']
                label = self.nodes[current_state]['label']
            # ----
            u, m = self.act_by_plan(best_all_plan, current_state)
            X.append(mdp_state)
            PX.append(current_state)
            L.append(set(label))
            U.append(u)
            M.append(m)
            t += 1
        return X, L, U, M, PX

    def act_by_plan(self, best_plan, prod_state):
        # ----choose the randomized action by the optimal policy----
        # recall that {best_plan = [plan_prefix, prefix_cost, prefix_risk, y_in_sf],
        # [plan_suffix, suffix_cost, suffix_risk], [MEC[0], MEC[1], Sr, Sd], plan_bad]}
        plan_prefix = best_plan[0][0]
        plan_suffix = best_plan[1][0]
        plan_bad = best_plan[3]
        if (prod_state in plan_prefix):
            # print 'In prefix'
            U = plan_prefix[prod_state][0]
            P = plan_prefix[prod_state][1]
            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
            return U[k], 0
        elif (prod_state in plan_suffix):
            # print 'In suffix'
            U = plan_suffix[prod_state][0]
            P = plan_suffix[prod_state][1]
            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
            if prod_state in best_plan[2][1]:
                return U[
                    k], 10  # it is strange for best_plan[2][1] is for state set I_p, i.e., the states that Ap is satisfied
                # return U[k], 1
            else:
                return U[k], 1
        elif (prod_state in plan_bad):
            # print 'In bad states'
            U = plan_bad[prod_state][0]
            P = plan_bad[prod_state][1]
            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            # print 'action chosen: %s' %str(U[k])
            return U[k], 2
        else:
            print_c("Warning, current state %s is outside prefix and suffix !" % (str(prod_state),), color=33)
            return None, 4

    def act_by_plan_in_observer(self, state, policy_prefix, policy_suffix, is_set_in_suffix_cycle=False):
        # ----choose the randomized action by the optimal policy----
        if not is_set_in_suffix_cycle and state in policy_prefix:
            # print 'In prefix'
            U = policy_prefix[state][0]
            P = policy_prefix[state][1]
            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
            return U[k], 0
        elif is_set_in_suffix_cycle or state in policy_suffix:
            # print 'In suffix'
            U = policy_suffix[state][0]
            P = policy_suffix[state][1]
            rdn = random.random()
            pc = 0
            for k, p in enumerate(P):
                pc += p
                if pc > rdn:
                    break
            # print('%s action chosen: %s' % (str(prod_state), str(U[k], )))
            return U[k], 1
        else:
            print_c("Warning, current state %s is outside prefix and suffix !" % (str(state),), color=33)
            return None, 4

    def get_individual_state_from_observers(self, observer_state):
        is_ip_state = observer_state[2].__len__() > 0
        prod_state     = observer_state[0]
        mdp_state      = prod_state[0]
        observed_state = list(set(observer_state[1]).union(observer_state[2]))
        if is_ip_state:
            opacity_state = list(set(observer_state[1]).difference(set(observer_state[0])))
        else:
            opacity_state = []
        #
        current_label  = set(prod_state[1])
        observed_label = list([set(state_t[1]) for state_t in observer_state[1]]) + [ current_label ]

        return observer_state, prod_state, mdp_state, observed_state, opacity_state, current_label, observed_label



    def replace_list_items(self, lst, mapping):
        return [mapping.get(item, item) for item in lst]


