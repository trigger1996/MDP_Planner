#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import fabs, sqrt

import networkx as nx
from networkx import DiGraph, is_connected
from MDP_TG.mdp import find_MECs, find_SCCs
from MDP_TG.dra import Product_Dra
from User.vis2 import print_c

import random


def observation_func_1(state, observation='X'):
    if observation == 'X':
        return state[0][0]
    if observation == 'y':
        return state[0][1]     # x and y are NOT inverted here, they are only inverted in plotting

def observation_func_2(state):
    observation_set = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                       (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                       (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                       (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                       (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
                       (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),]
    min_dist = 1e6
    observed_state = (0, 0)
    for state_obs_t in observation_set:
        x_obs = state_obs_t[0]
        y_obs = state_obs_t[1]
        #
        dx = state[0][0] - x_obs
        dy = state[0][1] - y_obs
        dist = sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            observed_state = state_obs_t
    return observed_state

def is_with_identical_observation_1(state, state_prime):
    #
    # the observation is X coordinate of the state
    if state[0][0] == state_prime[0][0]:
        return True
    else:
        return False

def is_with_identical_observation_2(state, state_prime):
    #
    # the observation is Y coordinate of the state
    if state[0][1] == state_prime[0][1]:
        return True
    else:
        return False

def is_with_identical_observation_3(state, state_prime, dist_threshold=0.5):
    #
    # if the Cartesian distance is smaller than the given threshold
    dx = state[0][0] - state_prime[0][0]
    dy = state[0][1] - state_prime[0][1]
    dist = sqrt(dx * dx + dy * dy)
    if dist <= dist_threshold:
        return True
    else:
        return False

def is_with_identical_observation_4(state, state_prime, dist_threshold=1.):
    #
    # if the Manhattan distance is smaller than the given threshold
    dx = state[0][0] - state_prime[0][0]
    dy = state[0][1] - state_prime[0][1]
    dist = fabs(dx) + fabs(dy)
    if dist <= dist_threshold:
        return True
    else:
        return False

def get_ap_from_product_mdp(state):
    return list(state[1])

def is_ap_identical(state_pi, state_gamma):
    if list(state_pi[1]).__len__() == 0 and list(state_gamma[1]).__len__() == 0:
        return True
    elif set(state_pi[1]) == set(state_gamma[1]):
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

def find_initial_state(y_in_sf_pi, y_in_sf_gamma, state_set_gamma, observation_func=observation_func_1):
    state_list = []
    for state_pi_t in y_in_sf_pi.keys():
        # TODO
        # to explain reasons
        # if y_in_sf_pi[state_pi_t] == 0:
        #     continue
        for state_gamma_t in y_in_sf_gamma:
            # if y_in_sf_gamma[state_gamma_t] == 0:
            #     continue
            #
            if observation_func(state_pi_t[0]) == observation_func(state_gamma_t[0]):
                if is_ap_identical(state_pi_t, state_gamma_t):
                    state_list.append((state_pi_t, state_gamma_t, ))
    return state_list

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

def sort_scc_list(x):
    return x.__len__()

def act_by_plan(prod_mdp, best_plan, prod_state):
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
            return  U[k], 10                    # it is strange for best_plan[2][1] is for state set I_p, i.e., the states that Ap is satisfied
            #return U[k], 1
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
        print_c("Warning, current state %s is outside prefix and suffix !"  % (str(prod_state), ), color=33)
        return None, 4

def get_edge_props(g, u, v):
    prop_list = []
    for edge_t in list(g.out_edges(u, data=True)):
        if v == edge_t[1]:
            prop_list.append(edge_t[2]['prop'])
    return prop_list

class product_mdp2(Product_Dra):
    def __init__(self, mdp=None, dra=None):
        if mdp == None and dra == None:
            print_c("[product mdp 2] Initializing subgraph for product mdp ...", color='yellow')
        else:
            Product_Dra.__init__(self, mdp, dra)
            self.compute_S_f()
            #
            self.sync_amec_set    = list()
            self.mec_observer_set = list()
            #
            self.current_sync_amec_index = 0
            self.mec_observer_index      = 0

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

    def construct_opaque_subgraph_2_amec(self, product_mdp_gamma:Product_Dra, sync_amec_3, sync_amec_graph, mec_observer, ap_pi, ap_gamma, observation_func, ctrl_obs_dict):
        #
        # 目标是生成从初始状态到达sync_amec的通路
        ip = sync_amec_3[1]
        initial_sync_state = list({(a, b) for a in self.graph['initial'] for b in product_mdp_gamma.graph['initial']})
        stack_t = [state_t for state_t in initial_sync_state]       # make a copy
        visited = set()
        subgraph_2_amec_t = DiGraph()

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

                    # 2. 检查观测同步
                    if observation_func(next_state_pi[0]) != observation_func(next_state_gamma[0]):
                        continue

                    # 3. opacity requirement
                    # if not is_ap_satisfy_opacity(next_state_pi, next_state_gamma, ap_pi, ap_gamma):
                    #     continue

                    # 4. 构建转移属性
                    # if is_ap_identical(next_state_pi, next_state_gamma):
                    if True:
                        trans_pr_cost_list = {}
                        diff_expected_cost_list = {}
                        for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                            diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                            trans_pr_cost_list[action] = (prob, cost)
                            diff_expected_cost_list[action] = diff_cost

                        is_next_state_in_sync_amec = next_sync_state in sync_amec_graph.nodes()
                        is_next_state_in_mec_observer = next_sync_state in mec_observer.nodes()
                        is_next_state_in_ip = next_sync_state in ip
                        #
                        is_opacity = is_next_state_in_sync_amec and is_next_state_in_mec_observer

                        if is_opacity == False:
                            debug_var = 1

                        if is_next_state_in_sync_amec and not is_next_state_in_mec_observer:
                            debug_var = 2

                        if is_next_state_in_ip:
                            debug_var = 3

                        # 添加转移
                        subgraph_2_amec_t.add_edge(current_state, next_sync_state,
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
                        if not is_next_state_in_ip:
                            stack_t.append(next_sync_state)
                            stack_t = list(set(stack_t))
                            stack_t.sort()

        return subgraph_2_amec_t, initial_sync_state

    def construct_fullgraph_4_amec(self, initial_subgraph, product_mdp_gamma: Product_Dra, sync_amec, mec_observer, ap_pi, ap_gamma, observation_func, ctrl_obs_dict):
            #
            # 目标是生成从初始状态到达sync_amec的通路
            stack_t = [state_t for state_t in initial_subgraph.nodes()]  # make a copy
            visited = set()
            fullgraph_t = DiGraph(initial_subgraph)

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

                        if not initial_subgraph.has_edge(current_state, next_sync_state):
                            if mec_observer.has_edge(current_state, next_sync_state):
                                print(233)

                        #
                        # 有些时候点都有但边没有
                        # if next_sync_state in initial_subgraph.nodes():
                        #     #visited.add(next_sync_state)
                        #     continue

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
                        if True:
                            trans_pr_cost_list = {}
                            diff_expected_cost_list = {}
                            for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                                diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                                trans_pr_cost_list[action] = (prob, cost)
                                diff_expected_cost_list[action] = diff_cost

                            is_next_state_in_sync_amec = next_sync_state in sync_amec.nodes()
                            is_next_state_in_mec_observer = next_sync_state in mec_observer.nodes()
                            is_opacity = is_next_state_in_sync_amec and is_next_state_in_mec_observer

                            if is_opacity == False:
                                debug_var = 1

                            if is_next_state_in_sync_amec and not is_next_state_in_mec_observer:
                                debug_var = 2

                            # 添加转移
                            fullgraph_t.add_edge(current_state, next_sync_state,
                                                       prop=trans_pr_cost_list,
                                                       diff_exp=diff_expected_cost_list,
                                                       is_opacity=is_opacity)

                            stack_t.append(next_sync_state)
                            stack_t = list(set(stack_t))
                            stack_t.sort()

            return fullgraph_t

    def re_synthesize_sync_amec(self, y_in_sf_pi, y_in_sf_gamma, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi    = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf_pi, y_in_sf_gamma, list(MEC_gamma[0]), observation_func=observation_func)
        stack_t = list(set(stack_t))
        visited = set()

        sync_mec_t = DiGraph()
        # for state_t in stack_t:
        #     sync_mec_t.add_node(state_t)
        while stack_t.__len__():
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            #
            next_state_list_pi    = list(self.out_edges(current_state[0], data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1], data=True))
            #
            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    #
                    next_state_pi    = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)
                    #
                    u_pi    = list(edge_t_pi[2]['prop'].keys())[0]
                    u_gamma = list(edge_t_gamma[2]['prop'].keys())[0]
                    #
                    if u_pi != u_gamma:
                        continue
                    #
                    is_next_state_pi_in_amec    = next_state_pi    in mec_state_set_pi
                    is_next_state_gamma_in_amec = next_state_gamma in mec_state_set_gamma
                    if not is_next_state_pi_in_amec or not is_next_state_gamma_in_amec:
                        continue
                    #
                    if observation_func(next_state_pi) == observation_func(next_state_gamma):
                        # TODO
                        # it seems that if the AP are required identical, then the AP-none_AP states will NOT taken into consideration
                        # in other words, the successive states will be synchronized by AP-AP states
                        #
                        # ON THE OTHER HAND
                        # on the aspect of the intruders, the only information he can receive is the observation sequences
                        # so the sync-MEC (, or the observer) should focus only on the observation sequences
                        # NOT the aps
                        #
                        #if is_ap_identical(next_state_pi, next_state_gamma):
                        if True:

                            #
                            trans_pr_cost_list = dict()
                            diff_expected_cost_list = dict()
                            for current_action_t in edge_t_pi[2]['prop'].keys():
                                #
                                transition_prop = edge_t_pi[2]['prop'][current_action_t][0]
                                transition_cost = edge_t_pi[2]['prop'][current_action_t][1]
                                diff_expected_cost = obtain_differential_expected_cost(current_action_t, edge_t_pi, edge_t_gamma)
                                #
                                trans_pr_cost_list[current_action_t] = (transition_prop, transition_cost, )
                                diff_expected_cost_list[current_action_t] = diff_expected_cost
                            #
                            sync_mec_t.add_edge(current_state, next_sync_state, prop=trans_pr_cost_list, diff_exp=diff_expected_cost_list)
                            stack_t.append(next_sync_state)
                            stack_t = list(set(stack_t))
                            stack_t.sort()
        #
        print_c("[synthesize_w_opacity] DFS completed, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),), color=33)
        if sync_mec_t.edges().__len__():
            #
            # TODO
            # 检查连接性
            scc_list = list(nx.strongly_connected_components(sync_mec_t))
            scc_list.sort(key=sort_scc_list,reverse=True)
            if scc_list.__len__() == 0:
                print_c("[synthesize_w_opacity] NO SCCs found ..." , color=33)

            num_node_removed = 0
            for i, state_list_t in enumerate(scc_list):
                if i == 0:
                    continue
                for state_t in state_list_t:
                    try:
                        sync_mec_t.remove_node(state_t)
                        num_node_removed += 1
                        print_c("[synthesize_w_opacity] removing node: " + str(state_t), color=35)
                    except:
                        pass

            print_c("[synthesize_w_opacity] number of state to remove: %d" % (num_node_removed,), color=33)

            #
            # for debugging
            # state_pi_list_in_amec = []
            # for sync_state_t in (sync_mec_t.nodes()):
            #     state_pi_list_in_amec.append(sync_state_t[0])
            # #
            # state_pi_list_not_in_amec = []
            # for state_pi_t in MEC_pi[0]:
            #     if state_pi_t not in state_pi_list_in_amec:
            #         state_pi_list_not_in_amec.append(state_pi_t)

            #
            self.sync_amec_set.append(sync_mec_t)
            self.current_sync_amec_index = self.sync_amec_set.__len__() - 1
        #
        print_c("[synthesize_w_opacity] Generated sync_amec, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),))

    def re_synthesize_sync_amec2(self, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        mec_state_set_pi = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = []
        for state_pi_t in mec_state_set_pi:
            for state_gamma_t in mec_state_set_gamma:
                ap_state_pi    = set(state_pi_t[1])
                ap_state_gamma = set(state_gamma_t[1])
                if ap_pi in ap_state_pi:
                    if ap_gamma in ap_state_gamma:
                        stack_t.append((state_pi_t, state_gamma_t, ))

        #
        stack_t = list(set(stack_t))
        visited = set()
        sync_mec_t = DiGraph()

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

                    # 2. 检查状态是否在AMEC中
                    if (next_state_pi not in mec_state_set_pi or
                            next_state_gamma not in mec_state_set_gamma):
                        continue

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
        print_c(f"[synthesize_w_opacity] DFS completed, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}",
                color=33)

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

            self.synthesize_mec_observer(product_mdp_gamma, sync_mec_t, MEC_pi, MEC_gamma, observation_func, ctrl_obs_dict)

        print_c(f"[synthesize_w_opacity] Generated sync_amec, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}")

    def synthesize_mec_observer(self, product_mdp_gamma:Product_Dra, sync_amec, MEC_pi, MEC_gamma, observation_func, ctrl_obs_dict):

        stack_t = [ node_t for node_t in sync_amec.nodes() ]
        visited = set()
        mec_observer_t = DiGraph()

        while stack_t.__len__():
            current_sync_state = stack_t.pop()
            if current_sync_state in visited:
                continue
            visited.add(current_sync_state)

            # 获取当前状态的出边
            current_state_pi, current_state_gamma = current_sync_state
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

                    # 3. 检查观测同步
                    if observation_func(next_state_pi[0]) != observation_func(next_state_gamma[0]):
                        continue

                    trans_pr_cost_list = {}
                    diff_expected_cost_list = {}
                    for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                        diff_cost = obtain_differential_expected_cost(action, edge_t_pi, edge_t_gamma)
                        trans_pr_cost_list[action] = (prob, cost)
                        diff_expected_cost_list[action] = diff_cost

                    #
                    mec_observer_t.add_edge(current_sync_state, next_sync_state,
                                        prop=trans_pr_cost_list,
                                        diff_exp=diff_expected_cost_list)
                    stack_t.append(next_sync_state)
                    stack_t = list(set(stack_t))
                    stack_t.sort()


        if mec_observer_t.edges:
            # 检查强连通分量并保留最大的SCC
            scc_list = list(nx.strongly_connected_components(mec_observer_t))
            scc_list.sort(key=lambda x: len(x), reverse=True)

            if not scc_list:
                print_c("[synthesize_w_opacity] NO SCCs found...", color=33)
            else:
                # 移除非最大SCC的节点
                largest_scc = scc_list[0]
                nodes_to_remove = [node for scc in scc_list[1:] for node in scc]
                num_removed = 0

                for node in nodes_to_remove:
                    if node in mec_observer_t:
                        mec_observer_t.remove_node(node)
                        num_removed += 1
                        print_c(f"[synthesize_w_opacity] removing node: {node}", color=36)

        self.mec_observer_set.append(mec_observer_t)
        self.mec_observer_index += 1


    def re_synthesize_sync_amec_rex(self, y_in_sf_pi, y_in_sf_gamma, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi    = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf_pi, y_in_sf_gamma, list(MEC_gamma[0]), observation_func=observation_func)
        stack_t = list(set(stack_t))
        visited = set()



        #
        # scc_list_pi = list(nx.strongly_connected_components(self))
        # matching_sccs = [scc for scc in scc_list_pi if scc & set(mec_state_set_pi)]
        #
        # scc_list_gamma = list(nx.strongly_connected_components(product_mdp_gamma))
        # matching_sccs = [scc for scc in scc_list_gamma if scc & set(mec_state_set_gamma)]
        #
        #
        # 20240403后面发现这个事情不能用笛卡尔积算,  因为笛卡尔积必须保证有一个点不动, 所以最后只能用我们自己的办法来计算
        #
        # scc转图
        # 然后图求笛卡尔积
        #
        #
        # rex用subgraph计算, non-rex用scc计算
        # An end component (EC) of M is a sub-MDP (S, A) such that the digraph G (S,A) induced by (S, A) is strongly connected.
        # a strongly connected component (SCC) of the digraph G M induced by M is a set of states S ⊆ X, so that there exists a path in each direction between any pair of states in S.
        # the main difference between an MEC (S, A) and an SCC S is that the SCC does not restrict the set of actions U(s) that can be taken at each state s ∈ S.
        # subgraph_mec_pi = self.subgraph(mec_state_set_pi)
        # subgraph_mec_gamma = self.subgraph(mec_state_set_gamma)
        #
        # sync_subgraph = nx.cartesian_product(subgraph_mec_pi, subgraph_mec_gamma)
        #
        # for u, v, data in sync_subgraph.edges(data=True):
        #     node_u = u[0]
        #     node_v = v[0]
        #
        #     # 获取原子图中的边属性
        #     data_pi = subgraph_mec_pi[node_u][node_v] if subgraph_mec_pi.has_edge(node_u, node_v) else {}
        #     data_gamma = subgraph_mec_gamma[u[1]][v[1]] if subgraph_mec_gamma.has_edge(u[1], v[1]) else {}
        #
        #
        #
        # # to remove nodes
        # node_to_remove = []
        # edge_to_remove = []
        # for sync_edge_t in sync_subgraph.edges(data=True):
        #     #
        #     current_state_pi    = sync_edge_t[0][0]
        #     current_state_gamma = sync_edge_t[0][1]
        #     next_state_pi       = sync_edge_t[1][0]
        #     next_state_gamma    = sync_edge_t[1][1]
        #     #
        #     u_pi = get_edge_props(self, current_state_pi, next_state_pi)
        #     u_gamma = get_edge_props(product_mdp_gamma, current_state_gamma, next_state_gamma)
        #
        #     # 作笛卡尔积的时候会有一边不走的情况, 这种是不被允许的, 因为只有控制相同才能连通
        #     if u_pi.__len__() == 0 or u_gamma.__len__() == 0:
        #         edge_to_remove.append(sync_edge_t)
        #
        #     # 要求控制相同
        #     # TODO
        #     # 如果控制序列分为可观和不可观, 则要求可观相同即可
        #     if (u_pi.__len__() and u_gamma.__len__()):
        #         intersection_t = set(u_pi[0].keys()) & set(u_gamma[0].keys())
        #         if not intersection_t:
        #             edge_to_remove.append(sync_edge_t)
        #
        #     # 要求观测相同
        #     # TODO
        #     # 其实点边观测可以一起做
        #     o_curr_state_pi = observation_func(current_state_pi[0])
        #     o_curr_state_gamma = observation_func(current_state_gamma[0])
        #     o_next_state_pi = observation_func(next_state_pi[0])
        #     o_next_state_gamma = observation_func(next_state_gamma[0])
        #     if o_curr_state_pi == None or o_curr_state_gamma == None or o_next_state_pi == None or o_next_state_gamma == None:
        #         raise TypeError
        #
        #     if o_curr_state_pi != o_curr_state_gamma:
        #         node_to_remove.append(sync_edge_t[0])
        #     if o_next_state_pi != o_next_state_gamma:
        #         node_to_remove.append(sync_edge_t[1])
        #
        #     # AP
        #     ap_state = get_ap_from_product_mdp(current_state_pi)
        #     ap_obs_state = get_ap_from_product_mdp(current_state_gamma)
        #     ap_state_p = get_ap_from_product_mdp(current_state_pi)
        #     ap_obs_state_p = get_ap_from_product_mdp(current_state_gamma)
        #     if ap_pi in ap_state:
        #         if not ap_gamma in ap_obs_state:
        #             node_to_remove.append(ap_pi)
        #
        #     if ap_pi in ap_state_p:
        #         if not ap_gamma in ap_obs_state_p:
        #             node_to_remove.append(ap_gamma)
        #
        #
        #
        # node_to_remove = list(set(node_to_remove))
        # edge_to_remove = list(set(edge_to_remove))
        #
        # for node_t in node_to_remove:
        #     try:
        #         sync_subgraph.remove_node(node_t)
        #     except:
        #         pass
        # for edge_t in edge_to_remove:
        #     try:
        #         sync_subgraph.remove_edge(edge_t)
        #     except:
        #         pass


        #
        #
        sync_mec_t = DiGraph()
        # for state_t in stack_t:
        #     sync_mec_t.add_node(state_t)
        while stack_t.__len__():
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            if current_state[0][0] == '0' and current_state[1][0] == '0':           # for debugging: current_state[0][0] == '2' and current_state[1][0] == '3'
                print_c("var 1")

            #
            next_state_list_pi    = list(self.out_edges(current_state[0], data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1], data=True))
            #
            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    #
                    next_state_pi    = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)

                    if next_state_pi[0] == '2' and next_state_gamma[0] == '3':      # for debugging
                        print_c("var 2")

                    #
                    u_pi    = list(edge_t_pi[2]['prop'].keys())[0]
                    u_gamma = list(edge_t_gamma[2]['prop'].keys())[0]
                    if type(u_pi) == tuple:
                        u_pi = u_pi[0]
                    if type(u_gamma) == tuple:
                        u_gamma = u_gamma[0]
                    #

                    if ctrl_obs_dict == None and u_pi != u_gamma:
                        continue
                    elif u_pi != u_gamma and ctrl_obs_dict[u_pi] == True and ctrl_obs_dict[u_gamma] == True:
                        continue
                    #
                    is_next_state_pi_in_amec    = next_state_pi    in mec_state_set_pi
                    is_next_state_gamma_in_amec = next_state_gamma in mec_state_set_gamma
                    if not is_next_state_pi_in_amec or not is_next_state_gamma_in_amec:
                        continue

                    # forall ap in pi
                    # exists ap in gamma
                    # TODO
                    # 到底是建立过程中删除, 还是建立后统一删除
                    # ap_list_t = list(next_state_pi[1])
                    # if ap_list_t.__len__() == 0:
                    #     ap_t = None
                    # else:
                    #     ap_t = list(next_state_pi[1])[0]
                    # ap_list_p_t = list(next_state_gamma[1])
                    # if ap_list_p_t.__len__() == 0:
                    #     ap_p_t = None
                    # else:
                    #     ap_p_t = list(next_state_gamma[1])[0]
                    # if ap_t == ap_pi:
                    #     if not ap_p_t == ap_gamma:
                    #         continue

                    #
                    # if observation_func(next_state_pi) == observation_func(next_state_gamma):         # moved to below
                    if True:
                        # TODO
                        # it seems that if the AP are required identical, then the AP-none_AP states will NOT taken into consideration
                        # in other words, the successive states will be synchronized by AP-AP states
                        #
                        # ON THE OTHER HAND
                        # on the aspect of the intruders, the only information he can receive is the observation sequences
                        # so the sync-MEC (, or the observer) should focus only on the observation sequences
                        # NOT the aps
                        #
                        # 2025.1.29
                        # Trick:
                        # Use small probability to maintain connectivity
                        #
                        #if is_ap_identical(next_state_pi, next_state_gamma):
                        if True:

                            #
                            trans_pr_cost_list = dict()
                            diff_expected_cost_list = dict()
                            for current_action_t in edge_t_pi[2]['prop'].keys():
                                #
                                transition_prop = edge_t_pi[2]['prop'][current_action_t][0]
                                transition_cost = edge_t_pi[2]['prop'][current_action_t][1]
                                if observation_func(next_state_pi[0]) == observation_func(next_state_gamma[0]):
                                    diff_expected_cost = obtain_differential_expected_cost(current_action_t, edge_t_pi, edge_t_gamma)
                                else:
                                    #
                                    # maintain connectivity
                                    diff_expected_cost = 1.e6
                                    #
                                trans_pr_cost_list[current_action_t] = (transition_prop, transition_cost, )
                                diff_expected_cost_list[current_action_t] = diff_expected_cost

                            #
                            sync_mec_t.add_edge(current_state, next_sync_state, prop=trans_pr_cost_list, diff_exp=diff_expected_cost_list)
                            stack_t.append(next_sync_state)
                            stack_t = list(set(stack_t))
                            stack_t.sort()
        #
        print_c("[synthesize_w_opacity] DFS completed, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),), color=33)
        if sync_mec_t.edges().__len__():
            #
            # 检查连接性
            scc_list = list(nx.strongly_connected_components(sync_mec_t))
            scc_list.sort(key=sort_scc_list,reverse=True)
            if scc_list.__len__() == 0:
                print_c("[synthesize_w_opacity] NO SCCs found ..." , color=33)

            num_node_removed = 0
            for i, state_list_t in enumerate(scc_list):
                if i == 0:
                    continue
                for state_t in state_list_t:
                    try:
                        sync_mec_t.remove_node(state_t)
                        num_node_removed += 1
                        print_c("[synthesize_w_opacity] removing node: " + str(state_t), color=35)
                    except:
                        pass

            # 然后去除其中不满足观测条件的点
            node_to_remove = []
            for node_t in sync_mec_t:
                ap_list_t = list(node_t[0][1])
                if ap_list_t.__len__() == 0:
                    ap_t = None
                else:
                    ap_t = list(node_t[0][1])[0]
                ap_list_p_t = list(node_t[1][1])
                if ap_list_p_t.__len__() == 0:
                    ap_p_t = None
                else:
                    ap_p_t = list(node_t[1][1])[0]
                if ap_t == ap_pi:
                    if not ap_p_t == ap_gamma:
                        node_to_remove.append(node_t)
            for node_t in node_to_remove:
                try:
                    sync_mec_t.remove_node(node_t)
                    print_c("[synthesize_w_opacity] removing node for inconsistent ap: " + str(node_t), color=35)
                except:
                    pass

            print_c("[synthesize_w_opacity] number of state to remove: %d" % (num_node_removed,), color=33)

            #
            # for debugging
            # state_pi_list_in_amec = []
            # for sync_state_t in (sync_mec_t.nodes()):
            #     state_pi_list_in_amec.append(sync_state_t[0])
            # #
            # state_pi_list_not_in_amec = []
            # for state_pi_t in MEC_pi[0]:
            #     if state_pi_t not in state_pi_list_in_amec:
            #         state_pi_list_not_in_amec.append(state_pi_t)

            #
            self.sync_amec_set.append(sync_mec_t)
            self.current_sync_amec_index = self.sync_amec_set.__len__() - 1
        #
        print_c("[synthesize_w_opacity] Generated sync_amec, states: %d, edges: %d" % (sync_mec_t.nodes.__len__(), sync_mec_t.edges.__len__(),))

    def re_synthesize_sync_amec_rex2(self, y_in_sf_pi, y_in_sf_gamma, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set
        #
        mec_state_set_pi = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = find_initial_state(y_in_sf_pi, y_in_sf_gamma, list(MEC_gamma[0]), observation_func=observation_func)
        stack_t = list(set(stack_t))
        visited = set()

        sync_mec_t = DiGraph()
        while stack_t:
            current_state = stack_t.pop()
            if current_state in visited:
                continue
            visited.add(current_state)

            next_state_list_pi = list(self.out_edges(current_state[0], data=True))
            next_state_list_gamma = list(product_mdp_gamma.out_edges(current_state[1], data=True))

            for edge_t_pi in next_state_list_pi:
                for edge_t_gamma in next_state_list_gamma:
                    next_state_pi = edge_t_pi[1]
                    next_state_gamma = edge_t_gamma[1]
                    next_sync_state = (next_state_pi, next_state_gamma)

                    # 1. 检查控制观测同步
                    u_pi = edge_t_pi[2]['prop'].keys()[0] if edge_t_pi[2]['prop'] else None
                    u_gamma = edge_t_gamma[2]['prop'].keys()[0] if edge_t_gamma[2]['prop'] else None
                    if isinstance(u_pi, tuple):
                        u_pi = u_pi[0]
                    if isinstance(u_gamma, tuple):
                        u_gamma = u_gamma[0]

                    if u_pi != u_gamma:
                        if ctrl_obs_dict is None:
                            continue
                        if (str(u_pi) not in ctrl_obs_dict or str(u_gamma) not in ctrl_obs_dict or
                                (not ctrl_obs_dict[str(u_pi)] and not ctrl_obs_dict[str(u_gamma)])):
                            continue

                    # 2. 检查状态是否在 AMEC 中
                    if next_state_pi not in mec_state_set_pi or next_state_gamma not in mec_state_set_gamma:
                        continue

                    # 3. 观测同步检查
                    obs_pi = observation_func(next_state_pi[0])
                    obs_gamma = observation_func(next_state_gamma[0])
                    if obs_pi != obs_gamma:
                        continue  # 或赋予高代价：diff_expected_cost = 1.e6

                    # 4. AP 同步检查（如需）
                    # if not is_ap_identical(next_state_pi, next_state_gamma):
                    #     continue

                    # 添加转移
                    trans_pr_cost_list = {
                        action: (prob, cost)
                        for action, (prob, cost) in edge_t_pi[2]['prop'].items()
                    }
                    diff_expected_cost = obtain_differential_expected_cost(...)
                    sync_mec_t.add_edge(current_state, next_sync_state,
                                        prop=trans_pr_cost_list,
                                        diff_exp=diff_expected_cost)
                    stack_t.append(next_sync_state)
                    stack_t = list(set(stack_t))
                    stack_t.sort()


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
            u, m = act_by_plan(self, best_all_plan, current_state)
            X.append(mdp_state)
            PX.append(current_state)
            L.append(set(label))
            U.append(u)
            M.append(m)
            t += 1
        return X, L, U, M, PX

class Sync_AMEC(DiGraph):
    def init(self):
        pass

    def synthesize_from_prod_mdps(self, amec_pi, amec_gamma, mdp_pi:Product_Dra, mdp_gamma:Product_Dra):
        # find out all amecs
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set

        #
        # 1 find all proper initial states
        stack_t = []
        visited = []

        for state_pi_t in amec_pi[0]:
            for state_gamma_t in amec_gamma[0]:
                pass

    '''
    def synthesize_from_sync_mdp(self, prod_mdp, amec, amec_prime, initial_state, observation_function=is_with_identical_observation_2):
        # amec:
        #   [0] amec
        #   [1] amec ^ Ip
        #   [2] action set

        #
        # 1 find all proper initial states
        stack_t = []
        visited = []

        # 1.1 check the available state as initial sets
        #     初始状态如何确定? 由prefix可达?
        # 1.2 based on initial sets, find all states with identical observations
        for state_i in initial_state:
            for state_j in amec_prime[0]:
                if observation_function(state_i, state_j):
                    stack_t.append((state_i, state_j))

        while stack_t.__len__():
            current_state = stack_t.pop()           # (state_in_amec, ref_state_in_amec', )
            state_in_amec = current_state[0]
            state_in_amec_prime = current_state[1]

            #
            # 3. find successive states
            # conditions:
            #   a. successor state must be in AMEC
            #   b. there exists the corresponding successor state in AMEC', s.t. the observations in a. and b. are identical
            #   c. 定量关系最后放到求解中作为约束, 不在这里求解
            #   d. the state pair must NOT be visited
            #      if visited, then check whether the edge is visited
            for state_p_in_amec in prod_mdp.successors(state_in_amec):
                for state_p_in_amec_p in prod_mdp.successors(state_in_amec_prime):          # b.1
                    if state_p_in_amec in amec[0]:                                          # a
                        if observation_function(state_p_in_amec, state_p_in_amec_p):        # b.2
                            state_tuple_p = (state_p_in_amec, state_p_in_amec_p)
                            if state_tuple_p not in visited:
                                # add edge
                                # aa. obtain probability
                                # ab. calculate differential probability


                                # add to visited
                                visited.append(state_tuple_p)
                            else:
                                # to check the edge
                                # ba. the probability is identical
                                # bb. differential probability is identical
                                pass

                # edge information
                # a. transition probabilistic of states in AMEC
                # b. absolute differential transition probabilistic between two transitions

        #
        # remeber:
        #   finally, only the AMEC with proper structure can be applied for suffix synthesis
        #   otherwise, a modified version of synthesis method must be given
        return None
    '''