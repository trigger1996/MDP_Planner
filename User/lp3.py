#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pickle
import networkx as nx

from collections import defaultdict
from ortools.linear_solver import pywraplp

from MDP_TG.dra import Dra, Product_Dra
from User.dra3  import product_mdp3, project_sync_states_2_observer_states, project_sync_mec_3_2_observer_mec_3
from User.utils import ltl_convert
from User.vis2  import print_c

from rich.progress import Progress, track

import sys
stack_size_t = 500000
sys.setrecursionlimit(stack_size_t)         # change stack size to guarantee runtime stability, default: 3000
print_c("stack size changed %d ..." % (stack_size_t, ))
#
sys.setrecursionlimit(100000)               # 设置递归深度为100000

def find_states_satisfying_opt_prop(opt_prop, Se):
    S_pi = []
    for s in Se:
        if opt_prop in s[1]:
            S_pi.append(s)
    return S_pi

def exp_weight(u, v, d):
    val_list = []
    for u_t in d['prop'].keys():
        val_t = d['prop'][u_t][0] * d['prop'][u_t][1]
        val_list.append(val_t)
    return min(val_list)

def syn_plan_prefix_in_sync_amec(prod_mdp, initial_subgraph, initial_sync_state, sync_amec_graph, sync_amec_3, observer_mec, gamma):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    # ----with bounded risk and minimal expected total cost----
    print("===========[plan prefix synthesis starts]===========")
    #
    # sf对应MEC全集
    # ip对应MEC和DRA中接收集和MEC的交集
    sf = observer_mec[0]
    ip = observer_mec[1]  # force convergence to ip
    sf_sync_state = sync_amec_3[0]
    ip_sync_state = sync_amec_3[1]
    delta = 0.01

    for init_node in initial_sync_state:        # prod_mdp.graph['initial']
        #
        # Compute the shortest path between source and all other nodes reachable from source.
        path_init = nx.single_source_shortest_path(initial_subgraph, init_node)
        print_c("[Prefix Synthesis] Reachable from init size: %s" % len(list(path_init.keys())), color='blue')
        #
        # 能否到达当前MEC
        if not set(path_init.keys()).intersection(sf):
            print_c("[Prefix Synthesis] Initial node can not reach sf", color='red')
            return None, None, None, None, None, None
        #
        # path_init.keys(): 路径的初态, 和sf的差集, 求解可以到达MEC, 但是初态不在MEC内的状态
        Sn = set(path_init.keys()).difference(sf)
        # ----find bad states that can not reach MEC
        #simple_digraph = nx.DiGraph()
        #simple_digraph.add_edges_from(((v, u) for u, v in sync_amec_graph.edges()))                # 原product_mdp所有的边组成的图
        #simple_digraph.add_edges_from(((v, u) for u, v in initial_subgraph.edges()))               # TODO
        #
        # ip <- MEC[1] 这个东西应该是MEC本身的状态
        # 之所以可以用随机状态，是因为MEC内的状态是可以互相到达的，所以只要一个能到剩下都能到
        Sd = set()
        Sr = set()
        Sr_good = set()
        Sr_bad  = set()
        for observer_state_t in ip_observer:
            try:
                #path_inv = single_source_shortest_path(simple_digraph, state_ip_t)                      # path = single_source_shortest_path(simple_digraph, random.sample(ip, 1)[0])                                     # 为什么这边要随机初始状态?
                path = nx.single_target_shortest_path(initial_subgraph, target=observer_state_t)         # 哦其实原来的代码是对的, 上面建立的是一个反向图
            except nx.NetworkXError:
                continue
        for state_ip_t in ip:
            try:
                #
                # Added
                path_p = nx.single_target_shortest_path(sync_amec_graph, target=state_ip_t)
            except nx.NetworkXError:
                continue
            reachable_set = set(path.keys())
            print('States that can reach sf, size: %s' % str(len(reachable_set)))
            Sd = Sn.difference(reachable_set)                                                   # Sn \ { 可达状态 } -> 不可以到达MEC的状态,  可以由初态s0到达, 但不可到达MEC的状态
            Sr = Sn.intersection(reachable_set)                                                 # Sn ^ { 可达状态 } -> 可以到达MEC的所有状态, 论文里是所有可以由s0到达的状态

            reachable_set_p = project_sync_states_2_observer_states(initial_subgraph, list(path_p.keys()))
            Sr_good = Sn.intersection(set(reachable_set_p))

            # Added
            if Sr_good.__len__() and Sr.__len__():
                break
            else:
                continue

        if not Sr.__len__():
            print_c("[Warning] initial states are not compatible with Ip ...", color='yellow')
            Sr.add(init_node)

        if not Sr_good.__len__():
            print_c("[Warning] initial GOOD states are not compatible with Ip ...", color='yellow')
            #return None, None, None, None, None, None
            # TODO
            # 问题： 初始状态在AMEC内
            # PLAN A 将初始状态加入待求解, 这样能解但是会多个点
            # PLAN B 直接不要Prefix, 在suffix内处理这个状态, 但但是会怕其他问题
            #
            # PLAN A
            Sr_good.add(init_node)

        Sr_bad = Sr.difference(Sr_good)

        # #--------------
        print('Sn size: %s; Sd inside size: %s; Sr inside size: %s' %
              (len(Sn), len(Sd), len(Sr)))
        # ---------solve lp------------
        print('-----')
        print('ORtools for prefix starts now')
        print('-----')
        #try:
        if True:                    # Added, for debugging
            # if True:
            Y = defaultdict(float)
            prefix_solver = pywraplp.Solver.CreateSolver('GLOP')
            # create variables
            for s in Sr:

                act_pi_list = []
                for sync_u, sync_v, attr in initial_subgraph.edges(s, data=True):
                    # if not attr['is_opacity']:
                    #     continue
                    act_pi_list += list(attr['prop'].keys())
                act_pi_list = list(set(act_pi_list))
                for u in act_pi_list:
                    Y[(s, u)] = prefix_solver.NumVar(
                        0, 1000, 'y[(%s, %s)]' % (s, u))  # 下界，上界，名称
                #
                # for u in initial_subgraph.nodes[s]['act'].copy():
                #     Y[(s, u)] = prefix_solver.NumVar(
                #         0, 1000, 'y[(%s, %s)]' % (s, u))        # 下界，上界，名称
            print('Variables added')
            # set objective
            obj = 0
            for s in Sr:
                for t in initial_subgraph.successors(s):
                    is_opacity = initial_subgraph[s][t]['is_opacity']
                    # if not is_opacity:
                    #     continue
                    #
                    # s -> t, \forall s \in Sr 相当于这里直接把图给进去了?
                    prop = initial_subgraph[s][t]['prop'].copy()
                    for u in prop.keys():
                        pe = prop[u][0]
                        ce = prop[u][1]
                        #
                        # \mu * 状态转移概率p_E * 代价c_E
                        # 1 如果Y[(s, u)] = 0, 是不是所有的都到最小值了?
                        #   所以需要in_flow和out_flow的约束, 而且必须满足policy之和为1, 即所有策略必须选一个
                        # 2 解的历史相关性
                        #   a. 根据Principle of model checking里的结论, 历史无关的策略能做到和历史有关策略一样的最优性？
                        #      （Lemma 10.102)
                        #   b. 这里的Y[(s, u)]其实是对整体进行求解, 一边动了根据约束条件另一边也得动
                        #      是不是这样也能理解为是一种memoryless policy
                        obj += Y[(s, u)]*pe*ce
            prefix_solver.Minimize(obj)
            print('Objective function set')
            # add constraints
            # ------------------------------
            y_to_sd = 0.0
            y_to_sf = 0.0
            for s in Sr:
                for t in initial_subgraph.successors(s):
                    # is_opacity = initial_subgraph[s][t]['is_opacity']
                    # if not is_opacity:
                    #     continue
                    if t in Sd:
                        prop = initial_subgraph[s][t]['prop'].copy()
                        for u in prop.keys():
                            pe = prop[u][0]
                            y_to_sd += Y[(s, u)]*pe                                 # Sd: 可由s_0到达, 但不可以到达MEC的状态集合
                    elif t in sf:
                        prop = initial_subgraph[s][t]['prop'].copy()
                        for u in prop.keys():
                            pe = prop[u][0]
                            y_to_sf += Y[(s, u)]*pe                                 # Sf <- MEC[0]
                    else:
                        # TODO, critical
                        # 如果这些状态被搞成round robin那就
                        #print(233)
                        pass
            # prefix_solver.Add(y_to_sf+y_to_sd >= delta)                           # old result
            #
            # delta = 0.01, 松弛变量?
            # Sr: 可由s_0到达的状态
            # y_to_sf = \sum_{s \in Sr} \mu(s, u) * pe, 就是这一条路径的概率
            # y_to_sd : 到不了MEC的概率
            # 所以y_to_sf + y_to_sd是所有概率之和
            # 所以不等式可以理解为: 到达MEC的概率 >= (1 - gamma) * 从s_0出发的所有概率
            # 这么干是make sense的
            # 但是为什么不能 y_to_sf >= 1 - gamma - delta?
            prefix_solver.Add(y_to_sf >= (1.0-gamma-delta)*(y_to_sf+y_to_sd))
            print_c('Risk constraint added')
            # --------------------
            for s in Sr:
                node_y_in = 0.
                node_y_out = 0.
                if s in Sr_good:
                    act_pi_list = []
                    for sync_u, sync_v, attr in initial_subgraph.edges(s, data=True):
                        # if not attr['is_opacity']:
                        #     continue
                        act_pi_list += list(attr['prop'].keys())
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        node_y_out += Y[(s, u)]
                    #
                    # for u in initial_subgraph.nodes[s]['act']:
                    #     node_y_out += Y[(s, u)]
                    #
                    for f in initial_subgraph.predecessors(s):
                        if f in Sr:
                            prop = initial_subgraph[f][s]['prop'].copy()
                            # is_opacity = initial_subgraph[f][s]['is_opacity']
                            # if not is_opacity:
                            #     continue
                            for uf in prop.keys():
                                node_y_in += Y[(f, uf)] * prop[uf][0]
                    #
                    # 对应论文中公式 (8c)
                    # 其实可以理解为, 初始状态in_flow就是1? node_y_in = 0
                    if s == init_node:
                        prefix_solver.Add(node_y_out == 1.0 + node_y_in)
                        print_c("[prefix solver] Initial flow constraint added: " + str(node_y_in) + " + 1.0 == " + str(node_y_out), color='blue')
                    else:
                        prefix_solver.Add(node_y_out == node_y_in)
                        print_c("[prefix solver] Middle flow constraint added: " + str(node_y_in) + " == " + str(node_y_out), color='green')
                elif s in Sr_bad:
                    act_pi_list = []
                    for sync_u, sync_v, attr in initial_subgraph.edges(s, data=True):
                        # if not attr['is_opacity']:
                        #     continue
                        act_pi_list += list(attr['prop'].keys())
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        node_y_out += Y[(s, u)]
                    #
                    # for u in initial_subgraph.nodes[s]['act']:
                    #     node_y_out += Y[(s, u)]
                    #
                    for f in initial_subgraph.predecessors(s):
                        if f in Sr:
                            prop = initial_subgraph[f][s]['prop'].copy()
                            # is_opacity = initial_subgraph[f][s]['is_opacity']
                            # if not is_opacity:
                            #     continue
                            for uf in prop.keys():
                                node_y_in += Y[(f, uf)] * prop[uf][0]

                    if (type(node_y_in) != float and type(node_y_out) != float) or (type(node_y_in) != float and node_y_out != 0.0) or (type(node_y_out) != float and node_y_in != 0.0):
                        if s == init_node:
                            prefix_solver.Add(node_y_in == 1.0 + node_y_out)
                            print_c("[prefix solver] Recovery constraint added: " + str(node_y_in) + " ==  1.0 + " + str(node_y_out), color='magenta')
                        else:
                            prefix_solver.Add(node_y_in == node_y_out)
                            print_c("[prefix solver] Recovery constraint added: " + str(node_y_in) + " == " + str(node_y_out), color='magenta')
            print_c('Recovery constraint added')

            print('Initial node flow balanced')
            print('Middle node flow balanced')
            # ----------------------
            # solve
            print('--optimization starts--')
            #
            # 求解在这里
            status = prefix_solver.Solve()
            #
            # for debugging
            Y_val = dict()
            for s_u in Y.keys():
                Y_val[s_u] = Y[s_u].solution_value()
            #
            if status == pywraplp.Solver.OPTIMAL:
                print('Solution:')
                print('Objective value =', prefix_solver.Objective().Value())
                print('Advanced usage:')
                print('Problem solved in %f milliseconds' %
                      prefix_solver.wall_time())
                print('Problem solved in %d iterations' %
                      prefix_solver.iterations())
            else:
                print('The problem does not have an optimal solution.')
                return None, None, None, None, None, None
            #
            # 这后面是输出了应该
            #
            # plan
            plan_prefix = dict()
            for s in Sr:
                norm = 0
                U = []
                P = []
                # U_total = initial_subgraph.nodes[s]['act'].copy()
                U_total = list()
                for sync_u, sync_v, attr in initial_subgraph.edges(s, data=True):
                    #
                    # 这边就不用is_opacity了
                    U_total += list(attr['prop'].keys())
                    #
                U_total = list(set(U_total))
                U_total.sort()
                for u in U_total:
                    if type(Y[(s, u)]) == float:
                        norm += 0.                              # now, Y[(s, u)] is not considered as a feasible solution for opacity
                    else:
                        norm += Y[(s, u)].solution_value()
                #
                # TODO
                if norm > 0.01:
                    for u in U_total:
                        U.append(u)
                        if type(Y[(s, u)]) != float:
                            P.append(Y[(s, u)].solution_value()/norm)
                else:
                    path_t = nx.single_source_shortest_path(initial_subgraph, s)
                    reachable_set_t = set(path_t).intersection(set(sf))
                    dist_val_dict = {}
                    for tgt_t in reachable_set_t:
                        dist_val_dict[tgt_t] = nx.shortest_path_length(initial_subgraph, s, tgt_t,
                                                                             weight=exp_weight)
                    #
                    min_dist_target = min(dist_val_dict, key=dist_val_dict.get)
                    #
                    if len(path_t[min_dist_target]) > 1:
                        successor_state = path_t[min_dist_target][1]
                        edge_data = initial_subgraph.edges[s, successor_state]
                        for u_p in edge_data['prop'].keys():
                            U.append(u_p)
                            P.append(1.0 / len(edge_data['prop'].keys()))
                        for u_p in U_total:
                            if u_p in edge_data['prop'].keys():
                                continue
                            U.append(u_p)
                            P.append(0.)
                    else:
                        # the old round_robin
                        U.append(U_total)
                        P.append(1.0 / len(U_total))
                #
                # for u in U_total:
                #     U.append(u)
                #     if type(Y[(s, u)]) != float:
                #         if norm > 0.01:
                #             P.append(Y[(s, u)].solution_value()/norm)
                #         else:
                #             P.append(1.0/len(U_total))
                #     else:
                #         P.append(0.)
                plan_prefix[s] = [U, P]
            print("----Prefix plan generated")
            cost = prefix_solver.Objective().Value()
            print("----Prefix cost computed, cost: %.2f" % cost)
            # compute the risk given the plan prefix
            risk = 0.0
            y_to_sd = 0.0
            y_to_sf = 0.0
            for s in Sr:
                for t in initial_subgraph.successors(s):
                    if t in Sd:
                        prop = initial_subgraph[s][t]['prop'].copy()
                        for u in prop.keys():
                            pe = prop[u][0]
                            if type(Y[(s, u)]) != float:
                                y_to_sd += Y[(s, u)].solution_value()*pe
                            else:
                                y_to_sd += 0.0
                    elif t in sf:
                        prop = initial_subgraph[s][t]['prop'].copy()
                        for u in prop.keys():
                            pe = prop[u][0]
                            if type(Y[(s, u)]) != float:
                                y_to_sf += Y[(s, u)].solution_value()*pe
                            else:
                                y_to_sd += 0.0
            if (y_to_sd+y_to_sf) > 0:
                risk = y_to_sd/(y_to_sd+y_to_sf)
            print('y_to_sd: %s; y_to_sf: %s, y_to_sd+y_to_sf: %s' %
                  (y_to_sd, y_to_sf, y_to_sd+y_to_sf))
            print("----Prefix risk computed: %s" % str(risk))
            # compute the input flow to the suffix
            y_in_sf = dict()                                            # Sn -> Sf, 从S0能到达但不在AMEC的状态, 到达AMEC的状态, 其中keys为AMEC的状态
            for s in Sn:
                for t in initial_subgraph.successors(s):
                    if t in sf:
                        prop = initial_subgraph[s][t]['prop'].copy()
                        for u in prop.keys():
                            if type(Y[(s, u)]) == float:
                                continue                                        # TODO
                            pe = prop[u][0]
                            if t not in y_in_sf:
                                y_in_sf[t] = Y[(s, u)].solution_value()*pe      # TODO 概率对吗, 核心在于, 此时的不可达区域除去不可到达AMEC区域的，还有一部分属于AMEC但是不属于opacity的坏状态
                            else:
                                y_in_sf[t] += Y[(s, u)].solution_value()*pe
            # normalize the input flow
            y_total = 0.0
            for s, y in y_in_sf.items():
                y_total += y
            print('actual y_total: %s' % str(y_total))
            if y_total > 0:
                for s in y_in_sf.keys():
                    y_in_sf[s] = y_in_sf[s]/y_total
                print("----Y in Sf computed and normalized")
            # print y_in_sf
            return plan_prefix, cost, risk, y_in_sf, Sr, Sd
        # except:
        #     print("ORTools Error reported")
        #     return None, None, None, None, None, None

def synthesize_full_plan_w_opacity3(mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, observation_func,
                                    ctrl_obs_dict, alpha=1, is_enable_inter_state_constraints=True):
    t2 = time.time()

    task_pi = task + ' & GF ' + optimizing_ap
    ltl_converted_pi = ltl_convert(task_pi)

    dra = Dra(ltl_converted_pi)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3 - t2))

    # ----
    prod_dra_pi = product_mdp3(mdp, dra)
    prod_dra_pi.compute_S_f()  # for AMECs, finished in building
    # prod_dra.dotify()
    t41 = time.time()
    print('Product DRA done, time: %s' % str(t41 - t3))

    pickle.dump((nx.get_edge_attributes(prod_dra_pi, 'prop'),
                 prod_dra_pi.graph['initial']), open('prod_dra_edges.p', "wb"))
    print('prod_dra_edges.p saved')

    # new main loop
    for l, S_fi_pi in enumerate(prod_dra_pi.Sf):  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC_pi in enumerate(S_fi_pi):
            #
            # finding states that satisfying optimizing prop
            S_pi = find_states_satisfying_opt_prop(optimizing_ap, MEC_pi[0])

            for ap_4_opacity in ap_list:
                if ap_4_opacity == optimizing_ap:
                    continue
                # synthesize product mdp for opacity
                task_gamma = task + ' & GF ' + ap_4_opacity
                ltl_converted_gamma = ltl_convert(task_gamma)
                dra = Dra(ltl_converted_gamma)
                prod_dra_gamma = Product_Dra(mdp, dra)
                prod_dra_gamma.compute_S_f()  # for AMECs

                #
                for p, S_fi_gamma in enumerate(prod_dra_gamma.Sf):
                    for q, MEC_gamma in enumerate(S_fi_gamma):
                        prod_dra_pi.re_synthesize_sync_amec(optimizing_ap, ap_4_opacity, MEC_pi, MEC_gamma,
                                                             prod_dra_gamma, observation_func=observation_func,
                                                             ctrl_obs_dict=ctrl_obs_dict)

                        sync_mec_t = prod_dra_pi.project_sync_amec_back_to_mec_pi(
                            prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index], MEC_pi)
                        if not sync_mec_t[1].__len__():
                            print_c("[prefix synthesis] AP: %s do not have a satisfying Ip in DRA!" % (ap_4_opacity,),
                                    color='yellow')
                            continue

                        initial_subgraph, initial_sync_state = prod_dra_pi.construct_opaque_subgraph_2_amec(
                                                                                    prod_dra_gamma,
                                                                                    sync_mec_t,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    MEC_pi, MEC_gamma,
                                                                                    optimizing_ap, ap_4_opacity, observation_func, ctrl_obs_dict)

                        observer_mec_3 = project_sync_mec_3_2_observer_mec_3(initial_subgraph, sync_mec_t)

                        plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync, Sr, Sd = syn_plan_prefix_in_sync_amec(
                                                                                    prod_dra_pi, initial_subgraph, initial_sync_state,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t, observer_mec_3, risk_pr)

                        opaque_full_graph = prod_dra_pi.construct_fullgraph_4_amec(initial_subgraph,
                                                                                    prod_dra_gamma,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    MEC_pi, MEC_gamma,
                                                                                    optimizing_ap, ap_4_opacity,
                                                                                    observation_func, ctrl_obs_dict)

                        plan_suffix, suffix_cost, suffix_risk, suffix_opacity_threshold = synthesize_suffix_cycle_in_sync_amec(
                                                                                    prod_dra_pi,
                                                                                    prod_dra_pi.mec_observer_set[prod_dra_pi.current_sync_amec_index],
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t,
                                                                                    y_in_sf_sync,
                                                                                    opaque_full_graph,  # 用来判断Sn是否可达, 虽然没啥意义但是还是可以做,
                                                                                    initial_sync_state,
                                                                                    differential_exp_cost)

                        plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync],
                                     [plan_suffix, suffix_cost, suffix_risk],
                                     [MEC_pi[0], MEC_pi[1], Sr, Sd],
                                     [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index,
                                      MEC_gamma[0],
                                      MEC_gamma[1]],
                                     prod_dra_pi])

        if plan:
            print("=========================")
            print(" || Final compilation  ||")
            print("=========================")
            best_all_plan = min(plan, key=lambda p: p[0][1] + alpha * p[1][1])
            prod_dra_pi = best_all_plan[4]
            best_all_plan = best_all_plan[0:4]
            print('Best plan prefix obtained for %s states in Sr' %
                  str(len(best_all_plan[0][0])))
            print('cost: %s; risk: %s ' %
                  (best_all_plan[0][1], best_all_plan[0][2]))
            print('Best plan suffix obtained for %s states in Sf' %
                  str(len(best_all_plan[1][0])))
            print('cost: %s; risk: %s ' %
                  (best_all_plan[1][1], best_all_plan[1][2]))
            print('Total cost:%s' %
                  (best_all_plan[0][1] + alpha * best_all_plan[1][1]))
            print_c('Opacity threshold %f <= %f' % (best_all_plan[3][1], differential_exp_cost,))
            #
            # TODO
            '''
            plan_bad = syn_plan_bad(prod_mdp, best_all_plan[2])
            print('Plan for bad states obtained for %s states in Sd' %
                  str(len(best_all_plan[2][3])))
            best_all_plan.append(plan_bad)
            '''
            return best_all_plan, prod_dra_pi
        else:
            print("No valid plan found")
            return None
#
