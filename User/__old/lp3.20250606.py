#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pickle
import networkx as nx
import numpy    as np

from collections import defaultdict
from ortools.linear_solver import pywraplp

from MDP_TG.dra import Dra, Product_Dra
from User.dra3  import product_mdp3, project_sync_states_2_observer_states, project_observer_state_2_sync_state, project_sync_mec_3_2_observer_mec_3
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


def find_scc_with_known_states(
        G: nx.MultiDiGraph,
        known_states: set,
        match_type: str = "partial"
) -> list:
    """
    查找图中包含指定状态集合的强连通分量。

    参数:
        G: 有向图 (nx.MultiDiGraph)
        known_states: 已知状态集合（节点名集合）
        match_type: 匹配类型，可选：
            - "partial": 至少包含 known_states 中一个节点（默认）
            - "full": 包含所有 known_states 中的节点
            - "exact": 分量和 known_states 完全一致

    返回:
        list[set]：满足条件的强连通分量列表，每个分量是一个节点集合
    """
    sccs = list(nx.strongly_connected_components(G))

    if match_type == "partial":
        matched = [scc for scc in sccs if known_states & scc]
    elif match_type == "full":
        matched = [scc for scc in sccs if known_states <= scc]
    elif match_type == "exact":
        matched = [scc for scc in sccs if known_states == scc]
    else:
        raise ValueError(f"未知匹配类型: {match_type}，应为 'partial'、'full' 或 'exact'")

    return matched

def get_action_from_successor_edge(g, s):
    act_list = []
    for sync_u, sync_v, attr in g.edges(s, data=True):
        act_list += list(attr['prop'].keys())
    act_list = list(set(act_list))
    act_list.sort()
    return act_list

def exp_weight(u, v, d):
    val_list = []
    for key_t in d:
        for u_t in d[key_t]['prop'].keys():
            val_t = d[key_t]['prop'][u_t][0] * d[key_t]['prop'][u_t][1]
            val_list.append(val_t)
    return min(val_list)

def print_analyze_constraints_matrix_form(solver):
    variables = solver.variables()
    constraints = solver.constraints()

    # 创建 A 和 b 的零矩阵
    num_vars = len(variables)
    num_constraints = len(constraints)

    A = np.zeros((num_constraints, num_vars))
    b = np.zeros(num_constraints)

    print("\n=== 变量顺序（列）：===")
    for idx, var in enumerate(variables):
        print(f"  x[{idx}] = {var.name()}")

    print("\n=== 等式约束形式 Ax = b：===")

    for i, ct in enumerate(constraints):
        coeffs = []
        for j, var in enumerate(variables):
            coeff = ct.GetCoefficient(var)
            coeffs.append(coeff)
            A[i, j] = coeff  # 填充 A 矩阵
        rhs = ct.ub()  # 等式约束中 ub = lb = b
        b[i] = rhs  # 填充 b 向量
        equation = " + ".join(
            [f"{coeff:.3g}*x[{j}]" for j, coeff in enumerate(coeffs) if coeff != 0]
        )
        print(f"  Row {i}: {equation} = {rhs}")

    # 输出 A 和 b 的矩阵
    print("\n=== 约束矩阵 A 和常数向量 b：===")
    print("A = ")

    # 遍历 A 矩阵的每一行，并逐行输出，每个数字对齐
    for row in A:
        print("[" + ", ".join([f"{val:>8.3f}" for val in row]) + "]")

    print("\nb = ")
    # 输出 b 向量，每个数字对齐
    print("[" + ", ".join([f"{val:>8.3f}" for val in b]) + "]")

    # 计算矩阵 A 的秩
    rank_A = np.linalg.matrix_rank(A)
    print_c(f"\n矩阵 A 的秩为：{rank_A}", color='bg_cyan', style='bold')

    try:
        # 计算 A 的伪逆
        A_pseudo_inv = np.linalg.pinv(A)

        # 使用伪逆计算 x
        x = np.dot(A_pseudo_inv, b)
        print(x)
    except:
        print_c("Balance constraints may not be feasible ...", color='red')

def syn_plan_prefix_in_sync_amec(prod_mdp, initial_subgraph, initial_sync_state, sync_amec_graph, sync_amec_3, observer_mec_3, gamma):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    # ----with bounded risk and minimal expected total cost----
    print("===========[plan prefix synthesis starts]===========")
    #
    # sf对应MEC全集
    # ip对应MEC和DRA中接收集和MEC的交集
    sf = observer_mec_3[0]
    ip = observer_mec_3[1]  # force convergence to ip
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
        Sn = set(path_init.keys()).difference(sf)                                                   # Sr0 \backslash Sf: 可由s0到达, 但不属于MEC的状态
        # ----find bad states that can not reach MEC
        Sd = set()
        Sr = set()
        Sr_good = set()
        Sr_bad  = set()
        #
        for observer_state_t in ip:
            path = nx.single_target_shortest_path(initial_subgraph, target=observer_state_t)         # 可以到达product_mdp中可以到达ip的状态, 对应论文中Sc
            Sc = set(path.keys())

            sync_state_ip_list_t = project_observer_state_2_sync_state(sync_amec_3[0], [observer_state_t])
            path_p = dict()
            for sync_state_ip_t in sync_state_ip_list_t:
                path_p.update(nx.single_target_shortest_path(sync_amec_graph, target=sync_state_ip_t))

            print('States that can reach sf, size: %s' % str(len(Sc)))
            Sd = Sn.difference(Sc)                                                                   # Sn \backslash { Sc } -> 不在MEC内，可由s0到达的状态除去可以到达ip的状态, 这个就是论文的Sd
            Sr = Sn.intersection(Sc)                                                                 # Sn \cap       { Sc } -> 不在MEC内，可以s0到达且能到达ip的状态

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
                    for key_t in initial_subgraph[s][t]:
                        is_opacity = initial_subgraph[s][t][key_t]['is_opacity']
                        # if not is_opacity:
                        #     continue
                        #
                        # s -> t, \forall s \in Sr 相当于这里直接把图给进去了?
                        prop = initial_subgraph[s][t][key_t]['prop'].copy()
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
                    for key_t in initial_subgraph[s][t]:
                        # is_opacity = initial_subgraph[s][t][key_t]['is_opacity']
                        # if not is_opacity:
                        #     continue
                        if t in Sd:
                            prop = initial_subgraph[s][t][key_t]['prop'].copy()
                            for u in prop.keys():
                                pe = prop[u][0]
                                y_to_sd += Y[(s, u)]*pe                                 # Sd: 可由s_0到达, 但不可以到达MEC的状态集合
                        elif t in sf:
                            prop = initial_subgraph[s][t][key_t]['prop'].copy()
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
                        if not attr['is_opacity']:                                      # TODO, to check, 好的状态只让走满足opacity的边
                            continue
                        act_pi_list += list(attr['prop'].keys())
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        node_y_out += Y[(s, u)]
                    #
                    # for u in initial_subgraph.nodes[s]['act']:
                    #     node_y_out += Y[(s, u)]
                    #
                    for f in initial_subgraph.predecessors(s):
                        for key_t in initial_subgraph[f][s]:
                            if f in Sr:
                                prop = initial_subgraph[f][s][key_t]['prop'].copy()
                                is_opacity = initial_subgraph[f][s][key_t]['is_opacity']           # TODO, to check, 好的状态只让走满足opacity的边
                                if not is_opacity:
                                    continue
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
                        for u in attr['prop'].keys():
                            # 只有当这个动作通往的是 good 状态才保留
                            if sync_v in observer_mec_3[0]:                                  # TODO, to check
                                act_pi_list.append(u)
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        node_y_out += Y[(s, u)]
                    #
                    # for u in initial_subgraph.nodes[s]['act']:
                    #     node_y_out += Y[(s, u)]
                    #
                    for f in initial_subgraph.predecessors(s):
                        for key_t in initial_subgraph[f][s]:
                            if f in Sr:
                                prop = initial_subgraph[f][s][key_t]['prop'].copy()
                                # is_opacity = initial_subgraph[f][s][key_t]['is_opacity']
                                # if not is_opacity:
                                #     continue
                                for uf in prop.keys():                                          # TODO, to check, 所有坏状态的出都要回到sync_amec的对应状态
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
                        for key_t in initial_subgraph[s][successor_state]:
                            edge_data = initial_subgraph[s][successor_state][key_t]
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
                    for key_t in initial_subgraph[s][t]:
                        if t in Sd:
                            prop = initial_subgraph[s][t][key_t]['prop'].copy()
                            for u in prop.keys():
                                pe = prop[u][0]
                                if type(Y[(s, u)]) != float:
                                    y_to_sd += Y[(s, u)].solution_value()*pe
                                else:
                                    y_to_sd += 0.0
                        elif t in sf:
                            prop = initial_subgraph[s][t][key_t]['prop'].copy()
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
                        for key_t in initial_subgraph[s][t]:
                            prop = initial_subgraph[s][t][key_t]['prop'].copy()
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

def synthesize_suffix_cycle_in_sync_amec3(prod_mdp, sync_amec_graph, sync_mec_3, observer_mec_3, y_in_sf_sync, opaque_full_graph, initial_sync_state, differential_expected_cost=1.55):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print_c("===========[plan suffix synthesis starts]", color=32)
    print_c("[synthesize_w_opacity] differential exp cost: %f" % (differential_expected_cost, ), color=32)

    sf = observer_mec_3[0]                      # MEC
    ip = observer_mec_3[1]                      # MEC 和 ip 的交集
    act = observer_mec_3[2].copy()              # 所有状态的动作集合全集
    delta = 0.01                                # 松弛变量?
    gamma = 0.00                                # 根据(11), 整个系统进入MEC内以后就不用概率保证了?
    for init_node in initial_sync_state:

        # paths = nx.single_source_shortest_path(prod_mdp, init_node)                       # path.keys(): 可以通过初始状态到达的状态
        # Sn = set(paths.keys()).intersection(sf)                                           # Sn, 可以由初始状态到达的MEC状态(此时的sf都满足opacity requirement且强连通)
        reachable_sync_states = nx.single_source_shortest_path(opaque_full_graph, init_node)
        #
        # TODO
        mec_full_in_observer = find_scc_with_known_states(opaque_full_graph, set(y_in_sf_sync.keys()))
        mec_full_in_observer = mec_full_in_observer[0]                                      # 这里只取最大的那个

        Sn_good = set(reachable_sync_states.keys()).intersection(sf)                        # 满足Opacity requirement的MEC子集, 且满足强连通
        Sn_bad  = set(reachable_sync_states.keys()).intersection(mec_full_in_observer)      # 可能在运行过程中到达的坏状态
        Sn_bad  = Sn_bad.difference(Sn_good)
        Sn      = Sn_good.union(Sn_bad)                                                     # 相当于此时Sn是MEC_observer的状态全集
        print('Sf size: %s' % len(sf))
        print('reachable sf size: %s' % len(Sn))
        print('Ip size: %s' % len(ip))
        print('Ip and sf intersection size: %s' % len(Sn.intersection(ip)))
        # ---------solve lp------------
        print('------')
        print('ORtools for suffix starts now')
        print('------')
        #try:
        if True:
            Y = defaultdict(float)
            suffix_solver = pywraplp.Solver.CreateSolver('GLOP')
            # create variables
            #
            # 注意这里和prefix不同的是, prefix
            # prefix -> Sr:
            #       path = single_source_shortest_path(simple_MultiDiGraph, random.sample(ip, 1)[0]), 即可以到达MEC的状态
            #       Sn: path_init.keys() 和 Sf的交集, 两变Sf定义是一样的都是MEC
            #       Sr: path.keys() 和 Sn相交, 也就是从初始状态可以到达MEC的状态
            # suffix -> Sn:
            #       Sn = set(paths.keys()).intersection(sf)
            #       这里Sn和之前定义其实是一样的
            #       可由初态到达, 且可到达MEC但是不在MEC内
            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))
                for k, s in enumerate(Sn):
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    act_pi_list = []
                    for sync_u, sync_v, attr in opaque_full_graph.edges(s, data=True):
                        # if not attr['is_opacity']:
                        #     continue
                        act_pi_list += list(attr['prop'].keys())
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        Y[(s, u)] = suffix_solver.NumVar(
                            0, 1000, 'y[(%s, %s)]' % (s, u))  # 下界，上界，名称
                    # for u in act[s]:
                    #     Y[(s, u)] = suffix_solver.NumVar(0, 1000, 'y[(%s, %s)]' % (s, u))
            print('Variables added: %d' % len(Y))
            # set objective
            obj = 0
            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))
                for k, s in enumerate(Sn):
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    act_pi_list = []
                    for sync_u, sync_v, attr in opaque_full_graph.edges(s, data=True):
                        # if not attr['is_opacity']:
                        #     continue
                        act_pi_list += list(attr['prop'].keys())
                    act_pi_list = list(set(act_pi_list))
                    for u in act_pi_list:
                        for t in opaque_full_graph.successors(s):
                            for key_t in opaque_full_graph[s][t]:
                                prop = opaque_full_graph[s][t][key_t]['prop'].copy()
                                if u in list(prop.keys()):
                                    pe = prop[u][0]
                                    ce = prop[u][1]
                                    obj += Y[(s, u)]*pe*ce
            suffix_solver.Minimize(obj)
            print('Objective added')
            # add constraints
            # --------------------
            #
            # description of each constraints
            constr_descrip = []

            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))

                #
                for k, s in enumerate(Sn):
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    # for debugging
                    if s == (('0', frozenset({'upload'}), 2), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2)), ()):
                        debug_var = 1
                    if s == (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2), ('0', frozenset({'upload'}), 3)), ()):
                        debug_var = 2

                    #
                    # constr3: sum of outflow
                    # constr4: sum of inflow
                    constr3 = 0.
                    constr4 = 0.
                    if s in Sn_good:
                        act_s_list = get_action_from_successor_edge(opaque_full_graph, s)
                        # TODO
                        # 好状态不能走到坏状态
                        # 一个从好状态出发的动作有三种可能性
                        # 1. 到达一个好状态
                        # 2. 到达一个坏状态
                        # 3. 既可能到达好状态,  又可能到达坏状态
                        # 那么其实为了安全约束, 我们只能允许1？
                        # 如果解释不通, 那么则允许1和2？
                        for u in act_s_list:
                            for v, w, attr in opaque_full_graph.out_edges(s, data=True):
                                u_p_list = attr['prop'].keys()
                                if u in u_p_list:
                                    if w in Sn_good:                        # 好状态只能到达好状态
                                        constr3 += Y[(s, u)]
                                    elif w in Sn_bad:
                                        pass
                            # the old solution
                            #constr3 += Y[(s, u)]
                        for f in opaque_full_graph.predecessors(s):
                            for key_t in opaque_full_graph[f][s]:
                                if f == (('0', frozenset({'upload'}), 2), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2)), ()):
                                    debug_var = 3
                                if f == (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2), ('0', frozenset({'upload'}), 3)), ()):
                                    debug_var = 4
                                #
                                # 这里也有不同
                                # prefix
                                #       if f in Sr:
                                # suffix
                                #       if f in Sn 且 s in Sn 且 s not in ip
                                if (f in Sn_good) and (s not in ip):
                                    prop = opaque_full_graph[f][s][key_t]['prop'].copy()
                                    act_f_list = get_action_from_successor_edge(opaque_full_graph, f)
                                    for uf in act_f_list:
                                        if uf in list(prop.keys()):
                                            constr4 += Y[(f, uf)] * prop[uf][0]
                                        else:
                                            #constr4 += Y[(f, uf)] * 0.00
                                            pass
                                if (f in Sn_good) and (s in ip) and (f != s):
                                    prop = opaque_full_graph[f][s][key_t]['prop'].copy()
                                    act_f_list = get_action_from_successor_edge(opaque_full_graph, f)
                                    for uf in act_f_list:
                                        if uf in list(prop.keys()):
                                            constr4 += Y[(f, uf)] * prop[uf][0]
                                        else:
                                            #constr4 += Y[(f, uf)] * 0.00
                                            pass


                        # Added for debugging
                        if s in ip:
                            debug_var = 4

                        #
                        # 在debug的时候, 如果有约束条件不可解, 专心看rhs有没有相同的, 而非lhs, 因为lhs肯定都是和s挂钩所以都不一样
                        # 而且不可解多半是因为flow balance constraint
                        #
                        # y_in_sf input flow of sf, sf的输入状态转移概率
                        #     if t not in y_in_sf:
                        #         y_in_sf[t] = Y[(s, u)].solution_value()*pe
                        #     else:
                        #         y_in_sf[t] += Y[(s, u)].solution_value()*pe
                        #     最后还需要归一化
                        #     当单状态状态转移Sn -> sf时会被记录到Sf
                        #
                        # 如果 s in Sf且上一时刻状态属于Sn, 但不在MEC内
                        if (s in list(y_in_sf_sync.keys())) and (s not in ip):
                            suffix_solver.Add(constr3 == constr4 + y_in_sf_sync[s])  # 可能到的了的要计算?
                            #
                            # Added, for debugging
                            print_c("constraint: %d" % (k,), color=32)               # green / yellow
                            print_c(constr3, color=32)
                            print_c(constr4, color=33)
                            print_c(y_in_sf_sync[s], color=33)
                            print_c(" ")
                        #
                        # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                        if (s in list(y_in_sf_sync.keys())) and (s in ip):
                            suffix_solver.Add(constr3 == y_in_sf_sync[s])  # 在里面的永远到的了?
                            #
                            # Added, for debugging
                            print_c("constraint: %d" % (k,), color=34)
                            print_c(constr3, color=34)
                            print_c(y_in_sf_sync[s], color=35)
                            print_c(" ")

                        # 如果s不在Sf内且不在MEC内
                        if (s not in list(y_in_sf_sync.keys())) and (s not in ip):
                            suffix_solver.Add(constr3 == constr4)  # 到不了的永远到不了?
                            #
                            # Added, for debugging
                            print_c("constraint: %d" % (k,), color=36)
                            print_c(constr3, color=36)
                            print_c(constr4, color=37)
                            print_c(" ")                                            # cyan / gray
                    elif s in Sn_bad:
                        # TODO
                        # to recover
                        # 所有坏状态都要回到好状态
                        # f -> s
                        # 除了限制s, 也需要限制f?
                        act_pi_list = get_action_from_successor_edge(opaque_full_graph, s)
                        for u in act_pi_list:
                            constr3 += Y[(s, u)]
                        #
                        # for u in initial_subgraph.nodes[s]['act']:
                        #     node_y_out += Y[(s, u)]
                        #
                        for f in opaque_full_graph.predecessors(s):
                            for key_t in opaque_full_graph[f][s]:
                                if f == (('0', frozenset({'upload'}), 2), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2)), ()):
                                    debug_var = 5
                                if f == (('0', frozenset({'upload'}), 1), (('0', frozenset({'upload'}), 1), ('0', frozenset({'upload'}), 2), ('0', frozenset({'upload'}), 3)), ()):
                                    debug_var = 6

                                if f in Sn_good:
                                    prop = opaque_full_graph[f][s][key_t]['prop'].copy()
                                    for uf in prop.keys():
                                        suffix_solver.Add(Y[(f, uf)] * prop[uf][0] == 0)
                                elif f in Sn_bad:
                                    if f in Sn:
                                        prop = opaque_full_graph[f][s][key_t]['prop'].copy()
                                        for uf in prop.keys():
                                            constr4 += Y[(f, uf)] * prop[uf][0]

                                #
                                # the old solution
                                # if f in Sn:
                                #     prop = opaque_full_graph[f][s][key_t]['prop'].copy()
                                #     # is_opacity = initial_subgraph[f][s][key_t]['is_opacity']
                                #     # if not is_opacity:
                                #     #     continue
                                #     for uf in prop.keys():
                                #         constr4 += Y[(f, uf)] * prop[uf][0]

                        if (type(constr3) != float and type(constr4) != float) or (
                                type(constr3) != float and constr4 != 0.0) or (
                                type(constr4) != float and constr3 != 0.0):
                            if (s in list(y_in_sf_sync.keys())) and (s not in ip):
                                suffix_solver.Add(constr3 == constr4 + y_in_sf_sync[s])  # 可能到的了的要计算?
                                #
                                # Added, for debugging
                                print_c("[sf bad] constraint: %d" % (k,), color=32)  # green / yellow
                                print_c(constr3, color=32)
                                print_c(constr4, color=33)
                                print_c(y_in_sf_sync[s], color=33)
                                print_c(" ")
                            #
                            # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                            if (s in list(y_in_sf_sync.keys())) and (s in ip):
                                suffix_solver.Add(constr3 == y_in_sf_sync[s])  # 在里面的永远到的了?
                                #
                                # Added, for debugging
                                print_c("[sf bad] constraint: %d" % (k,), color=34)
                                print_c(constr3, color=34)
                                print_c(y_in_sf_sync[s], color=35)
                                print_c(" ")

                            # 如果s不在Sf内且不在MEC内
                            if (s not in list(y_in_sf_sync.keys())) and (s not in ip):
                                suffix_solver.Add(constr3 == constr4)  # 到不了的永远到不了?
                                #
                                # Added, for debugging
                                print_c("[sf bad] constraint: %d" % (k,), color=36)
                                print_c(constr3, color=36)
                                print_c(constr4, color=37)
                                print_c(" ")
                                #
            print('Balance condition added')
            print('Initial sf condition added')
            print('Recovery of Sf_bad added')
            #
            # for debugging
            print_analyze_constraints_matrix_form(suffix_solver)
            #
            # --------------------
            y_to_ip = 0.0
            y_out = 0.0
            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))

                for s in Sn:
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    for t in opaque_full_graph.successors(s):
                        for key_t in opaque_full_graph[s][t]:
                            act_s_list = get_action_from_successor_edge(opaque_full_graph, s)
                            if t not in Sn:
                                prop = opaque_full_graph[s][t][key_t]['prop'].copy()
                                for u in prop.keys():
                                    if u in act_s_list:
                                        pe = prop[u][0]
                                        #
                                        # Sn里出Sn的
                                        y_out += Y[(s, u)]*pe
                            elif t in ip:
                                prop = opaque_full_graph[s][t][key_t]['prop'].copy()
                                for u in prop.keys():
                                    if u in act_s_list:
                                        #
                                        # Sn里进Ip的
                                        pe = prop[u][0]
                                        y_to_ip += Y[(s, u)]*pe
            # suffix_solver.Add(y_to_ip+y_out >= delta)
            suffix_solver.Add(y_to_ip >= (1.0-gamma-delta)*(y_to_ip+y_out))
            print_c("y_2_ip:", color=35)
            print_c(y_to_ip, color=35)
            print_c("y_out:", color=36)
            print_c(y_out, color=36)
            print('Risk constraint added')
            #
            # TODO opacity constraint
            for s in Sn:
                pass

            constr_opacity_lhs = []
            constr_opacity_rhs = differential_expected_cost

            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))

                for k, s in enumerate(Sn):
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    for t in opaque_full_graph.successors(s):
                        for key_t in opaque_full_graph[s][t]:
                            if opaque_full_graph.has_edge(s, t):
                                prop = opaque_full_graph[s][t][key_t]['diff_exp'].copy()
                                for u in prop.keys():
                                    if (s, u) in Y:
                                        max_diff_exp_cost = max(prop[u].values())                   # TODO 最小化上界?
                                        y_t = Y[(s, u)] * max_diff_exp_cost                         # Y[(s, u)] * prop[u]
                                        constr_opacity_lhs.append(y_t)

            sum_opacity_lhs = suffix_solver.Sum(constr_opacity_lhs)
            suffix_solver.Add(sum_opacity_lhs <= constr_opacity_rhs)
            constr_descrip.append("differential_expected_cost")

            print_c("opacity constraint added ...", color=46)
            print_c(f'number of constraints: {suffix_solver.NumConstraints()}', color=46)
            constr_opacity_index = suffix_solver.NumConstraints() - 1

            #
            # ------------------------------
            # solve
            print('--optimization for suffix starts--')
            status = suffix_solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                print('Solution:')
                print('Objective value =', suffix_solver.Objective().Value())
                print('Advanced usage:')
                print('Problem solved in %f milliseconds' %
                      suffix_solver.wall_time())
                print('Problem solved in %d iterations' %
                      suffix_solver.iterations())
            else:
                print('The problem does not have an optimal solution.')
                return None, None, None, None
            #
            # for debugging
            Y_val = dict()
            for s_u in Y.keys():
                Y_val[s_u] = Y[s_u].solution_value()
            #
            # compute optimal plan suffix given the LP solution
            plan_suffix = dict()
            for s in Sn:

                #
                # added for debugging
                #if s == (('0', frozenset({'upload'}), 3), (('0', frozenset({'upload'}), 3), ('0', frozenset({'upload'}), 1)), ()):
                if s[0] == ('0', frozenset({'upload'}), 3):
                    debug_var = 20

                norm = 0
                U = []
                P = []
                # U_total = initial_subgraph.nodes[s]['act'].copy()
                U_total = list()
                for sync_u, sync_v, attr in opaque_full_graph.edges(s, data=True):
                    #
                    # 这边就不用is_opacity了
                    U_total += list(attr['prop'].keys())
                    #
                U_total = list(set(U_total))
                U_total.sort()
                for u in U_total:
                    if type(Y[(s, u)]) == float:
                        norm += 0.  # now, Y[(s, u)] is not considered as a feasible solution for opacity
                    else:
                        norm += Y[(s, u)].solution_value()
                #
                # TODO
                if norm > 0.01:
                    for u in U_total:
                        U.append(u)
                        if type(Y[(s, u)]) != float:
                            P.append(Y[(s, u)].solution_value() / norm)
                else:
                    if s not in sf:
                        # 当s不在sync_amec内时, 引导s到达sync_amec
                        path_t = nx.single_source_shortest_path(opaque_full_graph, s)
                        reachable_set_t = set(path_t).intersection(set(sf))
                        dist_val_dict = {}
                        for tgt_t in reachable_set_t:
                            dist_val_dict[tgt_t] = nx.shortest_path_length(opaque_full_graph, s, tgt_t,
                                                                                 weight=exp_weight)
                        #
                        min_dist_target = min(dist_val_dict, key=dist_val_dict.get)
                        #
                        if len(path_t[min_dist_target]) > 1:
                            successor_state = path_t[min_dist_target][1]
                            for key_t in opaque_full_graph[s][successor_state]:
                                edge_data = opaque_full_graph[s][successor_state][key_t]
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
                            for u in U_total:
                                P.append(1.0 / len(U_total))        # modified the number of P
                    else:
                        # 当在amec内
                        U_total_p_good = list()
                        U_total_p_bad  = list()
                        #for sync_u, sync_v, attr in sync_amec_graph.edges(s, data=True):
                        for sync_u, sync_v, attr in opaque_full_graph.edges(s, data=True):
                            if sync_v in sf:
                                U_total_p_good += list(attr['prop'].keys())
                            else:
                                U_total_p_bad  += list(attr['prop'].keys())
                        #
                        U_total_p_good = list(set(U_total_p_good))
                        U_total_p_good.sort()
                        for u in U_total_p_good:
                            U.append(u)
                            P.append(1.0 / len(U_total_p_good))
                        #
                        # TODO
                        # 这个会影响概率连贯性?
                        U_total_p_bad = list(set(U_total_p_bad))
                        U_total_p_bad.sort()
                        for u in U_total_p_bad:
                            U.append(u)
                            P.append(0.)

                if U.__len__() == 0:
                    debug_var = 4.1

                plan_suffix[s] = [U, P]

            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")
            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                act_s_list = get_action_from_successor_edge(opaque_full_graph, s)
                for t in opaque_full_graph.successors(s):
                    for key_t in opaque_full_graph[s][t]:
                        if t not in Sn:
                            prop = opaque_full_graph[s][t][key_t]['prop'].copy()
                            for u in prop.keys():
                                if u in act_s_list:
                                    pe = prop[u][0]
                                    y_out += Y[(s, u)].solution_value()*pe
                        elif t in ip:
                            prop = opaque_full_graph[s][t][key_t]['prop'].copy()
                            for u in prop.keys():
                                if u in act_s_list:
                                    pe = prop[u][0]
                                    y_to_ip += Y[(s, u)].solution_value()*pe
            if (y_to_ip+y_out) > 0:
                risk = y_out/(y_to_ip+y_out)
            print('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip+y_out))
            print("----Suffix risk computed")

            constr_t = suffix_solver.constraint(constr_opacity_index)
            opacity_val = 0.
            for s_sync_t in Sn:
                act_s_list = get_action_from_successor_edge(opaque_full_graph, s)
                for u_t in act_s_list:
                    Y_t = Y[(s_sync_t, u_t)]  # 单独拿出来是为了debugging
                    if type(Y_t) != float:
                        ki_t = constr_t.GetCoefficient(Y_t)
                        opacity_val += ki_t * Y_t.solution_value()
            print_c("opacity_value: %f <= %f" % (opacity_val, differential_expected_cost,), color=35)
            print_c("----Suffix opacity threshold computed", color=35)

            return plan_suffix, cost, risk, opacity_val
        # except:
        #     print("ORtools Error reported")
        #     return None, None, None

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
    plan = []
    for l, S_fi_pi in enumerate(prod_dra_pi.Sf):  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
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
                prod_dra_gamma = product_mdp3(mdp, dra)
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
                        if len(initial_subgraph.nodes()) == 0:
                            print_c("[prefix synthesis] failed to construct initial subgraph， AP: %s" % (ap_4_opacity,),
                                    color='yellow')
                            continue

                        # TODO
                        # 这个东西干吗的
                        observer_mec_3 = project_sync_mec_3_2_observer_mec_3(initial_subgraph, sync_mec_t)

                        plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync, Sr, Sd = syn_plan_prefix_in_sync_amec(
                                                                                    prod_dra_pi, initial_subgraph, initial_sync_state,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t, observer_mec_3, risk_pr)

                        if plan_prefix == None:
                            print_c("[prefix synthesis] failed to synthesize prefix plan， AP: %s" % (ap_4_opacity,),
                                    color='yellow')
                            continue

                        opaque_full_graph = prod_dra_pi.construct_fullgraph_4_amec(initial_subgraph,
                                                                                    prod_dra_gamma,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    MEC_pi, MEC_gamma,
                                                                                    optimizing_ap, ap_4_opacity,
                                                                                    observation_func, ctrl_obs_dict)
                        # TODO
                        # To get rid of mec_observer
                        plan_suffix, suffix_cost, suffix_risk, suffix_opacity_threshold = synthesize_suffix_cycle_in_sync_amec3(
                                                                                    prod_dra_pi,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t, observer_mec_3,
                                                                                    y_in_sf_sync,
                                                                                    opaque_full_graph,  # 用来判断Sn是否可达, 虽然没啥意义但是还是可以做,
                                                                                    initial_sync_state,
                                                                                    differential_exp_cost)

                        plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync],
                                     [plan_suffix, suffix_cost, suffix_risk],
                                     [MEC_pi[0], MEC_pi[1], Sr, Sd],
                                     [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index, MEC_gamma],
                                     [initial_subgraph, initial_sync_state, opaque_full_graph],
                                     [sync_mec_t, observer_mec_3]
                                     ])

    if plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        # best_all_plan = min(plan, key=lambda p: p[0][1] + alpha * p[1][1])
        valid_plans = [p for p in plan if p[0][1] is not None and p[1][1] is not None]
        best_all_plan = min(valid_plans, key=lambda p: p[0][1] + alpha * p[1][1])
        #
        prod_dra_pi.update_best_all_plan(best_all_plan, is_print_policy=True)
        # print('Best plan prefix obtained for %s states in Sr' %
        #       str(len(best_all_plan[0][0])))
        # print('cost: %s; risk: %s ' %
        #       (best_all_plan[0][1], best_all_plan[0][2]))
        # print('Best plan suffix obtained for %s states in Sf' %
        #       str(len(best_all_plan[1][0])))
        # print('cost: %s; risk: %s ' %
        #       (best_all_plan[1][1], best_all_plan[1][2]))
        # print('Total cost:%s' %
        #       (best_all_plan[0][1] + alpha * best_all_plan[1][1]))
        # print_c('Opacity threshold %f <= %f' % (best_all_plan[3][1], differential_exp_cost,))
        # #


        return best_all_plan, prod_dra_pi
    else:
        print("No valid plan found")
        return None

