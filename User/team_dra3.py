#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import textwrap

from collections import Counter, deque
from networkx import MultiDiGraph
from networkx import strongly_connected_components_recursive
from MDP_TG.dra import Product_Dra
from User.dra3 import product_mdp3
from User.dra3 import obtain_differential_expected_cost, is_state_satisfy_ap, is_ap_satisfy_opacity
from User.utils import print_c

def find_MECs(mdp, Sneg):
    # ----implementation of Alg.47 P866 of Baier08----
    print('Remaining states size', len(Sneg))
    U = mdp.graph['U']
    A = dict()
    for s in Sneg:
        A[s] = mdp.nodes[s]['act'].copy()
        if not A[s]:
            print("Isolated state")
    MEC = set()
    MECnew = set()
    MECnew.add(frozenset(Sneg))
    # ----
    k = 0
    while MEC != MECnew:
        print("<============iteration %s============>" % k)
        k += 1
        MEC = MECnew
        MECnew = set()
        print("MEC size: %s" % len(MEC))
        print("MECnew size: %s" % len(MECnew))
        for T in MEC:
            R = set()
            T_temp = set(T)
            simple_MultiDiGraph = MultiDiGraph()
            for s_f in T_temp:
                if s_f not in simple_MultiDiGraph:
                    simple_MultiDiGraph.add_node(s_f)
                for s_t in mdp.successors(s_f):
                    if s_t in T_temp:
                        simple_MultiDiGraph.add_edge(s_f, s_t)
            print("SubGraph of one MEC: %s states and %s edges" % (
                str(len(simple_MultiDiGraph.nodes())), str(len(simple_MultiDiGraph.edges()))))
            i = 0
            for Scc in strongly_connected_components_recursive(simple_MultiDiGraph):
                i += 1
                if (len(Scc) >= 1):
                    for s in Scc:
                        U_to_remove = set()
                        for u in A[s]:
                            for t in mdp.successors(s):
                                if ((u in list(mdp[s][t]['prop'].keys())) and (t not in Scc)):
                                    U_to_remove.add(u)
                        A[s].difference_update(U_to_remove)
                        if not A[s]:
                            R.add(s)
            while R:
                s = R.pop()
                T_temp.remove(s)
                for f in mdp.predecessors(s):
                    if f in T_temp:
                        A[f].difference_update(
                            set(mdp[f][s]['prop'].keys()))
                        if not A[f]:
                            R.add(f)
            j = 0
            for Scc in strongly_connected_components_recursive(simple_MultiDiGraph):
                j += 1
                if (len(Scc) >= 1):
                    common = set(Scc).intersection(T_temp)
                    if common:
                        MECnew.add(frozenset(common))
    # ---------------
    print('Final MEC and MECnew size:', len(MEC))
    return MEC, A


def find_SCCs(mdp, Sneg):
    # ----simply find strongly connected components----
    print('Remaining states size', len(Sneg))
    SCC = set()
    simple_MultiDiGraph = MultiDiGraph()
    A = dict()
    for s in mdp.nodes():
        A[s] = mdp.nodes[s]['act'].copy()
    for s_f in Sneg:
        if s_f not in simple_MultiDiGraph:
            simple_MultiDiGraph.add_node(s_f)
        for s_t in mdp.successors(s_f):
            if s_t in Sneg:
                simple_MultiDiGraph.add_edge(s_f, s_t)
    print("SubGraph of one Sf: %s states and %s edges" %
          (str(len(simple_MultiDiGraph.nodes())), str(len(simple_MultiDiGraph.edges()))))
    for scc in strongly_connected_components_recursive(simple_MultiDiGraph):
        SCC.add(frozenset(scc))
    return SCC, A

class product_team_mdp3(product_mdp3):
    def __init__(self, mdp=None, dra=None):
        product_mdp3.__init__(self, mdp, dra)

    # TODO
    # 个体的函数在实际使用过程中会遇到一些问题
    def build_full(self):
        # ----construct full product----（构建 MDP 和 DRA 的乘积图）

        # 遍历所有 MDP 节点
        for f_mdp_node in self.graph['mdp']:
            # 遍历该节点的所有标签及其概率（标签是 frozenset，概率是 float）
            f_mdp_label = []
            for f_mdp_label_t, f_label_prob_t in self.graph['mdp'].nodes[f_mdp_node]['label'].items():
                # TODO 1
                f_mdp_label.append(next(iter(f_mdp_label_t)))
            f_mdp_label = list(set(f_mdp_label))
            if f_mdp_label.__len__() > 1 and str('') in f_mdp_label:        # TODO, 整合为一个函数
                f_mdp_label.remove(str(''))                                 # TODO, 其他类型空
            f_mdp_label.sort()
            f_mdp_label = tuple(f_mdp_label)
            # 遍历所有 DRA 节点
            for f_dra_node in self.graph['dra']:
                # 构造当前的乘积节点（前缀）
                f_prod_node = self.composition(f_mdp_node, f_mdp_label, f_dra_node)

                # 遍历 MDP 中该节点的后继节点
                for t_mdp_node in self.graph['mdp'].successors(f_mdp_node):
                    # 获取 MDP 边的属性信息（通常包含转移概率和成本）
                    mdp_edge = self.graph['mdp'][f_mdp_node][t_mdp_node]

                    # 遍历目标 MDP 节点的所有标签及其概率
                    t_mdp_label = []
                    for t_mdp_label_t, t_label_prob in self.graph['mdp'].nodes[t_mdp_node]['label'].items():
                        # TODO 2
                        t_mdp_label.append(next(iter(t_mdp_label_t)))
                    t_mdp_label = list(set(t_mdp_label))
                    if t_mdp_label.__len__() > 1 and str('') in t_mdp_label:    # TODO, 整合为一个函数
                        t_mdp_label.remove(str(''))                             # TODO, 其他类型空, 比如frozenset, 而且现在这边是写死了, 为了frozenset -> set
                    t_mdp_label.sort()
                    t_mdp_label = tuple(t_mdp_label)
                    # 遍历 DRA 中当前 DRA 状态的所有后继 DRA 状态
                    for t_dra_node in self.graph['dra'].successors(f_dra_node):
                        # 构造下一个乘积节点（后缀）
                        t_prod_node = self.composition(t_mdp_node, t_mdp_label, t_dra_node)

                        # 检查当前标签是否满足 DRA 的转移条件（根据边标签判断是否允许转移）
                        truth = self.graph['dra'].check_label_for_dra_edge(
                            f_mdp_label, f_dra_node, t_dra_node)

                        if truth:
                            prob_cost = dict()
                            # 遍历该 MDP 边上的所有 action u 和其属性 attri = (prob, cost)
                            for u, attri in mdp_edge['prop'].items():
                                # attri[0] 是转移概率，attri[1] 是代价
                                if t_label_prob * attri[0] != 0:
                                    # 将乘积边的属性记录为字典：{action: (prob, cost)}
                                    prob_cost[u] = (t_label_prob * attri[0], attri[1])

                            # 如果有合法的转移，则在乘积图中添加边
                            if list(prob_cost.keys()):
                                self.add_edge(f_prod_node, t_prod_node, prop=prob_cost)

        # 构建接受状态信息（通常用于后续做模型检测或策略合成）
        self.build_acc()

        # 输出构建完成信息
        print("-------Prod DRA Constructed-------")
        print("%s states, %s edges and %s accepting pairs" % (
            str(len(self.nodes())), str(len(self.edges())), str(len(self.graph['accept']))))

        # TODO, for reference, 165 nodes and 955 edges

    def check_label_for_dra_edge(self, label, f_dra_node, t_dra_node):
        # ----check if a label satisfies the guards on one dra edge----
        guard_string_list = self[f_dra_node][t_dra_node]['guard_string']
        guard_int_list = []
        for st in guard_string_list:
            int_st = []
            for l in st:
                int_st.append(int(l))
            guard_int_list.append(int_st)
        for guard_list in guard_int_list:
            valid = True
            for k, ap in enumerate(self.graph['symbols']):
                if (guard_list[k] == 1) and (ap not in label):
                    valid = False
                if (guard_list[k] == 0) and (ap in label):
                    valid = False
            if valid:
                return True
        return False

    # 构造乘积节点
    def composition(self, mdp_node, mdp_label, dra_node):
        # 乘积节点由 MDP 节点、其标签、DRA 节点构成三元组
        prod_node = (mdp_node, mdp_label, dra_node)

        # 如果该乘积节点还不存在，就添加它
        if not self.has_node(prod_node):
            # 复制该 MDP 节点的可用动作集合
            Us = self.graph['mdp'].nodes[mdp_node]['act'].copy()
            # 添加乘积节点，并记录其属性（源 MDP 节点、标签、DRA 状态、可用动作）
            self.add_node(prod_node, mdp=mdp_node,
                          label=mdp_label, dra=dra_node, act=Us)

            # 如果该节点是起始状态（即 MDP 初始节点、标签正确、且 DRA 初始状态包含该 DRA 节点）
            if ((mdp_node == self.graph['mdp'].graph['init_state']) and
                #(mdp_label == self.graph['mdp'].graph['init_label']) and       # ADDED， REMOVED
                (dra_node in self.graph['dra'].graph['initial'])):
                # 将其加入乘积初始状态集合
                self.graph['initial'].add(prod_node)

        return prod_node

    def compute_S_f(self):
        # ----find all accepting End components----
        S = set(self.nodes())
        acc_pairs = self.graph['accept']        # TODO
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

    def re_synthesize_sync_amec(self, ap_pi, ap_gamma, MEC_pi, MEC_gamma, product_mdp_gamma:Product_Dra, observation_func, ctrl_obs_dict, is_re_compute_Sf=True):
        #
        # 想了一下这个还是沿用旧的
        # sync_amec的最大问题在于它的初始状态只凭借两个集合是很难确定的
        mec_state_set_pi = MEC_pi[0]
        mec_state_set_gamma = MEC_gamma[0]

        stack_t = deque()                                   # Added
        stack_visited = set()                               # Added 表示当前状态是否在stack内, 用于减少stack_t = list(set(stack_t))的计算
        for state_pi_t in mec_state_set_pi:
            for state_gamma_t in mec_state_set_gamma:
                ap_state_pi = set(state_pi_t[1])
                ap_state_gamma = set(state_gamma_t[1])
                if ap_pi in ap_state_pi:
                    if ap_gamma in ap_state_gamma:
                        stack_t.append((state_pi_t, state_gamma_t,))
                        stack_visited.add((state_pi_t, state_gamma_t,))
        #
        # stack_t = list(set(stack_t))
        visited = set()
        sync_mec_t = MultiDiGraph()
        #
        # Added
        for node_t in stack_t:
            sync_mec_t.add_node(node_t)

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
                        #
                        # Added, Removed
                        # if isinstance(u_pi, tuple):
                        #     u_pi = u_pi[0]
                        # if isinstance(u_gamma, tuple):
                        #     u_gamma = u_gamma[0]

                        # TODO, Added, to verify
                        u_pi_equal_to_u_gamma = Counter(u_pi) == Counter(u_gamma)

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        # TODO, 注意这里输入的ctrl_obs_dict是一个tuple, 所以需要单独生成一个team_ctrl_obs_dict
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
                            # Added
                            if next_sync_state not in visited and next_sync_state not in stack_visited:
                                stack_t.append(next_sync_state)
                                stack_visited.add(next_sync_state)
                                # stack_t = list(set(stack_t))
                                # stack_t.sort()

        # 完成后处理
        print_c(
            f"[synthesize_w_opacity] DFS completed, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}",
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

        print_c(f"[synthesize_w_opacity] Generated sync_amec, states: {len(sync_mec_t.nodes)}, edges: {len(sync_mec_t.edges)}")

    #
    # Seems identical to the function for individual system
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
        for state_pi in self.graph['initial']:
            observed_state_list = []
            for state_gamma in product_mdp_gamma.graph['initial']:
                observed_state_list.append(state_gamma)
            initial_sync_state.append((state_pi, tuple(observed_state_list), tuple()))
        stack_t = [state_t for state_t in initial_sync_state]       # make a copy
        stack_t = deque(stack_t)                                    # TODO
        stack_visited = set(stack_t)
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
                if ('0', '5') in current_state[0]:
                    if ('0', '5') in next_state_pi:
                        debug_var = 1.1
                #
                for edge_t_gamma in next_state_list_gamma:
                    # 获取下一状态
                    next_state_gamma = edge_t_gamma[1]

                    # for debugging
                    if next_state_gamma[0] == next_state_pi[0]:
                        debug_var = 1.5

                    # 1. 检查控制动作同步
                    try:
                        u_pi = next(iter(edge_t_pi[2]['prop'].keys()))
                        u_gamma = next(iter(edge_t_gamma[2]['prop'].keys()))
                        #
                        # Added, Removed
                        # if isinstance(u_pi, tuple):
                        #     u_pi = u_pi[0]
                        # if isinstance(u_gamma, tuple):
                        #     u_gamma = u_gamma[0]

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        # TODO, 注意这里输入的ctrl_obs_dict是一个tuple, 所以需要单独生成一个team_ctrl_obs_dict
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
                next_observed_states = tuple(set(next_observed_states))
                next_states_3        = tuple(set(next_states_3))
                next_sync_state = (next_state_pi, next_observed_states, next_states_3, )
                #
                mapping_t = {}
                for observed_state_t in subgraph_2_amec_t.nodes():
                    if ('0', '5') in next_state_pi and ('0', '5') in observed_state_t[0]:
                        debug_var = 3.1

                    #
                    # 几点思考
                    # 1 相同MDP状态, 不同DRA状态的乘积状态能否合并？
                    #       不能, 因为表示不同任务状态, 且状态转移关系主要依靠同步深度搜索, 即符合原prod_mdp_pi的状态转移关系
                    # 2 观测相同的点可以合并吗, 如果可以合并, 其历史状态如何表述?
                    #       可以, 在从prod_mdp_gamma搜索观测状态时, 满足o(x, u) = o(x', u'), x, u为prod_mdp_pi的状态和动作, x', u'为prod_mdp_gamma的状态和动作
                    #       这里只有观测相同才会被同步(现有工作主要和u无关, 以后观测可以加上u), 这里的u可以用于表示和区分不同的历史动作
                    #       同时, 观测器的状态转移关系可以表示系统的历史的evolution, 所以不用担心系统的历史不被表示
                    # Some Thoughts
                    #
                    # 1. Can product states with the same MDP state but different DRA states be merged?
                    #    No, because they represent different task states, and the state transition
                    #    relationships mainly rely on synchronized deep search, meaning they must follow
                    #    the transition structure of the original `prod_mdp_pi`.
                    #
                    # 2. Can points with the same observation be merged? If so, how should their historical
                    #    states be represented?
                    #    Yes, when searching for observation states from `prod_mdp_gamma`, the condition
                    #    o(x, u) = o(x', u') must be satisfied, where x, u are the state and action from
                    #    `prod_mdp_pi`, and x', u' are from `prod_mdp_gamma`.
                    #    Here, only identical observations will be synchronized (in current work, this is
                    #    mostly independent of u, but in the future, the observation could include u).
                    #    The action u can help indicate and differentiate different historical actions.
                    #    Moreover, the transition structure of the observer can capture the historical
                    #    evolution of the system, so there's no need to worry that the system's history
                    #    won't be represented.
                    if observed_state_t[0] == next_state_pi:
                        next_observed_states = list(next_observed_states)
                        next_observed_states = next_observed_states + list(observed_state_t[1])
                        #
                        next_states_3 = list()
                        next_states_3 = next_states_3 + list(observed_state_t[2])
                        #
                        next_observed_states = tuple(set(next_observed_states))
                        next_states_3 = tuple(set(next_states_3))
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
                    #
                    # for debugging
                    if ('0', '5') in next_sync_state[0]:
                        debug_var = 3.1

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
                    if next_sync_state not in visited and next_sync_state not in stack_visited:
                        stack_t.append(next_sync_state)
                        stack_visited.add(next_sync_state)
                        # stack_t = list(set(stack_t))
                        # stack_t.sort()

        return subgraph_2_amec_t, initial_sync_state

    #
    # Seems identical to the function for individual system
    def construct_fullgraph_4_amec(self, initial_subgraph, product_mdp_gamma: Product_Dra, sync_amec_graph, mec_pi_3, mec_gamma_3, ap_pi, ap_gamma, observation_func, ctrl_obs_dict):
        #
        # 目标是生成从初始状态到达sync_amec的通路
        stack_t = [state_t for state_t in initial_subgraph.nodes()]  # make a copy
        stack_t = deque(stack_t)
        stack_visited = set(stack_t)
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

                        # Added, Removed
                        # if isinstance(u_pi, tuple):
                        #     u_pi = u_pi[0]
                        # if isinstance(u_gamma, tuple):
                        #     u_gamma = u_gamma[0]

                        # 如果不考虑可观性
                        if ctrl_obs_dict == None and u_pi != u_gamma:
                            continue
                        # 如果考虑可观性
                        # TODO, 注意这里输入的ctrl_obs_dict是一个tuple, 所以需要单独生成一个team_ctrl_obs_dict
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

                # next_observed_states.sort()
                # next_states_3.sort()

                for action, (prob, cost) in edge_t_pi[2]['prop'].items():
                    trans_pr_cost_list[action] = (prob, cost)  # it is evident that this only corresponds to state_pi

                # 添加转移
                next_observed_states = tuple(set(next_observed_states))
                next_states_3 = tuple(set(next_states_3))
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
                        next_observed_states = tuple(set(next_observed_states))
                        next_states_3 = tuple(set(next_states_3))
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
                if next_sync_state not in visited and next_sync_state not in stack_visited:
                    stack_t.append(next_sync_state)
                    stack_visited.add(next_sync_state)
                    # stack_t = list(set(stack_t))
                    # stack_t.sort()

        return fullgraph_t

    def print_policy(self, plan_prefix, plan_suffix):
        def format_policy_entry(state, actions, probs, width=120, indent="  "):
            # 自动换行三个部分：STATE、ACTIONS、PROBS
            state_str = str(state)
            actions_str = str(actions)
            probs_str = str(probs)

            state_lines = textwrap.wrap(state_str, width=width)
            actions_lines = textwrap.wrap(actions_str, width=30)
            probs_lines = textwrap.wrap(probs_str, width=40)

            max_lines = max(len(state_lines), len(actions_lines), len(probs_lines))
            result_lines = []

            for i in range(max_lines):
                state_part = state_lines[i] if i < len(state_lines) else ""
                actions_part = actions_lines[i] if i < len(actions_lines) else ""
                probs_part = probs_lines[i] if i < len(probs_lines) else ""
                result_lines.append("{:<{w1}} {:<30}: {}".format(state_part, actions_part, probs_part, w1=width))

            return "\n".join(result_lines)

        # 打印 Prefix 部分
        print_c("\nPrefix", color='bg_magenta', style='bold')
        header = "{:<120} {:<30} {}".format("STATE", "ACTIONS", ": PROBS")
        print_c(header, color='magenta', style='bold')
        for state_t in plan_prefix:
            actions, probs = plan_prefix[state_t]
            line = format_policy_entry(state_t, actions, probs)
            print_c(line, color='magenta')

        # 打印 Suffix 部分
        print_c("\nSuffix", color='bg_cyan', style='bold')
        header = "{:<120} {:<30} {}".format("STATE", "ACTIONS", ": PROBS")
        print_c(header, color='blue', style='bold')
        for state_t in plan_suffix:
            actions, probs = plan_suffix[state_t]
            line = format_policy_entry(state_t, actions, probs)
            print_c(line, color='blue')