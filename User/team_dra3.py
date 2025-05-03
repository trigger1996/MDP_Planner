#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import textwrap

from collections import Counter
from networkx import DiGraph
from networkx import strongly_connected_components_recursive
from MDP_TG.dra import Product_Dra
from User.dra3 import product_mdp3
from User.dra3 import obtain_differential_expected_cost, is_ap_satisfy_opacity
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
            simple_digraph = DiGraph()
            for s_f in T_temp:
                if s_f not in simple_digraph:
                    simple_digraph.add_node(s_f)
                for s_t in mdp.successors(s_f):
                    if s_t in T_temp:
                        simple_digraph.add_edge(s_f, s_t)
            print("SubGraph of one MEC: %s states and %s edges" % (
                str(len(simple_digraph.nodes())), str(len(simple_digraph.edges()))))
            i = 0
            for Scc in strongly_connected_components_recursive(simple_digraph):
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
            for Scc in strongly_connected_components_recursive(simple_digraph):
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
    simple_digraph = DiGraph()
    A = dict()
    for s in mdp.nodes():
        A[s] = mdp.nodes[s]['act'].copy()
    for s_f in Sneg:
        if s_f not in simple_digraph:
            simple_digraph.add_node(s_f)
        for s_t in mdp.successors(s_f):
            if s_t in Sneg:
                simple_digraph.add_edge(s_f, s_t)
    print("SubGraph of one Sf: %s states and %s edges" %
          (str(len(simple_digraph.nodes())), str(len(simple_digraph.edges()))))
    for scc in strongly_connected_components_recursive(simple_digraph):
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
                        elif u_pi != u_gamma and ctrl_obs_dict[str(u_pi)] == False and ctrl_obs_dict[u_gamma] == False:
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