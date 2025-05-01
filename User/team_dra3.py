#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx

from networkx import DiGraph
from networkx import strongly_connected_components_recursive
from User.dra3 import product_mdp3
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
                f_mdp_label.append(f_mdp_label_t)
            f_mdp_label = list(set(f_mdp_label))
            f_mdp_label.sort()
            f_mdp_label = tuple(f_mdp_label)
            # 遍历所有 DRA 节点
            # TODO 问题1, 如何确定dra node和mdp node的关联性
            # TODO 问题2, 如果不存在关联性, 那么是否可以将team mdp的prop直接全部取出来?
            for f_dra_node in self.graph['dra']:
                # 构造当前的乘积节点（前缀）
                f_prod_node = self.composition(f_mdp_node, f_mdp_label, f_dra_node)

                # 遍历 MDP 中该节点的后继节点
                for t_mdp_node in self.graph['mdp'].successors(f_mdp_node):
                    # 获取 MDP 边的属性信息（通常包含转移概率和成本）
                    mdp_edge = self.graph['mdp'][f_mdp_node][t_mdp_node]

                    # 遍历目标 MDP 节点的所有标签及其概率
                    t_mdp_label = set()
                    for t_mdp_label_t, t_label_prob in self.graph['mdp'].nodes[t_mdp_node]['label'].items():
                        # TODO 2
                        t_mdp_label.add(t_mdp_label_t)
                    t_mdp_label = list(set(t_mdp_label))
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
        self.build_acc()        # TODO

        # 输出构建完成信息
        print("-------Prod DRA Constructed-------")
        print("%s states, %s edges and %s accepting pairs" % (
            str(len(self.nodes())), str(len(self.edges())), str(len(self.graph['accept']))))

        # TODO, for reference, 165 nodes and 955 edges

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
