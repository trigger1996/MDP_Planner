from MDP_TG import lp
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import find_twin_states, find_original_states, find_corresponding_outgoing_states, find_corresponding_incoming_states, is_state_equal
from MDP_TG.lp import syn_plan_prefix, syn_plan_suffix, syn_plan_suffix2, syn_plan_bad
from User.dra2 import product_mdp2
from User.team_mdp_dra import Team_MDP, Team_Product_Dra
import User.dra2 as dra2

from copy import deepcopy
from collections import defaultdict
import random
from ortools.linear_solver import pywraplp
from networkx import single_source_shortest_path

from subprocess import check_output
from User.vis2  import print_c

from functools import cmp_to_key
from User.grid_utils import sort_grids, sort_sync_grid_states

import pickle
import time
import networkx

from rich.progress import Progress, track


import sys
stack_size_t = 500000
sys.setrecursionlimit(stack_size_t)         # change stack size to guarantee runtime stability, default: 3000
print_c("stack size changed %d ..." % (stack_size_t, ))
#
sys.setrecursionlimit(100000)               # 设置递归深度为100000

def ltl_convert(task, is_display=True):
    #
    # https://www.ltl2dstar.de/docs/ltl2dstar.html#:~:text=ltl2dstar%20is%20designed%20to%20use%20an%20external%20tool%20to%20convert
    # LTL是有两个格式的, 一个ltl2dstar notation, 一个spin notation
    # 后者是常用的, 要从后者转到前者ltl2dstar才能用
    cmd_ltl_convert = 'ltlfilt -l -f \'%s\'' % (task, )
    ltl_converted = str(check_output(cmd_ltl_convert, shell=True))
    ltl_converted = ltl_converted[2 : len(ltl_converted) - 3]               # tested
    if is_display:
        print_c('converted ltl: ' + ltl_converted)

    return ltl_converted

def state_action_sets_pi_from_sync_mec(sync_mec, MEC_pi):
    sf = []
    sf_pi = []
    ip = []
    ip_pi = []
    act = dict()            # act is identical to act_pi, but keys are different
    act_pi = dict()
    for sync_state_t in sync_mec.nodes:
        #
        state_pi_t = sync_state_t[0]
        #
        if state_pi_t in MEC_pi[0]:
            sf.append(sync_state_t)
            if state_pi_t not in sf_pi:
                sf_pi.append(state_pi_t)
        #
        if state_pi_t in MEC_pi[1]:
            ip.append(sync_state_t)
            if state_pi_t not in ip_pi:
                ip_pi.append(state_pi_t)

    for edge_t in sync_mec.edges(data=True):
        sync_state_t = edge_t[0]
        state_pi_t   = edge_t[0][0]
        act_t = list(edge_t[2]['prop'].keys())
        #
        if sync_state_t not in act.keys():
            act[sync_state_t] = act_t
        else:
            act[sync_state_t] = act[sync_state_t] + act_t
            act[sync_state_t] = list(set(act[sync_state_t]))
        #
        if state_pi_t not in act_pi.keys():
            act_pi[state_pi_t] = act_t
        else:
            act_pi[state_pi_t] = act_pi[state_pi_t] + act_t
            act_pi[state_pi_t] = list(set(act_pi[state_pi_t]))

    for sync_state_t in sync_mec.nodes:
        #
        state_pi_t = sync_state_t[0]
        if sync_state_t not in act.keys():
            act[sync_state_t]  = []
            act_pi[state_pi_t] = []

    return sf, sf_pi, ip, ip_pi, act, act_pi

def sn_pi_2_sync_sn(sn_pi, sync_mec):
    sync_sn = set()                     # use set instead of list will boost the performance
    for state_pi_t in sn_pi:
        for sync_state_t in sync_mec.nodes:
            if state_pi_t == sync_state_t[0]:
                if sync_state_t not in sync_sn:
                    sync_sn.add(sync_state_t)

    return list(sync_sn)

def y_in_sf_2_sync_states(sync_mec, y_in_sf):
    y_in_sf_sync = dict()

    # use averaged probability
    # is there any better ideas?
    number_shared_states_list = dict()              # for further possible improvement
    for state_pi_t in y_in_sf.keys():
        # if y_in_sf[state_pi_t] == 0.:
        #     continue

        number_shared_states = 0
        shared_state_list = []
        # find the number
        for sync_state_t in sync_mec.nodes:
            #
            if sync_state_t[0] == state_pi_t:
                number_shared_states += 1
                shared_state_list.append(sync_state_t)
        #
        number_shared_states_list[state_pi_t] = shared_state_list
        #
        for sync_state_t in shared_state_list:
            # take average value
            y_in_sf_sync[sync_state_t] = y_in_sf[state_pi_t] / number_shared_states

    return y_in_sf_sync

def find_i_in_and_i_out_in_amec(prod_mdp, mec_pi):
    ip_pi = mec_pi[1]
    i_in  = ip_pi
    i_out = ip_pi
    transitions_i_in  = dict()
    transitions_i_out = dict()

    for state_pi_t in ip_pi:

        for out_edge_t in prod_mdp.out_edges(state_pi_t, data=True):
            if state_pi_t not in transitions_i_out.keys():
                transitions_i_out[state_pi_t] = [ out_edge_t ]
            else:
                transitions_i_out[state_pi_t].append(out_edge_t)
        for in_edge_t  in prod_mdp.in_edges(state_pi_t, data=True):
            if state_pi_t not in transitions_i_in.keys():
                transitions_i_in[state_pi_t] = [ in_edge_t ]
            else:
                transitions_i_in[state_pi_t].append(in_edge_t)

    return i_in, i_out, transitions_i_in, transitions_i_out

def find_i_in_and_i_out_in_sync_amec(prod_mdp, sync_mec, sync_ip):
    ip = sync_ip.copy()
    i_in  = ip
    i_out = ip
    transitions_i_in  = dict()
    transitions_i_out = dict()

    for sync_state_t in ip:

        for out_edge_t in sync_mec.out_edges(sync_state_t, data=True):
            if sync_state_t not in transitions_i_out.keys():
                transitions_i_out[sync_state_t] = [ out_edge_t ]
            else:
                transitions_i_out[sync_state_t].append(out_edge_t)
        for in_edge_t  in sync_mec.in_edges(sync_state_t, data=True):
            if sync_state_t not in transitions_i_in.keys():
                transitions_i_in[sync_state_t] = [ in_edge_t ]
            else:
                transitions_i_in[sync_state_t].append(in_edge_t)

    return i_in, i_out, transitions_i_in, transitions_i_out

def find_states_satisfying_opt_prop(opt_prop, Se):
    S_pi = []
    for s in Se:
        if opt_prop in s[1]:
            S_pi.append(s)
    return S_pi

def print_policies_w_opacity(ap_4_opacity, plan_prefix, plan_suffix):
    # Added
    # for printing policies
    print_c("policy for AP: %s" % str(ap_4_opacity))
    print_c("state action: probabilities")
    print_c("Prefix", color=42)
    #
    state_in_prefix = [ state_t for state_t in plan_prefix ]
    #state_in_prefix.sort(key=cmp_to_key(sort_grids))
    #for state_t in plan_prefix:
    for state_t in state_in_prefix:
        print_c("%s, %s: %s" % (str(state_t), str(plan_prefix[state_t][0]), str(plan_prefix[state_t][1]), ), color=43)
    #
    print_c("Suffix", color=45)
    state_in_suffix = [ state_t for state_t in plan_suffix ]
    #state_in_suffix.sort(key=cmp_to_key(sort_grids))
    #for state_t in plan_suffix:
    for state_t in state_in_suffix:
        print_c("%s, %s: %s" % (str(state_t), str(plan_suffix[state_t][0]), str(plan_suffix[state_t][1]), ), color=46)

def synthesize_suffix_cycle_in_sync_amec(prod_mdp, sync_mec, MEC_pi, y_in_sf, S_pi, differential_expected_cost=1.55):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print_c("===========[plan suffix synthesis starts]", color=32)
    print_c("[synthesize_w_opacity] differential exp cost: %f" % (differential_expected_cost, ), color=32)
    # step 1: find states
    # sf:  states in MEC -> states in sync MEC
    # ip:  MEC states intersects with Ip -> Accepting states Ip in sync MEC / state[0] intersects with IP
    # act: actions available for each state
    sf, sf_pi, ip, ip_pi, act, act_pi = state_action_sets_pi_from_sync_mec(sync_mec, MEC_pi)

    # distribute the initial probability distributions of origin AMEC to the corresponding states in sync_amec
    # currently, considering the observer takes the statistics results, so we omit the effect of initial distributions
    # therefore, the averaged initial distributions is applied
    y_in_sf_sync = y_in_sf_2_sync_states(sync_mec, y_in_sf)

    # find S_e', I_in, and I_out
    # in Guo et. al. Probabilstic, I_c' denotes the accepting states in AMECs
    i_in_pi, i_out_pi, transitions_i_in_pi, transitions_i_out_pi = find_i_in_and_i_out_in_amec(prod_mdp, MEC_pi)
    i_in, i_out,       transitions_i_in,    transitions_i_out    = find_i_in_and_i_out_in_sync_amec(prod_mdp, sync_mec, ip)

    delta = 0.01                    # 松弛变量?
    gamma = 0.05                    # 根据(11), 整个系统进入MEC内以后就不用概率保证了?
    for init_node in prod_mdp.graph['initial']:
        # find states that reachable from initial state
        paths = single_source_shortest_path(prod_mdp, init_node)
        Sn_pi = set(paths.keys()).intersection(sf_pi)
        print('Sf_pi size: %s' % len(sf_pi))                                                # sf: MEC中状态
        print('reachable sf_pi size: %s' % len(Sn_pi))                                      # Sn: 可由当前状态到达的MEC中的状态
        print('Ip_pi size: %s' % len(ip_pi))                                                # Ip: 可被接收的MEC的状态
        print('Ip_pi and sf intersection size: %s' % len(Sn_pi.intersection(ip_pi)))        # 可达的MEC中的状态
        #
        print_c("Sn_pi: %d / sync_mec: %d" % (Sn_pi.__len__(), sync_mec.__len__(),))
        #
        # back to sync_mec
        Sn = sn_pi_2_sync_sn(Sn_pi, sync_mec)

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
            #       path = single_source_shortest_path(simple_digraph, random.sample(ip, 1)[0]), 即可以到达MEC的状态
            #       Sn: path_init.keys() 和 Sf的交集, 两变Sf定义是一样的都是MEC
            #       Sr: path.keys() 和 Sn相交, 也就是从初始状态可以到达MEC的状态
            # suffix -> Sn:
            #       Sn = set(paths.keys()).intersection(sf)
            #       这里Sn和之前定义其实是一样的
            #       可由初态到达, 且可到达MEC但是不在MEC内
            for s in Sn:
                for u in act[s]:
                    Y[(s, u)] = suffix_solver.NumVar(0, 1000, 'y[(%s, %s)]' % (s, u))
            print('Variables added: %d' % len(Y))
            # set objective
            obj = 0
            for s in Sn:
                for u in act[s]:
                    for t in sync_mec.successors(s):
                        prop = sync_mec[s][t]['prop'].copy()
                        if u in list(prop.keys()):
                            pe = prop[u][0]
                            ce = prop[u][1]
                            obj += Y[(s, u)] * pe * ce
            suffix_solver.Minimize(obj)
            print('Objective added')

            # add constraints
            # --------------------
            #
            # description of each constraints
            constr_descrip = []
            #
            # Added
            # constraint 1
            # 由于sync_mec内所有sync_state[0]对应的状态是一个状态, 所以其对应概率之和就应该是1
            # it seems that there are some issues on the model, so this constraint cannot be applied,
            # but we use the normalized result as policies such that all values can be constrainted into [0, 1]
            '''
            for s_pi_t in Sn_pi:
                constr_s_pi = []
                for s_sync_t in Sn:
                    for u_t in act[s_sync_t]:
                        if s_sync_t[0] == s_pi_t:
                            Y_t = Y[(s_sync_t, u_t)]            # 单独拿出来是为了debugging
                            constr_s_pi.append(Y_t)
                if constr_s_pi.__len__():
                    sum_t = suffix_solver.Sum(constr_s_pi)
                    suffix_solver.Add(sum_t == 1.)
                    constr_descrip.append("merge of " + str(s_pi_t))
            print_c('inter-state constraints added', color=42)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(), ), color=42)
            '''
            # 这个约束就算错的
            # 真正的flow是由balance, 即贝尔曼方程保证的
            # 不是这个东西

            # Constraint 2 / 11b
            constr_f_in = []
            constr_f_out = []
            constr_s_in = []
            constr_s_out = []
            constr_s_in_f_in = []
            constr_y0_2_f_rhs = 0.0

            with Progress() as progress:
                task_id = progress.add_task("Constraint 2 / 11b ...", total=len(Sn))

                for k, sync_s in enumerate(Sn):
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing constraint 2 / 11b ... {k + 1}/{len(Sn)}")
                    # for f_in and s_in
                    # in this way, the constaint will not be repeated
                    successors_to_visit = set(sync_mec.successors(sync_s))
                    for t in successors_to_visit:               # for t in list(sync_mec.successors(sync_s)):
                        # 检查 t 是否可索引且长度大于 0
                        if isinstance(t, (tuple, list)) and len(t) > 0:
                            if t in Sn and (t in ip or t[0] in S_pi):
                                # 检查边是否存在
                                if sync_mec.has_edge(sync_s, t):
                                    prop = sync_mec[sync_s][t]['prop'].copy()
                                    for u in prop.keys():
                                        # 检查 (sync_s, u) 是否在 Y 中
                                        if (sync_s, u) in Y:
                                            y_t = Y[(sync_s, u)] * prop[u][0]
                                            constr_s_in_f_in.append(y_t)

                    # for f_in
                    for t in successors_to_visit:               # for t in list(sync_mec.successors(sync_s)):
                        if t in Sn and t in ip:
                            if sync_mec.has_edge(sync_s, t):
                                prop = sync_mec[sync_s][t]['prop'].copy()
                                for u in prop.keys():
                                    if (sync_s, u) in Y:
                                        y_t = Y[(sync_s, u)] * prop[u][0]
                                        constr_f_in.append(y_t)

                    # for y_0
                    if sync_s in y_in_sf_sync:
                        constr_y0_2_f_rhs += y_in_sf_sync[sync_s]

                    # for f_out
                    if sync_s in Sn and sync_s in ip:
                        for t in successors_to_visit:           # for t in sync_mec.successors(sync_s):
                            if sync_mec.has_edge(sync_s, t):
                                prop = sync_mec[sync_s][t]['prop'].copy()
                                for u in prop.keys():
                                    if sync_s in act and u in act[sync_s]:
                                        #
                                        # Sn里进Ip的
                                        pe = prop[u][0]
                                        Y_t = Y[(sync_s, u)] * pe
                                        constr_f_out.append(Y_t)

                    # for s_in
                    for t in successors_to_visit:
                        if isinstance(t, (tuple, list)) and len(t) > 0:
                            if t in Sn and t[0] in S_pi:
                                if sync_mec.has_edge(sync_s, t):
                                    prop = sync_mec[sync_s][t]['prop'].copy()
                                    for u in prop.keys():
                                        if sync_s in act and u in act[sync_s]:
                                            #
                                            # Sn里进Ip的
                                            pe = prop[u][0]
                                            Y_t = Y[(sync_s, u)] * pe
                                            constr_s_in.append(Y_t)

                    # for s_out
                    if sync_s in Sn and isinstance(sync_s, (tuple, list)) and len(sync_s) > 0 and sync_s[0] in S_pi:
                        for t in successors_to_visit:
                            if sync_mec.has_edge(sync_s, t):
                                prop = sync_mec[sync_s][t]['prop'].copy()
                                for u in prop.keys():
                                    if sync_s in act and u in act[sync_s]:
                                        pe = prop[u][0]
                                        Y_t = Y[(sync_s, u)] * pe
                                        constr_s_out.append(Y_t)

            # Sum constraints
            sum_s_in = suffix_solver.Sum(constr_s_in)
            sum_s_out = suffix_solver.Sum(constr_s_out)
            sum_f_in = suffix_solver.Sum(constr_f_in)
            sum_f_out = suffix_solver.Sum(constr_f_out)
            #
            sum_s_in_f_in = suffix_solver.Sum(constr_s_in_f_in)

            # Add constraints
            suffix_solver.Add(sum_s_in_f_in == constr_y0_2_f_rhs)
            constr_descrip.append('I_in for (11b)')

            print_c("Reachability constraint added ...", color=44)
            print_c(f"left:  {sum_s_in_f_in} \n right: {constr_y0_2_f_rhs}", color=44)
            print_c(f"number of states in lhs: {len(constr_s_in_f_in)}", color=44)
            print_c(f'number of constraints: {suffix_solver.NumConstraints()}', color=44)

            suffix_solver.Add(sum_s_in == sum_f_out)
            constr_descrip.append("f_out -> s_in")
            suffix_solver.Add(sum_f_in == sum_s_out)
            constr_descrip.append("s_out -> f_in")

            print("Repeated reachability constraint added")
            print_c('f_out -> s_in', color=34)
            print_c(constr_f_out, color=34)
            print_c(constr_s_in, color=34)
            print_c('s_out -> f_in', color=35)
            print_c(constr_s_out, color=35)
            print_c(constr_f_in, color=35)

            # Constraint 3 / 11c
            #
            # here we treat all sync states that represents original states togetherly,
            # Plan B is to treat these states in sync mec separately, which may make it infeasible to solve
            # due to too much constraints
            nonzero_constr_num_11c = 0
            nonzero_balance_constr_list = []

            with Progress() as progress:
                task_id = progress.add_task("Constraint 3 / 11c ...", total=len(Sn_pi))
                for k, s_pi in enumerate(Sn_pi):
                    #
                    # for debugging
                    # if s_pi == ((1.25, 2.25, 'W'), frozenset({'supply'}), 7):
                    #     debug_var = 1
                    # if s_pi == ((1.25, 2.25, 'W'), frozenset({'supply'}), 2):
                    #     debug_var = 2
                    progress.update(task_id, advance=1, description=f"Processing constraint 3 / 11c ... {k + 1}/{len(Sn_pi)}")

                    #
                    constr_11c_lhs = []
                    constr_11c_rhs = []

                    for l, sync_s in enumerate(Sn):
                        if s_pi != sync_s[0]:
                            continue
                        for u in act.get(sync_s, []):
                            if (sync_s, u) in Y:
                                y_t = Y[(sync_s, u)]
                                constr_11c_lhs.append(y_t)

                        for f in sync_mec.predecessors(sync_s):  # 求解对象不一样了, product mdp -> sync_mec
                            if (f in Sn and sync_s not in ip) or (f in Sn and sync_s in ip and f != sync_s):
                                if sync_mec.has_edge(f, sync_s):
                                    prop = sync_mec[f][sync_s]['prop'].copy()
                                    for uf in act.get(f, []):
                                        if uf in prop:
                                            y_t_p_e = Y[(f, uf)] * prop[uf][0]
                                            constr_11c_rhs.append(y_t_p_e)
                                    else:
                                        y_t_p_e = Y[(f, uf)] * 0.00
                                        # constr_11c_rhs.append(y_t_p_e)

                    sum_11c_lhs = suffix_solver.Sum(constr_11c_lhs)
                    sum_11c_rhs = suffix_solver.Sum(constr_11c_rhs)
                    #
                    if (s_pi in list(y_in_sf.keys())) and (s_pi not in ip_pi):
                        suffix_solver.Add(sum_11c_lhs == sum_11c_rhs + y_in_sf[s_pi])
                        #
                        # for debugging
                        constr_descrip.append(str(s_pi))
                        #
                        # Added, for debugging
                        print_c("constraint: %d" % (k,), color=37)
                        print_c(sum_11c_lhs, color=38)
                        print_c(sum_11c_rhs, color=39)
                        print_c(y_in_sf[s_pi], color=39)
                        print_c(" ")
                    #
                    # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                    if (s_pi in list(y_in_sf.keys())) and (s_pi in ip_pi):
                        suffix_solver.Add(sum_11c_lhs == y_in_sf[s_pi])
                        #
                        # for debugging
                        constr_descrip.append(str(s_pi))
                        #
                        # Added, for debugging
                        print_c("constraint: %d" % (k,), color=37)
                        print_c(sum_11c_lhs, color=38)
                        print_c(y_in_sf[s_pi], color=39)
                        print_c(" ")
                    #
                    # 如果s不在Sf内且不在MEC内
                    if (s_pi not in list(y_in_sf.keys())) and (s_pi not in ip_pi):
                        suffix_solver.Add(sum_11c_lhs == sum_11c_rhs)
                        #
                        # for debugging
                        constr_descrip.append(str(s_pi))
                        #
                        # Added, for debugging
                        print_c("constraint: %d" % (k,), color=37)
                        print_c(sum_11c_lhs, color=38)
                        print_c(sum_11c_rhs, color=39)
                        print_c(" ")

                    if s_pi in y_in_sf and y_in_sf[s_pi] != 0.0:
                        nonzero_constr_num_11c += 1
                        print_c(
                            f"NON-zero balance constraint {nonzero_constr_num_11c}: {s_pi} - \n left:  {sum_11c_lhs} \n right: {sum_11c_rhs} \n {y_in_sf[s_pi]}",
                            color=45)
                        current_constr_index_t = suffix_solver.NumConstraints() - 1
                        nonzero_balance_constr_list.append(current_constr_index_t)

            print_c("balance and initial distribution constraint added ...", color=44)
            print_c(f'number of constraints: {suffix_solver.NumConstraints()}', color=42)

            # Risk constraints
            y_to_ip = 0.0
            y_out = 0.0
            #
            with Progress() as progress:
                task_id = progress.add_task("Risk constraints ...", total=len(Sn))

                for k, s_sync in enumerate(Sn):
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing risk constraints ... {k + 1}/{len(Sn)}")

                    for t_sync in sync_mec.successors(s_sync):
                        if sync_mec.has_edge(s_sync, t_sync):
                            prop = sync_mec[s_sync][t_sync]['prop'].copy()
                            for u in prop.keys():
                                if u in act.get(s_sync, []):
                                    pe = prop[u][0]
                                    Y_t = Y[(s_sync, u)]
                                    if t_sync not in Sn:
                                        y_out += Y_t * pe
                                    elif t_sync in ip:
                                        y_to_ip += Y_t * pe

            suffix_solver.Add(y_to_ip >= (1.0 - gamma - delta) * (y_to_ip + y_out))
            constr_descrip.append('risk constraints')
            print_c('Risk constraint added', color=47)

            # Opacity constraints
            #
            # CRITICAL
            constr_opacity_lhs = []
            constr_opacity_rhs = differential_expected_cost

            with Progress() as progress:
                task_id = progress.add_task("Opacity constraints ...", total=len(Sn))

                for k, s in enumerate(Sn):
                    # 更新进度条的描述
                    progress.update(task_id, advance=1, description=f"Processing ... {k + 1}/{len(Sn)}")

                    for t in sync_mec.successors(s):
                        if sync_mec.has_edge(s, t):
                            prop = sync_mec[s][t]['diff_exp'].copy()
                            for u in prop.keys():
                                if (s, u) in Y:
                                    y_t = Y[(s, u)] * prop[u]
                                    constr_opacity_lhs.append(y_t)

            sum_opacity_lhs = suffix_solver.Sum(constr_opacity_lhs)
            suffix_solver.Add(sum_opacity_lhs <= constr_opacity_rhs)
            constr_descrip.append("differential_expected_cost")

            print_c("opacity constraint added ...", color=46)
            print_c(f'number of constraints: {suffix_solver.NumConstraints()}', color=46)
            constr_opacity_index = suffix_solver.NumConstraints() - 1

            # ------------------------------
            # solve
            print('--optimization for suffix starts--')
            status = suffix_solver.Solve()
            #
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
            # v = dict()
            # for sync_state_u_t in list(Y.keys()):
            #     state_pi_t = sync_state_u_t[0][0]
            #     u_t = sync_state_u_t[0][0]
            #     if state_pi_t == ((1.25, 2.25, 'W'), frozenset({'supply'}), 7):
            #         debug_var = 3
            #         v[sync_state_u_t] = Y[sync_state_u_t].solution_value()
            #     if state_pi_t == ((1.25, 2.25, 'W'), frozenset({'supply'}), 2):
            #         debug_var = 4
            #         v[sync_state_u_t] = Y[sync_state_u_t].solution_value()

            #
            #
            #
            plan_suffix_non_round_robin_list = []
            raw_value = dict()
            for s_u_t in Y.keys():
                s_t = s_u_t[0]
                u_t = s_u_t[1]
                val_t = Y[s_u_t].solution_value()
                #
                if s_t not in raw_value.keys():
                    act_value_tuple = [[], []]
                    act_value_tuple[0].append(u_t)
                    act_value_tuple[1].append(val_t)
                    raw_value[s_t] = act_value_tuple
                else:
                    raw_value[s_t][0].append(u_t)
                    raw_value[s_t][1].append(val_t)

            plan_suffix = dict()
            for k, s_pi_t in enumerate(Sn_pi):
                #
                norm = 0
                U = []
                P = []
                for s_sync_t in Sn:
                    for u_t in act[s_sync_t]:
                        if s_pi_t == s_sync_t[0]:
                            Y_t = Y[(s_sync_t, u_t)].solution_value()
                            norm += Y_t

                norm_sync_state_t = dict()
                for s_sync_t in Sn:
                    if s_pi_t != s_sync_t[0]:
                        continue

                    for u_t in act[s_sync_t]:
                        if u_t not in U:
                            U.append(u_t)
                        if u_t not in norm_sync_state_t.keys():
                            norm_sync_state_t[u_t]  = Y[(s_sync_t, u_t)].solution_value()
                        else:
                            norm_sync_state_t[u_t] += Y[(s_sync_t, u_t)].solution_value()
                if norm > 0.01:
                    for u_t in norm_sync_state_t.keys():
                        # U.append(u_t)
                        P.append(norm_sync_state_t[u_t] / norm)
                        #
                        # for debugging
                        plan_suffix_non_round_robin_list.append(k)
                else:
                    for u_t in norm_sync_state_t.keys():
                        P.append(1.0 / len(norm_sync_state_t.keys()))          # the length of act_pi[s_pi_t] is equal to that of norm_sync_state_t.keys()
                    # #P.append(1.0 / len(act_pi[s_pi_t]))             # round robin
                plan_suffix[s_pi_t] = [U, P]
            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")

            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
            for s_sync_t in Sn:
                s_pi_t = s_sync_t[0]
                for t_pi_t in prod_mdp.successors(s_pi_t):
                    prop = prod_mdp[s_pi_t][t_pi_t]['prop'].copy()
                    for u_t in prop.keys():
                        if u_t in act[s_sync_t]:
                            pe = prop[u_t][0]
                            Y_t = Y[(s_sync_t, u_t)].solution_value()
                            if t_pi_t not in Sn_pi:
                                y_out += Y_t * pe
                            elif t_pi_t in ip_pi:
                                y_to_ip += Y_t * pe
            #
            if (y_to_ip + y_out) > 0:
                risk = y_out / (y_to_ip + y_out)
            #
            print_c('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip + y_out,), color=32)
            print_c("----Suffix risk computed", color=32)

            constr_t = suffix_solver.constraint(constr_opacity_index)
            opacity_val = 0.
            for s_sync_t in Sn:
                for u_t in act[s_sync_t]:
                    Y_t = Y[(s_sync_t, u_t)]  # 单独拿出来是为了debugging
                    if type(Y_t) != float:
                        ki_t = constr_t.GetCoefficient(Y_t)
                        opacity_val += ki_t * Y_t.solution_value()
            print_c("opacity_value: %f <= %f" % (opacity_val, differential_expected_cost, ), color=35)
            print_c("----Suffix opacity threshold computed", color=35)

            #
            # for display
            for k in range(0, suffix_solver.NumConstraints()):
                try:
                    constr_t = suffix_solver.constraint(k)
                    sum_ret_t = 0.
                    for s_sync_t in Sn:
                        for u_t in act[s_sync_t]:
                            Y_t = Y[(s_sync_t, u_t)]  # 单独拿出来是为了debugging
                            if type(Y_t) != float:
                                ki_t = constr_t.GetCoefficient(Y_t)
                                sum_ret_t += ki_t * Y_t.solution_value()
                    if k in nonzero_balance_constr_list:
                        print_c("constraint_%d %s: %f <= %f <= %f" % (k, constr_descrip[k], constr_t.lb(), sum_ret_t, constr_t.ub(),), color=45)
                    elif k == constr_opacity_index:
                        print_c("constraint_%d %s: %f <= %f <= %f" % (k, constr_descrip[k], constr_t.lb(), sum_ret_t, constr_t.ub(),), color=46)
                    else:
                        print("constraint_%d %s: %f <= %f <= %f"   % (k, constr_descrip[k], constr_t.lb(), sum_ret_t ,constr_t.ub(), ))
                except IndexError:
                    pass

            '''                
            print("optimal policies: ")
            Sn_pi_sorted = list(Sn_pi)
            #Sn_pi_sorted.sort(key=cmp_to_key(sort_grids))
            for k, s_pi_t in enumerate(Sn_pi_sorted):
                #
                # for s_sync_t in Sn:
                #     if s_sync_t[0] == s_pi_t:
                #         print(str(s_sync_t) + " " + str(plan_suffix[s_sync_t]))
                if k in plan_suffix_non_round_robin_list:
                    print_c(str(s_pi_t) + " " + str(plan_suffix[s_pi_t]), color=46)
                else:
                    print(str(s_pi_t) + " " + str(plan_suffix[s_pi_t]))
            '''

            return plan_suffix, cost, risk, opacity_val
        '''
        except:
            print("ORtools Error reported")
            return None, None, None
        '''

def syn_plan_suffix_repeated(prod_mdp, MEC, S_pi, y_in_sf):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print("===========[plan suffix synthesis starts]")
    sf = MEC[0]  # MEC
    ip = MEC[1]  # MEC 和 ip 的交集
    act = MEC[2].copy()  # 所有状态的动作集合全集
    delta = 0.01  # 松弛变量?
    gamma = 0.00  # 根据(11), 整个系统进入MEC内以后就不用概率保证了?
    for init_node in prod_mdp.graph['initial']:
        paths = single_source_shortest_path(prod_mdp, init_node)
        Sn = set(paths.keys()).intersection(sf)
        print('Sf size: %s' % len(sf))
        print('reachable sf size: %s' % len(Sn))
        print('Ip size: %s' % len(ip))
        print('Ip and sf intersection size: %s' % len(Sn.intersection(ip)))
        # ---------solve lp------------
        print('------')
        print('ORtools for suffix starts now')
        print('------')
        # TODO for debugging
        #try:
        if True:
            Y = defaultdict(float)
            suffix_solver = pywraplp.Solver.CreateSolver('GLOP')
            # create variables
            #
            # 注意这里和prefix不同的是, prefix
            # prefix -> Sr:
            #       path = single_source_shortest_path(simple_digraph, random.sample(ip, 1)[0]), 即可以到达MEC的状态
            #       Sn: path_init.keys() 和 Sf的交集, 两变Sf定义是一样的都是MEC
            #       Sr: path.keys() 和 Sn相交, 也就是从初始状态可以到达MEC的状态
            # suffix -> Sn:
            #       Sn = set(paths.keys()).intersection(sf)
            #       这里Sn和之前定义其实是一样的
            #       可由初态到达, 且可到达MEC但是不在MEC内
            for s in Sn:
                for u in act[s]:
                    Y[(s, u)] = suffix_solver.NumVar(0, 1000, 'y[(%s, %s)]' % (s, u))
            print('Variables added: %d' % len(Y))
            # set objective
            obj = 0
            for s in Sn:
                for u in act[s]:
                    for t in prod_mdp.successors(s):
                        prop = prod_mdp[s][t]['prop'].copy()
                        if u in list(prop.keys()):
                            pe = prop[u][0]
                            ce = prop[u][1]
                            obj += Y[(s, u)] * pe * ce
            suffix_solver.Minimize(obj)
            print('Objective added')
            # add constraints
            # --------------------
            #
            # constraint 11b
            constr_f_in  = 0.
            constr_f_out = 0.
            constr_s_in  = 0.
            constr_s_out = 0.
            #
            constr_s_in_f_in = 0.
            constr_y0 = 0.
            for k, s in enumerate(Sn):
                #
                # for modified 11b lhs, i.e., f_in
                for t in prod_mdp.successors(s):
                    if t in Sn and (t in ip or t in S_pi):
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                constr_s_in_f_in += Y[(s, u)] * pe

                # for f_in
                for t in prod_mdp.successors(s):
                    if t in Sn and t in ip:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                constr_f_in += Y[(s, u)] * pe
                #
                # for 11b rhs, i.e., y_0
                if (s in list(y_in_sf.keys())):
                    constr_y0 += y_in_sf[s]

                # for f_out
                if s in Sn and s in ip:
                    for t in prod_mdp.successors(s):
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                constr_f_out += Y[(s, u)] * pe
                # for s_in
                for t in prod_mdp.successors(s):
                    if t in Sn and t in S_pi:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                constr_s_in += Y[(s, u)] * pe
                # for s_out
                if s in Sn and s in S_pi:
                    for t in prod_mdp.successors(s):
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                constr_s_out += Y[(s, u)] * pe
            #
            #
            # 这一项对结果其实影响很大, 很容易让很多确定性策略变成随机策略
            suffix_solver.Add(constr_s_in_f_in == constr_y0)
            print("Reachability constraint added")
            print_c(constr_s_in_f_in, color=32)
            print_c(constr_y0, color=32)
            #
            suffix_solver.Add(constr_s_in == constr_f_out)
            suffix_solver.Add(constr_f_in == constr_s_out)
            #
            print("Repeated reachability constraint added")
            print_c('f_out -> s_in', color=34)
            print_c(constr_f_out, color=34)
            print_c(constr_s_in,  color=34)
            print_c('s_out -> f_in', color=35)
            print_c(constr_s_out, color=35)
            print_c(constr_f_in, color=35)
            #
            # constrain 11c
            for k, s in enumerate(Sn):
                #
                # constr3: sum of outflow
                # constr4: sum of inflow
                constr3 = 0
                constr4 = 0
                for u in act[s]:
                    #
                    # 这里也有不同
                    # prefix
                    #       s in Sr
                    # suffix
                    #       s in Sn
                    constr3 += Y[(s, u)]
                for f in prod_mdp.predecessors(s):
                    #
                    # 这里也有不同
                    # prefix
                    #       if f in Sr:
                    # suffix
                    #       if f in Sn 且 s in Sn 且 s not in ip
                    if (f in Sn) and (s not in ip):
                        prop = prod_mdp[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                constr4 += Y[(f, uf)] * prop[uf][0]
                            else:
                                constr4 += Y[(f, uf)] * 0.00
                    if (f in Sn) and (s in ip) and (f != s):
                        prop = prod_mdp[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                constr4 += Y[(f, uf)] * prop[uf][0]
                            else:
                                constr4 += Y[(f, uf)] * 0.00
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
                if (s in list(y_in_sf.keys())) and (s not in ip):
                    suffix_solver.Add(constr3 == constr4 + y_in_sf[s])  # 可能到的了的要计算?
                    #
                    # Added, for debugging
                    print_c("constraint: %d" % (k,), color=37)
                    print_c(constr3, color=38)
                    print_c(constr4, color=39)
                    print_c(y_in_sf[s], color=39)
                    print_c(" ")
                #
                # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                if (s in list(y_in_sf.keys())) and (s in ip):
                    suffix_solver.Add(constr3 == y_in_sf[s])  # 在里面的永远到的了?
                    #
                    # Added, for debugging
                    print_c("constraint: %d" % (k,), color=37)
                    print_c(constr3, color=38)
                    print_c(y_in_sf[s], color=39)
                    print_c(" ")
                #
                # 如果s不在Sf内且不在NEC内
                if (s not in list(y_in_sf.keys())) and (s not in ip):
                    suffix_solver.Add(constr3 == constr4)  # 到不了的永远到不了?
                    #
                    # Added, for debugging
                    print_c("constraint: %d" % (k,), color=37)
                    print_c(constr3, color=38)
                    print_c(constr4, color=39)
                    print_c(" ")
            print('Balance condition added')
            print('Initial sf condition added')
            # --------------------
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                for t in prod_mdp.successors(s):
                    if t not in Sn:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                #
                                # Sn里出Sn的
                                y_out += Y[(s, u)] * pe
                    elif t in ip:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #
                                # Sn里进Ip的
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)] * pe
            # suffix_solver.Add(y_to_ip+y_out >= delta)
            suffix_solver.Add(y_to_ip >= (1.0 - gamma - delta) * (y_to_ip + y_out))
            print_c("y_2_ip:", color=35)
            print_c(y_to_ip, color=35)
            print_c("y_out:", color=36)
            print_c(y_out, color=36)
            print('Risk constraint added')
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
                return None, None, None
            # compute optimal plan suffix given the LP solution
            plan_suffix = dict()
            for s in Sn:
                norm = 0
                U = []
                P = []
                for u in act[s]:
                    norm += Y[(s, u)].solution_value()
                for u in act[s]:
                    U.append(u)
                    if norm > 0.01:
                        P.append(Y[(s, u)].solution_value() / norm)
                    else:
                        P.append(1.0 / len(act[s]))
                plan_suffix[s] = [U, P]
            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")
            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                for t in prod_mdp.successors(s):
                    if t not in Sn:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_out += Y[(s, u)].solution_value() * pe
                    elif t in ip:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)].solution_value() * pe
            if (y_to_ip + y_out) > 0:
                risk = y_out / (y_to_ip + y_out)
            print('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip + y_out))
            print("----Suffix risk computed")
            return plan_suffix, cost, risk
        # TODO for debugging
        #except:
        #    print("ORtools Error reported")
        #    return None, None, None

def syn_full_plan(prod_mdp, gamma, alpha=1):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    Plan = []
    for l, S_fi in enumerate(prod_mdp.Sf):                                                  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC in enumerate(S_fi):
            plan_prefix, prefix_cost, prefix_risk, y_in_sf, Sr, Sd = syn_plan_prefix(
                prod_mdp, MEC, gamma)
            print("Best plan prefix obtained, cost: %s, risk %s" %
                  (str(prefix_cost), str(prefix_risk)))
            if y_in_sf:
                plan_suffix, suffix_cost, suffix_risk = syn_plan_suffix(
                    prod_mdp, MEC, y_in_sf)
                print("Best plan suffix obtained, cost: %s, risk %s" %
                      (str(suffix_cost), str(suffix_risk)))
            else:
                plan_suffix = None
            if plan_prefix and plan_suffix:
                plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf], [
                            plan_suffix, suffix_cost, suffix_risk], [MEC[0], MEC[1], Sr, Sd]])
        if plan:
            best_k_plan = min(plan, key=lambda p: p[0][1] + alpha*p[1][1])
            Plan.append(best_k_plan)
        else:
            print("No valid found!")
    if Plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        best_all_plan = min(Plan, key=lambda p: p[0][1] + alpha*p[1][1])
        print('Best plan prefix obtained for %s states in Sr' %
              str(len(best_all_plan[0][0])))
        print('cost: %s; risk: %s ' %
              (best_all_plan[0][1], best_all_plan[0][2]))
        print('Best plan suffix obtained for %s states in Sf' %
              str(len(best_all_plan[1][0])))
        print('cost: %s; risk: %s ' %
              (best_all_plan[1][1], best_all_plan[1][2]))
        print('Total cost:%s' %
              (best_all_plan[0][1] + alpha*best_all_plan[1][1]))
        plan_bad = syn_plan_bad(prod_mdp, best_all_plan[2])
        print('Plan for bad states obtained for %s states in Sd' %
              str(len(best_all_plan[2][3])))
        best_all_plan.append(plan_bad)
        return best_all_plan
    else:
        print("No valid plan found")
        return None

def syn_full_plan_repeated(prod_mdp, gamma, opt_prop, alpha=1):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    Plan = []
    for l, S_fi in enumerate(prod_mdp.Sf):                                                  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC in enumerate(S_fi):
            plan_prefix, prefix_cost, prefix_risk, y_in_sf, Sr, Sd = syn_plan_prefix(
                prod_mdp, MEC, gamma)
            print("Best plan prefix obtained, cost: %s, risk %s" %
                  (str(prefix_cost), str(prefix_risk)))
            if y_in_sf:
                # Added
                S_pi = find_states_satisfying_opt_prop(opt_prop, MEC[0])
                #
                plan_suffix, suffix_cost, suffix_risk = syn_plan_suffix(
                    prod_mdp, MEC, y_in_sf)
                plan_suffix, suffix_cost, suffix_risk = syn_plan_suffix_repeated(
                    prod_mdp, MEC, S_pi, y_in_sf)
                print("Best plan suffix obtained, cost: %s, risk %s" %
                      (str(suffix_cost), str(suffix_risk)))
            else:
                plan_suffix = None
            if plan_prefix and plan_suffix:
                plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf], [
                            plan_suffix, suffix_cost, suffix_risk], [MEC[0], MEC[1], Sr, Sd]])
        if plan:
            best_k_plan = min(plan, key=lambda p: p[0][1] + alpha*p[1][1])
            Plan.append(best_k_plan)
        else:
            print("No valid found!")
    if Plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        best_all_plan = min(Plan, key=lambda p: p[0][1] + alpha*p[1][1])
        print('Best plan prefix obtained for %s states in Sr' %
              str(len(best_all_plan[0][0])))
        print('cost: %s; risk: %s ' %
              (best_all_plan[0][1], best_all_plan[0][2]))
        print('Best plan suffix obtained for %s states in Sf' %
              str(len(best_all_plan[1][0])))
        print('cost: %s; risk: %s ' %
              (best_all_plan[1][1], best_all_plan[1][2]))
        print('Total cost:%s' %
              (best_all_plan[0][1] + alpha*best_all_plan[1][1]))
        plan_bad = syn_plan_bad(prod_mdp, best_all_plan[2])
        print('Plan for bad states obtained for %s states in Sd' %
              str(len(best_all_plan[2][3])))
        best_all_plan.append(plan_bad)
        return best_all_plan
    else:
        print("No valid plan found")
        return None

def synthesize_full_plan_w_opacity(mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, alpha=1, observation_func=dra2.observation_func_1, is_enable_inter_state_constraints=True):
    t2 = time.time()

    task_pi = task + ' & GF ' + optimizing_ap
    ltl_converted_pi = ltl_convert(task_pi)

    dra = Dra(ltl_converted_pi)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3-t2))

    # ----
    prod_dra_pi = product_mdp2(mdp, dra)
    prod_dra_pi.compute_S_f()                       # for AMECs, finished in building
    # prod_dra.dotify()
    t41 = time.time()
    print('Product DRA done, time: %s' % str(t41-t3))

    pickle.dump((networkx.get_edge_attributes(prod_dra_pi, 'prop'),
                prod_dra_pi.graph['initial']), open('prod_dra_edges.p', "wb"))
    print('prod_dra_edges.p saved')


    # new main loop
    for l, S_fi_pi in enumerate(prod_dra_pi.Sf):                                                  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC_pi in enumerate(S_fi_pi):
            #
            # finding states that satisfying optimizing prop
            S_pi = find_states_satisfying_opt_prop(optimizing_ap, MEC_pi[0])
            #
            plan_prefix, prefix_cost, prefix_risk, y_in_sf, Sr, Sd = syn_plan_prefix(
                prod_dra_pi, MEC_pi, risk_pr)
            print("Best plan prefix obtained, cost: %s, risk %s" %
                  (str(prefix_cost), str(prefix_risk)))
            if y_in_sf:

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
                    # y_in_sf will be used as initial distribution
                    for p, S_fi_gamma in enumerate(prod_dra_gamma.Sf):
                        for q, MEC_gamma in enumerate(S_fi_gamma):
                            plan_prefix_p, prefix_cost_p, prefix_risk_p, y_in_sf_gamma, Sr_p, Sd_p = syn_plan_prefix(
                                prod_dra_gamma, MEC_gamma, risk_pr)

                            #prod_dra_pi.re_synthesize_sync_amec(y_in_sf, y_in_sf_gamma, optimizing_ap, ap_4_opacity MEC_pi, MEC_gamma, prod_dra_gamma, observation_func=observation_func)
                            prod_dra_pi.re_synthesize_sync_amec_rex(y_in_sf, y_in_sf_gamma, optimizing_ap, ap_4_opacity, MEC_pi, MEC_gamma, prod_dra_gamma, observation_func=observation_func)

                            # LP
                            plan_suffix, suffix_cost, suffix_risk, suffix_opacity_threshold = synthesize_suffix_cycle_in_sync_amec(prod_dra_pi, prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index], MEC_pi, y_in_sf, S_pi, differential_exp_cost)
                            #
                            print_c("Best plan suffix obtained, cost: %s, risk %s" % (str(suffix_cost), str(suffix_risk)), color=36)
                            print_c("=-------------------------------------=", color=36)

                            if plan_prefix and plan_suffix:
                                plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf],
                                             [plan_suffix, suffix_cost, suffix_risk],
                                             [MEC_pi[0], MEC_pi[1], Sr, Sd],
                                             [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index, MEC_gamma[0], MEC_gamma[1]],
                                             prod_dra_pi])

                                print_policies_w_opacity(ap_4_opacity, plan_prefix, plan_suffix)
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
            print_c('Opacity threshold %f <= %f' % (best_all_plan[3][1], differential_exp_cost, ))
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

def synthesize_full_plan_w_opacity_4_Team_MDP(team_mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, alpha=1, observation_func=dra2.observation_func_1, is_enable_inter_state_constraints=True):
    t2 = time.time()

    task_pi = task + ' & GF ' + optimizing_ap
    ltl_converted_pi = ltl_convert(task_pi)

    dra = Dra(ltl_converted_pi)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3-t2))

    # ----
    prod_dra_pi = Team_Product_Dra(team_mdp, dra)
    prod_dra_pi.compute_S_f()                          # for AMECs
    # prod_dra.dotify()
    t41 = time.time()
    print('Product DRA done, time: %s' % str(t41-t3))

    pickle.dump((networkx.get_edge_attributes(prod_dra_pi, 'prop'),
                prod_dra_pi.graph['initial']), open('prod_dra_edges.p', "wb"))
    print('prod_dra_edges.p saved')


    # new main loop
    for l, S_fi_pi in enumerate(prod_dra_pi.Sf):                                                  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC_pi in enumerate(S_fi_pi):
            #
            # finding states that satisfying optimizing prop
            S_pi = find_states_satisfying_opt_prop(optimizing_ap, MEC_pi[0])
            #
            plan_prefix, prefix_cost, prefix_risk, y_in_sf, Sr, Sd = syn_plan_prefix(
                prod_dra_pi, MEC_pi, risk_pr)
            print("Best plan prefix obtained, cost: %s, risk %s" %
                  (str(prefix_cost), str(prefix_risk)))
            if y_in_sf:

                for ap_4_opacity in ap_list:
                    if ap_4_opacity == optimizing_ap:
                        continue
                    # synthesize product mdp for opacity
                    task_gamma = task + ' & GF ' + ap_4_opacity
                    ltl_converted_gamma = ltl_convert(task_gamma)
                    dra = Dra(ltl_converted_gamma)
                    prod_dra_gamma = Team_Product_Dra(team_mdp, dra)
                    prod_dra_gamma.compute_S_f()  # for AMECs

                    #
                    # y_in_sf will be used as initial distribution
                    for p, S_fi_gamma in enumerate(prod_dra_gamma.Sf):
                        for q, MEC_gamma in enumerate(S_fi_gamma):
                            plan_prefix_p, prefix_cost_p, prefix_risk_p, y_in_sf_gamma, Sr_p, Sd_p = syn_plan_prefix(
                                prod_dra_gamma, MEC_gamma, risk_pr)

                            #prod_dra_pi.re_synthesize_sync_amec(y_in_sf, y_in_sf_gamma, MEC_pi, MEC_gamma, prod_dra_gamma, observation_func=observation_func)
                            prod_dra_pi.re_synthesize_sync_amec_rex(y_in_sf, y_in_sf_gamma, MEC_pi, MEC_gamma, prod_dra_gamma, observation_func=observation_func)

                            if prod_dra_pi.sync_amec_set.__len__() == 0:
                                continue

                            # LP
                            plan_suffix, suffix_cost, suffix_risk, suffix_opacity_threshold = synthesize_suffix_cycle_in_sync_amec(prod_dra_pi, prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index], MEC_pi, y_in_sf, S_pi, differential_exp_cost)
                            #
                            print_c("Best plan suffix obtained, cost: %s, risk %s" % (str(suffix_cost), str(suffix_risk)), color=36)
                            print_c("=-------------------------------------=", color=36)

                            if plan_prefix and plan_suffix:
                                plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf],
                                             [plan_suffix, suffix_cost, suffix_risk],
                                             [MEC_pi[0], MEC_pi[1], Sr, Sd],
                                             [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index, MEC_gamma[0], MEC_gamma[1]],
                                             prod_dra_pi])

                                print_policies_w_opacity(ap_4_opacity, plan_prefix, plan_suffix)
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
            print_c('Opacity threshold %f <= %f' % (best_all_plan[3][1], differential_exp_cost, ))
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