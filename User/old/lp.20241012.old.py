from MDP_TG import lp
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_plan_prefix, syn_plan_suffix, syn_plan_bad
from User.dra2 import product_mdp2

from collections import defaultdict
import random
from ortools.linear_solver import pywraplp
from networkx import single_source_shortest_path

from subprocess import check_output
from User.vis2  import print_c

import pickle
import time
import networkx

import sys
sys.setrecursionlimit(3000)         # change stack size to guarantee runtime stability

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
    sync_sn = []
    for state_pi_t in sn_pi:
        for sync_state_t in sync_mec.nodes:
            if state_pi_t == sync_state_t[0]:
                if sync_state_t not in sync_sn:
                    sync_sn.append(sync_state_t)

    return sync_sn

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

def synthesize_suffix_cycle_in_sync_amec(prod_mdp, sync_mec, MEC_pi, y_in_sf):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print_c("===========[plan suffix synthesis starts]", color=32)
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
    gamma = 0.00                    # 根据(11), 整个系统进入MEC内以后就不用概率保证了?
    for init_node in prod_mdp.graph['initial']:
        # find states that reachable from initial state
        paths = single_source_shortest_path(prod_mdp, init_node)
        Sn_pi = set(paths.keys()).intersection(sf_pi)
        print('Sf_pi size: %s' % len(sf_pi))                                                # sf: MEC中状态
        print('reachable sf_pi size: %s' % len(Sn_pi))                                      # Sn: 可由当前状态到达的MEC中的状态
        print('Ip_pi size: %s' % len(ip_pi))                                                # Ip: 可被接收的MEC的状态
        print('Ip_pi and sf intersection size: %s' % len(Sn_pi.intersection(ip_pi)))        # 可达的MEC中的状态
        #
        # TODO
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

            # constraint 2 / 11b
            constr_11b_lhs = []
            constr_11b_rhs = 0.
            for k, s_pi in enumerate(Sn_pi):
                #
                for l, sync_s in enumerate(Sn):
                    if s_pi != sync_s[0]:
                        continue
                    for t in list(sync_mec.successors(sync_s)):
                        if t not in i_in:
                            continue
                        prop = sync_mec[sync_s][t]['prop'].copy()
                        for u in prop.keys():
                            try:
                                y_t = Y[(sync_s, u)] * prop[u][0]
                                constr_11b_lhs.append(y_t)
                            except KeyError:
                                pass
                    if sync_s in list(y_in_sf_sync.keys()):
                        constr_11b_rhs += y_in_sf_sync[sync_s]
            sum_11b_lhs = suffix_solver.Sum(constr_11b_lhs)
            #
            suffix_solver.Add(sum_11b_lhs == constr_11b_rhs)
            constr_descrip.append('I_in for (11b)')
            #
            print_c("reachibility constraint added ...", color=44)
            print_c("left:  %s \n right: %f" % (sum_11b_lhs, constr_11b_rhs,), color=44)
            print_c("number of states in lhs: %d" % (constr_11b_lhs.__len__(),), color=44)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(),), color=44)

            #
            # it seems that whether treat these states separately will not make a difference
            # because all states will be summed up together
            '''
            constr_11b_lhs = []
            constr_11b_rhs = 0.
            for k, s in enumerate(Sn):
                for t in list(sync_mec.successors(s)):
                    if t not in i_in:
                        continue
                    prop = sync_mec[s][t]['prop'].copy()
                    for u in prop.keys():
                        try:
                            y_t = Y[(s, u)] * prop[u][0]
                            constr_11b_lhs.append(y_t)
                        except KeyError:
                            pass
                if s in list(y_in_sf_sync.keys()):
                    constr_11b_rhs += y_in_sf_sync[s]
            sum_11b_lhs = suffix_solver.Sum(constr_11b_lhs)
            #
            suffix_solver.Add(sum_11b_lhs == constr_11b_rhs)
            #
            print_c("reachibility constraint added ...", color=44)
            print_c("left:  %s \n right: %f" % (sum_11b_lhs, constr_11b_rhs,), color=44)
            print_c("number of states in lhs: %d" % (constr_11b_lhs.__len__(),), color=44)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(),), color=44)
            '''

            # constraint 3 / 11c
            nonzero_constr_num_11c = 0
            nonzero_balance_constr_list = []
            for k, s_pi in enumerate(Sn_pi):
                #
                constr_11c_lhs = []
                constr_11c_rhs = []
                for l, sync_s in enumerate(Sn):
                    if s_pi != sync_s[0]:
                        continue
                    for u in act[sync_s]:
                        y_t = Y[(sync_s, u)]
                        constr_11c_lhs.append(y_t)
                        #
                    for f in sync_mec.predecessors(sync_s):  # 求解对象不一样了, product mdp -> sync_mec
                        if (f in Sn and sync_s not in ip) or (f in Sn and sync_s in ip and f != sync_s):
                            prop = sync_mec[f][sync_s]['prop'].copy()
                            for uf in act[f]:
                                if uf in list(prop.keys()):
                                    y_t_p_e = Y[(f, uf)] * prop[uf][0]
                                    constr_11c_rhs.append(y_t_p_e)
                                else:
                                    y_t_p_e = Y[(f, uf)] * 0.00
                                    # constr_11c_rhs.append(y_t_p_e)
                #
                sum_11c_lhs = suffix_solver.Sum(constr_11c_lhs)
                sum_11c_rhs = suffix_solver.Sum(constr_11c_rhs)
                #
                if (s_pi in list(y_in_sf.keys())) and (s_pi not in ip_pi):
                    suffix_solver.Add(sum_11c_lhs == sum_11c_rhs + y_in_sf[s_pi])
                    #
                    # for debugging
                    constr_descrip.append(str(s_pi))
                #
                # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                if (s_pi in list(y_in_sf.keys())) and (s_pi in ip_pi):
                    suffix_solver.Add(sum_11c_lhs == y_in_sf[s_pi])
                    #
                    # for debugging
                    constr_descrip.append(str(s_pi))
                #
                # 如果s不在Sf内且不在NEC内
                if (s_pi not in list(y_in_sf.keys())) and (s_pi not in ip_pi):
                    suffix_solver.Add(sum_11c_lhs == sum_11c_rhs)
                    #
                    # for debugging
                    constr_descrip.append(str(s_pi))

                if s_pi in list(y_in_sf.keys()) and y_in_sf[s_pi] != 0.:
                    nonzero_constr_num_11c += 1
                    print_c("NON-zero balance constraint %d: %s - \n left:  %s \n right: %s \n %f" % (
                    nonzero_constr_num_11c, str(s_pi), sum_11c_lhs, sum_11c_rhs, y_in_sf[s_pi]), color=45)
                    #
                    # record current constraint index in which y0 != 0.
                    current_constr_index_t = suffix_solver.NumConstraints() - 1
                    nonzero_balance_constr_list.append(current_constr_index_t)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(),), color=42)

            #
            # Plan B is to treat these states in sync mec separately, which may make it infeasible to solve
            # due to too much constraints
            '''
            nonzero_constr_num_11c = 0
            for k, s in enumerate(Sn):
                #
                constr_11c_lhs = []
                constr_11c_rhs = []
                for u in act[s]:
                    y_t = Y[(s, u)]
                    constr_11c_lhs.append(y_t)
                #
                for f in sync_mec.predecessors(s):                      # 求解对象不一样了, product mdp -> sync_mec
                    if (f in Sn and s not in ip) or (f in Sn and s in ip and f != s):
                        prop = sync_mec[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                y_t_p_e = Y[(f, uf)] * prop[uf][0]
                                constr_11c_rhs.append(y_t_p_e)
                            else:
                                y_t_p_e = Y[(f, uf)] * 0.00
                                #constr_11c_rhs.append(y_t_p_e)
                #
                sum_11c_lhs = suffix_solver.Sum(constr_11c_lhs)
                sum_11c_rhs = suffix_solver.Sum(constr_11c_rhs)
                #
                # TODO To remove comments
                if (s in list(y_in_sf_sync.keys())) and (s not in ip):
                    suffix_solver.Add(sum_11c_lhs == sum_11c_rhs + y_in_sf_sync[s])
                #
                # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                if (s in list(y_in_sf_sync.keys())) and (s in ip):
                    suffix_solver.Add(sum_11c_lhs == y_in_sf_sync[s])
                #
                # 如果s不在Sf内且不在MEC内
                if (s not in list(y_in_sf_sync.keys())) and (s not in ip):
                    suffix_solver.Add(sum_11c_lhs == sum_11c_rhs)


                if s in list(y_in_sf_sync.keys()) and y_in_sf_sync[s] != 0.:
                    nonzero_constr_num_11c += 1
                    print_c("NON-zero balance constraint %d: %s - \n left:  %s \n right: %s \n %f" % (
                    nonzero_constr_num_11c, str(s), sum_11c_lhs, sum_11c_rhs, y_in_sf_sync[s]), color=45)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(),), color=42)
            '''

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
                #
                # TODO
                # 整个cost要基于原product_mdp映射回来
                # 映射回来的概率要乘以gamma系统的状态转移概率
                # 后面这一步是会有很大问题的
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

            # TODO
            # compute optimal plan suffix given the LP solution
            '''
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
                        P.append(1.0 / len(act[s]))             # round robin
                plan_suffix[s] = [U, P]
            '''
            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")

            # TODO
            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
            for s_sync_t in Sn:
                s_pi_t = s_sync_t[0]
                for t_pi_t in prod_mdp.successors(s_pi_t):
                    if t_pi_t not in Sn_pi:
                        prop = prod_mdp[s_pi_t][t_pi_t]['prop'].copy()
                        for u_t in prop.keys():
                            if u_t in act[s_sync_t]:
                                pe = prop[u_t][0]
                                Y_t = Y[(s_sync_t, u_t)].solution_value()
                                y_out += Y_t * pe
                    elif t_pi_t in ip_pi:
                        prop = prod_mdp[s_pi_t][t_pi_t]['prop'].copy()
                        for u_t in prop.keys():
                            if u_t in act[s_sync_t]:
                                Y_t = Y[(s_sync_t, u_t)].solution_value()
                                y_to_ip += Y_t * pe
            if (y_to_ip + y_out) > 0:
                risk = y_out / (y_to_ip + y_out)

            '''
            for s in Sn:
                for t in sync_mec.successors(s):
                    if t not in Sn:
                        prop = sync_mec[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_out += Y[(s, u)].solution_value() * pe
                    elif t in ip:
                        prop = sync_mec[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)].solution_value() * pe
            if (y_to_ip + y_out) > 0:
                risk = y_out / (y_to_ip + y_out)
            '''
            print_c('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip + y_out,), color=32)
            print_c("----Suffix risk computed", color=32)


            #
            # TODO for debugging
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
                    else:
                        print("constraint_%d %s: %f <= %f <= %f"   % (k, constr_descrip[k], constr_t.lb(), sum_ret_t ,constr_t.ub(), ))
                except IndexError:
                    pass
            print("optimal policies: ")
            for k, s_pi_t in enumerate(Sn_pi):
                #
                # for s_sync_t in Sn:
                #     if s_sync_t[0] == s_pi_t:
                #         print(str(s_sync_t) + " " + str(plan_suffix[s_sync_t]))
                if k in plan_suffix_non_round_robin_list:
                    print_c(str(s_pi_t) + " " + str(plan_suffix[s_pi_t]), color=46)
                else:
                    print(str(s_pi_t) + " " + str(plan_suffix[s_pi_t]))

            return plan_suffix, cost, risk
        '''
        except:
            print("ORtools Error reported")
            return None, None, None
        '''

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


def synthesize_full_plan_w_opacity(mdp, task, optimizing_ap, ap_list, risk_pr, alpha=1):
    t2 = time.time()

    task_pi = task + ' & GF ' + optimizing_ap
    ltl_converted_pi = ltl_convert(task_pi)

    dra = Dra(ltl_converted_pi)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3-t2))

    # ----
    prod_dra_pi = product_mdp2(mdp, dra)
    prod_dra_pi.compute_S_f()                       # for AMECs
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
                            prod_dra_pi.re_synthesize_sync_amec(y_in_sf, MEC_pi, MEC_gamma, prod_dra_gamma)

                            # LP
                            plan_suffix, suffix_cost, suffix_risk = synthesize_suffix_cycle_in_sync_amec(prod_dra_pi, prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index], MEC_pi, y_in_sf)
                            #
                            print_c("Best plan suffix obtained, cost: %s, risk %s" % (str(suffix_cost), str(suffix_risk)), color=36)
                            print_c("=-------------------------------------=", color=36)

