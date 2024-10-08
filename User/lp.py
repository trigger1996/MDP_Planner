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
        sf.append(sync_state_t)
        sf_pi.append(state_pi_t)
        #
        # TODO to check
        if state_pi_t in MEC_pi[1]:
            if state_pi_t not in ip_pi:
                ip.append(sync_state_t)
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

def synthesize_suffix_cycle_in_sync_amec(prod_mdp, sync_mec, MEC_pi, y_in_sf):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print_c("===========[plan suffix synthesis starts]", color=32)
    # step 1: find states
    # sf:  states in MEC -> states in sync MEC
    # ip:  MEC states intersects with Ip -> Accepting states Ip in sync MEC / state[0] intersects with IP
    # act: actions available for each state
    sf, sf_pi, ip, ip_pi, act, act_pi = state_action_sets_pi_from_sync_mec(sync_mec, MEC_pi)

    state_action_sets_pi_from_sync_mec(sync_mec, MEC_pi)

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
            # Added
            # constraint 1
            # 由于sync_mec内所有sync_state[0]对应的状态是一个状态, 所以其对应概率之和就应该是1
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
            print_c('inter-state constraints added', color=42)
            print_c('number of constraints: %d' % (suffix_solver.NumConstraints(), ), color=42)

            # constraint 2
            # inflow and outflow
            # 是不是这个地方直接用他的约束也是可以的
            for s in Sn:
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
                for f in sync_mec.predecessors(s):                      # 求解对象不一样了, product mdp -> sync_mec
                    #
                    # 这里也有不同
                    # prefix
                    #       if f in Sr:
                    # suffix
                    #       if f in Sn 且 s in Sn 且 s not in ip
                    if (f in Sn) and (s not in ip):
                        prop = sync_mec[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                constr4 += Y[(f, uf)] * prop[uf][0]
                            else:
                                constr4 += Y[(f, uf)] * 0.00
                    if (f in Sn) and (s in ip) and (f != s):
                        prop = sync_mec[f][s]['prop'].copy()
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
                # 注意这里y_in_sf的keys是不一样的, 不是sync_states
                s_pi_t = s[0]
                #
                # if (s_pi_t in list(y_in_sf.keys())) and (s not in ip):
                #    suffix_solver.Add(constr3 == constr4 + y_in_sf[s_pi_t])  # 可能到的了的要计算?
                #
                # 如果 s in Sf, 且上一时刻状态在Sn，且在MEC内
                # if (s_pi_t in list(y_in_sf.keys())) and (s in ip):
                #     suffix_solver.Add(constr3 == y_in_sf[s_pi_t])  # 在里面的永远到的了?
                #
                # 如果s不在Sf内且不在NEC内
                #if (s_pi_t not in list(y_in_sf.keys())) and (s not in ip):
                #    suffix_solver.Add(constr3 == constr4)  # 到不了的永远到不了?
            print('Balance condition added')
            print('Initial sf condition added')
            # --------------------
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                for t in sync_mec.successors(s):
                    if t not in Sn:
                        prop = sync_mec[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                #
                                # Sn里出Sn的
                                y_out += Y[(s, u)] * pe
                    elif t in ip:
                        prop = sync_mec[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                #`
                                # Sn里进Ip的
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)] * pe
            # suffix_solver.Add(y_to_ip+y_out >= delta)                                 # 这个是原来就注释的
            # suffix_solver.Add(y_to_ip >= (1.0 - gamma - delta) * (y_to_ip + y_out))
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

            #
            # TODO for debugging
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
                        norm += Y[(s_sync_t, u_t)].solution_value()
                    for u_t in act[s_sync_t]:
                        U.append(u_t)
                        if norm > 0.01:
                            P.append(Y[(s_sync_t, u_t)].solution_value() / norm)
                        else:
                            P.append(1.0 / len(act[s_sync_t]))             # round robin
                plan_suffix[s_pi_t] = [U, P]

            # TODO
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
                        P.append(1.0 / len(act[s]))             # round robin
                plan_suffix[s] = [U, P]
            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")

            # TODO
            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
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
            print_c('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip + y_out,), color=32)
            print_c("----Suffix risk computed", color=32)


            #
            # TODO for debugging
            # for display
            for k, s_pi_t in enumerate(Sn_pi):
                #
                for s_sync_t in Sn:
                    if s_sync_t[0] == s_pi_t:
                        print(str(s_sync_t) + " " + str(plan_suffix[s_sync_t]))
                #
                try:
                    constr_t = suffix_solver.constraint(k)
                    sum_ret_t = 0.
                    for s_sync_t in Sn:
                        for u_t in act[s_sync_t]:
                            Y_t = Y[(s_sync_t, u_t)]
                            ki_t = constr_t.GetCoefficient(Y_t)
                            #
                            index_t = plan_suffix[s_sync_t][0].index(u_t)
                            prob_t  = plan_suffix[s_sync_t][1][index_t]
                            sum_ret_t += ki_t * prob_t
                    print("constraint_%d: %f <= %f <= %f", (k, constr_t.lb(), sum_ret_t ,constr_t.ub(), ))
                except IndexError:
                    pass
                #
                print("")

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

