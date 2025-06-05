#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import User.utils
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from functools import cmp_to_key
from subprocess import check_output
from Map.example_20250426_team_mdp import construct_team_mdp, team_observation_func_0426, control_observable_dict, run_2_observations_seqs, observation_seq_2_inference, calculate_cost_from_runs
from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra
from MDP_TG.lp  import syn_full_plan_rex
from User.team_dra3 import product_team_mdp3
from User.team_lp3  import synthesize_full_plan_w_opacity3
from User.grid_utils import sort_team_numerical_states
from User.vis2 import print_c, print_colored_sequence, print_highlighted_sequences
from User.plot import plot_cost_hist, plot_cost_hists_multi

# for debugging
# import random
# random.seed(42)         # for debugging, if for alternative solutions, this line can be removed

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

def obtain_all_aps_from_team_mdp(mdp:Motion_MDP, is_convert_to_str=True, is_remove_empty_ap=True):
    ap_list = []
    for state_t in mdp.nodes():
        state_attr_t = mdp.nodes()[state_t]
        label_t = state_attr_t['label']
        for ap_t in list(state_attr_t['label'])[0]:
            #
            # Added
            if is_convert_to_str and type(ap_t) == frozenset:
                ap_t = list(ap_t)
                if not ap_t.__len__():
                    ap_t = ''
                else:
                    ap_t = ap_t[0]
            #
            ap_list.append(ap_t)

    ap_list = list(set(ap_list))
    if is_remove_empty_ap:
        if '' in ap_list:
            ap_list.remove('')
        if ' ' in ap_list:
            ap_list.remove(' ')

    ap_list = list(set(ap_list))
    ap_list.sort()
    return ap_list

def print_best_all_plan(best_all_plan):
    # Added
    # for printing policies
    if best_all_plan.__len__() >= 4 and best_all_plan[3].__len__():
        print_c("optimal AP: %s" % (best_all_plan[3][0], ), color=47)
    print_c("state action: probabilities")
    print_c("Prefix", color=42)
    #
    state_in_prefix = [ state_t for state_t in best_all_plan[0][0] ]
    state_in_prefix.sort(key=cmp_to_key(sort_team_numerical_states))
    #for state_t in best_all_plan[0][0]:
    for state_t in state_in_prefix:
        print_c("%s, %s: %s" % (str(state_t), str(best_all_plan[0][0][state_t][0]), str(best_all_plan[0][0][state_t][1]), ), color=42)
    #
    print_c("Suffix", color=45)
    state_in_suffix = [ state_t for state_t in best_all_plan[1][0] ]
    state_in_suffix.sort(key=cmp_to_key(sort_team_numerical_states))
    #for state_t in best_all_plan[1][0]:
    for state_t in state_in_suffix:
        print_c("%s, %s: %s" % (str(state_t), str(best_all_plan[1][0][state_t][0]), str(best_all_plan[1][0][state_t][1]), ), color=45)

def execute_example_4_product_mdp3(N, total_T, prod_dra, best_all_plan, state_seq, label_seq, opt_prop, ap_gamma, attr='opaque'):
    XX  = []
    LL  = []
    UU  = []
    MM  = []
    OXX = []
    cost_list_pi = []
    cost_list_gamma = []
    for n in range(0, N):
        X, OX, O, X_OPA, L, OL, U, M = prod_dra.execution_in_observer_graph(total_T)

        XX.append(X)
        LL.append(L)
        UU.append(U)
        MM.append(M)
        OXX.append(OX)

    print('[Product Dra] process all done')

    color_init = 32
    for i in range(0, XX.__len__()):
        X_U = []
        for j in range(0, XX[i].__len__()):
            X_U.append(XX[i][j])
            if j < XX[i].__len__() - 1:
                X_U.append(UU[i][j])
        #
        Y = run_2_observations_seqs(X_U)
        X_INV, AP_INV = observation_seq_2_inference(Y)
        #
        cost_cycle = calculate_cost_from_runs(prod_dra, XX[i], LL[i], UU[i], opt_prop)
        cost_list_pi = cost_list_pi + cost_cycle
        #
        cost_cycle_p = calculate_cost_from_runs(prod_dra, XX[i], LL[i], UU[i], ap_gamma)
        cost_list_gamma = cost_list_gamma + cost_cycle_p
        #
        # print_c(X_U, color=color_init)
        # print_c(Y, color=color_init)
        # print_c(X_INV, color=color_init)
        # print_c(AP_INV, color=color_init)
        # print_c("[cost / achieved_index] " + str(cost_cycle), color=color_init)
        # color_init += 1
        #
        print_colored_sequence(X_U)
        print_colored_sequence(Y)
        print_colored_sequence(X_INV)
        print_colored_sequence(AP_INV)
        print_c("[cost / achieved_index] " + str(cost_cycle), color=color_init)
        #
        print_highlighted_sequences(X_U, Y, X_INV, AP_INV, marker1=opt_prop, marker2=ap_gamma, attr=attr)
    # fig = visualize_run_sequence(XX, LL, UU, MM, 'surv_result', is_visuaize=False)

    return cost_list_pi, cost_list_gamma

def execute_example_in_origin_product_mdp(N, total_T, prod_dra, best_all_plan, state_seq, label_seq, opt_prop, ap_gamma, attr):
    XX = []
    LL = []
    UU = []
    MM = []
    PP = []
    for n in range(0, N):
        X, L, U, M, PX = prod_dra.execution(best_all_plan, total_T, state_seq, label_seq)

        XX.append(X)
        LL.append(L)
        UU.append(U)
        MM.append(M)
        PP.append(PX)

    print('[Product Dra] process all done')

    cost_list_pi = []
    cost_list_gamma = []
    color_init = 32
    for i in range(0, XX.__len__()):
        X_U = []
        for j in range(0, XX[i].__len__()):
            X_U.append(XX[i][j])
            if j < XX[i].__len__() - 1:
                X_U.append(UU[i][j])
        #
        Y = run_2_observations_seqs(X_U)
        X_INV, AP_INV = observation_seq_2_inference(Y)
        #
        cost_cycle = calculate_cost_from_runs(prod_dra, XX[i], LL[i], UU[i], opt_prop)
        cost_list_pi = cost_list_pi + cost_cycle
        #
        cost_cycle_p = calculate_cost_from_runs(prod_dra, XX[i], LL[i], UU[i], ap_gamma)
        cost_list_gamma = cost_list_gamma + cost_cycle_p
        #
        # print_c(X_U, color=color_init)
        # print_c(Y, color=color_init)
        # print_c(X_INV, color=color_init)
        # print_c(AP_INV, color=color_init)
        # print_c("[cost / achieved_index] " + str(cost_cycle), color=color_init)
        # color_init += 1
        #
        print_colored_sequence(X_U)
        print_colored_sequence(Y)
        print_colored_sequence(X_INV)
        print_colored_sequence(AP_INV)
        print_c("[cost / achieved_index] " + str(cost_cycle), color=color_init)
        #
        print_highlighted_sequences(X_U, Y, X_INV, AP_INV, marker1=opt_prop, marker2=ap_gamma, attr=attr)
    # fig = visualize_run_sequence(XX, LL, UU, MM, 'surv_result', is_visuaize=False)

    return cost_list_pi, cost_list_gamma

if __name__ == "__main__":
    team_mdp, initial_node, initial_label = construct_team_mdp()
    ap_list                               = obtain_all_aps_from_team_mdp(team_mdp)

    # ltl_formula = 'GF (gather -> drop)'
    ltl_formula = 'GF (gather -> (!gather U drop))'  # 'GF (gather -> X(!gather U drop))'
    opt_prop = 'gather'
    ltl_formula_converted = ltl_convert(ltl_formula)

    dra = Dra(ltl_formula_converted)

    t42 = time.time()

    # ------
    gamma = 0.5
    d = 100
    risk_threshold = 0.05  # default:  0.1
    differential_exp_cost = 15  # 1.590106
    is_run_opaque_synthesis = True
    if is_run_opaque_synthesis:
        best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity3(team_mdp, ltl_formula, opt_prop, ap_list,
                                                                     risk_threshold,
                                                                     differential_exp_cost,
                                                                     observation_func=team_observation_func_0426,
                                                                     ctrl_obs_dict=control_observable_dict)
    ap_gamma = best_all_plan[3][0]

    prod_dra = product_team_mdp3(team_mdp, dra)
    best_all_plan_p = syn_full_plan_rex(prod_dra, gamma, d)
    # best_all_plan_p = syn_full_plan_repeated(prod_dra, gamma, opt_prop)

    # print_best_all_plan(best_all_plan)
    #
    # # for visualization
    total_T = 50
    state_seq = [initial_node, ]
    label_seq = [initial_label, ]
    N = 5
    is_average = True
    #
    # #
    # # Opaque runs
    if is_run_opaque_synthesis:
        # try:
        # TODO
        if True:
            cost_list_pi, cost_list_gamma = execute_example_4_product_mdp3(N, total_T, prod_dra_pi, best_all_plan,
                                                                           state_seq, label_seq, opt_prop, ap_gamma,
                                                                           attr='Opaque')

            plot_cost_hist(cost_list_pi, bins=25, is_average=is_average,
                           title="Cost for Satisfaction of AP \pi in Opaque runs")
            plot_cost_hist(cost_list_gamma, bins=25, color='r', is_average=is_average,
                           title="Cost for Satisfaction of AP \gamma in Opaque runs")
            plot_cost_hists_multi(cost_list_pi, cost_list_gamma, bins=25, colors=['r', 'magenta'], labels=['\pi', '\gamma'], is_average=is_average,
                           title="Cost for Satisfaction of APs in Opaque runs")

            # TODO
            # except:
            print_c("No best plan synthesized, try re-run this program", color=33)

    #
    print_c("\n\nFOR COMPARASION, NON_OPAQUE SYNTHESIS: \n", color=46)
    print_best_all_plan(best_all_plan_p)

    #
    # Non-opaque runs
    # try:
    state_seq = [initial_node, ]
    label_seq = [initial_label, ]
    if True:
        cost_list_pi_p, cost_list_gamma_p = execute_example_in_origin_product_mdp(N, total_T, prod_dra, best_all_plan_p,
                                                                                  state_seq, label_seq, opt_prop,
                                                                                  ap_gamma, attr='Opaque')
    # except:
    #     print_c("No best plan synthesized, try re-run this program", color=33)
    # is_average = True
    plot_cost_hist(cost_list_pi_p, bins=25, color='b', is_average=is_average,
                   title="Cost for Satisfaction of AP \pi in NON-Opaque runs")
    plot_cost_hist(cost_list_gamma_p, bins=25, color='cyan', is_average=is_average,
                   title="Cost for Satisfaction of AP \gamma in NON-Opaque runs")
    plot_cost_hists_multi(cost_list_pi_p, cost_list_gamma_p, bins=25, colors=['b', 'cyan'], labels=['\pi', '\gamma'], is_average=is_average,
                          title="Cost for Satisfaction of APs in NON-Opaque runs")

    # TODO 对比实验
    # 我的问题是, 入侵者到底拿到的是什么数据
    # 进而, 如何通过实验现象来描述opacity

    # TODO
    # draw_action_principle()
    # draw_mdp_principle()

    #
    #
    plt.show()
