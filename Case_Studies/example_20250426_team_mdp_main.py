#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import User.utils
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from subprocess import check_output
from Map.example_20250426_team_mdp import construct_team_mdp, observation_func_0425, control_observable_dict, calculate_cost_from_runs, plot_cost_hist
from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from User.lp3  import synthesize_full_plan_w_opacity3
from User.mdp3 import MDP3
from User.utils import print_c

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

def obtain_all_aps_from_mdp(mdp:Motion_MDP, is_convert_to_str=True, is_remove_empty_ap=True):
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
                    if is_remove_empty_ap:
                        continue                # remove empty aps
                else:
                    ap_t = ap_t[0]
            #
            ap_list.append(ap_t)

    return list(set(ap_list))

if __name__ == "__main__":
    team_mdp = construct_team_mdp()
    ap_list  = obtain_all_aps_from_mdp(team_mdp)

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
        # best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity(motion_mdp, ltl_formula, opt_prop, ap_list, risk_threshold,
        #                                                             differential_exp_cost,
        #                                                             observation_func=observation_func_0401,
        #                                                             ctrl_obs_dict=control_observable_dict)
        best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity3(team_mdp, ltl_formula, opt_prop, ap_list,
                                                                     risk_threshold,
                                                                     differential_exp_cost,
                                                                     observation_func=observation_func_0425,
                                                                     ctrl_obs_dict=control_observable_dict)
        ap_gamma = best_all_plan[3][0]
    else:
        ap_gamma = 'upload'
    #
    # # TODO
    # best_all_plan_p = syn_full_plan_rex(prod_dra, gamma, d)
    # # best_all_plan_p = syn_full_plan_repeated(prod_dra, gamma, opt_prop)
    #
    # # print_best_all_plan(best_all_plan)
    # #
    # # # for visualization
    # total_T = 50
    # state_seq = [initial_node, ]
    # label_seq = [initial_label, ]
    # N = 5
    # is_average = True
    # #
    # # #
    # # # Opaque runs
    # if is_run_opaque_synthesis:
    #     # try:
    #     # TODO
    #     if True:
    #         cost_list_pi, cost_list_gamma = execute_example_4_product_mdp3(N, total_T, prod_dra_pi, best_all_plan,
    #                                                                        state_seq, label_seq, opt_prop, ap_gamma,
    #                                                                        attr='Opaque')
    #
    #         plot_cost_hist(cost_list_pi, bins=25, is_average=is_average,
    #                        title="Cost for Satisfaction of AP \pi in Opaque runs")
    #         plot_cost_hist(cost_list_gamma, bins=25, color='r', is_average=is_average,
    #                        title="Cost for Satisfaction of AP \gamma in Opaque runs")
    #
    #         # TODO
    #         # except:
    #         print_c("No best plan synthesized, try re-run this program", color=33)
    #
    # #
    # print_c("\n\nFOR COMPARASION, NON_OPAQUE SYNTHESIS: \n", color=46)
    # print_best_all_plan(best_all_plan_p)
    #
    # #
    # # Non-opaque runs
    # # try:
    # state_seq = [initial_node, ]
    # label_seq = [initial_label, ]
    # if True:
    #     cost_list_pi_p, cost_list_gamma_p = execute_example_in_origin_product_mdp(N, total_T, prod_dra, best_all_plan_p,
    #                                                                               state_seq, label_seq, opt_prop,
    #                                                                               ap_gamma, attr='Opaque')
    # # except:
    # #     print_c("No best plan synthesized, try re-run this program", color=33)
    # # is_average = True
    # plot_cost_hist(cost_list_pi_p, bins=25, color='b', is_average=is_average,
    #                title="Cost for Satisfaction of AP \pi in NON-Opaque runs")
    # plot_cost_hist(cost_list_gamma_p, bins=25, color='cyan', is_average=is_average,
    #                title="Cost for Satisfaction of AP \gamma in NON-Opaque runs")
    #
    # # TODO 对比实验
    # # 我的问题是, 入侵者到底拿到的是什么数据
    # # 进而, 如何通过实验现象来描述opacity
    #
    # # TODO
    # # draw_action_principle()
    # # draw_mdp_principle()
    #
    # #
    # #
    # plt.show()

    print(233)