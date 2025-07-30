#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
#
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
from User.plot import plot_cost_hist, plot_cost_hists_multi, plot_cost_hists_together_4_comparision

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# 现在可以正常导入上一级模块
from ..example_20250426_team_mdp_main import obtain_all_aps_from_team_mdp, ltl_convert, execute_example_in_origin_product_mdp, execute_example_4_product_mdp3, print_best_all_plan


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
    gamma = 0.125
    d = 100
    risk_threshold = 0.05  # default:  0.1
    differential_exp_cost = 5  # 1.590106
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

            # plot_cost_hist(cost_list_pi, bins=25, is_average=is_average,
            #                title="Cost for Satisfaction of AP \pi in Opaque runs")
            # plot_cost_hist(cost_list_gamma, bins=25, color='r', is_average=is_average,
            #                title="Cost for Satisfaction of AP \gamma in Opaque runs")
            plot_cost_hists_multi(cost_list_pi, cost_list_gamma, bins=25, colors=["#C99E8C", "#465E65"], labels=[r"$\pi$", r"$\gamma$"], is_average=is_average,
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
    # plot_cost_hist(cost_list_pi_p, bins=25, color='b', is_average=is_average,
    #                title="Cost for Satisfaction of AP \pi in NON-Opaque runs")
    # plot_cost_hist(cost_list_gamma_p, bins=25, color='cyan', is_average=is_average,
    #                title="Cost for Satisfaction of AP \gamma in NON-Opaque runs")
    plot_cost_hists_multi(cost_list_pi_p, cost_list_gamma_p, bins=25, colors=["#57C3C2", "#FE4567"], labels=[r"$\pi$", r"$\gamma$"], is_average=is_average,
                          title="Cost for Satisfaction of APs in NON-Opaque runs")

    if is_run_opaque_synthesis:
        plot_cost_hists_together_4_comparision(
            [
                [cost_list_pi, cost_list_gamma],  # 方法 1
                [cost_list_pi_p, cost_list_gamma_p],  # 方法 2
            ],
            colors_pi=["#C99E8C", "#57C3C2"],
            colors_gamma=["#465E65", "#FE4567"],
            labels_pi=[r"$\pi$ in opaque run", r"$\pi$ in non-opaque run"],
            labels_gamma=[r"$\gamma$ in opaque run", r"$\gamma$ in non-opaque run"],
            title="Cost for Satisfaction of APs"
        )

    # TODO 对比实验
    # 我的问题是, 入侵者到底拿到的是什么数据
    # 进而, 如何通过实验现象来描述opacity

    # TODO
    # draw_action_principle()
    # draw_mdp_principle()

    #
    #
    plt.show()
