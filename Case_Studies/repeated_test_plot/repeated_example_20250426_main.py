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
from Map.example_20250426_team_mdp import construct_team_mdp, team_observation_func_0426, control_observable_dict, run_2_observations_seqs, observation_seq_2_inference
from User.evaluation_team_ts import calculate_cost_from_runs, calculate_observed_cost_from_runs, calculate_sync_observed_cost_from_runs, construct_policy_comparasion_tables
from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra
from MDP_TG.lp  import syn_full_plan_rex
from User.team_dra3 import product_team_mdp3
from User.team_lp3  import synthesize_full_plan_w_opacity3
from User.grid_utils import sort_team_numerical_states
from User.vis2 import print_c, print_colored_sequence, print_highlighted_sequences
from User.plot import plot_cost_hist, plot_cost_hists_multi, plot_cost_hists_together_4_comparision, plot_cost_hists_together_4_comparision_multi_groups

# 获取父目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# 现在可以正常导入上一级模块
from Case_Studies.example_20250426_team_mdp_main import obtain_all_aps_from_team_mdp, ltl_convert, execute_example_in_origin_product_mdp, execute_example_4_product_mdp3, print_best_all_plan

def run_one_param_group(team_mdp, ltl_formula_converted, ap_list, param_group, max_attempt=10):
    gamma, d, risk_threshold, differential_exp_cost = param_group

    dra = Dra(ltl_formula_converted)

    best_all_plan = None
    for i in range(max_attempt + 1):
        best_all_plan, prod_dra_pi = synthesize_full_plan_w_opacity3(
            team_mdp, ltl_formula, opt_prop, ap_list,
            risk_threshold,
            differential_exp_cost,
            observation_func=team_observation_func_0426,
            ctrl_obs_dict=control_observable_dict
        )
        if best_all_plan is not None:
            print_c(f"[Opaque Synthesis] plan synthesized        {i} / {max_attempt} ...", color='white', bg_color='cyan', style='bold')
            break
        else:
            print_c(f"[Opaque Synthesis] NO VALID plan, retrying {i} / {max_attempt} ...", color='white', bg_color='red', style='bold')
    if best_all_plan is None:
        raise RuntimeError("[Opaque Synthesis] Failed after all attempts.")

    ap_gamma = best_all_plan[3][0]

    best_all_plan_p = None
    for i in range(max_attempt + 1):
        prod_dra = product_team_mdp3(team_mdp, dra)
        best_all_plan_p = syn_full_plan_rex(prod_dra, gamma, d)
        if best_all_plan_p is not None:
            print_c(f"[NON-Opaque Synthesis] plan synthesized        {i} / {max_attempt} ...", color='white', bg_color='cyan', style='bold')
            break
        else:
            print_c(f"[NON-Opaque Synthesis] NO VALID plan, retrying {i} / {max_attempt} ...", color='white', bg_color='red', style='bold')

    return best_all_plan, best_all_plan_p, prod_dra_pi

if __name__ == "__main__":

    total_T = 150
    N = 500
    is_average = True

    team_mdp, initial_node, initial_label = construct_team_mdp()
    ap_list                               = obtain_all_aps_from_team_mdp(team_mdp)

    ltl_formula = 'GF (gather -> (!gather U drop))'
    opt_prop = 'gather'
    ltl_formula_converted = ltl_convert(ltl_formula)


    # gamma = 0.125
    # d = 100                                   # for rex
    # risk_threshold = 0.05  # default:  0.1
    # differential_exp_cost = 5  # 1.590106
    param_groups = [
        (0.125, 5,  0.05, 5),
        (0.125, 5,  0.1,  5),
        (0.25,  7,  0.25, 7),
        (0.5,   15, 0.5, 15),
    ]

    results = []
    for idx, param_group in enumerate(param_groups):
        print_c(f"\n===== Running param group {idx + 1}: gamma={param_group[0]}, d={param_group[1]}, "
                f"risk={param_group[2]}, Δexp_cost={param_group[3]} =====", color='black', bg_color='yellow',
                style='bold')

        try:
            best_plan_opq, best_plan_non_opq, prod_dra_pi = run_one_param_group(
                team_mdp, ltl_formula_converted, ap_list, param_group, max_attempt=1)
            results.append(((param_group, best_plan_opq, prod_dra_pi), (param_group, best_plan_non_opq)))
        except RuntimeError as e:
            print_c(str(e), color='white', bg_color='red', style='bold')
            results.append(((param_group, None, None), (param_group, None)))

    all_cost_groups = []

    for (param_group, best_plan_opq, prod_dra_pi), (_, best_plan_non_opq) in results:
        if best_plan_opq is None or best_plan_non_opq is None:
            continue

        dra = Dra(ltl_formula_converted)

        cost_list_pi, cost_list_gamma, diff_exp_list = execute_example_4_product_mdp3(
            N, total_T, prod_dra_pi, best_plan_opq,
            [initial_node], [initial_label], opt_prop, best_plan_opq[3][0], attr='Opaque'
        )

        prod_dra = product_team_mdp3(team_mdp, dra)
        cost_list_pi_p, cost_list_gamma_p = execute_example_in_origin_product_mdp(
            N, total_T, prod_dra, best_plan_non_opq,
            [initial_node], [initial_label], opt_prop, best_plan_opq[3][0], attr='Non-Opaque'
        )

        all_cost_groups.append([[cost_list_pi, cost_list_gamma], [cost_list_pi_p, cost_list_gamma_p]])

    labels_pi = [[r"$\pi$ opaque", r"$\pi$ non-opaque"]] * len(all_cost_groups)
    labels_gamma = [[r"$\gamma$ opaque", r"$\gamma$ non-opaque"]] * len(all_cost_groups)
    #titles = [f"Param Group {i+1}" for i in range(len(all_cost_groups))]
    titles = [f"" for i in range(len(all_cost_groups))]

    if len(param_groups) == 1:
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
    else:
        plot_cost_hists_together_4_comparision_multi_groups(
            all_cost_groups,
            bins=25,
            # colors_pi=["#C99E8C", "#57C3C2", "#C99E8C", "#57C3C2","#C99E8C", "#57C3C2","#C99E8C", "#57C3C2"],
            # colors_gamma=["#465E65", "#FE4567","#465E65", "#FE4567", "#465E65", "#FE4567", "#465E65", "#FE4567"],
            labels_pi=labels_pi,
            labels_gamma=labels_gamma,
            titles=titles,
            is_average=is_average,
        )

    plt.show()
