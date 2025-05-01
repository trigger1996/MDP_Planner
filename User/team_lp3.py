#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pickle
import networkx as nx
import numpy    as np

from collections import defaultdict
from ortools.linear_solver import pywraplp

from MDP_TG.dra import Dra, Product_Dra
from User.dra3  import project_sync_states_2_observer_states, project_observer_state_2_sync_state, project_sync_mec_3_2_observer_mec_3
from User.lp3 import find_states_satisfying_opt_prop, syn_plan_prefix_in_sync_amec, synthesize_suffix_cycle_in_sync_amec3
from User.team_dra3 import product_team_mdp3
from User.utils import ltl_convert
from User.vis2  import print_c


from rich.progress import Progress, track

import sys
stack_size_t = 500000
sys.setrecursionlimit(stack_size_t)         # change stack size to guarantee runtime stability, default: 3000
print_c("stack size changed %d ..." % (stack_size_t, ))
#
sys.setrecursionlimit(100000)               # 设置递归深度为100000


def synthesize_full_plan_w_opacity3(mdp, task, optimizing_ap, ap_list, risk_pr, differential_exp_cost, observation_func,
                                    ctrl_obs_dict, alpha=1, is_enable_inter_state_constraints=True):
    t2 = time.time()

    task_pi = task + ' & GF ' + optimizing_ap
    ltl_converted_pi = ltl_convert(task_pi)

    dra = Dra(ltl_converted_pi)
    t3 = time.time()
    print('DRA done, time: %s' % str(t3 - t2))

    # ----
    prod_dra_pi = product_team_mdp3(mdp, dra)
    prod_dra_pi.compute_S_f()  # for AMECs, finished in building
    # prod_dra.dotify()
    t41 = time.time()
    print('Product DRA done, time: %s' % str(t41 - t3))

    pickle.dump((nx.get_edge_attributes(prod_dra_pi, 'prop'),
                 prod_dra_pi.graph['initial']), open('prod_dra_edges.p', "wb"))
    print('prod_dra_edges.p saved')

    # new main loop
    plan = []
    for l, S_fi_pi in enumerate(prod_dra_pi.Sf):  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        for k, MEC_pi in enumerate(S_fi_pi):
            #
            # finding states that satisfying optimizing prop
            S_pi = find_states_satisfying_opt_prop(optimizing_ap, MEC_pi[0])

            for ap_4_opacity in ap_list:
                if ap_4_opacity == optimizing_ap:
                    continue
                # synthesize product mdp for opacity
                task_gamma = task + ' & GF ' + ap_4_opacity
                ltl_converted_gamma = ltl_convert(task_gamma)
                dra = Dra(ltl_converted_gamma)
                prod_dra_gamma = product_team_mdp3(mdp, dra)
                prod_dra_gamma.compute_S_f()  # for AMECs

                #
                for p, S_fi_gamma in enumerate(prod_dra_gamma.Sf):
                    for q, MEC_gamma in enumerate(S_fi_gamma):
                        prod_dra_pi.re_synthesize_sync_amec(optimizing_ap, ap_4_opacity, MEC_pi, MEC_gamma,
                                                             prod_dra_gamma, observation_func=observation_func,
                                                             ctrl_obs_dict=ctrl_obs_dict)

                        sync_mec_t = prod_dra_pi.project_sync_amec_back_to_mec_pi(
                            prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index], MEC_pi)
                        if not sync_mec_t[1].__len__():
                            print_c("[prefix synthesis] AP: %s do not have a satisfying Ip in DRA!" % (ap_4_opacity,),
                                    color='yellow')
                            continue

                        initial_subgraph, initial_sync_state = prod_dra_pi.construct_opaque_subgraph_2_amec(
                                                                                    prod_dra_gamma,
                                                                                    sync_mec_t,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    MEC_pi, MEC_gamma,
                                                                                    optimizing_ap, ap_4_opacity, observation_func, ctrl_obs_dict)

                        observer_mec_3 = project_sync_mec_3_2_observer_mec_3(initial_subgraph, sync_mec_t)

                        plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync, Sr, Sd = syn_plan_prefix_in_sync_amec(
                                                                                    prod_dra_pi, initial_subgraph, initial_sync_state,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t, observer_mec_3, risk_pr)

                        opaque_full_graph = prod_dra_pi.construct_fullgraph_4_amec(initial_subgraph,
                                                                                    prod_dra_gamma,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    MEC_pi, MEC_gamma,
                                                                                    optimizing_ap, ap_4_opacity,
                                                                                    observation_func, ctrl_obs_dict)

                        # TODO
                        # To get rid of mec_observer
                        plan_suffix, suffix_cost, suffix_risk, suffix_opacity_threshold = synthesize_suffix_cycle_in_sync_amec3(
                                                                                    prod_dra_pi,
                                                                                    prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index],
                                                                                    sync_mec_t, observer_mec_3,
                                                                                    y_in_sf_sync,
                                                                                    opaque_full_graph,  # 用来判断Sn是否可达, 虽然没啥意义但是还是可以做,
                                                                                    initial_sync_state,
                                                                                    differential_exp_cost)

                        plan.append([[plan_prefix, prefix_cost, prefix_risk, y_in_sf_sync],
                                     [plan_suffix, suffix_cost, suffix_risk],
                                     [MEC_pi[0], MEC_pi[1], Sr, Sd],
                                     [ap_4_opacity, suffix_opacity_threshold, prod_dra_pi.current_sync_amec_index, MEC_gamma],
                                     [initial_subgraph, initial_sync_state, opaque_full_graph],
                                     [sync_mec_t, observer_mec_3]
                                     ])

    if plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        best_all_plan = min(plan, key=lambda p: p[0][1] + alpha * p[1][1])
        prod_dra_pi.update_best_all_plan(best_all_plan, is_print_policy=True)
        # print('Best plan prefix obtained for %s states in Sr' %
        #       str(len(best_all_plan[0][0])))
        # print('cost: %s; risk: %s ' %
        #       (best_all_plan[0][1], best_all_plan[0][2]))
        # print('Best plan suffix obtained for %s states in Sf' %
        #       str(len(best_all_plan[1][0])))
        # print('cost: %s; risk: %s ' %
        #       (best_all_plan[1][1], best_all_plan[1][2]))
        # print('Total cost:%s' %
        #       (best_all_plan[0][1] + alpha * best_all_plan[1][1]))
        # print_c('Opacity threshold %f <= %f' % (best_all_plan[3][1], differential_exp_cost,))
        # #


        return best_all_plan, prod_dra_pi
    else:
        print("No valid plan found")
        return None