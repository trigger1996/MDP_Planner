from MDP_TG import lp
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_plan_prefix, syn_plan_suffix, syn_plan_bad
from User.dra2 import product_mdp2

from subprocess import check_output
from User.vis2  import print_c

import pickle
import time
import networkx

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

def synthesize_suffix_cycle_in_sync_amec(sync_amec):
    pass

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
                            # prod_dra_pi.sync_amec_set[prod_dra_pi.current_sync_amec_index]


                # TODO
                # if finished, change intend
                plan_suffix, suffix_cost, suffix_risk = syn_plan_suffix(
                    prod_dra_pi, MEC_pi, y_in_sf)
                print("Best plan suffix obtained, cost: %s, risk %s" %
                      (str(suffix_cost), str(suffix_risk)))
