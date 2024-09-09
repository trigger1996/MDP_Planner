from MDP_TG import lp
from MDP_TG.lp import syn_plan_prefix, syn_plan_suffix, syn_plan_bad
from User.dra2 import Sync_Product_Dra
import MDP_TG.dra

def syn_full_plan(prod_mdp, gamma, alpha=1):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    Plan = []
    for l, S_fi in enumerate(prod_mdp.Sf):                                                  # prod_mdp.Sf 对应所有的AMEC
        print("---for one S_fi---")
        plan = []
        for k, MEC in enumerate(S_fi):                                                      # 一个Sf的元素对应一个Accepting Pair, 一个Accepting Pair对应一组AMEC
            plan_prefix, prefix_cost, prefix_risk, y_in_sf, Sr, Sd = syn_plan_prefix(
                prod_mdp, MEC, gamma)
            print("Best plan prefix obtained, cost: %s, risk %s" %
                  (str(prefix_cost), str(prefix_risk)))
            if y_in_sf:

                # User Added
                # y_in_sf Sn -> Sf的概率, 从S0能到达但不在AMEC的状态, 到达AMEC的状态, 其中keys为AMEC的状态
                initial_state_amec = []
                for initial_state_t in y_in_sf.keys():
                    if y_in_sf[initial_state_t] > 0:                 # y_in_sf的keys()是能找到的所有Sn状态的后继状态，并不是所有后继状态的policy概率都不为0
                        initial_state_amec.append(initial_state_t)

                for l_prime, S_fi_prime in enumerate(prod_mdp.Sf):                          # 可以不是一个接收对的
                    for k_prime, MEC_prime in enumerate(S_fi_prime):
                        #
                        # synthesize
                        # it is admissible that S_fi and S_fj are identical
                        sync_product_mdp = Sync_Product_Dra()
                        sync_product_mdp.synthesize_from_sync_mdp(prod_mdp, MEC, MEC_prime, initial_state_amec)

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
