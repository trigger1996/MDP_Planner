#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def calculate_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []                      # [[value, current step, path_length], [value, current step, path_length], ...]
    cost_cycle = 0.
    #
    x_i_last = x[0]
    for i in range(1, x.__len__()):
        x_i = x[i]
        x_p = None
        l_i = list(l[i])
        if i < x.__len__() - 1:
            u_i = u[i]

        #
        if l_i.__len__() and opt_prop in l_i:
            if cost_list.__len__():
                path_length = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length = i
            #
            cost_current_step = [cost_cycle, i, path_length]
            cost_list.append(cost_current_step)
            cost_cycle = 0.
        #
        #
        for edge_t in list(product_mdp.graph['mdp'].edges(x_i_last, data=True)):
            if x_i == edge_t[1]:
                #
                event_t = list(edge_t[2]['prop'])[0]           # event_t = i_i???
                #
                cost_t = edge_t[2]['prop'][event_t][1]
                cost_cycle += cost_t

    if is_remove_zeros:
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass

    return cost_list

def calculate_sync_observed_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, ap_gamma, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []              # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    cost_list_gamma = []        # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    diff_exp_list = []          # [[value, current step pi, current step gamma, path_length], [value, current step, current step gamma, path_length], ...]
    #
    cost_cycle_pi = 0.
    cost_cycle_gamma = 0.
    #
    x_i_last = x[0]
    for i in range(1, x.__len__()):
        x_i_last = x[i - 1]
        o_i_last = o[i - 1]
        x_i = x[i]
        o_i = o[i]
        ol_i = list(ol_set[i])  # l_i = list(l[i])
        if i < x.__len__() - 1:
            # TODO
            u_i = u[i]

        #
        is_observed_ap_pi_gamma_found = False
        for o_t in ol_i:
            if ol_i.__len__() and opt_prop in o_t and ap_gamma in o_t:
                is_observed_ap_pi_gamma_found = True
                #break                                      # break不能加, 加了就出去了影响前后判断


        if is_observed_ap_pi_gamma_found:
            #
            # Add pi
            if cost_list.__len__():
                path_length_pi = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length_pi = i
            #
            cost_current_step = [cost_cycle_pi, i, path_length_pi]
            cost_list.append(cost_current_step)
            cost_cycle_pi = 0.
            #
            # Add gamma
            if cost_list_gamma.__len__():
                path_length_gamma = i - cost_list_gamma[cost_list_gamma.__len__() - 1][1]
            else:
                path_length_gamma = i
            #
            cost_current_step = [cost_cycle_gamma, i, path_length_gamma]
            cost_list_gamma.append(cost_current_step)
            cost_cycle_gamma = 0.
        #
        #
        available_event_cost_dict_pi = {}
        available_event_cost_dict_gamma = {}
        for o_last_t in o_i_last:
            for o_t in o_i:
                for edge_t in list(product_mdp.graph['mdp'].edges(o_last_t, data=True)):
                    if o_t == edge_t[1]:
                        label_o_t = list(product_mdp.graph['mdp'].nodes[o_t]['label'].keys())
                        label_o_t = [list(fs)[0] for fs in label_o_t if isinstance(fs, frozenset)]
                        #
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]
                        #
                        if opt_prop in label_o_t:
                            available_event_cost_dict_pi[event_t] = cost_t
                        if ap_gamma in label_o_t:
                            available_event_cost_dict_gamma[event_t] = cost_t
        if available_event_cost_dict_pi.__len__():
            min_event = min(available_event_cost_dict_pi, key=lambda e: available_event_cost_dict_pi[e])
            min_cost = available_event_cost_dict_pi[min_event]

            cost_cycle_pi += min_cost
        if available_event_cost_dict_gamma.__len__():
            min_event = min(available_event_cost_dict_gamma, key=lambda e: available_event_cost_dict_gamma[e])
            min_cost = available_event_cost_dict_gamma[min_event]

            cost_cycle_gamma += min_cost

        else:
            #
            # TODO
            # 用x代替?
            for edge_t in list(product_mdp.graph['mdp'].edges(x_i_last, data=True)):
                    if x_i == edge_t[1]:
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]

                        cost_cycle_gamma += cost_t
                        cost_cycle_pi += cost_t
            #print_c("[cost execution] warning ... using original cost instead of observed cost")


    used_gamma_indices = set()

    last_pi_step = 0
    last_gamma_step = 0

    for i in range(len(cost_list)):
        step_i = cost_list[i][1]

        # 在 gamma 列表中找与 step_i 最接近的 step_j，避免重复匹配
        min_j_index = -1
        min_step_diff = float('inf')
        for j in range(len(cost_list_gamma)):
            if j in used_gamma_indices:
                continue  # 防止重复匹配

            step_j = cost_list_gamma[j][1]
            diff = abs(step_i - step_j)
            if diff < min_step_diff:
                min_step_diff = diff
                min_j_index = j

        # 如果找不到匹配的 gamma（极端情况），跳过
        if min_j_index == -1:
            continue

        used_gamma_indices.add(min_j_index)

        cost_pi = cost_list[i]
        cost_gamma = cost_list_gamma[min_j_index]

        diff_cost = abs(cost_pi[0] - cost_gamma[0])
        path_length = max(cost_pi[1] - last_pi_step, cost_gamma[1] - last_gamma_step)

        diff_exp_list.append([
            diff_cost,
            cost_pi[1],
            cost_gamma[1],
            path_length
        ])

        last_pi_step = cost_pi[1]
        last_gamma_step = cost_gamma[1]

    if is_remove_zeros:
        #
        # pi
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass
        #
        # gamma
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list_gamma:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list_gamma.remove(cost_tuple_t)
            except:
                pass


    return cost_list, cost_list_gamma, diff_exp_list

def calculate_observed_cost_from_runs(product_mdp, x, o, l, u, ol, ol_set, opt_prop, ap_gamma, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []              # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    cost_list_gamma = []        # [[value, current step,                        path_length], [value, current step,                     path_length], ...]
    diff_exp_list = []          # [[value, current step pi, current step gamma, path_length], [value, current step, current step gamma, path_length], ...]
    #
    cost_cycle_pi = 0.
    cost_cycle_gamma = 0.
    for i in range(1, o.__len__()):
        x_i_last = x[i - 1]
        o_i_last = o[i - 1]
        x_i = x[i]
        o_i = o[i]
        ol_i = list(ol_set[i])  # l_i = list(l[i])
        if i < x.__len__() - 1:
            # TODO
            u_i = u[i]

        #
        is_observed_ap_pi_found = False
        is_observed_ap_gamma_found = False
        for o_t in ol_i:
            if ol_i.__len__() and opt_prop in o_t:
                is_observed_ap_pi_found = True
                #break                                      # break不能加, 加了就出去了影响前后判断
            #
            if ol_i.__len__() and ap_gamma in o_t:
                is_observed_ap_gamma_found = True
                #break

        if is_observed_ap_pi_found:
            if cost_list.__len__():
                path_length_pi = i - cost_list[cost_list.__len__() - 1][1]
            else:
                path_length_pi = i
            #
            cost_current_step = [cost_cycle_pi, i, path_length_pi]
            cost_list.append(cost_current_step)
            cost_cycle_pi = 0.
        if is_observed_ap_gamma_found:
            if cost_list_gamma.__len__():
                path_length_gamma = i - cost_list_gamma[cost_list_gamma.__len__() - 1][1]
            else:
                path_length_gamma = i
            #
            cost_current_step = [cost_cycle_gamma, i, path_length_gamma]
            cost_list_gamma.append(cost_current_step)
            cost_cycle_gamma = 0.
        #
        #
        available_event_cost_dict_pi = {}
        available_event_cost_dict_gamma = {}
        for o_last_t in o_i_last:
            for o_t in o_i:
                for edge_t in list(product_mdp.graph['mdp'].edges(o_last_t, data=True)):
                    if o_t == edge_t[1]:
                        label_o_t = list(product_mdp.graph['mdp'].nodes[o_t]['label'].keys())
                        label_o_t = [list(fs)[0] for fs in label_o_t if isinstance(fs, frozenset)]
                        #
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]
                        #
                        if opt_prop in label_o_t:
                            available_event_cost_dict_pi[event_t] = cost_t
                        if ap_gamma in label_o_t:
                            available_event_cost_dict_gamma[event_t] = cost_t
        if available_event_cost_dict_pi.__len__():
            min_event = min(available_event_cost_dict_pi, key=lambda e: available_event_cost_dict_pi[e])
            min_cost = available_event_cost_dict_pi[min_event]

            cost_cycle_pi    += min_cost
        if available_event_cost_dict_gamma.__len__():
            min_event = min(available_event_cost_dict_gamma, key=lambda e: available_event_cost_dict_gamma[e])
            min_cost = available_event_cost_dict_gamma[min_event]

            cost_cycle_gamma += min_cost

        else:
            #
            # TODO
            # 用x代替?
            for edge_t in list(product_mdp.graph['mdp'].edges(x_i_last, data=True)):
                    if x_i == edge_t[1]:
                        event_t = list(edge_t[2]['prop'])[0]  # event_t = i_i???
                        cost_t = edge_t[2]['prop'][event_t][1]

                        cost_cycle_gamma += cost_t
                        cost_cycle_pi += cost_t
            #print_c("[sync cost execution] warning ... using original cost instead of observed cost")


    #
    # 先全部算
    # 这个不同步的算法不准确, 但是能用
    used_gamma_indices = set()

    last_pi_step = 0
    last_gamma_step = 0

    for i in range(len(cost_list)):
        step_i = cost_list[i][1]

        # 在 gamma 列表中找与 step_i 最接近的 step_j，避免重复匹配
        min_j_index = -1
        min_step_diff = float('inf')
        for j in range(len(cost_list_gamma)):
            if j in used_gamma_indices:
                continue  # 防止重复匹配

            step_j = cost_list_gamma[j][1]
            diff = abs(step_i - step_j)
            if diff < min_step_diff:
                min_step_diff = diff
                min_j_index = j

        # 如果找不到匹配的 gamma（极端情况），跳过
        if min_j_index == -1:
            continue

        used_gamma_indices.add(min_j_index)

        cost_pi = cost_list[i]
        cost_gamma = cost_list_gamma[min_j_index]

        diff_cost = abs(cost_pi[0] - cost_gamma[0])
        path_length = max(cost_pi[1] - last_pi_step, cost_gamma[1] - last_gamma_step)

        diff_exp_list.append([
            diff_cost,
            cost_pi[1],
            cost_gamma[1],
            path_length
        ])

        last_pi_step = cost_pi[1]
        last_gamma_step = cost_gamma[1]

    if is_remove_zeros:
        #
        # pi
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list.remove(cost_tuple_t)
            except:
                pass
        #
        # gamma
        cost_tuple_to_remove = []
        for cost_tuple_t in cost_list_gamma:
            if cost_tuple_t[0] == 0.:
                cost_tuple_to_remove.append(cost_tuple_t)

        for cost_tuple_t in cost_tuple_to_remove:
            try:
                cost_list_gamma.remove(cost_tuple_t)
            except:
                pass


    return cost_list, cost_list_gamma, diff_exp_list