from MDP_TG.mdp import Motion_MDP
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

robot_nodes_w_aps = dict()
robot_edges = dict()
U = []
initial_node  = None
initial_label = None

#
# in simulations, we can let those states with identical APs carry identical observations, which is to simulate the APs are observed satisfied
observation_dict = {
    'u': ['0'],
    'v': ['1'],
    'w': ['2', '3'],
}

'''
    # 这个东西在lp_x.py是这么用的
    try:
        u_pi = next(iter(edge_t_pi[2]['prop'].keys()))
        u_gamma = next(iter(edge_t_gamma[2]['prop'].keys()))
        if isinstance(u_pi, tuple):
            u_pi = u_pi[0]
        if isinstance(u_gamma, tuple):
            u_gamma = u_gamma[0]

        # 如果不考虑可观性
        if ctrl_obs_dict == None and u_pi != u_gamma:
            continue
        # 如果考虑可观性
        elif u_pi != u_gamma and ctrl_obs_dict[u_pi] == True and ctrl_obs_dict[u_gamma] == True:
            continue
    except (StopIteration, KeyError):
        continue  # 如果没有动作则跳过
'''
# control_observable_dict = {
#     'a' : False,
#     'b' : True,
#     'c' : False
#
# }
control_observable_dict = None

def build_model():

    global robot_nodes_w_aps, robot_edges, U, initial_node, initial_label

    # robot nodes
    # the lower satisfaction probability may result in conflict/failure in opacity constraint in user/lp.py
    robot_nodes_w_aps['0'] = { frozenset({'upload'}) : 1.0 }
    robot_nodes_w_aps['1'] = { frozenset({'gather'}) : 1.  }
    robot_nodes_w_aps['2'] = { frozenset({'gather'}) : 1.  }
    robot_nodes_w_aps['3'] = {frozenset({'recharge'}): 1.  }
    #
    robot_edges = {
        # x,   a,   x'  : prob, cost
        ('0', 'a', '1') : (1, 1),            # gather
        ('1', 'a', '0') : (1, 1),            #
        ('0', 'b', '0') : (0.05, 2),

        ('0', 'b', '2') : (0.5, 3),
        ('2', 'b', '0') : (1, 2),

        ('0', 'b', '3') : (0.45, 3),
        ('3', 'b', '0') : (1, 2),
    }

    #
    U = [ 'b' ]

    #
    initial_node  = '0'
    initial_label = list(robot_nodes_w_aps[initial_node].keys())[0]


    return (robot_nodes_w_aps, robot_edges, U, initial_node, initial_label)


def observation_func_0401(x, u=None):
    global observation_dict

    for y in observation_dict.keys():
        if x in observation_dict[y]:
            return y

    print("[observation_func_0401] Please check input x !")
    raise TypeError

    return None

def observation_inv_func_0401(y):
    return observation_dict[y]

def run_2_observations_seqs(x_u_seqs):
    y_seq = []
    for i in range(0, x_u_seqs.__len__() - 1, 2):
        x_t = x_u_seqs[i]
        u_t = x_u_seqs[i + 1]
        y_t = observation_func_0401(x_t, u_t)
        y_seq.append(y_t)
        #y_seq.append(u_t)           # u is for display and NOT in actual sequences
    return y_seq

def observation_seq_2_inference(y_seq):
    global robot_nodes_w_aps
    x_inv_set_seq = []
    ap_inv_seq = []
    for i in range(0, y_seq.__len__()):
        x_inv_t = observation_inv_func_0401(y_seq[i])
        #
        ap_inv_t = []
        for state_t in x_inv_t:
            # ap_inv_t = ap_inv_t + list(robot_nodes_w_aps[state_t].keys())
            ap_list_t = list(robot_nodes_w_aps[state_t].keys())
            for ap_t in ap_list_t:
                ap_inv_t = ap_inv_t + list(ap_t)

        ap_inv_t = list(set(ap_inv_t))
        x_inv_set_seq.append(x_inv_t)
        ap_inv_seq.append(ap_inv_t)
    return x_inv_set_seq, ap_inv_seq

def calculate_cost_from_runs(product_mdp, x, l, u, opt_prop, is_remove_zeros=True):
    #
    # calculate the transition cost
    # if current state staifies optimizing AP
    # then zero the AP
    cost_list = []                      # [[value, current step, path_length], [value, current step, path_length], ...]
    cost_cycle = 0.
    #
    x_i_last = x[0]
    for i in range(0, x.__len__()):
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

def plot_cost_hist(cost_list, bins=25, color='g', is_average=True, title="Cost Distribution", xlabel="Cost", ylabel="Probability"):
    plt.figure()
    cost_list.sort(key=lambda x: x[0])
    data_list = []
    for i in range(0, cost_list.__len__()):
        if is_average:
            data_list.append(cost_list[i][0] / cost_list[i][2])
        else:
            data_list.append(cost_list[i][0])

    sns.histplot(data_list, bins=bins, kde=True, color=color, stat="probability")    # stat="density" "probability"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)