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
    'u': ['0', '1', '2', '11', '14', '15'],
    'v': ['8', '12'],
    'w': ['3', '4', '5', '6'],
    'x': ['7', '13'],
    'y': ['9', '10'],
}

# True = Observable
control_observable_dict = {
    'a' : False,
    'b' : False,
    'c' : True,
    'd' : True

}

def build_model():

    global robot_nodes_w_aps, robot_edges, U, initial_node, initial_label

    # robot nodes
    # the lower satisfaction probability may result in conflict/failure in opacity constraint in user/lp.py
    robot_nodes_w_aps['0']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['1']  = { frozenset({'drop'}): 0.85 }
    robot_nodes_w_aps['2']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['3']  = { frozenset({'recharge'}): 0.65 }
    robot_nodes_w_aps['4']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['5']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['6']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['7']  = { frozenset({'gather'}): 0.75 }
    robot_nodes_w_aps['8']  = { frozenset({'gather'}): 0.95 }
    robot_nodes_w_aps['9']  = { frozenset(): 1.0 }
    robot_nodes_w_aps['10'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['11'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['12'] = { frozenset({'gather'}): 0.85 }
    robot_nodes_w_aps['13'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['14'] = { frozenset({'recharge'}): 0.95 }
    robot_nodes_w_aps['15'] = { frozenset({'recharge'}): 0.5 }
    #
    robot_edges = {
        # x, a, x' :    prob, cost
        ('0', 'a', '1')  : (1,   1),
        ('0', 'b', '7')  : (1,   3),
        ('0', 'c', '13') : (1,   2),
        #
        #('1', 'c', '0') : (1,   3),
        ('1', 'a', '2') : (1,   3),
        ('1', 'b', '3') : (0.6, 1),
        ('1', 'b', '5') : (0.4, 2),
        #
        ('2', 'a', '1'):  (1,   3),
        ('2', 'b', '5') : (0.3, 1),
        ('2', 'b', '6') : (0.7, 4),
        #
        ('3', 'a', '5') : (0.9, 3),
        ('3', 'a', '4') : (0.1, 1),
        ('3', 'a', '9') : (1, 1),
        #
        ('4', 'a', '3') : (1,  4),
        ('4', 'd', '4') : (1,  1),
        #
        ('5', 'a', '6') : (0.5, 2),
        ('5', 'a', '8') : (0.5, 2),
        ('5', 'b', '10'): (0.3, 4),
        ('5', 'b', '14'): (0.7, 2),
        #
        ('6', 'a', '2'):  (0.8, 1),
        ('6', 'a', '5'):  (0.2, 3),
        ('6', 'd', '6') : (1,  1),
        #
        ('7', 'c', '2') : (0.5, 3),
        ('7', 'c', '6') : (0.2, 4),
        ('7', 'c', '8') : (0.2, 5),
        ('7', 'c', '13'): (0.1, 7),
        #
        ('8', 'd', '8')  : (1,   1),
        ('8', 'a', '5')  : (0.2, 3),
        ('8', 'a', '7')  : (0.8, 7),
        #
        ('9', 'b', '3')   : (0.9, 2),
        ('9', 'b', '11')  : (0.1, 2),               # IF NOT FEASIBLE TRY REMOVING THIS TRANSITION
        ('9', 'c', '9')   : (0.8, 1),
        ('9', 'c', '10')  : (0.2, 4),
        #
        ('10', 'a', '12') : (0.5, 2),
        ('10', 'a', '5')  : (0.5, 4),
        #
        ('11', 'a', '9')  : (1, 2),
        ('11', 'b', '13') : (1, 2),
        ('11', 'c', '15') : (1, 1),
        #
        ('12', 'c', '9')  : (1, 4),
        #
        ('13', 'a', '0')  : (0.5, 4),
        ('13', 'a', '7')  : (0.5, 8),
        #
        ('14', 'a', '8')  : (1,  2),
        #
        ('15', 'c', '15') : (1, 1),
    }

    #
    U = [ 'a', 'b', 'c', 'd' ]

    #
    initial_node  = '0'                                 # '0' and '11' available
    initial_label = frozenset()


    return (robot_nodes_w_aps, robot_edges, U, initial_node, initial_label)


def observation_func_0105(x, u=None):
    global observation_dict

    for y in observation_dict.keys():
        if x in observation_dict[y]:
            return y

    print("[observation_func_0105] Please check input x !")
    raise TypeError

    return None

def observation_inv_func_0105(y):
    return observation_dict[y]

def run_2_observations_seqs(x_u_seqs):
    y_seq = []
    for i in range(0, x_u_seqs.__len__() - 1, 2):
        x_t = x_u_seqs[i]
        u_t = x_u_seqs[i + 1]
        y_t = observation_func_0105(x_t, u_t)
        y_seq.append(y_t)
        #y_seq.append(u_t)           # u is for display and NOT in actual sequences
    return y_seq

def observation_seq_2_inference(y_seq):
    global robot_nodes_w_aps
    x_inv_set_seq = []
    ap_inv_seq = []
    for i in range(0, y_seq.__len__()):
        x_inv_t = observation_inv_func_0105(y_seq[i])
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
    for i in range(1, x.__len__()):
        x_i = x[i]
        x_p = None
        l_i = list(l[i])
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