from MDP_TG.mdp import Motion_MDP


robot_nodes_w_aps = dict()
robot_edges = dict()
U = []
initial_node  = None
initial_label = None

observation_dict = {
    'u': ['0', '1', '2'],
    'v': ['3', '7'],
    'w': ['4', '5', '6'],
    'x': ['8'],
}

def build_model():

    global robot_nodes_w_aps, robot_edges, U, initial_node, initial_label

    # robot nodes
    robot_nodes_w_aps['0'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['1'] = { frozenset({'gather'}): 0.85}
    robot_nodes_w_aps['2'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['3'] = { frozenset({'recharge'}): 0.75 }
    robot_nodes_w_aps['4'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['5'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['6'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['7'] = { frozenset({'drop'}): 0.65 }
    robot_nodes_w_aps['8'] = { frozenset({'drop'}): 0.95 }

    #
    robot_edges = {
        # x, a, x' :    prob, cost
        ('0', 'a', '1') : (1,   1),
        ('0', 'b', '7') : (1,   3),
        #
        #('1', 'c', '0') : (1,   3),
        ('1', 'a', '2') : (1,   3),
        ('1', 'b', '3') : (0.6, 1),
        ('1', 'b', '5') : (0.4, 2),
        #
        ('2', 'a', '1'):  (1,   3),
        ('2', 'b', '5') : (0.3, 1),
        ('2', 'b', '6') : (0.7, 1),
        #
        ('3', 'a', '6') : (0.9, 3),
        ('3', 'a', '4') : (0.1, 1),
        #
        ('4', 'a', '3') : (1,  4),
        ('4', 'd', '4') : (1,  1),
        #
        ('5', 'a', '6') : (0.5, 2),
        ('5', 'a', '8') : (0.5, 2),
        #
        ('6', 'a', '2'):  (0.8, 1),
        ('6', 'a', '5'):  (0.2, 3),
        ('6', 'd', '6') : (1,  1),
        #
        ('7', 'c', '2') : (0.6, 3),
        ('7', 'c', '6') : (0.2, 4),
        ('7', 'c', '8') : (0.2, 5),
        #
        ('8', 'd', '8') : (1,   1),
        ('8', 'a', '5') : (0.2, 3),
        ('8', 'a', '7') : (0.8, 7),
    }

    #
    U = [ 'a', 'b', 'c', 'd' ]

    #
    initial_node  = '0'
    initial_label = frozenset()


    return (robot_nodes_w_aps, robot_edges, U, initial_node, initial_label)


def observation_func_1018(x, u=None):
    global observation_dict

    for y in observation_dict.keys():
        if x in observation_dict[y]:
            return y

    return None

def observation_inv_func_1018(y):
    return observation_dict[y]

def run_2_observations_seqs(x_u_seqs):
    y_seq = []
    for i in range(0, x_u_seqs.__len__() - 1, 2):
        x_t = x_u_seqs[i]
        u_t = x_u_seqs[i + 1]
        y_t = observation_func_1018(x_t, u_t)
        y_seq.append(y_t)
        #y_seq.append(u_t)           # u is for display and NOT in actual sequences
    return y_seq

def observation_seq_2_inference(y_seq):
    global robot_nodes_w_aps
    x_inv_set_seq = []
    ap_inv_seq = []
    for i in range(0, y_seq.__len__()):
        x_inv_t = observation_inv_func_1018(y_seq[i])
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