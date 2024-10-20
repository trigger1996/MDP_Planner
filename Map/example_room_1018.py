from MDP_TG.mdp import Motion_MDP


def build_model():


    # robot nodes
    robot_nodes_w_aps = dict()
    robot_nodes_w_aps['0'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['1'] = { frozenset(): 0.0, frozenset({'gather'}): 0.85}
    robot_nodes_w_aps['2'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['3'] = { frozenset(): 0.0, frozenset({'recharge'}): 0.75 }
    robot_nodes_w_aps['4'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['5'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['6'] = { frozenset(): 1.0 }
    robot_nodes_w_aps['7'] = { frozenset(): 0.0, frozenset({'drop'}): 0.65 }
    robot_nodes_w_aps['8'] = { frozenset(): 0.0, frozenset({'drop'}): 0.95 }

    #
    robot_edges = {
        # x, a, x' :    prob, cost
        ('0', 'a', '1') : (1,   1),
        ('0', 'b', '7') : (1,   3),
        #
        ('1', 'a', '2') : (1,   1),
        ('1', 'b', '3') : (0.6, 2),
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
