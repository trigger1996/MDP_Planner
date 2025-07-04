#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MDP_Planner'))

import yaml
import math
from collections import defaultdict
from collections import OrderedDict
from example_20250506_team_mdp import build_mdp_with_grid

def grid_to_world(row, col, resolution=0.4, z=1.2, yaw=0.0):
    x = col * resolution
    y = row * resolution
    return x, y, z, yaw


def convert_nodes_to_waypoints(robot_nodes_w_aps, grid_nodes, resolution, transition_time=0):
    waypoints = OrderedDict()
    for idx, (node_id, attr) in enumerate(grid_nodes.items(), start=1):
        x, y = attr["pos"]
        z    = 1.2
        yaw  = 90.0
        x_w, y_w, z_w, yaw_w = grid_to_world(y, x, resolution, z, yaw)
        waypoint_id = str(node_id)                  #
        waypoints[waypoint_id] = {
            "pos": [x_w, y_w, z_w, yaw_w],
            "ap": [str(set(ap_t)) for ap_t in list(robot_nodes_w_aps[node_id].keys())],
            "transition": transition_time
        }
    return waypoints


def convert_edges_to_yaml_format(robot_edges):
    edges = []
    for (u, act, v), (prob, cost) in robot_edges.items():
        edges.append([u, v, {'control': act, 'weight': round(cost)}])
    return edges


def generate_full_yaml(x_len=5, y_len=5, output_path='./yaml/20250506_map_w_edges.yaml'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    robot_nodes_w_aps, robot_edges, _, grid_nodes, _, _ = build_mdp_with_grid(5, 5, start_position=(0, 0))

    #
    # Added
    # Critical
    resolution = 4

    yaml_data = OrderedDict()
    yaml_data['name'] = 'test2025'
    yaml_data['waypoint'] = convert_nodes_to_waypoints(robot_nodes_w_aps, grid_nodes, resolution, transition_time=0)
    yaml_data['edges'] = convert_edges_to_yaml_format(robot_edges)

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ YAML with nodes and edges saved to {output_path}")


if __name__ == '__main__':
    generate_full_yaml()
