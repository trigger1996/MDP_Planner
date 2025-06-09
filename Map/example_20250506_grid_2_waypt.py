#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import math
from collections import defaultdict
from collections import OrderedDict
from example_20250506_team_mdp import build_mdp_with_grid

def grid_to_world(row, col, resolution=0.4, z=1.2, yaw=0.0):
    x = col * resolution
    y = row * resolution
    return x, y, z, yaw


def convert_nodes_to_waypoints(grid_nodes, transition_time=3):
    waypoints = OrderedDict()
    for idx, (node_id, attr) in enumerate(grid_nodes.items(), start=1):
        x, y = attr["pos"]
        z, yaw = 1.2, 0.0
        waypoint_id = str(idx)
        waypoints[waypoint_id] = {
            "pos": [x, y, z, yaw],
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

    yaml_data = OrderedDict()
    yaml_data['name'] = 'test2025'
    yaml_data['waypoint'] = convert_nodes_to_waypoints(grid_nodes, transition_time=3)
    yaml_data['edges'] = convert_edges_to_yaml_format(robot_edges)

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ YAML with nodes and edges saved to {output_path}")


if __name__ == '__main__':
    generate_full_yaml()
