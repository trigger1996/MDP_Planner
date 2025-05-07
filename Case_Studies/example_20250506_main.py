#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt

from Map.example_20250506_team_mdp import construct_team_mdp

# ----------------------------
# 示例使用
if __name__ == "__main__":
    # 生成6x6栅格MDP
    team_mdp, initial_node, initial_label, node_positions = construct_team_mdp()

    # # 创建可视化图
    # G = nx.DiGraph()
    #
    # # 添加带位置的节点
    # for node, attrs in robot_nodes_w_aps.items():
    #     G.add_node(node, pos=node_positions[node]["pos"])  # 使用 node_positions 中的坐标
    #
    # # 添加带权重的边
    # for (u, a, v), (prob, cost) in robot_edges.items():
    #     G.add_edge(u, v, weight=cost, action=a)
    #
    # # 绘制图形
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(10, 8))
    # nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue', font_size=10)
    #
    # # 绘制边标签（显示动作和权重）
    # edge_labels = {(u, v): f"{a}\n{c:.1f}"
    #                for u, v, a, c in [(u, v, d['action'], d['weight'])
    #                                   for u, v, d in G.edges(data=True)]}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    #
    # plt.title("6x6 Grid MDP Visualization")
    # plt.tight_layout()
    # plt.show()

