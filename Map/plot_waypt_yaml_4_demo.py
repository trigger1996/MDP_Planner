import yaml
import matplotlib.pyplot as plt

# === 如果你已经有Python对象, 可以跳过yaml.safe_load ===
yaml_file = "./yaml/20250426_map_w_edges.yaml"              # TODO
#yaml_file = "./yaml/20250506_map_w_edges.yaml"
#
with open(yaml_file, "r") as f:
    data = yaml.load(f, Loader=yaml.UnsafeLoader)

waypoints = data["waypoint"]
edges = data["edges"]

plt.figure(figsize=(8, 6))

# 画节点 (转换到 NED: 用 y 作为横坐标, x 作为纵坐标)
for node_id, node_info in waypoints.items():
    x_enu, y_enu, z, yaw = node_info["pos"]
    ap = node_info["ap"][0]

    # ENU -> NED (swap)
    Xned = y_enu   # East -> X
    Yned = x_enu   # North -> Y

    plt.scatter(Xned, Yned, c="skyblue", edgecolors="black", s=200, zorder=3)
    plt.text(Xned, Yned + 0.1, f"{node_id}\n{ap}", ha="center", va="bottom", fontsize=8)

# 画边
for edge in edges:
    src, dst, attr = edge
    x1, y1, _, _ = waypoints[src]["pos"]
    x2, y2, _, _ = waypoints[dst]["pos"]

    # ENU -> NED
    X1, Y1 = y1, x1
    X2, Y2 = y2, x2

    control = attr["control"]
    weight = attr["weight"]

    plt.arrow(X1, Y1, X2 - X1, Y2 - Y1,
              length_includes_head=True,
              head_width=0.1, head_length=0.15,
              fc="gray", ec="gray", alpha=0.7, zorder=2)

    xm, ym = (X1 + X2) / 2, (Y1 + Y2) / 2
    plt.text(xm, ym, f"{control},{weight}", fontsize=7, color="red")

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title(data["name"])
plt.grid(True, linestyle="--", alpha=0.5)

# === 如果你希望严格符合 NED (Y 轴向下)，取消注释下面这行 ===
# plt.gca().invert_yaxis()

plt.show()

