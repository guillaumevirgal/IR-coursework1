import csv

import matplotlib.pyplot as plt
from vis2 import draw_robot, read_joints_from_csv

ax_pos = []
ay_pos = []
az_pos = []
errors = []

file_name = "robot_data_double_1e-3.csv"

with open(file_name, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        ax_pos.append(float(row["ax"]))
        ay_pos.append(float(row["ay"]))
        az_pos.append(float(row["az"]))
        errors.append(float(row["error"]) * 1000)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    ax_pos,
    ay_pos,
    az_pos,
    c=errors,
    cmap="RdYlGn_r",
    # vmin=0,
    # vmax=0.3,
    s=3
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"TCP Position Distribution ({len(ax_pos)} points)")

# Add colour bar showing error
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("TCP Error (mm)")

draw_robot(ax, *read_joints_from_csv(3, filename=file_name))

plt.show()