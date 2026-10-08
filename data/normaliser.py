import csv
import os
from itertools import islice

# ============================================================
# CONFIGURATION
# ============================================================

# Choose which TCP data to use:
#   "actual"   -> ax, ay, az
#   "expected" -> ex, ey, ez
TCP_SOURCE = "actual"
INPUT_FILE = "robot_data_1e-4.csv"
NUM_POINTS = 1000



OUTPUT_FILE = f"normalised/{TCP_SOURCE}_{NUM_POINTS}_normalized_{INPUT_FILE}"
INPUT_FILE = "csv/robot_data_1e-4.csv"

JOINT_LIMITS = {
    "joint_1": (-2.8, 2.8),
    "joint_2": (-0.8, 0.8),
    "joint_3": (0, 1.3),
}


# ============================================================
# FUNCTIONS
# ============================================================

def normalise_joint(q, q_min, q_max):
    """
    Normalise a joint value to [-1, 1]:

        q_norm = 2 * (q - q_min) / (q_max - q_min) - 1
    """
    return 2 * (q - q_min) / (q_max - q_min) - 1


def process_csv():
    if TCP_SOURCE not in ("actual", "expected"):
        raise ValueError(
            "TCP_SOURCE must be either 'actual' or 'expected'"
        )

    # Select TCP columns
    if TCP_SOURCE == "actual":
        x_col, y_col, z_col = "ax", "ay", "az"
    else:
        x_col, y_col, z_col = "ex", "ey", "ez"

    # Make sure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(f"{INPUT_FILE}", "r", newline="") as infile:
        reader = csv.DictReader(infile)

        with open(OUTPUT_FILE, "w", newline="") as outfile:
            fieldnames = [
                "joint1",
                "joint2",
                "joint3",
                "x",
                "y",
                "z",
            ]

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in islice(reader, NUM_POINTS):
                # Original joint values
                joint1 = float(row["joint1"])
                joint2 = float(row["joint2"])
                joint3 = float(row["joint3"])

                # Normalise joints
                joint1_norm = normalise_joint(
                    joint1,
                    *JOINT_LIMITS["joint_1"]
                )

                joint2_norm = normalise_joint(
                    joint2,
                    *JOINT_LIMITS["joint_2"]
                )

                joint3_norm = normalise_joint(
                    joint3,
                    *JOINT_LIMITS["joint_3"]
                )

                # Selected TCP coordinates
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col])

                writer.writerow({
                    "joint1": joint1_norm,
                    "joint2": joint2_norm,
                    "joint3": joint3_norm,
                    "x": x,
                    "y": y,
                    "z": z,
                })

    print(f"TCP source: {TCP_SOURCE}")
    print(f"Input:      {INPUT_FILE}")
    print(f"Output:     {OUTPUT_FILE}")


if __name__ == "__main__":
    process_csv()