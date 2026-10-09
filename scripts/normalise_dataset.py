"""
Task 2 - normalise Ned joint values to [-1, 1].

    q_norm = 2 * (q - q_min) / (q_max - q_min) - 1

q_min / q_max are the ranges the joints were *sampled* from (not the min/max
of the data), so the network output can be mapped back with denormalise().
joint_3 is restricted to positive values (elbow-up only) as required by Task 2.

Input : data/csv/<name>.csv  (joint1, joint2, joint3, ex, ey, ez, ax, ay, az, error)
Output: data/csv/<name>_normalised.csv  (q1, q2, q3, x, y, z)
        x, y, z are the FK positions (ex, ey, ez).

Usage: python scripts/normalise_dataset.py [robot_data_1e-3.csv]
"""
import csv
import sys
from pathlib import Path

CSV_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"

NED_RANGES = {
    "joint1": (-2.8, 2.8),
    "joint2": (-0.8, 0.8),
    "joint3": (0.0, 1.3),  # positive only (elbow-up)
}


def normalise(q, q_min, q_max):
    return 2.0 * (q - q_min) / (q_max - q_min) - 1.0


def denormalise(q_norm, q_min, q_max):
    return (q_norm + 1.0) / 2.0 * (q_max - q_min) + q_min


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "robot_data_1e-3_2.csv"
    src = CSV_DIR / name
    dst = src.with_name(src.stem + "_normalised.csv")

    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["q1", "q2", "q3", "x", "y", "z"])
        n = 0
        for row in reader:
            q = [normalise(float(row[j]), *NED_RANGES[j]) for j in NED_RANGES]
            if any(abs(v) > 1.0 for v in q):
                raise ValueError(f"row {n + 2}: joint outside sampling range: {row}")
            writer.writerow(q + [row["ex"], row["ey"], row["ez"]])
            n += 1

    print(f"Normalised {n} samples -> {dst}")


if __name__ == "__main__":
    main()
