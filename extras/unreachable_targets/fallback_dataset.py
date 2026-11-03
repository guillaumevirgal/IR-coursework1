"""Build the unreachable-target ("directional fallback") datasets.

Extra experiment, not part of the coursework deliverable. The datasets teach the
network what to do when asked for a point outside the workspace: aim in the right
direction and reach as close as possible.

Everything here is data preparation only. It never trains a model and never runs
Webots. Inputs are the already-generated reachable splits and their Webots audit
(see scripts/generate_dataset.py); the output is a train/validation/test set in
which each split mixes the reachable rows with certified-unreachable requests.

How one unreachable row is made:
1. Sample a point 1.5 cm to 60 cm away from a known reachable pose.
2. Keep it only if a conservative workspace certificate proves that no joint
   configuration can reach it (WorkspaceCertificate).
3. Label it with the closest recorded reachable pose that points within 2 degrees
   of the best available direction (choose_directional_poses). The label is a real
   Webots-checked pose, not a measurement at the impossible point.
"""

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXTRAS = Path(__file__).resolve().parent  # extras/unreachable_targets; its data/ holds these datasets
ROOT = EXTRAS.parents[1]
sys.path.insert(0, str(ROOT))
from scripts import forward_kinematics_ned as ned
from scripts import forward_kinematics_scara as scara
from scripts.task4_common import write_csv
from scripts.numpy_neighbors import NumpyNearestNeighbors as cKDTree
from scripts.robot_config import JOINT_LIMITS
from scripts.tcp_loss import tcp_and_jacobian
from scripts.train_mlp import OUTPUTS

ROBOTS = ("ned", "scara")
DIRECTION_TOLERANCE_DEG = 2.0
MIN_UNREACHABLE_CLEARANCE_M = 0.005
QUERY_OFFSETS_M = (0.015, 0.60)
NEAR_FAR_SPLIT_M = 0.10


@dataclass(frozen=True)
class Experiment:
    """One dataset variant; the fields are exactly what a config file stores."""

    name: str
    size: str
    seed: int
    train_extra: int          # unreachable training rows per robot
    test_extra: int           # unreachable test rows per robot
    reachable_splits: Path    # folder with <robot>/train_*.csv, validation_*.csv, test.csv
    reachable_audits: Path    # folder with <robot>_master_audit.csv
    test_requests: Path = None  # reuse another dataset's unreachable test points


def load_experiment(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    def folder(key):
        value = raw.get(key)
        return None if value is None else (ROOT / value).resolve()
    experiment = Experiment(
        name=raw["name"], size=raw["size"], seed=int(raw["seed"]),
        train_extra=int(raw["train_extra"]), test_extra=int(raw["test_extra"]),
        reachable_splits=folder("reachable_splits"), reachable_audits=folder("reachable_audits"),
        test_requests=folder("test_requests"),
    )
    if experiment.size not in ("small", "medium", "large"):
        raise ValueError(f"Unknown size {experiment.size!r} in {path}")
    if experiment.train_extra < 4 or experiment.test_extra < 2 or experiment.seed < 0:
        raise ValueError("train_extra must be >= 4, test_extra >= 2 and seed >= 0")
    return experiment


def resolve_dataset_dir(path):
    """Find a dataset folder whether it is recorded as repo-relative or extras-relative.

    Saved model configs record paths such as data/nn/directional_fallback from
    before these datasets moved into extras/unreachable_targets/data.
    """
    path = Path(path)
    for candidate in (path, ROOT / path, EXTRAS / path):
        if candidate.is_absolute() and candidate.exists():
            return candidate
    return ROOT / path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def array(rows, fields):
    return np.asarray([[float(r[key]) for key in fields] for r in rows])


def posix_relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def fingerprint(paths, root=ROOT):
    return {posix_relative(p, root): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def directions(robot, points):
    points = np.atleast_2d(points)
    offset = (points - [0, 0, ned.SHOULDER_HEIGHT] if robot == "ned"
              else points[:, :2] - [scara.BASE_TO_JOINT1_X, 0])
    radius = np.linalg.norm(offset, axis=1)
    valid = radius > 1e-10
    return offset / np.maximum(radius[:, None], 1e-10), valid


def direction_errors(robot, targets, positions):
    target_dir, target_valid = directions(robot, targets)
    position_dir, position_valid = directions(robot, positions)
    angles = np.degrees(np.arccos(np.clip(np.sum(target_dir * position_dir, axis=1), -1, 1)))
    return np.where(target_valid & position_valid, angles, np.nan)


def choose_directional_poses(robot, points, reference_xyz, tolerance_deg=DIRECTION_TOLERANCE_DEG):
    """Prioritise direction; minimise request distance within the angle band.

    A 2-degree band around the best recorded direction avoids selecting a
    much shorter pose just for a negligible improvement in angular alignment.
    SCARA direction is planar bearing; Ned uses the shoulder-to-TCP vector.
    At SCARA's exact yaw axis the bearing is undefined, so use distance alone.
    """
    reference_dir, _ = directions(robot, reference_xyz)
    requested_dir, valid = directions(robot, points)
    tree = cKDTree(reference_dir)
    chord, _ = tree.query(requested_dir)
    best_angle = 2 * np.arcsin(np.minimum(chord / 2, 1.0))
    radius = 2 * np.sin(np.minimum(best_angle + np.radians(tolerance_deg), np.pi) / 2)
    candidates = tree.query_ball_point(requested_dir, radius + 1e-12)
    chosen = []
    for point, allowed, defined in zip(points, candidates, valid):
        allowed = np.asarray(allowed if defined else np.arange(len(reference_xyz)), dtype=int)
        distance = np.linalg.norm(reference_xyz[allowed] - point, axis=1)
        chosen.append(int(allowed[np.argmin(distance)]))
    return np.asarray(chosen), np.degrees(best_angle)


class WorkspaceCertificate:
    """Lower bound on distance to every kinematically achievable TCP.

    A complete joint grid covers joint space. Each q is at most half a step
    from a grid node on each axis. Global Jacobian norm bounds then bound the
    TCP displacement between those two configurations. Subtracting that cover
    radius from the nearest grid distance gives a conservative lower bound.
    The grid includes all joint combinations, even poses rejected for floor or
    collision reasons, so this certificate cannot label a reachable hole as OOW.
    """

    def __init__(self, robot, resolution=65):
        self.robot = robot
        limits = np.asarray(JOINT_LIMITS[robot])
        axes = [np.linspace(lo, hi, resolution) for lo, hi in limits]
        joints = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.tree = cKDTree(tcp_and_jacobian(robot, joints)[0])
        if robot == "ned":
            self.forearm = np.hypot(ned.FOREARM_REACH, ned.FOREARM_RISE)
            self.radius = np.hypot(ned.UPPER_ARM_REACH, ned.UPPER_ARM_RISE) + self.forearm
            derivative_bounds = np.array([self.radius, self.radius, self.forearm])
        else:
            self.forearm = np.hypot(scara.JOINT2_TO_TCP_X, scara.JOINT2_TO_TCP_Y)
            self.radius = np.hypot(scara.JOINT1_TO_JOINT2_X, scara.JOINT1_TO_JOINT2_Y) + self.forearm
            derivative_bounds = np.array([self.radius, self.forearm, 1.0])
        self.cover_radius = float(np.sum(derivative_bounds * (limits[:, 1] - limits[:, 0])
                                         / (2 * (resolution - 1))))

    def lower_bound(self, points):
        points = np.atleast_2d(points)
        grid_bound = self.tree.query(points)[0] - self.cover_radius
        if self.robot == "ned":
            offset = points - np.array([0, 0, ned.SHOULDER_HEIGHT])
            analytic = np.linalg.norm(offset, axis=1) - self.radius
        else:
            planar = points[:, :2] - [scara.BASE_TO_JOINT1_X, 0]
            z_base = scara.BASE_TO_JOINT1_Z + scara.JOINT1_TO_JOINT2_Z + scara.TCP_Z_OFFSET
            lo, hi = JOINT_LIMITS["scara"][2]
            analytic = np.maximum.reduce((np.linalg.norm(planar, axis=1) - self.radius,
                                          points[:, 2] - z_base - hi,
                                          z_base + lo - points[:, 2]))
        return np.maximum.reduce((grid_bound, analytic, np.zeros(len(points))))


def sample_requests(robot, count, seed, certificate, reference_xyz):
    """Equal near/far coverage, with independent seeds for every data split."""
    rng = np.random.default_rng(seed)
    reference_tree = cKDTree(reference_xyz)
    selected = [[], []]
    quotas = [count // 2, count - count // 2]
    for _ in range(1000):
        if all(len(selected[i]) >= quotas[i] for i in (0, 1)):
            break
        anchor = reference_xyz[rng.integers(len(reference_xyz), size=4096)]
        direction = rng.normal(size=anchor.shape)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        candidates = anchor + direction * rng.uniform(0.015, 0.60, (len(anchor), 1))
        lower = certificate.lower_bound(candidates)
        distance, nearest = reference_tree.query(candidates)
        floor = 0.14 if robot == "ned" else 0.02
        valid = (lower > 0.005) & (candidates[:, 2] >= floor)
        for band in (0, 1):
            indices = np.flatnonzero(valid & ((distance < 0.10) if band == 0 else (distance >= 0.10)))
            remaining = quotas[band] - len(selected[band])
            for index in indices[:max(0, remaining)]:
                selected[band].append((candidates[index], int(nearest[index]),
                                       float(lower[index]), "near" if band == 0 else "far"))
    result = selected[0] + selected[1]
    if len(result) != count:
        raise RuntimeError(f"Could not obtain {count} certified requests for {robot}")
    return [result[index] for index in rng.permutation(len(result))]


def build_dataset(experiment, dataset_dir, audit_dir, root=ROOT):
    """Write <dataset_dir>/<robot>/*.csv, <audit_dir>/<robot>.csv and manifest.json.

    `root` only decides how paths are written inside the manifest and README, so
    the repository layout is recorded the same wherever the files are built.
    """
    dataset_dir, audit_dir = Path(dataset_dir), Path(audit_dir)
    size = experiment.size
    manifest = {"seed": experiment.seed, "size": size,
                "policy": "closest accepted pose within 2 degrees of best recorded request direction",
                "direction": {"ned": "3D shoulder-to-request direction", "scara": "planar yaw-axis-to-request bearing"},
                "direction_tolerance_deg": DIRECTION_TOLERANCE_DEG,
                "certificate": "65^3 full joint grid with global Jacobian cover bound; analytic sphere/cylinder bounds",
                "minimum_unreachable_clearance_m": MIN_UNREACHABLE_CLEARANCE_M,
                "query_offsets_m": list(QUERY_OFFSETS_M), "near_far_split_m": NEAR_FAR_SPLIT_M,
                "label_bank": f"original train_{size}.csv only; no validation/test poses",
                "shared_test_requests": posix_relative(experiment.test_requests, ROOT) if experiment.test_requests else None,
                "robots": {}}
    for robot_index, robot in enumerate(ROBOTS):
        reference = read_csv(experiment.reachable_splits / robot / f"train_{size}.csv")
        reference_xyz = array(reference, "xyz")
        certificate = WorkspaceCertificate(robot)
        # Ensure the conservative certificate does not reject any known valid pose.
        if np.max(certificate.lower_bound(reference_xyz)) > 1e-8:
            raise AssertionError("Workspace certificate rejected a recorded reachable pose")
        original_audit = read_csv(experiment.reachable_audits / f"{robot}_master_audit.csv")
        accepted = {tuple(float(row[f"e{axis}"]) for axis in "xyz"): (line, row)
                    for line, row in enumerate(original_audit, start=2) if row["accepted"] == "1"}
        audit_rows = []
        counts = {}
        seen = set()
        for split_index, (split, extra_count) in enumerate((
                (f"train_{size}", experiment.train_extra),
                (f"validation_{size}", (experiment.train_extra + 2) // 4),
                ("test", experiment.test_extra))):
            original_rows = read_csv(experiment.reachable_splits / robot / f"{split}.csv")
            rows = [{**row, "kind": "reachable", **{f"desired_{a}": row[a] for a in "xyz"}}
                    for row in original_rows]
            seed = np.random.SeedSequence([experiment.seed, robot_index, split_index])
            if split == "test" and experiment.test_requests:
                # Reuse only requested XYZ. Desired poses and joint labels are
                # recomputed from this size's training bank, never copied.
                shared = [r for r in read_csv(experiment.test_requests / robot / "test.csv") if r["kind"] == "unreachable"]
                if len(shared) != extra_count:
                    raise ValueError(f"Expected {extra_count} shared test requests for {robot}")
                points = array(shared, "xyz")
                lower_bounds = certificate.lower_bound(points)
                distances, indices = cKDTree(reference_xyz).query(points)
                if np.any(lower_bounds <= 0.005):
                    raise AssertionError("Shared test query is not certified unreachable")
                queries = [(point, int(index), float(lower), "near" if distance < 0.10 else "far")
                           for point, index, lower, distance in zip(points, indices, lower_bounds, distances)]
            else:
                queries = sample_requests(robot, extra_count, seed, certificate, reference_xyz)
            points = np.array([query[0] for query in queries])
            chosen, best_angles = choose_directional_poses(robot, points, reference_xyz)
            selected_angles = direction_errors(robot, points, reference_xyz[chosen])
            if np.any(selected_angles > best_angles + 2.0 + 1e-8):
                raise AssertionError("Chosen pose violates the direction priority")
            for query_number, (point, nearest_index, lower, band) in enumerate(queries):
                index = int(chosen[query_number])
                if tuple(point) in seen:
                    raise AssertionError("Duplicate unreachable request across splits")
                seen.add(tuple(point))
                ref = reference[index]
                source_line, audit = accepted[tuple(reference_xyz[index])]
                row = {**ref, **dict(zip("xyz", point)), "kind": "unreachable",
                       **{f"desired_{a}": ref[a] for a in "xyz"}}
                rows.append(row)
                audit_rows.append({"robot": robot, "split": split, "nn_csv_row": len(rows) + 1,
                    "band": band, **dict(zip((f"requested_{a}" for a in "xyz"), point)),
                    **{f"desired_{a}": ref[a] for a in "xyz"},
                    **{f"label_{q}": ref[q] for q in OUTPUTS},
                    **{f"physical_q{i}": audit[f"joint{i}"] for i in range(1, 4)},
                    "certified_distance_lower_bound_m": lower,
                    "nearest_accepted_distance_m": float(np.linalg.norm(point - reference_xyz[nearest_index])),
                    "distance_to_accepted_pose_m": float(np.linalg.norm(point - reference_xyz[index])),
                    "best_recorded_direction_error_deg": best_angles[query_number],
                    "chosen_direction_error_deg": selected_angles[query_number],
                    "reference_training_csv_row": index + 2, "source_audit_csv_row": source_line,
                    "source_attempt": audit["attempt"],
                    **{f"recorded_webots_{a}": audit[f"a{a}"] for a in "xyz"},
                    "recorded_fk_webots_error_m": audit["error"]})
            write_csv(dataset_dir / robot / f"{split}.csv", rows)
            counts[split] = {"reachable": len(original_rows), "unreachable": extra_count}
        write_csv(audit_dir / f"{robot}.csv", audit_rows)
        manifest["robots"][robot] = {"counts": counts, "workspace_cover_radius_m": certificate.cover_radius}
        print(f"Created {robot} fallback dataset: {counts}", flush=True)
    paths = sorted(dataset_dir.rglob("*.csv")) + sorted(audit_dir.rglob("*.csv"))
    manifest["generated_file_sha256"] = fingerprint(paths, root)
    manifest["reachable_splits"] = posix_relative(experiment.reachable_splits, ROOT)
    manifest["reachable_audits"] = posix_relative(experiment.reachable_audits, ROOT)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dataset_dir / "README.md").write_text(
        "# Directional fallback dataset\n\n"
        "The network inputs are `is_scara,is_ned,x,y,z`; XYZ is the requested point in metres. "
        "Outputs `q1,q2,q3` are normalised joint labels in [-1,1]. The trainer standardises "
        "XYZ using this training split only.\n\n"
        "`kind` identifies reachable or certified unreachable requests. "
        "`desired_x,desired_y,desired_z` give the desired achieved TCP in metres. "
        "These four columns are for auditing and metrics, not additional network inputs. "
        "For reachable rows the desired and requested points agree. For unreachable rows "
        "the desired point is an accepted training pose chosen to point toward the request.\n\n"
        "Direction is 3D from Ned's shoulder, and horizontal bearing from SCARA's yaw axis. "
        "Among accepted poses within 2 degrees of the best recorded direction, select the "
        "closest TCP to the request. All labels use the original training bank.\n\n"
        f"The per-request policy audit is in `{posix_relative(audit_dir, root)}`; "
        "it links labels to the original training CSV row and Webots audit row. "
        "These are synthetic requests with recorded reference poses, not Webots readings "
        "at impossible requested points. `manifest.json` records counts, seeds and hashes.\n",
        encoding="utf-8")
    return manifest
