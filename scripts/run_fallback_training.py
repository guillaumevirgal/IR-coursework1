"""Build and train an independent directional fallback experiment.

python scripts/run_fallback_training.py --name directional_fallback_repeat --webots /path/to/webots

The default run writes new data, a new joint-loss checkpoint and a report.
Existing experiment paths are refused. Original dataset/model files are hashed
before and after the run. No existing checkpoint is fine-tuned or overwritten.
"""

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mlp import INPUTS, OUTPUTS, JOINT_LIMITS, denormalise
from scripts.numpy_neighbors import NumpyNearestNeighbors as cKDTree
from scripts.tcp_loss import tcp_and_jacobian
from scripts import forward_kinematics_ned as ned
from scripts import forward_kinematics_scara as scara
from scripts.evaluation_common import write_csv
from scripts.generate_mlp_diagnostics import load_model
from scripts.webots_mlp_runner import run_webots
from scripts.run_dataset_generation import find_webots

ROBOTS = ("ned", "scara")
BASELINE = "mlp_ned_scara_large_h64x64_cosine"
MODEL = BASELINE + "_directional_fallback"
ORIGINAL = ROOT / "data/nn/splits"
DATASET = ROOT / "data/nn/directional_fallback"
AUDIT = ROOT / "data/raw/directional_fallback"
REPORT = ROOT / "reports/directional_fallback"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def fingerprint(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def reproduction_options(args):
    options = (f"--size {args.size} --name {args.name}_repeat --seed {args.seed} "
               f"--train-extra {args.train_extra} --test-extra {args.test_extra} --epochs {args.epochs}")
    for flag, path in (("--reachable-splits", ORIGINAL), ("--reachable-audits", args.reachable_audits)):
        options += f' {flag} "{os.path.relpath(path, ROOT)}"'
    if args.test_requests:
        options += f' --test-requests "{args.test_requests}"'
    if args.dataset_only:
        options += " --dataset-only"
    return options


def array(rows, fields):
    return np.asarray([[float(r[key]) for key in fields] for r in rows])


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


def choose_directional_poses(robot, points, reference_xyz, tolerance_deg=2.0):
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


def build_dataset(args):
    manifest = {"seed": args.seed, "size": args.size, "policy": "closest accepted pose within 2 degrees of best recorded request direction",
                "direction": {"ned": "3D shoulder-to-request direction", "scara": "planar yaw-axis-to-request bearing"},
                "direction_tolerance_deg": 2.0,
                "certificate": "65^3 full joint grid with global Jacobian cover bound; analytic sphere/cylinder bounds",
                "minimum_unreachable_clearance_m": 0.005,
                "query_offsets_m": [0.015, 0.60], "near_far_split_m": 0.10,
                "label_bank": f"original train_{args.size}.csv only; no validation/test poses",
                "shared_test_requests": str(args.test_requests) if args.test_requests else None,
                "robots": {}}
    for robot_index, robot in enumerate(ROBOTS):
        reference = read_csv(ORIGINAL / robot / f"train_{args.size}.csv")
        reference_xyz = array(reference, "xyz")
        certificate = WorkspaceCertificate(robot)
        # Ensure the conservative certificate does not reject any known valid pose.
        if np.max(certificate.lower_bound(reference_xyz)) > 1e-8:
            raise AssertionError("Workspace certificate rejected a recorded reachable pose")
        original_audit = read_csv(args.reachable_audits / f"{robot}_master_audit.csv")
        accepted = {tuple(float(row[f"e{axis}"]) for axis in "xyz"): (line, row)
                    for line, row in enumerate(original_audit, start=2) if row["accepted"] == "1"}
        audit_rows = []
        counts = {}
        seen = set()
        for split_index, (split, extra_count) in enumerate((
                (f"train_{args.size}", args.train_extra), (f"validation_{args.size}", (args.train_extra + 2) // 4),
                ("test", args.test_extra))):
            original_rows = read_csv(ORIGINAL / robot / f"{split}.csv")
            rows = [{**row, "kind": "reachable", **{f"desired_{a}": row[a] for a in "xyz"}}
                    for row in original_rows]
            seed = np.random.SeedSequence([args.seed, robot_index, split_index])
            if split == "test" and args.test_requests:
                # Reuse only requested XYZ. Desired poses and joint labels are
                # recomputed from this size's training bank, never copied.
                shared = [r for r in read_csv(args.test_requests / robot / "test.csv") if r["kind"] == "unreachable"]
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
            write_csv(DATASET / robot / f"{split}.csv", rows)
            counts[split] = {"reachable": len(original_rows), "unreachable": extra_count}
        write_csv(AUDIT / f"{robot}.csv", audit_rows)
        manifest["robots"][robot] = {"counts": counts, "workspace_cover_radius_m": certificate.cover_radius}
        print(f"Created {robot} fallback dataset: {counts}", flush=True)
    paths = sorted(DATASET.rglob("*.csv")) + sorted(AUDIT.rglob("*.csv"))
    manifest["generated_file_sha256"] = fingerprint(paths)
    (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATASET / "README.md").write_text(
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
        f"The per-request policy audit is in `{AUDIT.relative_to(ROOT)}`; "
        "it links labels to the original training CSV row and Webots audit row. "
        "These are synthetic requests with recorded reference poses, not Webots readings "
        "at impossible requested points. `manifest.json` records counts, seeds and hashes.\n",
        encoding="utf-8")
    return manifest


def predict(name, robot, rows):
    model, mean, std = load_model(name, robot)
    return model.predict((array(rows, INPUTS) - mean) / std)


def evaluate(model_names=None):
    summaries, predictions = [], []
    for robot in ROBOTS:
        rows = read_csv(DATASET / robot / "test.csv")
        x = array(rows, INPUTS)
        labels = array(rows, OUTPUTS)
        desired = array(rows, ("desired_x", "desired_y", "desired_z"))
        for model_name in (model_names or (BASELINE, MODEL)):
            raw = predict(model_name, robot, rows)
            joints = denormalise(np.clip(raw, -1, 1), JOINT_LIMITS[robot])
            positions = tcp_and_jacobian(robot, joints)[0]
            policy_error = np.linalg.norm(positions - desired, axis=1) * 1000
            request_error = np.linalg.norm(positions - x[:, 2:5], axis=1) * 1000
            for kind in ("reachable", "unreachable"):
                mask = np.array([r["kind"] == kind for r in rows])
                summaries.append({"robot": robot, "model": model_name, "kind": kind,
                    "samples": int(mask.sum()),
                    "joint_mse_normalised": float(np.mean((raw[mask] - labels[mask]) ** 2)),
                    "desired_tcp_rmse_mm": float(np.sqrt(np.mean(policy_error[mask] ** 2))),
                    "desired_tcp_mean_mm": float(policy_error[mask].mean()),
                    "desired_tcp_std_mm": float(policy_error[mask].std()),
                    "desired_tcp_p95_mm": float(np.percentile(policy_error[mask], 95)),
                    "desired_tcp_max_mm": float(policy_error[mask].max()),
                    "requested_tcp_rmse_mm": float(np.sqrt(np.mean(request_error[mask] ** 2))),
                    "requested_tcp_mse_mm2": float(np.mean(request_error[mask] ** 2)),
                    "requested_tcp_mean_mm": float(request_error[mask].mean()),
                    "requested_tcp_std_mm": float(request_error[mask].std()),
                    "clipped_samples_pct": float(np.mean(np.any(np.abs(raw[mask]) > 1, axis=1)) * 100)})
            for index, row in enumerate(rows):
                predictions.append({"robot": robot, "model": model_name, "kind": row["kind"],
                    "test_csv_row": index + 2, **{a: row[a] for a in "xyz"},
                    **{f"desired_{a}": row[f"desired_{a}"] for a in "xyz"},
                    **{f"predicted_{a}": float(v) for a, v in zip("xyz", positions[index])},
                    **{f"label_q{j+1}_normalised": labels[index, j] for j in range(3)},
                    **{f"raw_q{j+1}_normalised": raw[index, j] for j in range(3)},
                    **{f"command_q{j+1}_physical": joints[index, j] for j in range(3)},
                    "desired_tcp_error_mm": policy_error[index], "requested_tcp_error_mm": request_error[index]})
    write_csv(REPORT / "test_summary.csv", summaries)
    write_csv(REPORT / "test_predictions.csv", predictions)
    failures = [max((r for r in predictions if r["robot"] == robot and r["kind"] == "unreachable"
                     and r["model"] == model), key=lambda r: r["desired_tcp_error_mm"])
                for robot in ROBOTS for model in (model_names or (MODEL,))]
    write_csv(REPORT / "failure_examples.csv", failures)
    return summaries


def evaluate_webots(args, model_names=None):
    measurements = []
    for robot in ROBOTS:
        rows = read_csv(DATASET / robot / "test.csv")
        selected = [r for r in rows if r["kind"] == "reachable"][:10]
        selected += [r for r in rows if r["kind"] == "unreachable"][:20]
        for model_name in (model_names or (BASELINE, MODEL)):
            cases = [{"label": f"case_{i}", "kind": row["kind"],
                      "point": tuple(float(row[a]) for a in "xyz")} for i, row in enumerate(selected)]
            measured = run_webots(robot, model_name, cases, args)
            if [r["label"] for r in measured] != [r["label"] for r in cases]:
                raise RuntimeError("Missing or reordered Webots measurements")
            for result in measured:
                row = selected[int(result["label"].split("_")[-1])]
                desired = np.array([float(row[f"desired_{a}"]) for a in "xyz"])
                actual = np.array([float(result[f"webots_{a}"]) for a in "xyz"])
                measurements.append({"model": model_name, **result,
                    **{f"desired_{a}": row[f"desired_{a}"] for a in "xyz"},
                    "desired_tcp_error_mm": float(np.linalg.norm(actual - desired) * 1000)})
    write_csv(REPORT / "webots_predictions.csv", measurements)
    return measurements


def draw_report(summaries, measurements, manifest, args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fallback-training-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.plot_mlp_evaluation import smooth_curve
    history = read_csv(ROOT / f"models/{MODEL}_history.csv")
    config = json.loads((ROOT / f"models/{MODEL}.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = [int(r["epoch"]) for r in history]
    ax.plot(epochs, smooth_curve([float(r["train_mse"]) for r in history]),
            color="#2878b5", label="Training MSE", linewidth=2)
    ax.plot(epochs, smooth_curve([float(r["val_mse"]) for r in history]),
            color="#e07a3f", label="Validation MSE", linewidth=2)
    test_rows = [r for r in summaries if r["model"] == MODEL]
    # Test has a 1:1 group mix while train/validation have a 2:1 mix. Match
    # validation weights in this loss plot; per-group result tables stay raw.
    weights = [manifest["robots"][r["robot"]]["counts"][f"validation_{args.size}"][r["kind"]]
               for r in test_rows]
    test_mse = sum(float(r["joint_mse_normalised"]) * w for r, w in zip(test_rows, weights)) / sum(weights)
    ax.scatter([config["best_epoch"]], [test_mse], marker="D", color="#4a9b67", edgecolor="#202020",
               label="Test MSE (same group mix as validation)", zorder=5)
    ax.set(xlabel="Epoch", ylabel="Joint MSE (normalised units²)", yscale="log",
           title="Directional fallback: training, validation and test loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.text(.5, .01, "Lines: 25-epoch averages. Test: weighted to validation group proportions; checkpoint chosen by validation only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(REPORT / "training_loss.png", dpi=160)
    plt.close(fig)
    def endpoint_graph(rows, filename, title, note):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for i, robot in enumerate(ROBOTS):
            for j, kind in enumerate(("reachable", "unreachable")):
                subset = [r for r in rows if r["robot"] == robot and r["kind"] == kind]
                values = [float(r["requested_tcp_rmse_mm"]) for r in subset]
                ax = axes[i, j]
                bars = ax.bar(["Original data", "With fallback examples"], values,
                              color=["#2878b5", "#e07a3f"], width=0.55)
                ax.bar_label(bars, fmt="%.2f mm", padding=3)
                ax.set_ylim(0, max(values) * 1.25)
                ax.set_ylabel("Requested-point TCP RMSE (mm)")
                ax.set_title(f"{robot.upper()} · {kind} requests", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(title)
        fig.text(.5, .01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(REPORT / filename, dpi=160)
        plt.close(fig)

    endpoint_graph(summaries, "test_comparison.png", "Held-out test: TCP endpoint RMSE to the requested point",
                   "All 1,000 test requests per group and arm. Unreachable target errors include unavoidable reach limits.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, robot in zip(axes, ROBOTS):
        subset = [r for r in summaries if r["robot"] == robot and r["kind"] == "unreachable"]
        values = [float(r["desired_tcp_rmse_mm"]) for r in subset]
        bars = ax.bar(["Original data", "With fallback examples"], values,
                      color=["#2878b5", "#e07a3f"], width=0.55)
        ax.bar_label(bars, fmt="%.2f mm", padding=3)
        ax.set_title(robot.upper())
        ax.set_ylabel("Desired-fallback TCP RMSE (mm)")
        ax.set_ylim(0, max(values) * 1.25)
    fig.suptitle("Unreachable test requests: how closely is the chosen fallback endpoint followed?")
    fig.text(.5, .01, "Reference: the accepted TCP chosen by the directional labelling rule. This measures learning of the fallback response.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(REPORT / "fallback_comparison.png", dpi=160)
    plt.close(fig)
    (REPORT / "direction_comparison.png").unlink(missing_ok=True)

    measured_summary = []
    if measurements:
        for robot in ROBOTS:
            for kind in ("reachable", "unreachable"):
                for model in (BASELINE, MODEL):
                    rows = [r for r in measurements if r["robot"] == robot and r["kind"] == kind and r["model"] == model]
                    requested = np.array([float(r["webots_error_mm"]) for r in rows])
                    desired = np.array([float(r["desired_tcp_error_mm"]) for r in rows])
                    measured_summary.append({"robot": robot, "kind": kind, "model": model, "samples": len(rows),
                        "settled": sum(str(r["settled"]).lower() == "true" for r in rows),
                        "requested_tcp_mse_mm2": float(np.mean(requested**2)),
                        "requested_tcp_rmse_mm": float(np.sqrt(np.mean(requested**2))),
                        "requested_tcp_mean_mm": float(requested.mean()),
                        "requested_tcp_std_mm": float(requested.std()),
                        "desired_tcp_rmse_mm": float(np.sqrt(np.mean(desired**2)))})
        write_csv(REPORT / "webots_summary.csv", measured_summary)
        endpoint_graph(measured_summary, "webots_comparison.png", "Measured in Webots: TCP endpoint RMSE to requested point",
                       "Same first 10 reachable + 20 unreachable test requests per arm/model; all measured cases are included.")

    label = lambda model: "Original" if model == BASELINE else "With fallback examples"
    lines = ["# Joint-loss directional fallback experiment", "",
        "The new dataset teaches the arm to point toward an unreachable request, then approach it. Ned uses the 3D direction from its shoulder; SCARA uses horizontal bearing because its links cannot tilt upward. Find the best recorded direction; among accepted poses within 2 degrees of that direction error, choose the TCP closest to the request. If SCARA's requested point is exactly on its yaw axis, bearing is undefined and distance alone chooses the label. This selects the best available recorded pose; a finite pose bank does not prove a globally optimal result.", "",
        "New requests are certified outside the full joint-limited FK workspace using a conservative grid-cover bound or an analytic enclosing volume, with a minimum 5 mm clearance. Half the generated requests are within 100 mm of the accepted pose bank and half farther away. Query displacements span 15–600 mm from bank positions. Every label uses an original training pose with a Webots audit row. The requests are synthetic; no simulated TCP reading is invented at an impossible requested location.", "",
        (f"The {args.size} experiment reuses only unreachable test requested XYZ from `{args.test_requests}` for comparison on identical inputs. Its fallback joint labels are recomputed using the {args.size} training bank only. The shared test group need not retain the medium bank's near/far proportions." if args.test_requests else "Unreachable test requests are sampled independently of training and validation queries."), "",
        "Original reachable rows retain their labels. New query splits have independent seeds, and validation/test poses never supply training labels. This tests new requests against a fixed training reference bank. Accepted static poses do not guarantee collision-free motion, and the MLP can predict an unrecorded configuration. Its response beyond the sampled query distribution is unverified.", "",
        f"Training uses the {args.size} split and reuses the 5→64→64→3 NumPy ReLU MLP, Adam and joint MSE, from scratch with seed {args.seed} and up to {args.epochs} epochs. Validation joint MSE selected epoch {config['best_epoch']}; training ended at epoch {history[-1]['epoch']}. Best validation joint MSE: {config['best_val_loss']:.8f}. Test data were evaluated afterward. This comparison changes data quantity and distribution together.", "",
        "Split counts per arm: " + json.dumps(manifest["robots"]["ned"]["counts"]), "",
        "## Requested-point endpoint error", "",
        "For each sample, e = Euclidean distance from final TCP to requested XYZ, in mm. MSE = mean(e²), in mm²; RMSE = sqrt(mean(e²)), in mm. Mean/std are statistics of e, using population std. Unreachable requests include an unavoidable geometric gap, so their RMSE cannot be interpreted as ordinary reachable IK accuracy.", "",
        "![Held-out requested-point TCP RMSE](test_comparison.png)", "",
        "| Arm | Group | Training data | MSE (mm²) | RMSE (mm) | Mean (mm) | Std (mm) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['robot']} | {row['kind']} | {label(row['model'])} | {row['requested_tcp_mse_mm2']:.2f} | {row['requested_tcp_rmse_mm']:.2f} | {row['requested_tcp_mean_mm']:.2f} | {row['requested_tcp_std_mm']:.2f} |")
    lines += ["", "## Error to the chosen fallback endpoint", "",
        "For unreachable inputs, this measures final TCP distance to the accepted fallback TCP used to create the joint label. It measures how accurately the new response was learned. The requested-point error above remains the main endpoint comparison.", "",
        "![Fallback endpoint RMSE](fallback_comparison.png)", "",
        "| Arm | Training data | Fallback RMSE (mm) | Mean (mm) | Std (mm) | P95 (mm) | Maximum (mm) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["kind"] == "unreachable":
            lines.append(f"| {row['robot']} | {label(row['model'])} | {row['desired_tcp_rmse_mm']:.2f} | {row['desired_tcp_mean_mm']:.2f} | {row['desired_tcp_std_mm']:.2f} | {row['desired_tcp_p95_mm']:.2f} | {row['desired_tcp_max_mm']:.2f} |")
    lines += ["", "Individual failures remain. A joint-MSE regressor can average distinct joint solutions where the selected label switches branch, particularly near Ned's rear yaw limits. `failure_examples.csv` records the worst fallback endpoint error for each arm. `test_predictions.csv` retains all requested/desired/predicted points, normalised labels, raw outputs and physical joint commands. Direction determines the labelling rule; evaluation uses TCP endpoint errors.", ""]
    if measurements:
        lines += ["## Measured in Webots", "",
            "The first 10 reachable and 20 unreachable test requests per arm were executed for each saved model. Settling required joint sensors to remain within tolerance for 0.25 simulated seconds, with a 10-second timeout. Every result is included, including any timeouts.", "",
            "![Measured requested-point TCP RMSE](webots_comparison.png)", "",
            "| Arm | Group | Training data | Samples | Settled | Requested TCP RMSE (mm) | Fallback TCP RMSE (mm) |",
            "|---|---|---|---:|---:|---:|---:|"]
        for row in measured_summary:
            lines.append(f"| {row['robot']} | {row['kind']} | {label(row['model'])} | {row['samples']} | {row['settled']} | {row['requested_tcp_rmse_mm']:.2f} | {row['desired_tcp_rmse_mm']:.2f} |")
    lines += ["", "## Training loss", "", "![Training, validation and test joint MSE](training_loss.png)", "",
        "The test marker is reweighted to validation group proportions (2 reachable to 1 unreachable in this run); the test files have an equal group mix. Per-group endpoint metrics above are unweighted. Curves are 25-epoch moving averages.", "",
        "## Run and use", "", f"Run `python scripts/run_fallback_training.py {reproduction_options(args)}` from the repository root; add `--webots PATH` for physics measurements. Existing paths/checkpoints are refused. Data: `{DATASET.relative_to(ROOT)}`; policy audit: `{AUDIT.relative_to(ROOT)}`. `manifest.json` records dataset hashes and `source_integrity.json` confirms original data and checkpoint hashes.", "",
        f'In Webots keep controller `mlp_ik` and append `"--model={MODEL}"` after `"ned"` or `"scara"` in `controllerArgs`. The network supplies the learned fallback response.', ""]
    (REPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def compare_sizes(reference_dataset):
    """Compare requested-point RMSE only, after checking identical test inputs.

    Fallback labels use different pose banks at different training sizes, so
    fallback-label imitation errors are not used to rank sizes here.
    """
    reference_report = ROOT / "reports" / reference_dataset.name
    if not (reference_report / "test_summary.csv").is_file():
        return
    for robot in ROBOTS:
        current = read_csv(DATASET / robot / "test.csv")
        reference = read_csv(reference_dataset / robot / "test.csv")
        if [r["kind"] for r in current] != [r["kind"] for r in reference] or not np.array_equal(
                array(current, INPUTS), array(reference, INPUTS)):
            raise ValueError("Size comparison requires identical test requested inputs")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fallback-training-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def collect(filename):
        collected = []
        for folder in (REPORT, reference_report):
            for row in read_csv(folder / filename):
                config = json.loads((ROOT / f"models/{row['model']}.json").read_text(encoding="utf-8"))
                size = config["size"]
                original = row["model"] == f"mlp_ned_scara_{size}_h64x64_cosine"
                collected.append({"size": size, "model": row["model"], "variant": "original" if original else "fallback",
                    "robot": row["robot"], "kind": row["kind"], "samples": int(row["samples"]),
                    "requested_tcp_rmse_mm": float(row["requested_tcp_rmse_mm"])})
        return collected

    def graph(rows, filename, title):
        sizes = [s for s in ("small", "medium", "large") if any(r["size"] == s for r in rows)]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        x = np.arange(len(sizes))
        for i, robot in enumerate(ROBOTS):
            for j, kind in enumerate(("reachable", "unreachable")):
                ax = axes[i, j]
                for k, (variant, color, label) in enumerate((
                        ("original", "#2878b5", "Original joint-loss model"),
                        ("fallback", "#e07a3f", "Joint-loss model with fallback data"))):
                    values = [next(r["requested_tcp_rmse_mm"] for r in rows if r["robot"] == robot
                                   and r["kind"] == kind and r["size"] == size and r["variant"] == variant) for size in sizes]
                    bars = ax.bar(x + (k - 0.5) * 0.36, values, width=0.34, color=color, label=label)
                    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
                ax.set_xticks(x, [s.capitalize() for s in sizes])
                ax.set_title(f"{robot.upper()} · {kind} requests")
                ax.set_ylabel("Requested-point TCP RMSE (mm)")
                ax.margins(y=0.2)
                ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .01), fontsize=9)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, .07, 1, .94))
        fig.savefig(REPORT / filename, dpi=160)
        plt.close(fig)

    rows = collect("test_summary.csv")
    sizes = [s for s in ("small", "medium", "large") if any(r["size"] == s for r in rows)]
    if len(sizes) != 2:
        return
    size_title = " versus ".join(sizes)
    write_csv(REPORT / "size_comparison.csv", rows)
    graph(rows, "size_comparison.png", f"Same held-out points: {size_title} joint-loss models")
    lines = ["# Dataset-size comparison on identical requested points", "",
             "Every model is evaluated on exactly the same 1,000 reachable and 1,000 unreachable requested points per arm. Each size retains its original reachable split and adds half as many synthetic training examples (medium: 4,000 + 2,000; large: 12,000 + 6,000). Both use the same architecture, joint loss and training settings. Fallback labels use only each size's own training pose bank, so bank density and training quantity both change.", "",
             "![Endpoint RMSE comparison](size_comparison.png)", "",
             "The table compares the two models trained with fallback examples; the graph also includes the original checkpoints.", "",
             f"| Arm | Requests | {sizes[0].capitalize()} with fallback RMSE (mm) | {sizes[1].capitalize()} with fallback RMSE (mm) | Lower RMSE |",
             "|---|---|---:|---:|---|"]
    for robot in ROBOTS:
        for kind in ("reachable", "unreachable"):
            subset = [r for r in rows if r["variant"] == "fallback" and r["robot"] == robot and r["kind"] == kind]
            values = {r["size"]: r["requested_tcp_rmse_mm"] for r in subset}
            lower = min(values, key=values.get).capitalize()
            lines.append(f"| {robot} | {kind} | {values[sizes[0]]:.2f} | {values[sizes[1]]:.2f} | {lower} |")
    if all((folder / "webots_summary.csv").is_file() for folder in (REPORT, reference_report)):
        # Verify measured requests too; historical simulator subsets can differ.
        for robot in ROBOTS:
            requests = []
            for folder in (REPORT, reference_report):
                measured = read_csv(folder / "webots_predictions.csv")
                first_model = next(r["model"] for r in measured if r["robot"] == robot)
                requests.append(array([r for r in measured if r["robot"] == robot and r["model"] == first_model], "xyz"))
            if not np.array_equal(*requests):
                raise ValueError("Webots size comparison requires identical measured requests")
        measured_rows = collect("webots_summary.csv")
        write_csv(REPORT / "webots_size_comparison.csv", measured_rows)
        graph(measured_rows, "webots_size_comparison.png", f"Same Webots targets: {size_title} joint-loss models")
        lines += ["", "![Measured endpoint RMSE comparison](webots_size_comparison.png)", "",
                  "Webots comparison uses identical first 10 reachable and 20 unreachable test requests per arm/model. Full measurements and settling status are retained in each experiment's report folder."]
    lines += ["", "The endpoint reference here is requested XYZ, including unreachable points. Fallback endpoints differ between the finite medium and large pose banks, so error to those different fallback labels is not used to rank sizes."]
    (REPORT / "size_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    link = f"\n[{size_title.capitalize()} comparison on the same test inputs](size_comparison.md).\n"
    if link not in (REPORT / "README.md").read_text(encoding="utf-8"):
        with (REPORT / "README.md").open("a", encoding="utf-8") as destination:
            destination.write(link)


def compare_models(args, names):
    """Evaluate explicit checkpoints on this dataset, without training or selection."""
    from scripts.tcp_loss import tcp_loss_and_gradient
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fallback-training-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = {name: json.loads((ROOT / f"models/{name}.json").read_text()) for name in names}
    labels = {}
    for name, config in configs.items():
        directory = config.get("splits_dir", "data/nn/splits")
        size = config.get("size", "large").capitalize()
        if "directional_fallback_total_" in directory:
            training_rows = len(read_csv(ROOT / directory / "ned" / f"train_{config.get('size', 'large')}.csv"))
            group = f"{training_rows // 1000}k · 3:1"
        elif "fallback" in directory:
            training_rows = len(read_csv(ROOT / directory / "ned" / f"train_{config.get('size', 'large')}.csv"))
            group = f"{training_rows // 1000}k · 2:1"
        else:
            group = size
        loss_label = "TCP loss" if config.get("loss") == "tcp" else "Joint loss"
        labels[name] = f"{group}\n{loss_label}"
    summaries = evaluate(names)
    args.output, args.settle_time, args.timeout = REPORT, .25, 10.0
    measurements = evaluate_webots(args, names) if args.webots else []
    measured_summary = []
    for robot in ROBOTS:
        for name in names:
            for kind in ("reachable", "unreachable"):
                rows = [r for r in measurements if r["robot"] == robot and r["model"] == name and r["kind"] == kind]
                if not rows:
                    continue
                errors = np.array([float(r["webots_error_mm"]) for r in rows])
                measured_summary.append({"robot": robot, "model": name, "kind": kind, "samples": len(rows),
                    "settled": sum(str(r["settled"]) in ("True", "1", "true") for r in rows),
                    "requested_tcp_rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
                    "requested_tcp_mean_mm": float(errors.mean())})
    if measured_summary:
        write_csv(REPORT / "webots_summary.csv", measured_summary)

    def endpoint_plot(rows, filename, title):
        fig, axes = plt.subplots(2, 2, figsize=(17, 9))
        colors = ["#2878b5" if configs[name].get("loss") == "tcp" else "#e07a3f" for name in names]
        for i, robot in enumerate(ROBOTS):
            for j, kind in enumerate(("reachable", "unreachable")):
                ax = axes[i, j]
                values = [next(r["requested_tcp_rmse_mm"] for r in rows if r["robot"] == robot and r["kind"] == kind and r["model"] == name) for name in names]
                bars = ax.bar(range(len(names)), values, color=colors)
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
                ax.set_xticks(range(len(names)), [labels[n] for n in names], fontsize=8)
                ax.set_title(f"{robot.upper()}: {kind} requests")
                ax.set_ylabel("Requested-point TCP RMSE (mm)")
                ax.set_ylim(0, max(values) * 1.2)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
        fig.suptitle(title)
        fig.text(.5, .015, "Lower is better. Fallback labels show total training rows in thousands and reachable:unreachable ratio. Unreachable targets have a geometric gap. Orange: joint loss; blue: TCP loss.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(REPORT / filename, dpi=160)
        plt.close(fig)

    endpoint_plot(summaries, "test_comparison.png", "Same held-out requests: dataset size and loss comparison")
    if measured_summary:
        endpoint_plot(measured_summary, "webots_comparison.png", "Measured in Webots: same 10 reachable + 20 unreachable requests per arm/model")
    new_names = [n for n in names if "directional_fallback_total_" in configs[n].get("splits_dir", "")]
    fig, axes = plt.subplots(1, len(new_names), figsize=(7 * len(new_names), 4.5), squeeze=False)
    for ax, name in zip(axes[0], new_names):
        config = configs[name]
        history = read_csv(ROOT / f"models/{name}_history.csv")
        keys = ("train_loss_m2", "val_loss_m2") if config["loss"] == "tcp" else ("train_mse", "val_mse")
        epochs = [int(r["epoch"]) for r in history]
        ax.plot(epochs, [float(r[keys[0]]) for r in history], color="#2878b5", label="Training")
        ax.plot(epochs, [float(r[keys[1]]) for r in history], color="#e07a3f", label="Validation")
        heldout = 0.0
        for robot in ROBOTS:
            rows = read_csv(DATASET / robot / "test.csv")
            for kind, weight in (("reachable", .75), ("unreachable", .25)):
                subset = [r for r in rows if r["kind"] == kind]
                prediction = predict(name, robot, subset)
                if config["loss"] == "tcp":
                    objective = tcp_loss_and_gradient(prediction, array(subset, INPUTS), JOINT_LIMITS,
                        config["tcp_limit_weight"], array(subset, ("desired_x", "desired_y", "desired_z")))[0]
                else:
                    objective = float(np.mean((prediction - array(subset, OUTPUTS)) ** 2))
                heldout += .5 * weight * objective
        ax.scatter([config["best_epoch"]], [heldout], s=70, color="#33995c", marker="D", label="Test (one saved checkpoint)", zorder=5)
        ax.axvline(config["best_epoch"], color="#777777", alpha=.4)
        ax.set_title(labels[name].replace("\n", " · "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("TCP objective (m², includes limit penalty)" if config["loss"] == "tcp" else "Joint MSE (normalised units²)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Training, validation and test: different losses have different units")
    fig.text(.5, .01, "Raw epoch curves. Test loss is weighted 75% reachable / 25% unreachable to match validation. Test did not select the checkpoint.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(REPORT / "training_loss.png", dpi=160)
    plt.close(fig)
    lines = ["# Fallback dataset-size and loss comparison on shared test requests", "",
        "All models below are evaluated on the exact same held-out points: 1,000 reachable and 1,000 unreachable requests per robot. The original small, medium and large checkpoints were trained on reachable data only. The original large checkpoint used 12,000 training rows from the 16,000-row generated dataset. The new 16k-total models use 12,000 reachable + 4,000 unreachable training rows; the 20k-total models use 15,000 + 5,000. Both use a 3:1 ratio. The earlier 2:1 model uses 12,000 + 6,000 rows. Requested-point TCP RMSE is compared, independent of each experiment's fallback labels.", "",
        "The 16k and 20k training sets use different reachable pose banks as well as different row counts, so their comparison does not isolate dataset size alone. Small and medium models provide reachable-only baselines. Loss curves for both new total-dataset sizes are included below. Existing Webots measurements in this folder cover the five earlier large models; the newly trained 16k models have not been measured in Webots.", "",
        "## 3:1 training details", "",
        "For the 20k experiment: training per robot is 15,000 reachable + 5,000 certified unreachable requests (3:1); validation is 3,750 + 1,250. Test: 1,000 + 1,000. The 16k experiment uses 12,000 + 4,000 training and 3,000 + 1,000 validation rows. Both use the same 5→64→64→3 ReLU architecture, seed 2026, Adam, batch 32, cosine learning-rate schedule, 3,000-epoch maximum and patience 300. Best checkpoints are chosen by validation only.", "",
        "Joint loss learns the normalised joint label. TCP loss learns the corresponding desired endpoint through the robot-specific differentiable FK, plus the existing joint-limit penalty. For unreachable requests the desired endpoint is the accepted fallback pose, while the network input remains the requested point. The original TCP datasets have no fallback targets and retain their original requested-point objective.", "",
        "The held-out inputs are identical across models. Earlier datasets differ in both row count and reachable-pose-bank size, so the 3:1 versus 2:1 comparison does not isolate ratio alone.", "",
        "![Endpoint RMSE on all test requests](test_comparison.png)", "",
        "| Robot | Requests | Training / loss | Requested TCP RMSE (mm) | Mean error (mm) | Chosen target TCP RMSE (mm) |",
        "|---|---|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['robot']} | {r['kind']} | {labels[r['model']].replace(chr(10), ' / ')} | {r['requested_tcp_rmse_mm']:.2f} | {r['requested_tcp_mean_mm']:.2f} | {r['desired_tcp_rmse_mm']:.2f} |")
    lines += ["", "Requested-point RMSE is the main endpoint metric. Unreachable requests include an unavoidable geometric gap. The chosen-target RMSE compares predictions with the requested point for the original reachable-only models, and with the learned fallback endpoint for the new models. FK test results are calculated from clipped model commands; Webots results below are measured after settling. Bank size, data quantity and ratio change together compared with previous models, so this is not an isolated test of the ratio.", "",
        "![Training, validation and test losses](training_loss.png)", "",
        "| New model | Best epoch | Training stopped at epoch | Best validation objective |",
        "|---|---:|---:|---:|"]
    for name in new_names:
        history = read_csv(ROOT / f"models/{name}_history.csv")
        lines.append(f"| {labels[name].replace(chr(10), ' / ')} | {configs[name]['best_epoch']} | {history[-1]['epoch']} | {configs[name]['best_val_loss']:.8g} |")
    if measured_summary:
        lines += ["", "![Measured endpoint error](webots_comparison.png)", "",
            "Webots uses the same first 10 reachable and 20 unreachable requests per robot/model. It requires sensors within tolerance for 0.25 simulated seconds, with a 10-second timeout. `webots_summary.csv` retains settling counts; `webots_predictions.csv` retains every measurement."]
    lines += ["", "In Webots select controller `mlp_ik` and use the robot argument followed by one of:", "", "```text"]
    lines += [f"--model={name}" for name in new_names]
    lines += ["```", "", "Reproduce this saved-checkpoint comparison:", "", "```bash",
        f"python scripts/run_fallback_training.py --name {args.name} --evaluate-models " + " ".join(names), "```",
        "Add `--webots PATH` to include the simulator measurements. Existing models are never retrained by this command."]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT / "model_provenance.json").write_text(json.dumps(fingerprint([ROOT / f"models/{n}.{ext}" for n in names for ext in ("json", "npz")]), indent=2), encoding="utf-8")


def main():
    global BASELINE, MODEL, ORIGINAL, DATASET, AUDIT, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="large")
    parser.add_argument("--name", help="unique experiment name; defaults to directional_fallback (plus size for medium/small)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-extra", type=int, help="unreachable training rows per arm; validation adds a quarter of this, rounded to the nearest integer")
    parser.add_argument("--test-requests", type=Path, help="reuse only unreachable test inputs from another experiment, with fresh labels")
    parser.add_argument("--test-extra", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--webots", type=Path)
    parser.add_argument("--reachable-splits", type=Path, default=ORIGINAL,
                        help="parent of robot folders containing reachable train/validation/test CSVs")
    parser.add_argument("--reachable-audits", type=Path, default=ROOT / "data/raw",
                        help="folder containing the matching robot_master_audit.csv files")
    parser.add_argument("--dataset-only", action="store_true", help="create and audit data without training a model")
    parser.add_argument("--evaluate-models", nargs="+", help="compare named saved checkpoints on an existing experiment dataset")
    args = parser.parse_args()
    ORIGINAL = args.reachable_splits.resolve()
    args.reachable_audits = args.reachable_audits.resolve()
    args.name = args.name or ("directional_fallback" if args.size == "large" else f"directional_fallback_{args.size}")
    if args.train_extra is None:
        counts = [len(read_csv(ORIGINAL / robot / f"train_{args.size}.csv")) for robot in ROBOTS]
        if len(set(counts)) != 1:
            parser.error("reachable training counts differ between robots; specify --train-extra")
        args.train_extra = counts[0] // 2
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("name must contain only letters, numbers, underscores or hyphens")
    BASELINE = f"mlp_ned_scara_{args.size}_h64x64_cosine"
    MODEL = BASELINE + "_" + args.name
    DATASET, AUDIT, REPORT = (ROOT / f"data/nn/{args.name}", ROOT / f"data/raw/{args.name}", ROOT / f"reports/{args.name}")
    if args.train_extra < 4 or args.test_extra < 2 or args.epochs < 1 or args.seed < 0:
        parser.error("train-extra must be >= 4; test-extra >= 2; epochs >= 1; seed >= 0")
    if args.webots:
        args.webots = find_webots(args.webots)
    if args.evaluate_models:
        if args.dataset_only:
            parser.error("evaluate-models cannot be combined with dataset-only")
        if not (DATASET / "manifest.json").is_file():
            parser.error("evaluate-models needs an existing dataset for --name")
        REPORT.mkdir(exist_ok=True, parents=True)
        compare_models(args, args.evaluate_models)
        return
    for path in (DATASET, AUDIT, REPORT, *(ROOT / f"models/{MODEL}{suffix}" for suffix in (".npz", ".json", "_history.csv"))):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite experiment output: {path}")
    sources = sorted(p for directory in (ROOT / "data", ROOT / "models") for p in directory.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    before = fingerprint(sources)
    REPORT.mkdir(parents=True)
    try:
        manifest = build_dataset(args)
        manifest["reachable_splits"] = os.path.relpath(ORIGINAL, ROOT)
        manifest["reachable_audits"] = os.path.relpath(args.reachable_audits, ROOT)
        (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.dataset_only:
            (REPORT / "README.md").write_text(
                "# Dataset generation only\n\nNo model was trained. See "
                f"`{DATASET.relative_to(ROOT)}/manifest.json` for counts and provenance.\n\n"
                "Split counts per robot:\n\n```json\n" + json.dumps(manifest["robots"]["ned"]["counts"], indent=2) + "\n```\n\n"
                f"Reproduce with a fresh experiment name:\n\n```bash\npython scripts/run_fallback_training.py {reproduction_options(args)}\n```\n",
                encoding="utf-8")
            print(f"Dataset created: {DATASET}", flush=True)
            return
        command = [sys.executable, str(ROOT / "scripts/train_mlp.py"), "--size", args.size, "--hidden", "64", "64",
                   "--loss", "joints", "--epochs", str(args.epochs), "--seed", str(args.seed),
                   "--splits-dir", str(DATASET), "--model-name", MODEL]
        print(f"Training {MODEL}; progress is saved in {REPORT / 'training_output.txt'}", flush=True)
        with (REPORT / "training_output.txt").open("w", encoding="utf-8") as training_log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=training_log, stderr=subprocess.STDOUT)
        summaries = evaluate()
        args.output, args.settle_time, args.timeout = REPORT, 0.25, 10.0
        measurements = evaluate_webots(args) if args.webots else []
        draw_report(summaries, measurements, manifest, args)
        if args.test_requests:
            compare_sizes(args.test_requests)
        model_paths = [ROOT / f"models/{name}.{ext}" for name in (BASELINE, MODEL) for ext in ("json", "npz")]
        (REPORT / "model_provenance.json").write_text(json.dumps(fingerprint(model_paths), indent=2), encoding="utf-8")
        print(f"Experiment report: {REPORT / 'README.md'}", flush=True)
    finally:
        after = fingerprint(sources)
        unchanged = before == after
        (REPORT / "source_integrity.json").write_text(json.dumps({
            "all_existing_data_and_models_unchanged": unchanged, "source_sha256": before}, indent=2), encoding="utf-8")
        if not unchanged:
            raise AssertionError("An existing dataset or checkpoint changed")


if __name__ == "__main__":
    main()
