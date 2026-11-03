"""Build, train and report an independent unreachable-target ("directional fallback") experiment.

python extras/unreachable_targets/run_fallback_training.py --name directional_fallback_repeat --webots /path/to/webots

Extra experiment, separate from the coursework model. To only build a dataset
from a config, use extras/unreachable_targets/build_dataset.py instead.

The default run writes new data, a new joint-loss checkpoint and a report.
Existing experiment paths are refused. Original dataset/model files are hashed
before and after the run. No existing checkpoint is fine-tuned or overwritten.
Datasets live under extras/unreachable_targets/data; reports stay in reports/.
"""

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import numpy as np

from fallback_dataset import (
    EXTRAS, ROBOTS, ROOT, Experiment, array, build_dataset as build_experiment_dataset,
    fingerprint, read_csv, resolve_dataset_dir,
)
from scripts.robot_config import JOINT_LIMITS, denormalise
from scripts.train_mlp import INPUTS, OUTPUTS
from scripts.tcp_loss import tcp_and_jacobian
from scripts.evaluation_common import write_csv
from scripts.generate_mlp_diagnostics import load_model
from scripts.webots_mlp_runner import run_webots
from scripts.generate_dataset import find_webots

BASELINE = "mlp_ned_scara_large_h64x64_cosine"
MODEL = BASELINE + "_directional_fallback"
ORIGINAL = ROOT / "data/nn/splits"
DATASET = EXTRAS / "data/nn/directional_fallback"
AUDIT = EXTRAS / "data/raw/directional_fallback"
REPORT = ROOT / "reports/directional_fallback"


def build_dataset(args):
    """Build this run's dataset with the shared, config-driven builder."""
    experiment = Experiment(
        name=args.name, size=args.size, seed=args.seed, train_extra=args.train_extra,
        test_extra=args.test_extra, reachable_splits=ORIGINAL,
        reachable_audits=args.reachable_audits,
        test_requests=args.test_requests.resolve() if args.test_requests else None)
    return build_experiment_dataset(experiment, DATASET, AUDIT, EXTRAS)


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
        "## Run and use", "", f"Run `python extras/unreachable_targets/run_fallback_training.py {reproduction_options(args)}` from the repository root; add `--webots PATH` for physics measurements. Existing paths/checkpoints are refused. Data: `{DATASET.relative_to(ROOT)}`; policy audit: `{AUDIT.relative_to(ROOT)}`. `manifest.json` records dataset hashes and `source_integrity.json` confirms original data and checkpoint hashes.", "",
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
            training_rows = len(read_csv(resolve_dataset_dir(directory) / "ned" / f"train_{config.get('size', 'large')}.csv"))
            group = f"{training_rows // 1000}k · 3:1"
        elif "fallback" in directory:
            training_rows = len(read_csv(resolve_dataset_dir(directory) / "ned" / f"train_{config.get('size', 'large')}.csv"))
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
        f"python extras/unreachable_targets/run_fallback_training.py --name {args.name} --evaluate-models " + " ".join(names), "```",
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
    DATASET, AUDIT, REPORT = (EXTRAS / f"data/nn/{args.name}", EXTRAS / f"data/raw/{args.name}", ROOT / f"reports/{args.name}")
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
    sources = sorted(p for directory in (ROOT / "data", EXTRAS / "data", ROOT / "models") for p in directory.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    before = fingerprint(sources)
    REPORT.mkdir(parents=True)
    try:
        manifest = build_dataset(args)
        if args.dataset_only:
            (REPORT / "README.md").write_text(
                "# Dataset generation only\n\nNo model was trained. See "
                f"`{DATASET.relative_to(ROOT)}/manifest.json` for counts and provenance.\n\n"
                "Split counts per robot:\n\n```json\n" + json.dumps(manifest["robots"]["ned"]["counts"], indent=2) + "\n```\n\n"
                f"Reproduce with a fresh experiment name:\n\n```bash\npython extras/unreachable_targets/run_fallback_training.py {reproduction_options(args)}\n```\n",
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
