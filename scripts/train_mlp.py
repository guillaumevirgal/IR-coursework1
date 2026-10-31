"""
feed-forward MLP that maps [is_scara, is_ned, x, y, z] to
the normalised joint values [q1, q2, q3] in [-1, 1].


Usage:
    python scripts/train_mlp.py                       # small combined model
    python scripts/train_mlp.py --robots ned          # small Ned only
    python scripts/train_mlp.py --size medium --hidden 64 64 --lr 1e-3
    python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.robot_config import FORWARD_KINEMATICS, JOINT_LIMITS, denormalise  # noqa: F401 (re-exported)
from scripts.tcp_loss import tcp_loss_and_gradient


SPLITS = ROOT / "data" / "nn" / "splits"
MODELS = ROOT / "models"
INPUTS = ("is_scara", "is_ned", "x", "y", "z")
OUTPUTS = ("q1", "q2", "q3")

def load_split(robots, kind, size, directory=SPLITS):
    """Stack the requested robots' CSVs into input and target arrays."""

    rows = []
    for robot in robots:
        path = Path(directory) / robot / f"{kind}_{size}.csv"
        with path.open(newline="", encoding="utf-8") as source:
            rows.extend(csv.DictReader(source))
    x = np.array([[float(row[name]) for name in INPUTS] for row in rows])
    y = np.array([[float(row[name]) for name in OUTPUTS] for row in rows])
    return x, y


class MLP:
    """Fully connected ReLU hidden layers, linear output, trained with Adam."""

    def __init__(self, sizes, rng):
        # sizes = (n_in, *hidden, n_out). He initialisation suits ReLU layers;
        # Xavier-style init for the linear output layer.
        self.params = {}
        self.n_layers = len(sizes) - 1
        for index, (n_from, n_to) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
            gain = 1.0 if index == self.n_layers else 2.0
            self.params[f"W{index}"] = rng.normal(0.0, math.sqrt(gain / n_from), (n_from, n_to))
            self.params[f"b{index}"] = np.zeros(n_to)
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step = 0

    def forward(self, x):
        """Return the output and each layer's (input, pre-activation) for backprop."""

        cache = []
        activation = x
        for index in range(1, self.n_layers + 1):
            pre = activation @ self.params[f"W{index}"] + self.params[f"b{index}"]
            cache.append((activation, pre))
            activation = pre if index == self.n_layers else np.maximum(pre, 0.0)
        return activation, cache

    def predict(self, x):
        return self.forward(x)[0]

    def train_batch(self, x, y, lr, beta1=0.9, beta2=0.999, eps=1e-8, loss_fn=None):
        """One Adam step; optional loss_fn returns (loss, output gradient)."""

        output, cache = self.forward(x)
        if loss_fn is None:
            error = output - y
            loss = float(np.mean(error ** 2))
            d_pre = 2.0 * error / error.size
        else:
            loss, d_pre = loss_fn(output)

        # Walk the selected loss gradient back through the same layers.
        grads = {}
        for index in range(self.n_layers, 0, -1):
            layer_input, _ = cache[index - 1]
            grads[f"W{index}"] = layer_input.T @ d_pre
            grads[f"b{index}"] = d_pre.sum(axis=0)
            if index > 1:
                d_pre = (d_pre @ self.params[f"W{index}"].T) * (cache[index - 2][1] > 0.0)

        self.step += 1
        for key, grad in grads.items():
            self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * grad
            self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * grad ** 2
            m_hat = self.m[key] / (1.0 - beta1 ** self.step)
            v_hat = self.v[key] / (1.0 - beta2 ** self.step)
            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss


def position_errors(x_raw, q_pred, robots):
    """FK position error (m) per sample, using each row's robot flag."""

    errors = []
    for inputs, q_norm in zip(x_raw, q_pred):
        robot = "scara" if inputs[0] == 1.0 else "ned"
        if robot not in robots:
            continue
        joints = denormalise(np.clip(q_norm, -1.0, 1.0), JOINT_LIMITS[robot])
        predicted = np.array(FORWARD_KINEMATICS[robot](joints))
        errors.append(np.linalg.norm(predicted - inputs[2:5]))
    return np.array(errors)


def learning_rate(args, epoch):
    """Cosine decay from lr at epoch 1 to lr_min at the last epoch."""

    if args.decay == "none" or args.epochs == 1:
        return args.lr
    progress = (epoch - 1) / (args.epochs - 1)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", nargs="+", choices=tuple(JOINT_LIMITS), default=["ned", "scara"])
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="units per hidden layer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5, help="final learning rate of the cosine decay")
    parser.add_argument("--decay", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=300, help="epochs without validation improvement")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--loss", choices=("joints", "tcp"), default="joints",
                        help="joint-label MSE (baseline) or squared TCP distance through FK")
    parser.add_argument("--tcp-limit-weight", type=float, default=0.01,
                        help="TCP loss joint-limit penalty weight (m² per squared normalised joint unit)")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS,
                        help="separate dataset experiment; defaults to the original splits")
    parser.add_argument("--model-name", help="distinct checkpoint name; refuses to replace existing files")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 1:
        parser.error("epochs, batch and patience must be positive")
    if not math.isfinite(args.tcp_limit_weight) or args.tcp_limit_weight < 0:
        parser.error("tcp-limit-weight must be finite and nonnegative")
    if args.model_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_name):
        parser.error("model-name must contain only letters, numbers, underscores or hyphens")
    if args.splits_dir.resolve() != SPLITS.resolve() and not args.model_name:
        parser.error("a separate splits-dir requires model-name to protect original checkpoints")
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    name = args.model_name or f"mlp_{'_'.join(args.robots)}_{args.size}_h{'x'.join(map(str, args.hidden))}_{args.decay}"
    if args.loss == "tcp" and not args.model_name:
        name += "_tcp"
    if args.model_name and any((MODELS / f"{name}{suffix}").exists()
                               for suffix in (".npz", ".json", "_history.csv")):
        raise FileExistsError(f"Checkpoint {name} already exists; choose a new model-name")

    x_train_raw, y_train = load_split(args.robots, "train", args.size, args.splits_dir)
    x_val_raw, y_val = load_split(args.robots, "validation", args.size, args.splits_dir)

    def tcp_targets(kind, raw_inputs):
        targets = []
        for robot in args.robots:
            with (args.splits_dir / robot / f"{kind}_{args.size}.csv").open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    fields = ("desired_x", "desired_y", "desired_z")
                    fields = fields if all(key in row for key in fields) else ("x", "y", "z")
                    targets.append([float(row[key]) for key in fields])
        targets = np.asarray(targets)
        if targets.shape != (len(raw_inputs), 3) or not np.all(np.isfinite(targets)):
            raise ValueError("Invalid TCP target rows")
        return targets

    train_tcp_targets = tcp_targets("train", x_train_raw)
    val_tcp_targets = tcp_targets("validation", x_val_raw)

    # Standardise x, y, z with training statistics only; keep the flags 0/1.
    mean = np.zeros(len(INPUTS))
    std = np.ones(len(INPUTS))
    mean[2:] = x_train_raw[:, 2:].mean(axis=0)
    std[2:] = x_train_raw[:, 2:].std(axis=0)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std

    model = MLP((len(INPUTS), *args.hidden, len(OUTPUTS)), rng)
    best = {"val_loss": math.inf, "epoch": 0, "params": None}
    history = []
    loss_label = "joint MSE" if args.loss == "joints" else "TCP objective (m², includes limit penalty)"
    print(f"Training with {loss_label}", flush=True)

    def objective(prediction, labels, raw_inputs, target_xyz):
        if args.loss == "tcp":
            return tcp_loss_and_gradient(prediction, raw_inputs, JOINT_LIMITS,
                                         args.tcp_limit_weight, target_xyz)[0]
        return float(np.mean((prediction - labels) ** 2))

    for epoch in range(1, args.epochs + 1):
        lr = learning_rate(args, epoch)
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            loss_fn = None
            if args.loss == "tcp":
                loss_fn = lambda prediction: tcp_loss_and_gradient(
                    prediction, x_train_raw[batch], JOINT_LIMITS, args.tcp_limit_weight, train_tcp_targets[batch])
            model.train_batch(x_train[batch], y_train[batch], lr, loss_fn=loss_fn)

        train_loss = objective(model.predict(x_train), y_train, x_train_raw, train_tcp_targets)
        val_loss = objective(model.predict(x_val), y_val, x_val_raw, val_tcp_targets)
        history.append((epoch, lr, train_loss, val_loss))

        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "params": {key: value.copy() for key, value in model.params.items()},
            }
        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  lr {lr:.1e}  train loss {train_loss:.8f}  val loss {val_loss:.8f}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"Early stop at epoch {epoch}; best epoch {best['epoch']}")
            break

    model.params = best["params"]
    q_val = model.predict(x_val)
    print(f"\nBest validation {loss_label}: {best['val_loss']:.8f} at epoch {best['epoch']}")
    validation_metrics = {}
    # A fallback dataset's requested XYZ can be unreachable. Its desired TCP
    # is stored explicitly, so its unavoidable request gap is not model error.
    metric_inputs = x_val_raw.copy()
    offset = 0
    has_policy_targets = False
    for robot in args.robots:
        with (args.splits_dir / robot / f"validation_{args.size}.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if rows and all(key in rows[0] for key in ("desired_x", "desired_y", "desired_z")):
            metric_inputs[offset:offset + len(rows), 2:5] = [
                [float(row[f"desired_{axis}"]) for axis in "xyz"] for row in rows]
            has_policy_targets = True
        offset += len(rows)
    for robot in args.robots:
        errors = position_errors(metric_inputs, q_val, [robot])
        validation_metrics[robot] = {
            "mean_tcp_error_mm": float(errors.mean() * 1000),
            "median_tcp_error_mm": float(np.median(errors) * 1000),
            "p95_tcp_error_mm": float(np.percentile(errors, 95) * 1000),
            "max_tcp_error_mm": float(errors.max() * 1000),
            "rmse_tcp_error_mm": float(np.sqrt(np.mean(errors ** 2)) * 1000),
        }
        print(f"{robot}: validation {'desired TCP' if has_policy_targets else 'position'} error {errors.mean() * 1000:.2f} "
              f"± {errors.std() * 1000:.2f} mm (n={len(errors)})")

    MODELS.mkdir(exist_ok=True)
    np.savez(MODELS / f"{name}.npz", mean=mean, std=std, **model.params)
    with (MODELS / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        columns = ("train_mse", "val_mse") if args.loss == "joints" else ("train_loss_m2", "val_loss_m2")
        writer.writerow(("epoch", "lr", *columns))
        writer.writerows(history)
    with (MODELS / f"{name}.json").open("w", encoding="utf-8") as destination:
        dataset_path = args.splits_dir.resolve()
        dataset_path = dataset_path.relative_to(ROOT) if dataset_path.is_relative_to(ROOT) else dataset_path
        metadata = {**vars(args), "splits_dir": dataset_path.as_posix(), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                    "training_tcp_target": "desired policy TCP when present, otherwise requested TCP",
                    "validation_tcp_target": "desired policy TCP" if has_policy_targets else "requested TCP",
                    "validation_fk_metrics": validation_metrics}
        if args.loss == "joints":
            metadata["best_val_mse"] = best["val_loss"]
        json.dump(metadata, destination, indent=2)
    print(f"Saved {MODELS / name}.npz")


if __name__ == "__main__":
    main()
