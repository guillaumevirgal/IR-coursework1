"""
feed-forward MLP that maps [is_scara, is_ned, x, y, z] to
the normalised joint values [q1, q2, q3] in [-1, 1].


Usage:
    python scripts/train_mlp.py                       # small combined model
    python scripts/train_mlp.py --robots ned          # small Ned only
    python scripts/train_mlp.py --size medium --hidden 64 64 --lr 1e-3
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara


SPLITS = ROOT / "data" / "nn" / "splits"
MODELS = ROOT / "models"
INPUTS = ("is_scara", "is_ned", "x", "y", "z")
OUTPUTS = ("q1", "q2", "q3")

# Must match the sampling ranges in controllers/main/main.py (not imported
# directly because that module needs the Webots controller API).
JOINT_LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
}
FORWARD_KINEMATICS = {"ned": forward_kinematics_ned, "scara": forward_kinematics_scara}


def denormalise(q_norm, limits):
    """Map normalised joints in [-1, 1] back to motor units (rad or m)."""

    lower = np.array([low for low, _ in limits])
    upper = np.array([high for _, high in limits])
    return (q_norm + 1.0) / 2.0 * (upper - lower) + lower


def load_split(robots, kind, size):
    """Stack the requested robots' CSVs into input and target arrays."""

    rows = []
    for robot in robots:
        path = SPLITS / robot / f"{kind}_{size}.csv"
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

    def train_batch(self, x, y, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """One Adam step on the MSE loss; returns the batch loss."""

        output, cache = self.forward(x)
        error = output - y
        loss = float(np.mean(error ** 2))

        # Gradients of mean((output - y)^2) over all batch elements, walked
        # back from the linear output layer through each ReLU layer.
        grads = {}
        d_pre = 2.0 * error / error.size
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
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_train_raw, y_train = load_split(args.robots, "train", args.size)
    x_val_raw, y_val = load_split(args.robots, "validation", args.size)

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

    for epoch in range(1, args.epochs + 1):
        lr = learning_rate(args, epoch)
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            model.train_batch(x_train[batch], y_train[batch], lr)

        train_loss = float(np.mean((model.predict(x_train) - y_train) ** 2))
        val_loss = float(np.mean((model.predict(x_val) - y_val) ** 2))
        history.append((epoch, lr, train_loss, val_loss))

        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "params": {key: value.copy() for key, value in model.params.items()},
            }
        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  lr {lr:.1e}  train MSE {train_loss:.6f}  val MSE {val_loss:.6f}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"Early stop at epoch {epoch}; best epoch {best['epoch']}")
            break

    model.params = best["params"]
    q_val = model.predict(x_val)
    print(f"\nBest validation MSE {best['val_loss']:.6f} at epoch {best['epoch']}")
    for robot in args.robots:
        errors = position_errors(x_val_raw, q_val, [robot])
        print(f"{robot}: validation position error {errors.mean() * 1000:.2f} "
              f"± {errors.std() * 1000:.2f} mm (n={len(errors)})")

    name = f"mlp_{'_'.join(args.robots)}_{args.size}_h{'x'.join(map(str, args.hidden))}_{args.decay}"
    MODELS.mkdir(exist_ok=True)
    np.savez(MODELS / f"{name}.npz", mean=mean, std=std, **model.params)
    with (MODELS / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("epoch", "lr", "train_mse", "val_mse"))
        writer.writerows(history)
    with (MODELS / f"{name}.json").open("w", encoding="utf-8") as destination:
        json.dump({**vars(args), "best_epoch": best["epoch"], "best_val_mse": best["val_loss"]},
                  destination, indent=2)
    print(f"Saved {MODELS / name}.npz")


if __name__ == "__main__":
    main()
