"""Regression checks for the handout's formulas and Task 4.B sample selection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task4_common import physical_joint_errors, position_statistics, write_csv
from scripts.evaluate_task4b import first_100, summarize, validate_measurements
from scripts.train_mlp import INPUTS, OUTPUTS, JOINT_LIMITS, denormalise


class CourseworkEvaluationTests(unittest.TestCase):
    def test_signed_angles_wrap_at_pi_and_lift_is_squared(self):
        true = np.array([[np.pi - 0.01, -np.pi + 0.02, 0.03]])
        predicted = np.array([[-np.pi + 0.01, np.pi - 0.02, 0.01]])
        errors, units = physical_joint_errors("scara", true, predicted)
        np.testing.assert_allclose(errors, [[-0.02, 0.04, 0.0004]], atol=1e-14)
        self.assertEqual(units, ["rad", "rad", "m^2"])
        ned, units = physical_joint_errors("ned", true, predicted)
        self.assertAlmostEqual(ned[0, 2], 0.02)
        self.assertEqual(units, ["rad", "rad", "rad"])

    def test_population_standard_deviation_and_rmse(self):
        result = position_statistics([0, 3, 6])
        self.assertEqual(result["mean_tcp_mm"], 3)
        self.assertAlmostEqual(result["std_tcp_mm"], np.sqrt(6))
        self.assertAlmostEqual(result["rmse_tcp_mm"], np.sqrt(15))

    def test_task4b_takes_first_100_not_evenly_spaced_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(zip((*INPUTS, *OUTPUTS), (0, 1, i / 1000, 0.2, 0.3, 0, 0, 0)))
                    for i in range(150)]
            write_csv(root / "ned/test.csv", rows)
            inputs, cases, reduced = first_100(root, "ned")
            self.assertEqual(len(cases), 100)
            self.assertEqual(len(reduced), 100)
            self.assertEqual(cases[0]["label"], "test_0001")
            self.assertEqual(cases[-1]["label"], "test_0100")
            np.testing.assert_allclose(inputs[:, 2], np.arange(100) / 1000)

    def measured_rows(self):
        # Synthetic controller output for testing aggregation, not physics evidence.
        cases = [{"label": f"test_{i+1:04d}", "point": (0.1, 0.2, 0.3)} for i in range(100)]
        raw = np.zeros((100, 3))
        q = denormalise(raw, JOINT_LIMITS["ned"])
        rows = []
        for i, case in enumerate(cases):
            row = {"label": case["label"], "robot": "ned", "settled": str(i != 0), "settle_seconds": "1"}
            for axis, value in zip("xyz", case["point"]):
                row[axis] = str(value)
                row[f"fk_{axis}"] = str(value)
                row[f"webots_{axis}"] = str(value + (0.01 if axis == "x" else 0))
            for j in range(3):
                row[f"raw_q{j+1}"] = "0"
                row[f"command_q{j+1}"] = str(q[i, j])
                row[f"measured_q{j+1}"] = str(q[i, j])
            rows.append(row)
        return rows, cases, raw

    def test_webots_error_uses_measurement_and_keeps_timeouts(self):
        rows, cases, raw = self.measured_rows()
        validate_measurements(rows, cases, raw, "ned")
        result = summarize("small", "tcp", "model", "ned", rows)
        self.assertAlmostEqual(result["mean_tcp_mm"], 10)
        self.assertAlmostEqual(result["fk_first100_mean_mm"], 0)
        self.assertEqual(result["samples"], 100)
        self.assertEqual(result["settled_count"], 99)
        self.assertEqual(result["timeout_count"], 1)
        self.assertFalse(result["all_samples_settled"])

    def test_incomplete_reordered_or_wrong_model_output_is_rejected(self):
        rows, cases, raw = self.measured_rows()
        for invalid in (rows[:-1], list(reversed(rows))):
            with self.assertRaises(RuntimeError):
                validate_measurements(invalid, cases, raw, "ned")
        rows[0]["raw_q1"] = "0.1"
        with self.assertRaises(RuntimeError):
            validate_measurements(rows, cases, raw, "ned")


if __name__ == "__main__":
    unittest.main()
