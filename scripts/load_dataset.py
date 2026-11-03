"""NumPy loaders for the coursework ANN CSV datasets."""

import csv
from pathlib import Path

import numpy as np


def load_inputs_targets(path, input_columns, target_columns):
    """Read selected numeric CSV columns into NumPy input and target arrays."""

    input_columns = tuple(input_columns)
    target_columns = tuple(target_columns)
    columns = input_columns + target_columns
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        values = [[float(row[column]) for column in columns] for row in reader]

    if not values:
        raise ValueError(f"{path} contains no data rows")
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains non-finite values in the selected columns")
    return data[:, :len(input_columns)], data[:, len(input_columns):]
