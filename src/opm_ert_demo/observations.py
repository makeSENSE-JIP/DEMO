"""Observations in date-major, alphabetical-key order throughout the demo."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Observations:
    keys: tuple[str, ...]
    dates: np.ndarray
    values: np.ndarray
    stds: np.ndarray

    def __post_init__(self):
        shape = (len(self.dates), len(self.keys))
        if not self.keys or len(set(self.keys)) != len(self.keys) or self.keys != tuple(sorted(self.keys)):
            raise ValueError("Observation keys must be unique and sorted")
        if not len(self.dates) or len(np.unique(self.dates)) != len(self.dates):
            raise ValueError("Observation dates must be nonempty and unique")
        if (self.values.shape != shape or self.stds.shape != shape
                or not np.isfinite(self.values).all() or not np.isfinite(self.stds).all()
                or np.any(self.stds <= 0) or np.isnat(self.dates).any()):
            raise ValueError("Invalid observation values, dates, or standard deviations")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        columns = np.argsort(data["keys"])
        return cls(
            tuple(data["keys"][i] for i in columns),
            np.array(data["dates"], dtype="datetime64[D]"),
            np.array(data["values"], dtype=float)[:, columns],
            np.array(data["stds"], dtype=float)[:, columns],
        )

    def misfit(self, predictions):
        predictions = np.asarray(predictions)
        if (predictions.ndim != 2 or predictions.shape[1] != self.values.size
                or not np.isfinite(predictions).all()):
            raise ValueError("Predictions must be a finite members x observations matrix")
        return 0.5 * np.sum(((predictions - self.values.ravel()) / self.stds.ravel()) ** 2, axis=1)

    def columns(self, key):
        return np.arange(len(self.dates)) * len(self.keys) + self.keys.index(key)

    def write_ert(self, path):
        entries = []
        for i, date in enumerate(self.dates):
            for j, key in enumerate(self.keys):
                entries.append(
                    f"SUMMARY_OBSERVATION {key.replace(':', '_')}_{date}\n{{\n"
                    f" KEY = {key};\n DATE = {date};\n"
                    f" VALUE = {self.values[i, j]:.17g};\n ERROR = {self.stds[i, j]:.17g};\n}};\n"
                )
        path.write_text("\n".join(entries))
