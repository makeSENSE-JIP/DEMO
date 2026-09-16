"""One validated configuration for case generation and both ERT methods."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GNSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_iterations: int = Field(default=15, ge=1)
    basis_members: int = Field(default=10, ge=2)
    rank: int = Field(default=40, ge=1)
    energy: float = Field(default=1, gt=0, le=1)
    damping: float = Field(default=10, gt=0)
    gradient_tolerance: float = Field(default=1e-3, gt=0)
    step_tolerance: float = Field(default=1e-6, gt=0)
    alphas: list[float] = [1, 0.5, 0.25, 0.125]
    log_bounds: tuple[float, float] = (-5, 10)

    @model_validator(mode="after")
    def validate_search(self):
        if (not self.alphas or any(not 0 < a <= 1 for a in self.alphas)
                or any(a <= b for a, b in zip(self.alphas, self.alphas[1:]))):
            raise ValueError("alphas must decrease strictly within (0, 1]")
        if self.log_bounds[0] >= self.log_bounds[1]:
            raise ValueError("log_bounds must increase")
        return self


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    members: int = Field(default=100, ge=2)
    jobs: int = Field(default=1, ge=1)
    inflation: list[float] = [5, 5, 5, 5, 5]
    gn: GNSettings = Field(default_factory=GNSettings)

    @model_validator(mode="after")
    def validate_ensemble(self):
        if self.gn.basis_members > self.members:
            raise ValueError("gn.basis_members must not exceed members")
        if (not self.inflation or any(a <= 0 for a in self.inflation)
                or abs(sum(1 / a for a in self.inflation) - 1) > 1e-12):
            raise ValueError("inflation must be positive and sum(1/alpha) must equal one")
        return self

    @classmethod
    def from_file(cls, path: Path, **overrides):
        def expand(key, value):
            head, _, tail = key.partition(".")
            return {head: expand(tail, value) if tail else value}

        def merge(base, extra):
            for key, value in extra.items():
                base[key] = merge(base.get(key, {}), value) if isinstance(value, dict) else value
            return base

        with path.open("rb") as stream:
            values = tomllib.load(stream)
        for key, value in overrides.items():
            if value is not None:
                merge(values, expand(key, value))
        return cls.model_validate(values)
