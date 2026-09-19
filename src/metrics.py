"""Lightweight runtime metrics for reproducible benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class TimingSummary:
    """Accumulates durations for one pipeline stage."""

    samples_seconds: list[float] = field(default_factory=list)

    def observe(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Duration cannot be negative.")
        self.samples_seconds.append(seconds)

    @property
    def count(self) -> int:
        return len(self.samples_seconds)

    @property
    def mean_ms(self) -> float:
        if not self.samples_seconds:
            return 0.0
        return 1000.0 * sum(self.samples_seconds) / self.count

    @property
    def fps(self) -> float:
        if not self.samples_seconds:
            return 0.0
        total = sum(self.samples_seconds)
        return self.count / total if total > 0 else 0.0


class Timer:
    """Context manager that records elapsed time in a TimingSummary."""

    def __init__(self, summary: TimingSummary) -> None:
        self.summary = summary
        self._started_at: float | None = None

    def __enter__(self) -> "Timer":
        self._started_at = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self._started_at is None:
            raise RuntimeError("Timer was not started.")
        self.summary.observe(perf_counter() - self._started_at)
