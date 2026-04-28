"""Continual-learning measurement primitives — BWT, FWT, average accuracy.

These follow the standard formulation from Lopez-Paz & Ranzato, *Gradient
Episodic Memory for Continual Learning* (NeurIPS 2017),
https://arxiv.org/abs/1706.08840.

We work with a `History` object: a list of "rows", one per stage. Stage 0
captures the model BEFORE any task-specific training (used for FWT). Stage
k > 0 captures it AFTER training task k-1. Each row maps task index → test
accuracy on that task at that stage.

```python
from shared.continual import History, summarize

h = History()
h.add_stage(0, accuracies={0: 0.25, 1: 0.30})        # before training
h.add_stage(1, accuracies={0: 0.92, 1: 0.31})        # after task A
h.add_stage(2, accuracies={0: 0.55, 1: 0.88})        # after task B

print(summarize(h))                                  # avg_acc, BWT, FWT
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class History:
    """Accuracy matrix `R[stage][task]` collected during sequential training."""

    rows: list[dict[int, float]] = field(default_factory=list)

    def add_stage(self, stage: int, accuracies: dict[int, float]) -> None:
        if stage != len(self.rows):
            raise ValueError(
                f"stages must be added in order (got {stage}, expected {len(self.rows)})"
            )
        self.rows.append(dict(accuracies))

    def n_tasks(self) -> int:
        if not self.rows:
            return 0
        return max(len(row) for row in self.rows)

    def n_stages(self) -> int:
        return len(self.rows)

    def acc(self, stage: int, task: int) -> float:
        return float(self.rows[stage][task])

    def to_matrix(self) -> list[list[float | None]]:
        T = self.n_tasks()
        return [[row.get(t) for t in range(T)] for row in self.rows]


def average_accuracy(history: History) -> float:
    """Mean accuracy across all tasks at the FINAL stage (after the last task)."""
    if history.n_stages() == 0 or history.n_tasks() == 0:
        return float("nan")
    last = history.rows[-1]
    return float(mean(last.values()))


def backward_transfer(history: History) -> float:
    """BWT = mean over prior tasks i of `R[T][i] - R[i+1][i]`.

    Negative BWT means catastrophic forgetting; zero means no forgetting;
    positive means continued training on later tasks *helped* prior tasks.
    Stage 0 is "before training", so task i was trained at stage i+1.
    """
    T = history.n_tasks()
    if T < 2 or history.n_stages() < T + 1:
        return float("nan")
    final = T  # final stage index
    deltas = [history.acc(final, i) - history.acc(i + 1, i) for i in range(T - 1)]
    return float(mean(deltas))


def forward_transfer(history: History, baseline: dict[int, float] | None = None) -> float:
    """FWT = mean over future tasks i (i > 0) of `R[i][i] - baseline[i]`.

    `R[i][i]` is the accuracy on task i measured at stage i — i.e. BEFORE
    we have trained on task i. Default baseline is stage 0 (random init).
    Pass `baseline={i: 1/num_classes_i}` to compare against random instead
    of the as-loaded model.
    """
    T = history.n_tasks()
    if T < 2 or history.n_stages() < T + 1:
        return float("nan")
    base = baseline if baseline is not None else history.rows[0]
    deltas = [history.acc(i, i) - float(base.get(i, 0.0)) for i in range(1, T)]
    return float(mean(deltas))


def summarize(history: History, baseline: dict[int, float] | None = None) -> dict[str, Any]:
    """One-shot summary used by class metric blocks."""
    return {
        "n_tasks": history.n_tasks(),
        "n_stages": history.n_stages(),
        "avg_accuracy": average_accuracy(history),
        "BWT": backward_transfer(history),
        "FWT": forward_transfer(history, baseline=baseline),
        "matrix": history.to_matrix(),
    }
