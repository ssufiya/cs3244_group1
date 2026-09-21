"""A tiny audit log so every dropped row is accounted for."""
from __future__ import annotations

import json
import time
from pathlib import Path


class CleaningLog:
    def __init__(self, dataset: str, n_start: int):
        self.dataset = dataset
        self.n_start = n_start
        self.n_current = n_start
        self.steps: list[dict] = []
        self.stats: dict = {}
        self.t0 = time.time()

    def drop(self, reason: str, n_after: int, detail: str = "") -> None:
        removed = self.n_current - n_after
        self.steps.append({
            "step": len(self.steps) + 1,
            "reason": reason,
            "removed": removed,
            "pct_of_original": round(100 * removed / self.n_start, 4) if self.n_start else 0.0,
            "rows_remaining": n_after,
            "detail": detail,
        })
        self.n_current = n_after

    def note(self, key: str, value) -> None:
        self.stats[key] = value

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows_in": self.n_start,
            "rows_out": self.n_current,
            "retained_pct": round(100 * self.n_current / self.n_start, 2) if self.n_start else 0.0,
            "elapsed_sec": round(time.time() - self.t0, 1),
            "steps": self.steps,
            "stats": self.stats,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")

    def print_table(self) -> None:
        print(f"\n--- cleaning audit: {self.dataset} ---")
        print(f"{'step':<5}{'reason':<38}{'removed':>10}{'%orig':>9}{'left':>12}")
        for s in self.steps:
            print(f"{s['step']:<5}{s['reason']:<38}{s['removed']:>10,}"
                  f"{s['pct_of_original']:>8.2f}%{s['rows_remaining']:>12,}")
        print(f"{'':5}{'TOTAL RETAINED':<38}{'':>10}"
              f"{100 * self.n_current / self.n_start if self.n_start else 0:>8.2f}%"
              f"{self.n_current:>12,}")
