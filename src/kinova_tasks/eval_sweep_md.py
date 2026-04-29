"""Render breaking_points.md from an existing sweep_summary.csv.

Use when a sweep run completed but the markdown writer crashed (e.g. the
ASCII unicode bug fixed in eval_sweep.py:519). Reads the CSV, applies the
same envelope logic as the live driver, and writes breaking_points.md.

Usage:
    uv run python -m kinova_tasks.eval_sweep_md --output-dir docs/results/open_drawer_osc_phase1 --checkpoint-file logs/.../model_3800.pt
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import tyro

from kinova_tasks.eval_sweep import (
    DEGRADED_THRESHOLD,
    PASS_THRESHOLD,
    SUCCESS_THRESHOLD_M,
    SettingResult,
    _envelope_width,
    default_axes,
)


@dataclass
class Cfg:
    output_dir: str
    """Directory containing sweep_summary.csv; breaking_points.md is written here."""
    checkpoint_file: str = "(unknown — rendered from CSV)"
    """Optional checkpoint path string to record in the markdown header."""


def main(cfg: Cfg | None = None) -> None:
    if cfg is None:
        cfg = tyro.cli(Cfg)
    out = Path(cfg.output_dir)
    csv_path = out / "sweep_summary.csv"
    rows: list[SettingResult] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                SettingResult(
                    axis=r["axis"],
                    value=float(r["value"]),
                    success_rate=float(r["success_rate"]),
                    mean_error=float(r["mean_error_m"]),
                    mean_episode_length=float(r["mean_episode_length"]),
                    n_episodes=int(r["n_episodes"]),
                )
            )

    axes = default_axes()
    md_lines = [
        "# OOD sweep — breaking points",
        "",
        f"Checkpoint: `{cfg.checkpoint_file}`",
        f"`success_rate` threshold: {SUCCESS_THRESHOLD_M:.3f} m",
        f"Pass: success >= {PASS_THRESHOLD:.0%}, "
        f"degraded: >= {DEGRADED_THRESHOLD:.0%}, fail: below.",
        "",
        "| Axis | Nominal | Envelope (passes) | Width norm. | Status |",
        "|---|---|---|---|---|",
    ]
    widths: list[float] = []
    for axis in axes:
        axis_rows = [r for r in rows if r.axis == axis.name]
        if not axis_rows:
            md_lines.append(
                f"| {axis.name} ({axis.units}) | {axis.nominal:.4g} | (no data) "
                f"| 0.00 | n/a |"
            )
            widths.append(0.0)
            continue
        width_norm, passing = _envelope_width(axis, axis_rows)
        widths.append(width_norm)
        passing_str = (
            f"[{min(passing):.4g} .. {max(passing):.4g}]" if passing else "-"
        )
        nominal_row = next(
            (r for r in axis_rows if math.isclose(r.value, axis.nominal, rel_tol=1e-6, abs_tol=1e-9)),
            None,
        )
        nominal_status = nominal_row.status() if nominal_row else "n/a"
        md_lines.append(
            f"| {axis.name} ({axis.units}) | {axis.nominal:.4g} | {passing_str} "
            f"| {width_norm:.2f} | {nominal_status} |"
        )

    robustness = sum(widths) / len(widths) if widths else 0.0
    md_lines += [
        "",
        f"**Robustness score** (mean normalized envelope width): **{robustness:.3f}**",
        "",
        "Per-axis CSV: `sweep_summary.csv`.",
    ]

    md_path = out / "breaking_points.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[sweep_md] wrote {md_path}")
    print(f"[sweep_md] robustness_score = {robustness:.3f}")


if __name__ == "__main__":
    main()
