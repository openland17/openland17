"""
Render an SVG training-curves chart from HuggingFace Trainer's trainer_state.json.

This script produces the *same visual template* as the hand-crafted SVG in the
profile repo, but driven by real training data — so once you've actually
trained CamoNet, run this and commit the result.

USAGE:
    python scripts/render_training_chart.py \\
        --state checkpoints/camonet/trainer_state.json \\
        --out   ../profile/assets/training-curves.svg \\
        --title "training run · camonet · dinov2-base"

The HuggingFace Trainer writes `trainer_state.json` after every save; its
`log_history` contains step-level training loss and epoch-level eval metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Plot geometry — matches the hand-crafted SVG so style stays consistent.
# ---------------------------------------------------------------------------
W, H = 900, 320
PLOT_X0, PLOT_X1 = 70, 850
PLOT_Y0, PLOT_Y1 = 50, 270   # y0 = top (high loss), y1 = bottom (zero loss)


@dataclass
class TrainingPoint:
    epoch: float
    loss: float


@dataclass
class EvalPoint:
    epoch: float
    loss: float
    top1: float | None = None
    top3: float | None = None
    macro_f1: float | None = None


def parse_state(path: Path) -> tuple[list[TrainingPoint], list[EvalPoint]]:
    state = json.loads(path.read_text())
    log = state.get("log_history", [])
    train: list[TrainingPoint] = []
    eval_: list[EvalPoint] = []
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry and "epoch" in entry:
            train.append(TrainingPoint(epoch=float(entry["epoch"]),
                                        loss=float(entry["loss"])))
        if "eval_loss" in entry and "epoch" in entry:
            eval_.append(EvalPoint(
                epoch=float(entry["epoch"]),
                loss=float(entry["eval_loss"]),
                top1=entry.get("eval_top1"),
                top3=entry.get("eval_top3"),
                macro_f1=entry.get("eval_macro_f1"),
            ))
    return train, eval_


def make_scaler(epochs_max: float, loss_max: float):
    """Return (x_of, y_of) coordinate mappers."""
    def x_of(epoch: float) -> float:
        return PLOT_X0 + (epoch / epochs_max) * (PLOT_X1 - PLOT_X0)

    def y_of(loss: float) -> float:
        return PLOT_Y1 - (loss / loss_max) * (PLOT_Y1 - PLOT_Y0)

    return x_of, y_of


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render(
    train: list[TrainingPoint],
    eval_: list[EvalPoint],
    title: str,
    subtitle: str,
) -> str:
    if not train or not eval_:
        raise ValueError("need at least one train and one eval point")

    epochs_max = max(max(p.epoch for p in train), max(p.epoch for p in eval_))
    loss_max_data = max(max(p.loss for p in train), max(p.loss for p in eval_))
    # Round up to a clean ceiling: nearest 0.5 above data, min 4.0
    loss_max = max(4.0, (int(loss_max_data * 2) + 1) / 2)

    x_of, y_of = make_scaler(epochs_max, loss_max)

    # Best checkpoint = lowest eval loss (earliest epoch on ties).
    # Use the best checkpoint's metrics for the headline numbers because
    # `load_best_model_at_end=True` means that's what actually ships.
    best = min(eval_, key=lambda p: (p.loss, p.epoch))

    train_pts = [(x_of(p.epoch), y_of(p.loss)) for p in train]
    eval_pts = [(x_of(p.epoch), y_of(p.loss)) for p in eval_]

    # Y gridlines + labels at integer loss values up to ceiling
    grid_lines = []
    y_labels = []
    n_div = int(loss_max)
    for i in range(n_div + 1):
        loss_val = i
        y = y_of(loss_val)
        if i > 0:  # skip the bottom which is the axis itself
            grid_lines.append(
                f'  <line x1="{PLOT_X0}" y1="{y:.1f}" x2="{PLOT_X1}" y2="{y:.1f}" class="grid"/>'
            )
        y_labels.append(
            f'  <text x="{PLOT_X0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{loss_val:.1f}</text>'
        )

    # X ticks every 2 epochs (or every 1 if <8 epochs)
    step = 2 if epochs_max >= 8 else 1
    x_ticks = []
    e = 0
    while e <= epochs_max:
        x = x_of(e)
        x_ticks.append(
            f'  <line x1="{x:.1f}" y1="{PLOT_Y1}" x2="{x:.1f}" y2="{PLOT_Y1 + 3}" class="axis"/>\n'
            f'  <text x="{x:.1f}" y="{PLOT_Y1 + 15}" text-anchor="middle" class="label">{e}</text>'
        )
        e += step
    if e - step < epochs_max:  # add the last tick if missed
        x = x_of(epochs_max)
        x_ticks.append(
            f'  <line x1="{x:.1f}" y1="{PLOT_Y1}" x2="{x:.1f}" y2="{PLOT_Y1 + 3}" class="axis"/>\n'
            f'  <text x="{x:.1f}" y="{PLOT_Y1 + 15}" text-anchor="middle" class="label">{int(epochs_max)}</text>'
        )

    # Best-checkpoint marker
    bx, by = x_of(best.epoch), y_of(best.loss)
    best_label = f"★ best · ep {int(best.epoch)} · val {best.loss:.2f}"

    # Final metrics line — pulled from the best checkpoint (what ships).
    metric_bits = []
    if best.top1 is not None:
        metric_bits.append(f"top-1 · {best.top1:.3f}")
    if best.top3 is not None:
        metric_bits.append(f"top-3 · {best.top3:.3f}")
    if best.macro_f1 is not None:
        metric_bits.append(f"macro-f1 · {best.macro_f1:.2f}")
    metric_line = "    ".join(metric_bits) if metric_bits else f"best val · {best.loss:.3f}"

    aria = (
        f"Training run: train and validation loss across "
        f"{int(epochs_max)} epochs. Best checkpoint at epoch {int(best.epoch)} "
        f"with validation loss {best.loss:.2f}."
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     font-family="ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace"
     role="img"
     aria-label="{aria}">
  <style>
    .grid {{ stroke: #d8dee4; stroke-width: 0.5; }}
    .axis {{ stroke: #afb8c1; stroke-width: 1; }}
    .train {{ stroke: #0969da; stroke-width: 1.8; fill: none; }}
    .val {{ stroke: #cf222e; stroke-width: 1.8; fill: none; stroke-dasharray: 5 3; }}
    .marker {{ fill: #1a7f37; stroke: #1a7f37; }}
    .markerline {{ stroke: #1a7f37; stroke-width: 0.7; stroke-dasharray: 2 3; opacity: 0.65; fill: none; }}
    .label {{ fill: #656d76; font-size: 11px; }}
    .title {{ fill: #1f2328; font-size: 13px; font-weight: 600; }}
    .meta {{ fill: #656d76; font-size: 11px; }}
    .stat {{ fill: #1f2328; font-size: 11px; font-weight: 500; }}
    @media (prefers-color-scheme: dark) {{
      .grid {{ stroke: #30363d; }}
      .axis {{ stroke: #484f58; }}
      .train {{ stroke: #58a6ff; }}
      .val {{ stroke: #ff7b72; }}
      .marker {{ fill: #3fb950; stroke: #3fb950; }}
      .markerline {{ stroke: #3fb950; }}
      .label {{ fill: #8b949e; }}
      .title {{ fill: #e6edf3; }}
      .meta {{ fill: #8b949e; }}
      .stat {{ fill: #e6edf3; }}
    }}
  </style>

  <!-- Header -->
  <text x="{PLOT_X0}" y="22" class="title">{title}</text>
  <text x="{PLOT_X1}" y="22" text-anchor="end" class="meta">{subtitle}</text>

  <!-- Axes -->
  <line x1="{PLOT_X0}" y1="{PLOT_Y0}" x2="{PLOT_X0}" y2="{PLOT_Y1}" class="axis"/>
  <line x1="{PLOT_X0}" y1="{PLOT_Y1}" x2="{PLOT_X1}" y2="{PLOT_Y1}" class="axis"/>

  <!-- Gridlines -->
{chr(10).join(grid_lines)}

  <!-- Y labels -->
{chr(10).join(y_labels)}

  <!-- X ticks -->
{chr(10).join(x_ticks)}

  <text x="{(PLOT_X0 + PLOT_X1) // 2}" y="{PLOT_Y1 + 34}" text-anchor="middle" class="label">epoch</text>
  <text x="22" y="{(PLOT_Y0 + PLOT_Y1) // 2}" text-anchor="middle" class="label" transform="rotate(-90 22 {(PLOT_Y0 + PLOT_Y1) // 2})">loss</text>

  <!-- Best checkpoint guide line -->
  <line x1="{bx:.1f}" y1="{by:.1f}" x2="{bx:.1f}" y2="60" class="markerline"/>

  <!-- Curves -->
  <polyline class="train" points="{polyline(train_pts)}"/>
  <polyline class="val" points="{polyline(eval_pts)}"/>

  <!-- Best marker -->
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="3.5" class="marker"/>
  <text x="{bx:.1f}" y="55" text-anchor="middle" class="stat">{best_label}</text>

  <!-- Legend -->
  <line x1="700" y1="78" x2="730" y2="78" class="train"/>
  <text x="736" y="82" class="label">train</text>
  <line x1="700" y1="98" x2="730" y2="98" class="val"/>
  <text x="736" y="102" class="label">val</text>

  <!-- Final metrics -->
  <text x="{PLOT_X1}" y="318" text-anchor="end" class="stat">{metric_line}</text>
</svg>
'''
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True,
                        help="Path to trainer_state.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output SVG path")
    parser.add_argument("--title", default="training run")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()

    train, eval_ = parse_state(args.state)
    if not args.subtitle:
        # Auto-build subtitle from state metadata
        n_eval = len(eval_)
        max_epoch = max(p.epoch for p in eval_)
        args.subtitle = f"{n_eval} eval steps · {int(max_epoch)} epochs"

    svg = render(train, eval_, args.title, args.subtitle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"wrote {args.out}  ({len(train)} train pts, {len(eval_)} eval pts)")


if __name__ == "__main__":
    main()
