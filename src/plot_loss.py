"""Training-loss curves of the four OpenVid runs (high / low expert), from outputs/train/<run>/loss.csv.

    python3 src/plot_loss.py   -> results/figures/train_loss.png   (any env with matplotlib; p2v has none)

The per-step loss is one sample at a random sigma, so it is very noisy; we plot the mean per 100-step bin.
Rows logged twice around a resume keep the last copy (the run the final checkpoint came from). B0 high lost
loss.csv rows 2790-3500 in the 2026-11-22 ENOSPC crash; those bins come from the every-10-steps console log
(logs/train_openvid_b0_high_crash1122.log). Dotted lines mark resumes (AdamW moments restart there).
"""

import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [("main", "hybrid", "#2a78d6"), ("B0", "b0", "#eb6834"), ("B1", "b1", "#1baf7a"), ("zlat", "zlat", "#eda100")]
BIN = 100


def load(run):
    d = {}
    for r in csv.reader(open(f"outputs/train/{run}/loss.csv")):
        if r and r[0].isdigit():
            d[int(r[0])] = float(r[1])  # later rows overwrite earlier ones
    crash = f"logs/train_{run}_crash1122.log"
    if os.path.exists(crash):
        for m in re.finditer(r"^step (\d+) loss ([\d.]+)", open(crash).read(), re.M):
            d.setdefault(int(m[1]), float(m[2]))
    resumes = [int(re.search(r"resume(\d+)", f)[1]) for f in glob.glob(f"outputs/train/{run}/args_resume*.json")]
    return d, resumes


def main():
    os.chdir(ROOT)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    summary = {}
    for ax, expert in zip(axes, ("high", "low")):
        for label, key, color in RUNS:
            d, resumes = load(f"openvid_{key}_{expert}")
            steps = np.array(sorted(d))
            loss = np.array([d[s] for s in steps])
            bins = (steps - 1) // BIN
            xs = [(b + 0.5) * BIN for b in np.unique(bins)]
            ys = [loss[bins == b].mean() for b in np.unique(bins)]
            ax.plot(xs, ys, color=color, lw=2, label=label)
            for r in resumes:
                ax.axvline(r, color=color, lw=1, ls=":", alpha=.8)
            summary[f"{label}_{expert}"] = {"first500": float(loss[steps <= 500].mean()),
                                            "last500": float(loss[steps > 3500].mean()), "n_steps": int(len(steps))}
        ax.set_title(f"{expert}-noise expert" + (" (t ≥ 900)" if expert == "high" else " (t < 900)"), fontsize=11)
        ax.set_xlabel("step")
        ax.grid(alpha=.25, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"flow-matching loss (mean / {BIN} steps)")
    axes[1].legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/train_loss.png", dpi=160)
    json.dump(summary, open("results/figures/train_loss.json", "w"), indent=1)
    # main and zlat drew the exact same (clip, sigma) sequence (same seed and RNG use), so their per-step losses
    # pair up: (zlat - main) / main on the steps whose condition was not dropped removes the per-sample noise
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    for ax, expert in zip(axes, ("high", "low")):
        rows = {}
        for key in ("hybrid", "zlat"):
            rows[key] = {int(r[0]): (float(r[1]), r[2], r[4], r[3]) for r in
                         csv.reader(open(f"outputs/train/openvid_{key}_{expert}/loss.csv")) if r and r[0].isdigit()}
        m, z = rows["hybrid"], rows["zlat"]
        s = np.array([i for i in sorted(m) if i in z and m[i][1:3] == z[i][1:3] and m[i][3] == "1"])
        rel = np.array([(z[i][0] - m[i][0]) / m[i][0] for i in s]) * 100
        bins = (s - 1) // BIN
        xs, mean, lo, hi = [], [], [], []
        rng = np.random.default_rng(0)
        for b in np.unique(bins):
            d = rel[bins == b]
            bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            xs.append((b + .5) * BIN); mean.append(d.mean())
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.fill_between(xs, lo, hi, color="#eda100", alpha=.25, lw=0)
        ax.plot(xs, mean, color="#eda100", lw=2, label="zlat vs main (95% CI)")
        ax.axhline(0, color="#666", lw=1)
        ax.set_title(f"{expert}-noise expert: same samples, zlat loss relative to main", fontsize=10.5)
        ax.set_xlabel("step")
        ax.grid(alpha=.25, lw=.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        summary[f"zlat_vs_main_{expert}_last1000_pct"] = float(rel[s > 3000].mean())
    axes[0].set_ylabel("Δ loss (%), > 0 = zlat worse")
    fig.tight_layout()
    fig.savefig("results/figures/train_loss_zlat_vs_main.png", dpi=160)
    json.dump(summary, open("results/figures/train_loss.json", "w"), indent=1)
    for k, v in summary.items():
        if not isinstance(v, dict):
            print(f"{k}: {v:+.1f}%")
            continue
        print(f"{k:10s} first500 {v['first500']:.4f}  last500 {v['last500']:.4f}  steps {v['n_steps']}")
    plot_val()


def plot_val():
    """results/val/val_{high,low}.jsonl (src/val_loss.py) -> val_loss.png + val_cond_benefit.png."""
    rows = {e: [json.loads(line) for line in open(f"results/val/val_{e}.jsonl")]
            for e in ("high", "low") if os.path.exists(f"results/val/val_{e}.jsonl")}
    if not rows:
        return
    colors = {label: c for label, _, c in RUNS}
    rng = np.random.default_rng(0)
    out = {}

    # 1) val loss vs step (with the motion condition; B0 has none). step 0 = base Wan2.2 for every run
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, e in zip(axes, ("high", "low")):
        R = rows.get(e, [])
        base = next((r["mean"] for r in R if r["run"] == "base"), None)
        for label, _, color in RUNS:
            pts = sorted((r["step"], r["mean"]) for r in R if r["run"] == label and r["cond"] == (label != "B0"))
            if base is not None:
                pts = [(0, base)] + pts
            if len(pts) > 1:
                ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=4, label=label)
            nc = sorted((r["step"], r["mean"]) for r in R if r["run"] == label and not r["cond"] and label != "B0")
            if nc:
                ax.plot(*zip(*nc), color=color, lw=1.2, ls="--", marker="o", ms=5, mfc="none")
            out[f"{e}_{label}"] = {s: m for s, m in pts}
        ax.set_title(f"{e}-noise expert: held-out loss (20 clips × 2 σ)", fontsize=10.5)
        ax.set_xlabel("step  (0 = base Wan2.2)")
        ax.grid(alpha=.25, lw=.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("validation loss")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#666", ls="--", marker="o", mfc="none"))
    labels.append("same run, condition dropped")
    axes[1].legend(handles, labels, frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig("results/figures/val_loss.png", dpi=160)

    # 2) how much the motion condition helps: (no-cond - cond) / no-cond on the same items, paired bootstrap CI
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=False)
    for ax, e in zip(axes, ("high", "low")):
        R = rows.get(e, [])
        for k, (label, _, color) in enumerate(r for r in RUNS if r[0] != "B0"):
            xs, ms, los, his = [], [], [], []
            for step in sorted({r["step"] for r in R if r["run"] == label and not r["cond"]}):
                a = next((r for r in R if r["run"] == label and r["step"] == step and r["cond"]), None)
                b = next(r for r in R if r["run"] == label and r["step"] == step and not r["cond"])
                if a is None:
                    continue
                d = (np.array(b["losses"]) - np.array(a["losses"])) / np.array(b["losses"]) * 100
                bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                xs.append(step + (k - 1) * 60); ms.append(d.mean())
                los.append(d.mean() - np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5) - d.mean())
                out[f"{e}_{label}_cond_benefit_{step}_pct"] = float(d.mean())
            if xs:
                ax.errorbar(xs, ms, yerr=[los, his], color=color, lw=2, marker="o", ms=5, capsize=3, label=label)
        ax.axhline(0, color="#666", lw=1)
        ax.set_title(f"{e}-noise expert: loss reduction from the motion condition", fontsize=10.5)
        ax.set_xlabel("step")
        ax.grid(alpha=.25, lw=.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("(no cond − cond) / no cond  (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/val_cond_benefit.png", dpi=160)
    json.dump(out, open("results/figures/val_loss.json", "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k}: {v:+.1f}%")


if __name__ == "__main__":
    main()
