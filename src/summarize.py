"""Final tables for the Pexels benchmark (68 clips): EPE / MSE (bench_pexels.py eval) and VBench (vbench_eval.py),
with a paired bootstrap over clips against the main method (planner, same number of pokes).

    python src/summarize.py            -> results/summary.md (+ results/summary.json)

Rows whose result files do not exist yet are left out, so this can be rerun as results come in.
Delta columns: mean(method - main) over the clips both have, with a 95% CI from 10k paired bootstrap resamples;
'*' marks a CI that excludes 0.
"""

import json
import os

import numpy as np

from common import ROOT

# label, bench file, bench key prefix, vbench tag ({n} = number of pokes), poke-dependent?
METHODS = [
    ("Vanilla Wan2.2 (no fine-tune)", "vanilla__none", "none", "vanilla_none", False),
    ("B0: fine-tune, no motion", "b0__none", "none", "b0_none", False),
    ("B1: raw poke as straight line", "b1__line", "line", "b1_line_p{n}", True),
    ("Main: hybrid + ZipMo planner", "main__planner", "planner", "main_planner_p{n}", True),
    ("zlat: z as cross-attn tokens, planner", "zlat__planner", "planner", "zlat_planner_p{n}", True),
    ("Main, oracle z (upper bound)", "main__oracle", "oracle", "main_oracle", False),
    ("zlat, oracle z (upper bound)", "zlat__oracle", "oracle", "zlat_oracle", False),
    ("Static video (first frame repeated)", "static", "static", None, False),
    ("ZipMo planner tracks alone (no video)", "main__planner", "zipmo_only", None, True),
    ("Real video (GT)", None, None, "gt", False),
]
REF = "Main: hybrid + ZipMo planner"
MOTION = ["EPE", "Mean_MSE"]
VB = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree",
      "aesthetic_quality", "imaging_quality", "i2v_subject", "i2v_background"]
VB_SHORT = ["subj", "bg", "flicker", "smooth", "dynamic", "aesth", "imaging", "i2v-subj", "i2v-bg"]


def load(n):
    """-> {label: {metric: {clip: value}}} for n pokes."""
    out = {}
    for label, bf, key, vtag, _ in METHODS:
        m = {}
        if bf and os.path.exists(f"results/bench/{bf}.json"):
            per = json.load(open(f"results/bench/{bf}.json"))["per_clip"].get(f"{key}_p{n}", {})
            for k in MOTION:
                if per:
                    m[k] = {c: v[k] for c, v in per.items()}
        if vtag and os.path.exists(f"results/vbench/{vtag.format(n=n)}.json"):
            per = json.load(open(f"results/vbench/{vtag.format(n=n)}.json"))["per_video"]
            for k in VB:
                if k in per:
                    m[k] = per[k]
        if m:
            out[label] = m
    return out


def boot(a, b, iters=10000, seed=0):
    """paired: mean(a - b) and its 95% CI over clips present in both."""
    cs = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in cs])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (iters, len(d)))].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), len(d)


def fmt_delta(r, prec):
    m, lo, hi, _ = r
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{m:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]{star}"


def main():
    os.chdir(ROOT)
    md, js = [], {}
    for n in (1, 4):
        data = load(n)
        js[f"p{n}"] = {lab: {k: float(np.mean(list(v.values()))) for k, v in m.items()} for lab, m in data.items()}
        ref = data.get(REF, {})
        md += [f"## {n} poke{'s' if n > 1 else ''}", "",
               "### Motion following (lower is better)", "",
               "| Method | n | EPE (px) | Δ EPE vs main [95% CI] | MSE | Δ MSE vs main [95% CI] |",
               "|---|---|---|---|---|---|"]
        for lab, m in data.items():
            if "EPE" not in m:
                continue
            row = [lab, str(len(m["EPE"]))]
            for k, p in (("EPE", 2), ("Mean_MSE", 1)):
                row.append(f"{np.mean(list(m[k].values())):.{p}f}")
                row.append(fmt_delta(boot(m[k], ref[k]), p) if lab != REF and k in ref else "—")
            md.append("| " + " | ".join(row) + " |")
        md += ["", "### VBench (higher is better; dynamic = fraction of videos with motion)", "",
               "| Method | " + " | ".join(VB_SHORT) + " |", "|---|" + "---|" * len(VB)]
        for lab, m in data.items():
            if VB[0] not in m:
                continue
            cells = []
            for k in VB:
                if k not in m:
                    cells.append("")
                    continue
                v = np.mean(list(m[k].values()))
                s = f"{v:.2f}" if k == "imaging_quality" else f"{v:.3f}"
                if lab != REF and k in ref:
                    mm, lo, hi, _ = boot(m[k], ref[k])
                    s += "*" if lo > 0 or hi < 0 else ""
                cells.append(s)
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md += ["", "`*` = differs from the main method (planner) by a paired bootstrap 95% CI that excludes 0.", ""]
        if ref:
            md += ["VBench Δ vs main [95% CI]:", "", "| Method | " + " | ".join(VB_SHORT) + " |",
                   "|---|" + "---|" * len(VB)]
            for lab, m in data.items():
                if lab == REF or VB[0] not in m:
                    continue
                cells = [fmt_delta(boot(m[k], ref[k]), 2 if k == "imaging_quality" else 3)
                         if k in m and k in ref else "" for k in VB]
                md.append(f"| {lab} | " + " | ".join(cells) + " |")
            md.append("")
    open("results/summary.md", "w").write("# Pexels benchmark (68 clips)\n\n" + "\n".join(md))
    json.dump(js, open("results/summary.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
