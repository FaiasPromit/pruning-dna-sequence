#!/usr/bin/env python3
"""
Figure 6 of the paper: what pruning removes, and when, for two pretraining settings.

  row 1, DB2 on MSG   (a) unpruned   (b) pruned   (c) active corpus
  row 2, Hyena on HRG (d) unpruned   (e) pruned   (f) active corpus

The first two panels of each row show how many sequences fall in each accuracy
range at several epochs, with accuracy on the vertical axis. The third panel shows
how much of the corpus is still trained on at each epoch. The counts are numbers of
sequences rather than shares of the corpus, so the unpruned and pruned panels of a
row can be compared directly. The dotted lines mark the pruning threshold.

The inputs are the tracking folders written by the pretraining scripts
(mlm_tracking for DB2, clm_tracking for Hyena), which hold the per-epoch
epoch_*_accuracy_dist.txt files.

Usage
-----
    python3 plot_pruning_dynamics.py \\
        --db2-base /path/to/outputs/pretrain_DB2-MSG_Unpruned/mlm_tracking \\
        --db2-pruned "SP_20%:/path/to/outputs/pretrain_DB2-MSG_SP_20%/mlm_tracking" \\
        --hyena-base /path/to/outputs/pretrain_HY-HRG_Unpruned/clm_tracking \\
        --hyena-pruned "SP_50%:/path/to/outputs/pretrain_HY-HRG_SP_50%/clm_tracking" \\
        -o figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

ROW = re.compile(r"^\s*(\d+)%\s+[\d.]+%\s+([\d,]+)"
                 r"(?:\s+(?:KEEP|PRUNE CAND))?\s*$")
PRUNED_COLS = ["#C62828", "#EF6C00", "#6A1B9A"]


def parse_counts(path):
    c = np.zeros(101)
    for line in path.read_text().splitlines():
        m = ROW.match(line)
        if m:
            c[int(m.group(1))] = float(m.group(2).replace(",", ""))
    return c


def load_counts(d):
    """{epoch: counts per accuracy percent}. Uses the pruning table when there is
    no distribution file. Epochs with no active sequences left are kept, so a run
    that pruned its whole corpus still shows up."""
    out = {}
    d = Path(d)
    if not d.is_dir():
        raise SystemExit(f"not a directory: {d}")
    for pat in ("epoch_*_accuracy_dist.txt", "epoch_*_pruning_table.txt"):
        for f in sorted(d.glob(pat)):
            m = re.search(r"epoch_(\d+)_", f.name)
            if m and int(m.group(1)) not in out:
                out[int(m.group(1))] = parse_counts(f)
    if not out:
        raise SystemExit(f"no epoch_*.txt tracking files under {d}")
    return out


def parse_runs(spec):
    """'label:dir,label:dir' -> [(label, counts), ...]"""
    runs = []
    for part in spec.split(","):
        if not part.strip():
            continue
        label, _, path = part.partition(":")
        if not path:
            raise SystemExit(f"expected 'label:dir', got '{part}'")
        runs.append((label.strip(), load_counts(path.strip())))
    return runs


def split_legend(ax, first_col, **kw):
    """Two-column legend with exactly first_col entries on the left.

    Matplotlib fills the columns in order, ceil(n / 2) entries each, so the
    break cannot be set directly. Invisible entries are inserted to pad each
    column to the same height: after first_col to finish the left column, and
    at the end to finish the right one.
    """
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)
    if not first_col or first_col >= n:
        return ax.legend(handles, labels, **kw)
    height = max(first_col, n - first_col)
    mid, end = height - first_col, height - (n - first_col)
    blank = lambda k: [Line2D([], [], color="none") for _ in range(k)]
    handles = handles[:first_col] + blank(mid) + handles[first_col:] + blank(end)
    labels = labels[:first_col] + [""] * mid + labels[first_col:] + [""] * end
    return ax.legend(handles, labels, ncol=2, **kw)


def count_fmt(v, _=None):
    if v <= 0:
        return ""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


def style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman No9 L", "Nimbus Roman",
                       "Liberation Serif", "Times New Roman", "Times",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 6.7, "axes.titlesize": 6.7, "axes.labelsize": 6.7,
        "xtick.labelsize": 5.7, "ytick.labelsize": 5.7, "legend.fontsize": 6.2,
        "axes.linewidth": 0.5, "xtick.major.width": 0.4,
        "ytick.major.width": 0.4, "xtick.major.size": 1.6,
        "ytick.major.size": 1.6, "figure.dpi": 300, "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def binned(c, w):
    # The last bin includes 100%. The top few one-percent buckets are usually
    # empty, so a separate 100% bin would draw a dip at the edge that is not real.
    edges = np.append(np.arange(0, 100, w), 101)
    ctr = (edges[:-1] + np.minimum(edges[1:], 100)) / 2
    y = np.array([c[lo:hi].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    return ctr, y


def draw_row(fig, gs, r, base, runs, thr, epochs, metric, model, corpus,
             letters, w, ticks, orient, legend_loc, active_loc="best",
             active_ncol=1):
    N0 = max(c.sum() for c in base.values())
    plabel, pruned = runs[0]
    eps = [e for e in epochs if e in base and e in pruned
           and pruned[e].sum() > 0]
    missing = [e for e in epochs if e not in eps]
    if missing:
        print(f"  {model}: epochs not usable in both runs, skipped {missing}")
    cmap = plt.cm.viridis(np.linspace(0.08, 0.80, max(len(eps), 1)))

    seen, axes = [], []
    for k, (src, lab) in enumerate(((base, "Unpruned"),
                                    (pruned, plabel))):
        ax = fig.add_subplot(gs[r, k])
        for c, e in zip(cmap, eps):
            x, y = binned(src[e], w)
            seen.append(y)
            if orient == "acc-y":
                ax.plot(y, x, color=c, lw=0.75, label=f"ep {e}")
            else:
                ax.plot(x, y, color=c, lw=0.75, label=f"ep {e}")
        ax.set_title(f"({letters[k]}) {model} ({lab})", pad=2.0)
        ax.grid(True, alpha=0.15, lw=0.25)
        ax.tick_params(pad=1.0)
        axes.append(ax)

    allv = np.concatenate(seen)
    nz = allv[allv > 0]
    lim = (nz.min() / 3, nz.max() * 3) if nz.size else None
    for k, ax in enumerate(axes):
        if orient == "acc-y":
            ax.axhline(100 * thr, color="#C62828", ls=":", lw=0.6)
            ax.set_xscale("log")
            # Explicit minor locator, so every log axis in the paper shows
            # the same subdivisions regardless of how many decades it spans.
            ax.xaxis.set_minor_locator(
                mpl.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                      numticks=100))
            if ticks:
                ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(count_fmt))
            if lim:
                ax.set_xlim(*lim)
            ax.set_ylim(0, 100)
            # Ticks placed just inside the limits, labelled 0 and 100, so the
            # end labels do not collide with the title or the axis edge.
            ax.set_yticks([1, 50, 99])
            ax.set_yticklabels(["0", "50", "100"])
            ax.set_xlabel("Sequences", labelpad=0.5)
            if k == 0:
                ax.set_ylabel(f"{metric}" + r"$_{\rm(\%)}$", labelpad=0.5)
            else:
                ax.set_yticklabels([])
        else:
            ax.axvline(100 * thr, color="#C62828", ls=":", lw=0.6)
            ax.set_yscale("log")
            # Explicit minor locator, so every log axis in the paper shows
            # the same subdivisions regardless of how many decades it spans.
            ax.yaxis.set_minor_locator(
                mpl.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                      numticks=100))
            if ticks:
                ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(count_fmt))
            if lim:
                ax.set_ylim(*lim)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_xlabel(f"{metric}" + r"$_{\rm(\%)}$", labelpad=0.5)
            if k == 0:
                ax.set_ylabel("Sequences", labelpad=0.5)
            else:
                ax.set_yticklabels([])
    # Three epochs on the left, the rest in a second column.
    split_legend(axes[0], 3, frameon=False, loc=legend_loc, handlelength=0.8,
                 handletextpad=0.4,
                 columnspacing=0.6, borderpad=0.1, labelspacing=0.1,
                 borderaxespad=0.2)

    # Active corpus, with the epoch on the horizontal axis
    ax = fig.add_subplot(gs[r, 2])
    be = sorted(base)
    ax.plot(be, [100 * base[i].sum() / N0 for i in be], color="#777777",
            lw=0.9, ls="--", label="Baseline")
    for (lab, run), col in zip(runs, PRUNED_COLS):
        e = sorted(run)
        ax.plot(e, [100 * run[i].sum() / N0 for i in e], color=col, lw=1.0,
                label=lab)
    ax.set_xlabel("Epoch", labelpad=0.5)
    ax.set_ylabel("% still trained on", labelpad=0.5)
    ax.set_title(f"({letters[2]}) Active corpus - {model}", pad=2.0)
    ax.set_ylim(0, 112)
    ax.set_xlim(1, max(max(base), max(max(r) for _, r in runs)))
    ax.legend(frameon=False, loc=active_loc, ncol=active_ncol,
              handlelength=1.0, borderpad=0.1,
              labelspacing=0.15, borderaxespad=0.3)
    ax.grid(True, alpha=0.15, lw=0.25)
    ax.tick_params(pad=1.0)

    t = int(100 * thr)
    print(f"  {model}, % of original corpus (baseline / {plabel}):")
    for e in eps:
        hi = (100 * base[e][t:].sum() / N0, 100 * pruned[e][t:].sum() / N0)
        lo = (100 * base[e][:t].sum() / N0, 100 * pruned[e][:t].sum() / N0)
        print(f"    ep {e:>3}  above {t}%: {hi[0]:6.2f} / {hi[1]:6.2f}   "
              f"below: {lo[0]:6.2f} / {lo[1]:6.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db2-base", required=True)
    p.add_argument("--db2-pruned", required=True)
    p.add_argument("--hyena-base", required=True)
    p.add_argument("--hyena-pruned", required=True)
    p.add_argument("--db2-threshold", type=float, default=0.20)
    p.add_argument("--hyena-threshold", type=float, default=0.50)
    p.add_argument("--db2-epochs", default="1,5,10,20,45")
    p.add_argument("--hyena-epochs", default="1,3,7,14,28")
    p.add_argument("--db2-ticks", default="1000,100000,10000000")
    p.add_argument("--hyena-ticks", default="100,10000,1000000")
    p.add_argument("--db2-legend", default=None)
    p.add_argument("--hyena-legend", default=None)
    p.add_argument("--bin", type=int, default=5)
    p.add_argument("--height", type=float, default=1.7)
    p.add_argument("--name", default=None)
    p.add_argument("-o", "--outdir", default=".")
    a = p.parse_args()

    style()
    print(f"serif resolves to: "
          f"{Path(findfont(FontProperties(family='serif'))).name}")

    db, dr = load_counts(a.db2_base), parse_runs(a.db2_pruned)
    hb, hr = load_counts(a.hyena_base), parse_runs(a.hyena_pruned)
    print(f"DB2 baseline {len(db)} epochs; pruned "
          f"{', '.join(f'{l} {len(r)}' for l, r in dr)}")
    print(f"Hyena baseline {len(hb)} epochs; pruned "
          f"{', '.join(f'{l} {len(r)}' for l, r in hr)}")

    fig = plt.figure(figsize=(4.16, a.height), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])

    dleg = a.db2_legend or "upper left"
    hleg = a.hyena_legend or "upper left"
    draw_row(fig, gs, 0, db, dr, a.db2_threshold,
             [int(v) for v in a.db2_epochs.split(",")],
             "MLM accuracy", "DB2 on MSG", "MSG", "abc", a.bin,
             [float(v) for v in a.db2_ticks.split(",")], "acc-y", dleg)
    draw_row(fig, gs, 1, hb, hr, a.hyena_threshold,
             [int(v) for v in a.hyena_epochs.split(",")],
             "CLM accuracy", "Hyena on HRG", "HRG", "def", a.bin,
             [float(v) for v in a.hyena_ticks.split(",")], "acc-y", hleg,
             active_loc="lower right", active_ncol=2)

    fig.get_layout_engine().set(w_pad=0.004, h_pad=0.004,
                                wspace=0.02, hspace=0.02)
    stem = a.name or "pruning_dynamics_accy"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = out / f"{stem}.{ext}"
        fig.savefig(f)
        print(f"saved: {f}")


if __name__ == "__main__":
    main()
