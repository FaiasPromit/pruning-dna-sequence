#!/usr/bin/env python3
"""
Figures 1 and 2 of the paper: per-sequence accuracy as a signal of how easy a
sequence is, in the unpruned DB2-MSG and HY-HRG runs.

  <name>_seq.pdf     Figure 1: accuracy across epochs of ten sequences each from
                     DB2 on MSG (a) and Hyena on HRG (b), chosen to span the
                     accuracy range
  <name>_corpus.pdf  Figure 2: how many sequences of the whole corpus fall in each
                     accuracy range at selected epochs, with accuracy on the
                     vertical axis

The Hyena trajectories are smoother than the DB2 ones because next-token accuracy
is computed over every position, while masked-token accuracy uses only the masked
positions. The script prints the median within-sequence standard deviation of both.

The inputs come from the tracking folders of the unpruned runs (mlm_tracking for
DB2, clm_tracking for Hyena): tracked_sequences.xlsx for the trajectories and the
per-epoch epoch_*_accuracy_dist.txt files for the corpus panels.

Usage
-----
    python3 plot_accuracy_figures.py \\
        --db2-xlsx /path/to/outputs/pretrain_DB2-MSG_Unpruned/mlm_tracking/tracked_sequences.xlsx \\
        --db2-hist /path/to/outputs/pretrain_DB2-MSG_Unpruned/mlm_tracking \\
        --hyena-xlsx /path/to/outputs/pretrain_HY-HRG_Unpruned/clm_tracking/tracked_sequences.xlsx \\
        --hyena-hist /path/to/outputs/pretrain_HY-HRG_Unpruned/clm_tracking \\
        -o figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D

ORIENT = "acc-y"
DB2_LEGEND = "upper right"
HYENA_LEGEND = "upper left"
N_PER_MODEL = 10
N_ROWS = 5
MIN_GAP = 0.02
K_NEAR = 5


def load_xlsx(path):
    ws = openpyxl.load_workbook(path, read_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    ids, mat = [], []
    for r in rows[1:]:
        if r[0] is None:
            continue
        ids.append(r[0])
        mat.append([np.nan if (v is None or isinstance(v, str)) else float(v)
                    for v in r[1:]])
    return np.array(ids), np.array(mat, dtype=float)


def load_hist(d):
    """{epoch: counts per accuracy percent}"""
    out = {}
    if not d:
        return out
    row = re.compile(r"^\s*(\d+)%\s+[\d.]+%\s+([\d,]+)"
                     r"(?:\s+(?:KEEP|PRUNE CAND))?\s*$")
    for pat in ("epoch_*_accuracy_dist.txt", "epoch_*_pruning_table.txt"):
        for f in sorted(Path(d).glob(pat)):
            m = re.search(r"epoch_(\d+)_", f.name)
            if not m or int(m.group(1)) in out:
                continue
            c = np.zeros(101)
            for line in f.read_text().splitlines():
                mm = row.match(line)
                if mm:
                    c[int(mm.group(1))] = float(mm.group(2).replace(",", ""))
            if c.sum():
                out[int(m.group(1))] = c
    return out


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
    """1.2e6 -> 1.2M. Counts are easier to judge than percentages."""
    if v <= 0:
        return ""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


def select(A, n, explicit=None, ids=None):
    """Pick n sequences spread evenly over the accuracy range, preferring steady ones."""
    lvl = np.nanmean(A, axis=1)
    if explicit:
        pos = {int(v): k for k, v in enumerate(ids)}
        want = [int(v) for v in explicit.split(",") if v.strip()]
        miss = [w for w in want if w not in pos]
        if miss:
            print(f"    ids not in file, skipped: {miss}")
        sel = [pos[w] for w in want if w in pos][:n]
        sel.sort(key=lambda k: -lvl[k])
        return sel, lvl
    sd = np.array([np.nanstd(r[~np.isnan(r)]) for r in A])
    targets = np.linspace(np.nanmax(lvl), np.nanmin(lvl), n)
    tol = 0.5 * (targets[0] - targets[-1]) / max(n - 1, 1)
    used = []
    for t in targets:
        order = np.argsort(np.abs(lvl - t))
        free = [int(k) for k in order if k not in used
                and all(abs(lvl[k] - lvl[u]) >= MIN_GAP for u in used)]
        if not free:
            free = [int(k) for k in order if k not in used]
        if not free:
            break
        near = [k for k in free[:K_NEAR] if abs(lvl[k] - t) <= tol]
        used.append(min(near, key=lambda k: sd[k]) if near else free[0])
    used.sort(key=lambda k: -lvl[k])
    return used, lvl


def style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman No9 L", "Nimbus Roman",
                       "Liberation Serif", "Times New Roman", "Times",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "xtick.major.width": 0.4,
        "ytick.major.width": 0.4, "xtick.major.size": 1.6,
        "ytick.major.size": 1.6, "figure.dpi": 300, "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def draw_trajectories(fig, gs, col0, A, sel, lvl, n_ep, last_tick, title,
                      show_ylabel):
    """Ten sequences as a 5x2 block, on a percentage scale so the axis
    matches the population panels."""
    norm = mpl.colors.Normalize(vmin=lvl[sel[-1]], vmax=lvl[sel[0]])
    cmap = mpl.cm.viridis
    x = np.arange(1, n_ep + 1)
    xt = [t for t in (1, 15, 30, n_ep) if t <= n_ep]
    if n_ep <= 30:
        xt = [t for t in (1, 10, 20, n_ep) if t <= n_ep]
    xtl = [str(t) for t in xt]
    xtl[-1] = last_tick
    for i, k in enumerate(sel):
        row, col = i // 2, col0 + i % 2
        ax = fig.add_subplot(gs[row, col])
        colr = cmap(0.06 + 0.64 * (1 - norm(lvl[k])))
        y = A[k] * 100
        ok = ~np.isnan(y)
        ax.axhline(50, color="#C62828", ls=":", lw=0.4, zorder=1)
        ax.plot(x[ok], y[ok], color=colr, marker="o", ms=0.8, lw=0.45,
                zorder=3)
        ax.set_ylim(-4, 104)
        ax.set_xlim(1, n_ep)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(xt)
        ax.grid(True, alpha=0.15, lw=0.25)
        ax.tick_params(pad=1.0)
        ax.set_yticklabels(["0", "50", "100"]
                           if (show_ylabel and i % 2 == 0) else [])
        if row == N_ROWS - 1:
            ax.set_xticklabels(xtl)
            ax.set_xlabel("Epoch", labelpad=1.5)
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title(title, pad=2.0, x=1.0, ha="center")
        if show_ylabel and i == 4:
            ax.set_ylabel("Accuracy (%)", labelpad=1.5)


def draw_population(ax, hist, want, title, metric, orient, w, ticks,
                    legend_loc, first_col=None):
    eps = [e for e in want if e in hist]
    cmap = plt.cm.viridis(np.linspace(0.08, 0.85, max(len(eps), 1)))
    # The last bin includes 100%. The top few one-percent buckets are usually
    # empty, so a separate 100% bin would draw a dip at the edge that is not real.
    edges = np.append(np.arange(0, 100, w), 101)
    ctr = (edges[:-1] + np.minimum(edges[1:], 100)) / 2
    seen = []
    for c, e in zip(cmap, eps):
        y = np.array([hist[e][lo:hi].sum()
                      for lo, hi in zip(edges[:-1], edges[1:])])
        seen.append(y)
        if orient == "acc-y":
            ax.plot(y, ctr, color=c, lw=0.7, label=f"ep {e}")
        else:
            ax.plot(ctr, y, color=c, lw=0.7, label=f"ep {e}")

    lim = None
    if seen:
        allv = np.concatenate(seen)
        nz = allv[allv > 0]
        if nz.size:
            lim = (nz.min() / 3, nz.max() * 3)

    if orient == "acc-y":
        ax.axhline(50, color="#C62828", ls=":", lw=0.5)
        ax.set_xscale("log")
        # Set the minor locator explicitly. Left to matplotlib it drops the
        # minor ticks on axes spanning many decades and keeps them on
        # narrow ones, so panels in the same figure look different.
        ax.xaxis.set_minor_locator(
            mpl.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                  numticks=100))
        if ticks:
            ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(count_fmt))
        if lim:
            ax.set_xlim(*lim)
        ax.set_ylim(0, 100)
        # Ticks placed just inside the limits, labelled 0 and 100, so the end
        # labels do not collide with the title or the axis edge.
        ax.set_yticks([2, 50, 98])
        ax.set_yticklabels(["0", "50", "100"])
        ax.set_ylabel(f"{metric}(%)", labelpad=1.5)
        ax.set_xlabel("Sequences", labelpad=1.5)
    else:
        ax.axvline(50, color="#C62828", ls=":", lw=0.5)
        ax.set_yscale("log")
        # Set the minor locator explicitly. Left to matplotlib it drops the
        # minor ticks on axes spanning many decades and keeps them on
        # narrow ones, so panels in the same figure look different.
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
        ax.set_xlabel(f"{metric}(%)", labelpad=1.5)
        ax.set_ylabel("Sequences", labelpad=1.5)

    ax.set_title(title, pad=2.0)
    ax.grid(True, alpha=0.15, lw=0.25)
    ax.tick_params(pad=1.0)
    # Two legend columns when there are more entries than fit in one
    kw = dict(frameon=False, loc=legend_loc, handlelength=0.8, handletextpad=0.4,
              columnspacing=0.6, borderpad=0.1, labelspacing=0.1,
              borderaxespad=0.2)
    if first_col:
        split_legend(ax, first_col, **kw)
    else:
        ax.legend(ncol=2 if len(eps) > 5 else 1, **kw)
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db2-xlsx", required=True)
    p.add_argument("--db2-hist", required=True)
    p.add_argument("--hyena-xlsx", required=True)
    p.add_argument("--hyena-hist", required=True)
    p.add_argument("--db2-epochs", default="1,2,3,5,10,20,45")
    p.add_argument("--hyena-epochs", default="1,2,3,5,10,20,28")
    p.add_argument("--db2-last-tick", default="47")
    p.add_argument("--hyena-last-tick", default="28")
    p.add_argument("--ids-db2", default=None)
    p.add_argument("--ids-hyena", default=None)
    p.add_argument("--db2-ticks", default="1000,100000,10000000")
    p.add_argument("--hyena-ticks", default="100,10000,1000000")
    p.add_argument("--db2-legend", default=None)
    p.add_argument("--hyena-legend", default=None)
    p.add_argument("--bin", type=int, default=5)
    p.add_argument("--height", type=float, default=2.3,
                   help="Height of the trajectory figure.")
    p.add_argument("--corpus-width", type=float, default=1.55,
                   help="Width of the corpus figure. The default matches\n"
                        "one column of Figure 6 (plot_pruning_dynamics.py).")
    p.add_argument("--corpus-height", type=float, default=1.87,
                   help="Height of the corpus figure. The default is the\n"
                        "two rows of Figure 6 plus ten per cent.")
    p.add_argument("--name", default=None,
                   help="Output file name stem (default: accuracy_discriminator_accy).")
    p.add_argument("-o", "--outdir", default=".")
    a = p.parse_args()

    dids, D = load_xlsx(a.db2_xlsx)
    hids, H = load_xlsx(a.hyena_xlsx)
    DH, HH = load_hist(a.db2_hist), load_hist(a.hyena_hist)
    print(f"DB2 {D.shape[0]} seqs x {D.shape[1]} ep, {len(DH)} hist epochs")
    print(f"Hyena {H.shape[0]} seqs x {H.shape[1]} ep, {len(HH)} hist epochs")

    style()
    print(f"serif resolves to: "
          f"{Path(findfont(FontProperties(family='serif'))).name}")

    # Two separate figures. The trajectory figure (Figure 1) gets the full text
    # width; the corpus panels (Figure 2) are sized like one distribution panel of
    # Figure 6, which uses the same width in three columns and two rows.
    # The trajectory figure carries twenty small panels, so its text is set a
    # point larger than the shared default. Sizes are captured when artists
    # are created, so the whole figure is built inside the context.
    with mpl.rc_context({"font.size": 9.5, "axes.titlesize": 9.5,
                         "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 9}):
        fig1 = plt.figure(figsize=(5.5, a.height), layout="constrained")
        gs = fig1.add_gridspec(N_ROWS, 4)

        dsel, dlvl = select(D, N_PER_MODEL, a.ids_db2, dids)
        hsel, hlvl = select(H, N_PER_MODEL, a.ids_hyena, hids)
        draw_trajectories(fig1, gs, 0, D, dsel, dlvl, D.shape[1],
                          a.db2_last_tick, "(a) DB2 on MSG (10 sequences)", True)
        draw_trajectories(fig1, gs, 2, H, hsel, hlvl, H.shape[1],
                          a.hyena_last_tick, "(b) Hyena on HRG (10 sequences)",
                          False)
        fig1.get_layout_engine().set(w_pad=0.012, h_pad=0.008,
                                 wspace=0.035, hspace=0.06)

    # The corpus figure is smaller than the trajectory one, so its text is
    # set a point smaller. Sizes are captured when the artists are created,
    # so the whole figure is built inside the context.
    with mpl.rc_context({"font.size": 6.5, "axes.titlesize": 6.5,
                         "axes.labelsize": 6.5, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "legend.fontsize": 6}):
        fig2 = plt.figure(figsize=(a.corpus_width, a.corpus_height),
                          layout="constrained")
        pg = fig2.add_gridspec(2, 1)
        dleg = a.db2_legend or DB2_LEGEND
        hleg = a.hyena_legend or HYENA_LEGEND
        ax = fig2.add_subplot(pg[0, 0])
        de = draw_population(ax, DH, [int(v) for v in a.db2_epochs.split(",")],
                             "(a) DB2 on MSG (full corpus)", "MLM accuracy",
                             ORIENT, a.bin,
                             [float(v) for v in a.db2_ticks.split(",")], dleg,
                             first_col=2)
        ax2 = fig2.add_subplot(pg[1, 0])
        he = draw_population(ax2, HH, [int(v) for v in a.hyena_epochs.split(",")],
                             "(b) Hyena on HRG (full corpus)", "CLM accuracy",
                             ORIENT, a.bin,
                             [float(v) for v in a.hyena_ticks.split(",")], hleg)
        fig2.get_layout_engine().set(w_pad=0.012, h_pad=0.008, hspace=0.06)

    stem = a.name or "accuracy_discriminator_accy"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for fig, part in ((fig1, "seq"), (fig2, "corpus")):
        for ext in ("pdf", "png"):
            f = out / f"{stem}_{part}.{ext}"
            fig.savefig(f)
            print(f"saved: {f}")

    def sd(A):
        return float(np.median([np.nanstd(r[~np.isnan(r)]) for r in A]))
    print(f"\nmedian within-sequence sd: DB2 {sd(D):.3f}, "
          f"Hyena {sd(H):.3f}  (ratio {sd(D)/max(sd(H),1e-9):.1f}x)")
    print(f"panel (c) epochs: {de}")
    print(f"panel (d) epochs: {he}")


if __name__ == "__main__":
    main()
