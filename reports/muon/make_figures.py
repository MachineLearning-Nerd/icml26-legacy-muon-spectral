"""Generate the report figures from the committed result CSVs (static PNGs)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

def load_sweep():
    rows = []
    for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sweep_results.csv"))):
        r["lr_matrix"] = float(r["lr_matrix"]); r["final_val_loss"] = float(r["final_val_loss"])
        rows.append(r)
    return rows

rows = load_sweep()
by = {}
for r in rows:
    by.setdefault(r["variant"], []).append(r)
for v in by:
    by[v].sort(key=lambda r: r["lr_matrix"])

# Fig 1 — C3: mSGD vs mSGDZ val loss across LRs (robustness)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for v, marker, col in [("mSGD", "o", "#d62728"), ("mSGDZ", "s", "#2ca02c")]:
    xs = [r["lr_matrix"] for r in by[v]]; ys = [r["final_val_loss"] for r in by[v]]
    ax.plot(xs, ys, marker=marker, lw=2, ms=8, label=f"{v} ({'Muon' if v=='mSGDZ' else 'momentum SGD'})", color=col)
ax.axhline(4.575, ls=":", color="gray", label="random init (ln 97)")
ax.set_xscale("log"); ax.set_xlabel("matrix learning rate (log)"); ax.set_ylabel("final val loss")
ax.set_title("C3 — Muon stabilizes momentum across LRs\n(mSGD blows up; mSGDZ stays flat)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_stabilization.png"), dpi=130); plt.close(fig)

# Fig 2 — best val loss per variant (all 8), colored by family
order = ["Adam", "AdamS", "AdamQ", "AdamZ", "mSGD", "mSGDS", "mSGDQ", "mSGDZ"]
bests = {v: min(r["final_val_loss"] for r in by[v]) for v in order}
colors = ["#1f77b4"]*4 + ["#d62728"]*4
fig, ax = plt.subplots(figsize=(7.0, 4.0))
bars = ax.bar(order, [bests[v] for v in order], color=colors, edgecolor="black", lw=.5)
ax.set_ylabel("best val loss (tuned LR)")
ax.set_title("All 8 official optimizers @ best LR — RMS family (blue) vs momentum (red)")
for b, v in zip(bars, order):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{bests[v]:.2f}", ha="center", fontsize=8)
ax.set_ylim(2.0, 3.4); ax.grid(axis="y", alpha=0.3)
rms = [bests[v] for v in ["Adam","AdamS","AdamQ","AdamZ"]]
mom = [bests[v] for v in ["mSGD","mSGDS","mSGDQ","mSGDZ"]]
ax.text(0.01, 0.97, f"RMS spread {max(rms)-min(rms):.2f}\nMomentum spread {max(mom)-min(mom):.2f}",
        transform=ax.transAxes, va="top", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray"))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_variants.png"), dpi=130); plt.close(fig)

# Fig 3 — C4: RMS vs momentum family spread (bar)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
labels = ["RMS family\n(Adam p∈{1,½,¼,0})", "Momentum family\n(mSGD p∈{1,½,¼,0})"]
spreads = [max(rms)-min(rms), max(mom)-min(mom)]
ax.bar(labels, spreads, color=["#1f77b4", "#d62728"], edgecolor="black", lw=.5)
ax.set_ylabel("best-loss spread across the 4 variants")
ax.set_title("C4 — RMS normalization collapses\nthe spectral-variant gap (9×)")
for i, s in enumerate(spreads):
    ax.text(i, s+0.01, f"{s:.2f}", ha="center", fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c4_spread.png"), dpi=130); plt.close(fig)

# Fig 4 — C5: compute barrier (log time to 200k steps)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
steps = [30, 200000]
ax.bar(["Adam\n(12.0 s/step)", "mSGDZ / Muon\n(157.6 s/step)"],
       [12.0*200000/86400, 157.6*200000/86400],
       color=["#1f77b4", "#2ca02c"], edgecolor="black", lw=.5)
ax.axhline(1.0, ls=":", color="gray")
ax.set_ylabel("days to 200k steps (1-thread CPU)")
ax.set_title("C5 — GPT-2 124M compute barrier\n(paper used 8 GPUs; CPU ≈ infeasible)")
ax.set_yscale("log")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_barrier.png"), dpi=130); plt.close(fig)

print("wrote figures:", sorted(os.listdir(OUT)))
