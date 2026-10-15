# Constructor frontier figure: false-rejection rate (x) vs interception rate (y).
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
OUT = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(f"{OUT}/constructor_results_v4.json"))["constructors"]

fig, ax = plt.subplots(figsize=(4.9, 3.3))

def pt(name):
    r = C[name]
    return (r["fr"] * 100, r["int"] * 100,
            (r["fr"] - r["fr_ci"][0]) * 100, (r["fr_ci"][1] - r["fr"]) * 100,
            (r["int"] - r["int_ci"][0]) * 100, (r["int_ci"][1] - r["int"]) * 100)

# lead-k curve (text-hash baseline)
lk = ["lead1", "lead3", "lead5", "lead10"]
xs = [pt(k)[0] for k in lk]; ys = [pt(k)[1] for k in lk]
ax.plot(xs, ys, "^--", color="tab:gray", ms=6, lw=1.1, label="lead-$k$ exact matching", zorder=2)
for k in lk:
    x, y, xl, xh, yl, yh = pt(k)
    ax.errorbar([x], [y], xerr=[[xl], [xh]], yerr=[[yl], [yh]],
                color="tab:gray", alpha=0.55, capsize=2, lw=0.9, zorder=1)
    # lead-10 sits at the right edge: label above-left to avoid clipping and
    # to stay clear of its own error bars and the dashed lead-k line
    off, ha = ((-10, 7), "right") if k == "lead10" else ((5, -9), "left")
    ax.annotate(k.replace("lead", "lead-"), (x, y), textcoords="offset points",
                xytext=off, ha=ha, fontsize=6.5, color="tab:gray")

# constructor family
fam = [("cascade_only", "cascade (val/fuzzy-0.9;\nexact lead-3 fallback)", "tab:blue", "o", 7),
       ("casc_or_dense1", "cascade $\\vee$ dense-1", "tab:green", "s", 7),
       ("casc_or_lost12", "cascade $\\vee$ lost$\\geq$12", "tab:orange", "D", 6),
       ("casc_or_dense1_lost12", "cascade $\\vee$ dense-1 $\\vee$ lost$\\geq$12", "tab:red", "*", 13)]
for name, lab, col, mk, ms in fam:
    x, y, xl, xh, yl, yh = pt(name)
    ax.errorbar([x], [y], xerr=[[xl], [xh]], yerr=[[yl], [yh]],
                color=col, alpha=0.6, capsize=2.5, lw=1.0, zorder=3)
    ax.plot([x], [y], marker=mk, color=col, ms=ms, ls="none", label=lab, zorder=4)

# target region: interception > lead-5, FR ~ 10%
ax.axhline(33.85, color="tab:gray", ls=":", lw=0.9)
ax.annotate("lead-5 detection (33.8%)", xy=(3.2, 33.85), fontsize=6.3,
            color="tab:gray", va="bottom", ha="left")

ax.set_xlabel("proxy-negative flag rate (%)")
ax.set_ylabel("proxy-positive detection (%)")
ax.set_xlim(3, 20); ax.set_ylim(5, 55)
ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
# no in-figure title: the caption carries panel and experiment identity
fig.tight_layout()
fig.savefig(f"{OUT}/constructor_frontier_2026.pdf")
fig.savefig(f"{OUT}/constructor_frontier_2026.png", dpi=200)
print("saved constructor_frontier_2026.pdf/png")
