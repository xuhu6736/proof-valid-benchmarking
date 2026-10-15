#!/usr/bin/env python3
"""Regenerate fig3_or_strict_scis.pdf (Figure 4, disjunctive support) from the
E11 result JSON, reproducing the compact journal layout but with:
  - curve-end direct labels placed in empty space right of the last markers
    (fixes loose/strict OR text overlapping the curves);
  - legend entry updated to the current numbering: bookends (Prop. B.1).
Data: e11_or_strict_results.json (copy of e11_or_strict_results.json, seed 42, 20 reps).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("e11_or_strict_results.json") as f:
    d = json.load(f)

eps = d["eps"]
loose = d["loose"]["meas"]
strict = d["strict"]["meas"]
lower = d["strict"]["lower"]
upper = d["strict"]["upper"]
floor = d["strict"]["and_floor"]

# 167.04 x 104.4 pt  ==  2.32 x 1.45 in  (original compact geometry)
fig, ax = plt.subplots(figsize=(2.32, 1.45))

C_LOOSE, C_STRICT = "#2ca02c", "#1f77b4"

ax.fill_between(eps, np.clip(lower, 0, 1), np.clip(upper, 0, 1),
                color=C_STRICT, alpha=0.15, lw=0, label="strict-family bound refs")
ax.plot(eps, floor, "k--", lw=0.5, label="AND-only exact law")
ax.plot(eps, strict, "-o", color=C_STRICT, ms=2.6, lw=0.5)
ax.plot(eps, loose, "-s", color=C_LOOSE, ms=2.4, lw=0.5)

# direct curve-end labels in the empty margin right of the data
ax.text(0.515, loose[-1], "loose OR", color=C_LOOSE, fontsize=5.5,
        va="center", ha="left")
ax.text(0.515, strict[-1], "strict OR", color=C_STRICT, fontsize=5.5,
        va="center", ha="left")

ax.set_xlim(0.04, 0.64)  # labels end near 0.62; keep clear of the right spine
ax.set_ylim(0.0, 1.0)
ax.set_xticks(eps)
ax.set_xticklabels([str(e) for e in eps], fontsize=5.5)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis="y", labelsize=5.5, length=1.5)
ax.tick_params(axis="x", length=1.5)
ax.set_xlabel(r"erasure probability $\varepsilon$", fontsize=6)
ax.set_ylabel("evidence-family\navailability", fontsize=6)
ax.grid(alpha=0.25, lw=0.3)
ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9,
          borderpad=0.3, labelspacing=0.25, handlelength=1.6)
for s in ax.spines.values():
    s.set_linewidth(0.4)

fig.tight_layout(pad=0.15)
fig.savefig("fig3_or_strict_scis.pdf")
fig.savefig("_z_fig3_regen.png", dpi=300)
print("saved")
