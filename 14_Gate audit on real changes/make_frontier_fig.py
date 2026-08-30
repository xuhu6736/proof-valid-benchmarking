# Constructor frontier figure: false-rejection rate (x) vs interception rate (y).
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/agents/output/pvb_pilot"
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
ax.plot(xs, ys, "^--", color="tab:gray", ms=6, lw=1.1, label="lead-$k$ text hash", zorder=2)
for k in lk:
    x, y, xl, xh, yl, yh = pt(k)
    ax.errorbar([x], [y], xerr=[[xl], [xh]], yerr=[[yl], [yh]],
                color="tab:gray", alpha=0.55, capsize=2, lw=0.9, zorder=1)
    ax.annotate(k.replace("lead", "lead-"), (x, y), textcoords="offset points",
                xytext=(5, -9), fontsize=6.5, color="tab:gray")

# constructor family
fam = [("cascade_only", "cascade (val+fuzzy0.9+lead-3)", "tab:blue", "o", 7),
       ("casc_or_dense1", "cascade $\\vee$ dense-1", "tab:green", "s", 7),
       ("casc_or_lost12", "cascade $\\vee$ lost$\\geq$12", "tab:orange", "D", 6),
       ("casc_or_dense1_lost12", "cascade $\\vee$ dense-1 $\\vee$ lost$\\geq$12", "tab:red", "*", 13)]
for name, lab, col, mk, ms in fam:
    x, y, xl, xh, yl, yh = pt(name)
    ax.errorbar([x], [y], xerr=[[xl], [xh]], yerr=[[yl], [yh]],
                color=col, alpha=0.6, capsize=2.5, lw=1.0, zorder=3)
    ax.plot([x], [y], marker=mk, color=col, ms=ms, ls="none", label=lab, zorder=4)

# explored but not headline variants
for name in ["casc_or_lost8", "casc_or_lost9", "casc_or_lost10", "casc_or_dense1_lost10"]:
    x, y, *_ = pt(name)
    ax.plot([x], [y], marker=".", color="tab:red", ms=5, alpha=0.35, zorder=3)
ax.plot([], [], marker=".", color="tab:red", ms=5, alpha=0.5, ls="none",
        label="other OR-trigger variants")

# target region: interception > lead-5, FR ~ 10%
ax.axhline(33.85, color="tab:gray", ls=":", lw=0.9)
ax.annotate("lead-5 interception (33.9%)", xy=(17.3, 33.85), fontsize=6.3,
            color="tab:gray", va="bottom", ha="right")
ax.axvspan(8.0, 12.0, color="tab:blue", alpha=0.06)
ax.annotate("FR $\\approx$ 10% target band", xy=(10.0, 7.5), fontsize=6.3,
            color="tab:blue", ha="center")

ax.set_xlabel("false-rejection rate on unchanged evidence (%)")
ax.set_ylabel("interception rate on harmed evidence (%)")
ax.set_xlim(3, 20); ax.set_ylim(5, 55)
ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)
ax.set_title("Post-hoc certificate frontier, HotpotQA panel ($n_{\\rm harm}{=}130$, $n_{\\rm unch}{=}1{,}381$)",
             fontsize=7.6)
fig.tight_layout()
fig.savefig(f"{OUT}/constructor_frontier_2026.pdf")
fig.savefig(f"{OUT}/constructor_frontier_2026.png", dpi=200)
print("saved constructor_frontier_2026.pdf/png")
