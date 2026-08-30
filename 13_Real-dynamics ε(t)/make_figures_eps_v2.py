# Regenerate eps_t_real figure with CORRECTED (v2) statement-layer series.
# v1 statement numbers were computed on wrong entities (pageid != QID bug) and are superseded.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/agents/output/pvb_pilot"
eps_t = json.load(open(f"{OUT}/eps_t.json"))
eps_text = eps_t["text"]                      # text layer: unaffected by the bug
stmt = json.load(open(f"{OUT}/eps_stmt_v2.json"))  # corrected statement layer (v2, n=1739)
wfd = json.load(open(f"{OUT}/wfd_popularity_workload.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# ---- (a) eps(t) ----
m_t = [r["months"] for r in eps_text]
e_t = [r["eps_text"] * 100 for r in eps_text]
ci_t = [r["eps_text_ci95"] for r in eps_text]
ax1.errorbar(m_t, e_t,
             yerr=[[e - c[0] * 100 for e, c in zip(e_t, ci_t)],
                   [c[1] * 100 - e for e, c in zip(e_t, ci_t)]],
             marker="o", ms=4, lw=1.4, color="tab:blue", capsize=2.5,
             label="sentence-level $\\varepsilon(t)$ (enwiki text)")

m_s = [r["months"] for r in stmt]
e_s = [r["eps_stmt_exschema"] * 100 for r in stmt]
ci_s = [r["eps_stmt_exschema_ci95"] for r in stmt]
ax1.errorbar(m_s, e_s,
             yerr=[[e - c[0] * 100 for e, c in zip(e_s, ci_s)],
                   [c[1] * 100 - e for e, c in zip(e_s, ci_s)]],
             marker="s", ms=4, lw=1.4, color="tab:red", capsize=2.5,
             label="statement-level $\\varepsilon(t)$ ex-schema (Wikidata)")

l5 = [r["lead5_cert_break"] * 100 for r in eps_text]
ax1.plot(m_t, l5, "^:", color="tab:gray", ms=4, lw=1.2,
         label="lead-5 certificate breakage")

ax1.set_xlabel("staleness horizon $t$ (months)")
ax1.set_ylabel("erasure rate (%)")
ax1.set_xticks([1, 2, 3, 4])
ax1.set_title("(a) Measured premise erasure $\\varepsilon(t)$", fontsize=9)
ax1.legend(fontsize=7.0, loc="upper left")
ax1.set_xlim(0.8, 4.2); ax1.set_ylim(0, 21)

# ---- (b) popularity deciles (WikiFactDiff, unaffected) ----
rows = wfd["rows"]
dec = [r["decile"] for r in rows]
rate = [r["replace_rate"] * 100 for r in rows]
medpop = [r["median_log_pop"] for r in rows]
colors = ["tab:purple" if 2 <= d <= 5 else "tab:gray" for d in dec]
ax2.bar(dec, rate, color=colors, width=0.72)
for d, r_, p_ in zip(dec, rate, medpop):
    ax2.text(d, r_ + 0.35, f"{p_:.1f}", ha="center", fontsize=6.5, color="0.35")
ax2.set_xlabel("subject popularity decile")
ax2.set_ylabel("fact replacement rate (%)")
ax2.set_title("(b) Fact replacements by popularity", fontsize=9)
ax2.set_xticks(dec)
ax2.set_ylim(0, 26)
ax2.annotate(f"overall {wfd['unweighted']['rate_26mo']*100:.1f}% / 26 mo\n(labels: median log page views)",
             xy=(0.97, 0.95), xycoords="axes fraction", fontsize=6.8, ha="right", va="top")

fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/eps_t_real_2026_v2.pdf")
fig.savefig(f"{OUT}/eps_t_real_2026_v2.png", dpi=200)
print("saved eps_t_real_2026_v2.pdf/png")
print("stmt ex-schema series (%):", [f"{r['months']}mo {r['eps_stmt_exschema']*100:.3f}" for r in stmt])
print("entity-level ex-schema (%):", [f"{r['months']}mo {r['entity_any_removal_exschema']*100:.2f}" for r in stmt])
