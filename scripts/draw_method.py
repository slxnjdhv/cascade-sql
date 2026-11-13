"""Method overview figure for the paper (vector PDF + PNG preview).
usage: python scripts/draw_method.py ../paper/figures/method_overview"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "method_overview"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")

INK, MUTED = "#1f2933", "#52606d"
FILL = {"in": "#eef2f7", "model": "#dbe8f6", "sig": "#e8f3ea", "gate": "#fff2d6", "out": "#f6e3e3", "ok": "#e3f1e6"}


def box(x, y, w, h, title, body="", kind="in"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=FILL[kind], ec=MUTED, lw=0.8))
    if body:
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", color=INK, fontweight="bold")
        ax.text(x + w / 2, y + (h - 3.2) / 2, body, ha="center", va="center", color=INK, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color=INK, fontweight="bold", linespacing=1.3)


def arrow(x1, y1, x2, y2, text=None, tx=None, ty=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 color=MUTED, lw=0.9, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx, ty, text, ha="center", va="center", color=MUTED, fontsize=6.8, style="italic")


Y, H = 38, 14
box(0, Y, 13, H, "Input", "question\n+ evidence\n+ schema", "in")
box(16, Y, 19, H, "Small model $M_s$", "0.5B / 1.5B, 4-bit\ngreedy + 4 samples\n+ one repair", "model")
box(38, Y, 15, H, "Execute", "all candidates\n(read-only DB)", "model")
box(56, Y, 23, H, "Confidence signals", "SAC / FTC / SLC (tokens)\nresult consistency\nsub-clause agreement\nexecution flags, priors", "sig")
box(82, Y, 18, H, "Gate $p=g(x)$", "logistic regression\n+ Platt calibration", "gate")
for x1, x2 in ((13, 16), (35, 38), (53, 56), (79, 82)):
    arrow(x1 + 0.4, Y + H / 2, x2 - 0.4, Y + H / 2)

Y2, H2 = 19, 11
box(82, Y2, 18, H2, "Answer locally", "$p \\geq \\tau_1$", "ok")
box(56, Y2, 23, H2, "Escalate to $M_\\ell$", "3B, 4-bit, greedy + repair", "model")
box(34, Y2, 18, H2, "Stage-2 gate", "$p_2=g_2(z)$\nscores $M_\\ell$'s answer", "gate")
box(10, Y2 + 6, 20, 8.5, "Show $M_\\ell$ answer", "$p_2 \\geq \\tau_2$", "ok")
box(10, Y2 - 5.5, 20, 8.5, "Abstain", "$p_2 < \\tau_2$", "out")
arrow(91, Y - 0.4, 91, Y2 + H2 + 0.4, "yes", 93.5, (Y + Y2 + H2) / 2)
arrow(86, Y - 0.4, 72, Y2 + H2 + 0.4, "no", 77, (Y + Y2 + H2) / 2 + 1, rad=0.1)
arrow(55.6, Y2 + H2 / 2, 52.4, Y2 + H2 / 2)
arrow(33.6, Y2 + H2 / 2 + 1.5, 30.4, Y2 + 10)
arrow(33.6, Y2 + H2 / 2 - 1.5, 30.4, Y2 - 1.5)

ax.add_patch(FancyBboxPatch((0, 0), 100, 9.5, boxstyle="round,pad=0.2,rounding_size=1.0", fc=FILL["in"], ec=MUTED, lw=0.8))
ax.text(50, 4.75, "Thresholds $(\\tau_1, \\tau_2)$ are chosen on held-out calibration data: a fixed escalation budget,\n"
        "the utility $U$ (+1 correct, $-\\lambda$ wrong, 0 abstain, $-\\kappa$ per escalation), or a Clopper\u2013Pearson risk certificate.\n"
        "All models run in 4-bit on an 8 GB laptop; no data leaves the device.", ha="center", va="center", color=INK, linespacing=1.35)
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print(out)
