"""第 5 步（可选，与数据无关）：条形码与稳定秩的最小教学例子。

6 个点，位置 0, 1, 2, 6, 7, 12，距离就是坐标差，全部可手算。
四格图：点云 -> 单连接树状图 -> H0 条形码 -> 稳定秩。
给 paper 里解释概念用，几秒钟。
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from pathlib import Path

OUT_DIR = Path("figs"); OUT_DIR.mkdir(exist_ok=True)

P = np.array([0., 1., 2., 6., 7., 12.])
NAMES = list("ABCDEF")
TEAL, BROWN, GREY = "#1f5c54", "#d4762a", "#9aa0a6"

Z = linkage(pdist(P.reshape(-1, 1)), method="single")
deaths = sorted(Z[:, 2])                      # [1, 1, 1, 4, 5]
TMAX = 6.5

fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))

# (a) 点云 + 两个尺度下的球
a = ax[0, 0]
for t, yy, col in ((1.0, 0.55, BROWN), (4.0, 0.0, TEAL)):
    for p in P:
        a.add_patch(plt.Circle((p, yy), t / 2, color=col, alpha=0.13, lw=0))
    a.plot(P, [yy] * 6, "o", color=col, ms=7, zorder=3)
    a.text(-2.4, yy, f"$t={t:g}$", color=col, va="center", fontsize=10)
for p, nm in zip(P, NAMES):
    a.text(p, 1.05, nm, ha="center", fontsize=9)
a.set_xlim(-3, 14); a.set_ylim(-1.5, 1.5); a.set_yticks([])
a.set_xlabel("position  (think: HP)")
a.set_title("(a) six runs; balls of radius $t/2$", fontsize=11, loc="left")
a.set_aspect("equal", adjustable="box")

# (b) 树状图
b = ax[0, 1]
dendrogram(Z, labels=NAMES, ax=b, color_threshold=0, above_threshold_color=TEAL)
b.set_ylim(0, TMAX); b.set_ylabel("$t$ (merge height)")
b.set_title("(b) single linkage: blocks only merge", fontsize=11, loc="left")
for h in (1, 4, 5):
    b.axhline(h, color=GREY, ls=":", lw=0.8)

# (c) 条形码
c = ax[1, 0]
bars = deaths + [np.inf]
for i, d in enumerate(bars):
    end = TMAX if np.isinf(d) else d
    c.plot([0, end], [i, i], color=TEAL, lw=5, solid_capstyle="butt")
    if np.isinf(d):
        c.annotate("", xy=(TMAX + 0.45, i), xytext=(TMAX, i),
                   arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
        c.text(TMAX - 0.2, i + 0.34, "infinite bar", fontsize=8,
               color=TEAL, ha="right")
    else:
        c.text(end + 0.15, i, f"length {d:g}", fontsize=8, va="center")
c.set_xlim(0, TMAX + 0.8); c.set_ylim(-0.7, 5.9)
c.set_yticks(range(6)); c.set_yticklabels([f"bar {i+1}" for i in range(6)], fontsize=8)
c.set_xlabel("$t$")
c.set_title(r"(c) $H_0$ barcode: all bars born at $t=0$", fontsize=11, loc="left")

# (d) 稳定秩
d = ax[1, 1]
ts = np.linspace(0, TMAX, 2000)
f = np.array([sum(1 for x in bars if x > t) for t in ts])
d.step(ts, f, where="post", color=BROWN, lw=2.2)
for h, lab in ((1, "3 bars die"), (4, "1 dies"), (5, "1 dies")):
    d.axvline(h, color=GREY, ls=":", lw=0.8)
    d.text(h + 0.1, 5.6, lab, fontsize=8, color=GREY)
d.set_xlim(0, TMAX); d.set_ylim(0, 6.6); d.set_yticks(range(7))
d.set_xlabel("$t$"); d.set_ylabel(r"$f(t)$")
d.set_title(r"(d) stable rank  $f(t)=\#\{$bars longer than $t\}=|X/d_t|$",
            fontsize=11, loc="left")

fig.tight_layout()
fig.savefig(OUT_DIR / "barcode_demo.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "barcode_demo.png", dpi=170, bbox_inches="tight")
print("deaths:", deaths)
print("f(t) at t = 0, 0.5, 1, 2, 4, 5, 6:",
      [sum(1 for x in bars if x > t) for t in (0, .5, 1, 2, 4, 5, 6)])
print(f"图写入 {OUT_DIR}/barcode_demo.pdf")