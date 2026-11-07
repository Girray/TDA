"""第 4 步：画 21 个进阶等级的通关率曲线（带标准误）。

默认读 pools/_meta.json，也就是**剔除超上限作弊局之后**的计数，
与第 2 步的拓扑分析用的是同一批局，口径完全一致。
若还没跑第 1 步，退回读 clean2/_clean_stats.json（剔除前的数，差约 0.01pp）。

通关 = success = victory 或 floor_reached >= 52（打穿第三幕）。
几秒钟。输出 figs/clear_rate.pdf 和 .png，并打印一张表。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

META    = Path("pools/_meta.json")
STATS   = Path("clean2/_clean_stats.json")
OUT_DIR = Path("figs")
OUT_DIR.mkdir(exist_ok=True)

# ---------- 读计数 ----------
rows = {}                      # a -> (n_keep, n_succ, n_over_cap)
if META.exists():
    src = "pools/_meta.json（已剔除超上限作弊局）"
    per = json.loads(META.read_text(encoding="utf-8"))["per_level"]
    for a in range(21):
        s = per.get(f"a{a:02d}")
        if s:
            rows[a] = (s["n_all"], s["n_clear"], s.get("n_over_cap", 0))
elif STATS.exists():
    src = "clean2/_clean_stats.json（剔除前）"
    per = json.loads(STATS.read_text(encoding="utf-8"))["summary"]
    for a in range(21):
        s = per.get(str(a)) or per.get(a)
        if s:
            rows[a] = (s["n_keep"], s["n_succ"], 0)
else:
    raise SystemExit("找不到 pools/_meta.json 或 clean2/_clean_stats.json")

print(f"数据来源：{src}\n")

lv = np.array(sorted(rows))
n  = np.array([rows[a][0] for a in lv], dtype=float)
k  = np.array([rows[a][1] for a in lv], dtype=float)
bad = np.array([rows[a][2] for a in lv])
p  = k / n
se = np.sqrt(p * (1 - p) / n)
p, se = p * 100, se * 100

print(f"{'级':<5}{'保留':>12}{'通关':>10}{'通关率':>10}{'SE (pp)':>10}{'超上限':>9}")
for a, ni, ki, pi, si, bi in zip(lv, n, k, p, se, bad):
    print(f"A{a:02d} {ni:>12,.0f}{ki:>10,.0f}{pi:>9.2f}%{si:>10.3f}{bi:>9,}")
print(f"\n合计 {n.sum():,.0f} 局，其中通关 {k.sum():,.0f}，"
      f"超上限剔除 {bad.sum():,}")

# 真实难度必然随等级递增，所以观测通关率上升的台阶是反常点
print("\n上升的台阶（观测通关率反而变高）：")
for i in range(len(lv) - 1):
    d = p[i + 1] - p[i]
    if d > 0:
        sd = np.sqrt(se[i] ** 2 + se[i + 1] ** 2)
        print(f"  A{lv[i]:02d} -> A{lv[i+1]:02d}  {p[i]:6.2f}% -> {p[i+1]:6.2f}%"
              f"   +{d:.2f} pp  = {d/sd:5.1f} 个标准误")

# 最陡的三处下降，作为「哪条规则真正改变了难度」的候选
drops = sorted(((p[i] - p[i + 1], lv[i]) for i in range(len(lv) - 1)), reverse=True)
print("\n最陡的三处下降：")
for d, a in drops[:3]:
    print(f"  A{a:02d} -> A{a+1:02d}   -{d:.2f} pp")

# ---------- 作图 ----------
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.errorbar(lv, p, yerr=3 * se, fmt="o-", color="#1f5c54", ms=5, lw=1.6,
            capsize=3, ecolor="#9aa0a6", label=r"clear rate $\pm\,3\,$SE")
ax.set_xticks(lv)
ax.set_xticklabels([f"A{a}" for a in lv], fontsize=8)
ax.set_xlabel("Ascension level")
ax.set_ylabel("Clear rate (%)")
ax.set_ylim(bottom=0)
ax.set_title(f"Ironclad clear rate by Ascension level  "
             f"($n = {n.sum()/1e6:.1f}$M cleaned runs)")
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "clear_rate.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "clear_rate.png", dpi=160, bbox_inches="tight")
print(f"\n图写入 {OUT_DIR}/clear_rate.pdf")