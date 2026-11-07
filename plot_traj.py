"""从 clean2/ 随机抽若干局，画血量轨迹图（第 1-51 层，绝对血量）。

每级一张图：N 条随机轨迹，按结局着色，叠加该级的中位数曲线。
蓄水池抽样，所以是整个等级的均匀随机样本，不是文件开头那几局。

输出 figs/traj_aXX.pdf 和 .png
"""
import gzip, json, random, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR  = Path("clean2")
OUT_DIR = Path("figs")
LEVELS  = [0, 20]      # 要画的等级
N       = 30           # 每张图画几条轨迹
POOL    = 4000         # 蓄水池容量（中位数曲线用这个池算）
SEED    = 0

COLOR = {"success":    ("#2a7f62", "cleared Act 3"),
         "heart_died": ("#4c9f70", "reached Act 4, died"),
         "died":       ("#b3433b", "died"),
         "conceded":   ("#c89b3c", "conceded")}

OUT_DIR.mkdir(exist_ok=True)
t0 = time.time()

for a in LEVELS:
    src = IN_DIR / f"ironclad_a{a:02d}_clean.jsonl.gz"
    if not src.exists():
        print(f"A{a:02d} 缺文件，跳过"); continue

    rng = random.Random(SEED + a)
    res, seen = [], 0
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            seen += 1
            item = (r["traj"], r["end_type"])
            if len(res) < POOL:
                res.append(item)
            else:
                j = rng.randrange(seen)
                if j < POOL:
                    res[j] = item
    print(f"A{a:02d}  扫过 {seen:,} 局，池 {len(res):,}   已用 {(time.time()-t0)/60:.1f} 分")

    T = np.array([t for t, _ in res], dtype=float)
    med = np.median(T, axis=0)
    floors = np.arange(1, 52)

    pick = rng.sample(range(len(res)), min(N, len(res)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    drawn = set()
    for i in pick:
        traj, et = res[i]
        c, lab = COLOR.get(et, ("gray", et))
        ax.plot(floors, traj, color=c, lw=0.9, alpha=0.65,
                label=lab if et not in drawn else None)
        drawn.add(et)
    ax.plot(floors, med, color="black", lw=2.0, label="median (all runs)")

    for f in (16, 33, 50):            # 三个 Boss 层
        ax.axvline(f, color="gray", ls=":", lw=0.8, alpha=0.6)

    ax.set_xlim(1, 51); ax.set_ylim(bottom=0)
    ax.set_xlabel("Floor")
    ax.set_ylabel("HP")
    ax.set_title(f"Ironclad HP trajectories, Ascension {a}  "
                 f"({N} random runs of {seen:,})")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"traj_a{a:02d}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"traj_a{a:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分。图在 {OUT_DIR}/")