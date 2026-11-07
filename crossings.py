"""从 results/ 的 npz 读出 f_a 曲线，做两件事：

1. 检验「高进阶的对局是否彼此更不相像」。
   统计量取下降段中部的穿越点
       t_L(a) = inf{ t : f_a(t) <= L }
   L = 1000 时就是 MST 边长的中位数。它由一千条边决定，不是尾部那一两条
   离群局，所以稳健。t_L 随 a 上升 = 对局更分散；持平 = 假说不成立。

2. 重画 f_a 曲线图，横轴截断到信息区，去掉噪声尾巴。

不重算任何东西，只读 npz。几秒钟。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RES     = Path("results")
OUT     = Path("figs")
TAG     = "full_noT"          # 与 2_kappa.py 的 TAG 一致
STRATA  = ["all", "clear"]
LEVELS  = list(range(21))
CROSS   = [1000, 100, 10]     # 穿越水平（N=2000，所以 1000 是中位）
TRUNC   = 80                  # 横轴截断，单位血量

OUT.mkdir(exist_ok=True)

def load(stratum):
    p = RES / f"kappa_{stratum}_{TAG}.npz"
    if not p.exists():
        cands = sorted(RES.glob(f"kappa_{stratum}_*.npz"))
        if not cands:
            return None
        p = cands[0]
        print(f"（没找到 {TAG}，改用 {p.name}）")
    z = np.load(p)
    return {a: z[f"f{a:02d}"] for a in LEVELS if f"f{a:02d}" in z}

def crossing(arr, L):
    """第一个使 f(t) <= L 的 t。arr 是 (t, 值) 的断点表，f 非增。"""
    m = arr[:, 1] <= L
    return float(arr[m.argmax(), 0]) if m.any() else float("nan")

# ---------------------------------------------------------------- 表
store = {}
for stratum in STRATA:
    cur = load(stratum)
    if cur is None:
        print(f"{stratum}: 缺 npz，跳过"); continue
    store[stratum] = cur

    print(f"\n{'='*64}\n分层 {stratum}：f_a 的穿越点 t_L(a)，单位血量\n{'='*64}")
    print(f"{'级':<5}" + "".join(f"{'f<='+str(L):>12}" for L in CROSS))
    tab = {L: [] for L in CROSS}
    for a in sorted(cur):
        row = [crossing(cur[a], L) for L in CROSS]
        for L, v in zip(CROSS, row):
            tab[L].append(v)
        print(f"A{a:02d} " + "".join(f"{v:>12.1f}" for v in row))

    print(f"\n{'水平':<10}{'最小':>8}{'最大':>8}{'极差':>8}{'对 a 的斜率':>14}")
    for L in CROSS:
        v = np.array(tab[L], dtype=float)
        k = np.polyfit(np.arange(len(v)), v, 1)[0]
        print(f"f<={L:<7}{np.nanmin(v):>8.1f}{np.nanmax(v):>8.1f}"
              f"{np.nanmax(v)-np.nanmin(v):>8.1f}{k:>14.3f}")
    store[stratum + "_tab"] = tab

print("""
读法：
  斜率明显为正、且极差大于同级抽样波动  -> 高进阶的对局确实彼此更不相像
  斜率接近 0                            -> 假说不成立，各级的离散尺度一样
  f<=1000 那一行最可信（由一千条 MST 边决定）；f<=10 那一行已接近尾部，谨慎。
""")

# ---------------------------------------------------------------- 图
for stratum in STRATA:
    if stratum not in store:
        continue
    cur, tab = store[stratum], store[stratum + "_tab"]
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # 左：截断后的 f_a
    for a in sorted(cur):
        arr = cur[a]
        t = np.append(arr[:, 0], TRUNC)
        v = np.append(arr[:, 1], arr[-1, 1])
        ax1.step(t, v, where="post", color=cmap(a / 20), lw=1.1)
    for L in CROSS:
        ax1.axhline(L, color="0.6", ls=":", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_xlim(0, TRUNC)
    ax1.set_xlabel(r"$t$   (Chebyshev radius, absolute HP)")
    ax1.set_ylabel(r"$\bar f_a(t)$")
    ax1.set_title(f"Stable rank, truncated at $t={TRUNC}$  ({stratum})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 20))
    fig.colorbar(sm, ax=ax1, label="Ascension level $a$", pad=0.02)

    # 右：穿越点随 a
    for L, mk in zip(CROSS, ["o-", "s-", "^--"]):
        v = np.array(tab[L], dtype=float)
        ax2.plot(range(len(v)), v, mk, ms=4, lw=1.4,
                 label=rf"$t$ at $\bar f_a={L}$")
    ax2.set_xticks(range(0, 21, 2))
    ax2.set_xlabel("Ascension level $a$")
    ax2.set_ylabel(r"$t$  (HP)")
    ax2.set_title("Where the curve crosses a fixed height")
    ax2.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / f"fbar_trunc_{stratum}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fbar_trunc_{stratum}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"图写入 {OUT}/fbar_trunc_{stratum}.pdf")