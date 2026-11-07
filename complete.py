"""9_complete.py —— 用 complete linkage 重跑一遍，检验 §4.1 的「无平台」结论。

为什么要跑
  单链接只要有「一局」落在两坨中间就把它们连起来，是所有层次聚类里
  最不容易报告团块的方法。所以「单链接没看到平台」这个证据天然偏弱。
  complete linkage 只在「所有配对都近」时才合并，偏向正好相反——
  它倾向于找出紧凑的簇，哪怕数据里本来没有。

  于是逻辑是不对称的：
    complete 也没平台  ->  强证据，连最爱找簇的方法都找不到
    complete 有平台    ->  不确定，可能只是它的偏向

重要
  complete linkage 在 d_GH 下不连续，课程的稳定性定理不覆盖它。
  它只能当稳健性检查，不能当主干。末尾那个 kappa 同理，仅供参考。

两种 linkage 在「同一批子样本」上算，所以对比严格可比。
只读 pools/，不用 stablebear。整跑约 1-2 分钟。
"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POOL_DIR = Path("pools")
RES_DIR  = Path("results")
OUT_DIR  = Path("figs")
RES_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

LEVELS  = list(range(21))
STRATA  = ["all", "clear"]
METHODS = ["single", "complete"]

N     = 2000              # 与 2_kappa.py 一致
R     = 20
SEED  = 7
KS    = [2, 3, 5, 10, 20, 50]   # 在这些高度上查平台
KMIN  = 3                 # 下降跨度量到 f=KMIN 为止（再往下是一两个离群局）
NGRID = 1500

# ---------------------------------------------------------------- 诊断函数
def plateau_at(h, k):
    """f(t)=k 这个高度上曲线横着走多宽（单位：血）。

    h 是排序后的 N-1 个合并高度。f 在 [h[N-1-k], h[N-k]) 上取值 k，
    所以高度 k 处的平台宽度就是 h[N-k] - h[N-1-k]。
    """
    n = len(h) + 1
    if not (2 <= k <= n - 1):
        return np.nan
    return float(h[n - k] - h[n - 1 - k])

def descent_span(h, kmin=KMIN):
    """f 从 N-1 降到 kmin 所跨的 t，用来给平台宽度当参照。"""
    n = len(h) + 1
    return float(h[n - kmin] - h[0])

# ---------------------------------------------------------------- 计算
t0 = time.time()
heights = {}          # (stratum, method, a) -> R 个 ndarray

for stratum in STRATA:
    for a in LEVELS:
        f = POOL_DIR / f"a{a:02d}_{stratum}.npy"
        if not f.exists():
            print(f"缺 {f.name}，跳过"); continue
        P = np.load(f).astype(np.float64)
        n = min(N, P.shape[0])
        rng = np.random.default_rng(SEED * 1000 + a)
        for m in METHODS:
            heights[(stratum, m, a)] = []
        for _ in range(R):
            idx = rng.choice(P.shape[0], size=n, replace=False)
            cd = pdist(P[idx], metric="chebyshev")
            for m in METHODS:
                Z = linkage(cd.copy(), method=m)     # linkage 会改写输入
                heights[(stratum, m, a)].append(np.sort(Z[:, 2]))
    print(f"{stratum:<6} 完成   ({(time.time()-t0)/60:.1f} 分)")

# ---------------------------------------------------------------- 平台表
print(f"\n{'='*92}")
print("各高度上平台的宽度（血），R 次子抽样取中位数。span = f 从 N 降到 3 所跨的 t")
print("连续谱的平台宽应当远小于 span；真有 k 个分离团块时，k 那一列会接近 span")
print(f"{'='*92}")
for stratum in STRATA:
    for m in METHODS:
        if (stratum, m, 0) not in heights:
            continue
        print(f"\n分层 {stratum}   linkage = {m}")
        print(f"{'级':<5}{'span':>8}" + "".join(f"{'k='+str(k):>9}" for k in KS)
              + f"{'最大占比':>10}")
        worst = 0.0
        for a in LEVELS:
            hs = heights.get((stratum, m, a))
            if not hs:
                continue
            sp = float(np.median([descent_span(h) for h in hs]))
            w  = [float(np.median([plateau_at(h, k) for h in hs])) for k in KS]
            frac = max(x / sp for x in w) if sp > 0 else np.nan
            worst = max(worst, frac)
            print(f"A{a:02d} {sp:>8.1f}" + "".join(f"{x:>9.2f}" for x in w)
                  + f"{frac:>10.3f}")
        print(f"  -> 全部等级中最大的占比：{worst:.3f}")

print("""
怎么读
  占比 < 0.05  ->  没有平台，曲线光滑下降
  占比 > 0.3   ->  那个 k 上有真平台，说明数据分成 k 块
  对照：如果 complete 的占比也很小，§4.1 的结论就从「单链接没看见」
       升级成「两种方向相反的方法都没看见」。
""")

# ---------------------------------------------------------------- 平均曲线
def curve(hs, grid):
    n = len(hs[0]) + 1
    return np.mean([n - np.searchsorted(h, grid, side="right") for h in hs], axis=0)

grids, fbar = {}, {}
for stratum in STRATA:
    for m in METHODS:
        allh = [h for a in LEVELS for h in heights.get((stratum, m, a), [])]
        if not allh:
            continue
        tmax = float(np.percentile(np.concatenate(allh), 99.5))
        g = np.linspace(0, tmax, NGRID)
        grids[(stratum, m)] = g
        for a in LEVELS:
            hs = heights.get((stratum, m, a))
            if hs:
                fbar[(stratum, m, a)] = curve(hs, g)

np.savez_compressed(
    RES_DIR / "complete_vs_single.npz",
    **{f"{s}_{m}_grid": grids[(s, m)] for s, m in grids},
    **{f"{s}_{m}_f{a:02d}": fbar[(s, m, a)] for (s, m, a) in fbar})
print(f"曲线写入 {RES_DIR}/complete_vs_single.npz")

# ---------------------------------------------------------------- 图
for stratum in STRATA:
    if (stratum, "single") not in grids:
        continue
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, m in zip(axes, METHODS):
        g = grids[(stratum, m)]
        for a in LEVELS:
            if (stratum, m, a) in fbar:
                ax.step(g, fbar[(stratum, m, a)], where="post",
                        color=cmap(a / 20), lw=1.0)
        for L in (1000, 100, 10):
            ax.axhline(L, color="0.6", ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlim(0, g[-1])
        ax.set_xlabel(r"$t$   (Chebyshev radius, absolute HP)")
        ax.set_ylabel(r"$\bar f_a(t)$")
        ax.set_title(f"{m} linkage  ({stratum})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 20))
    fig.colorbar(sm, ax=axes, label="Ascension level $a$", pad=0.02)
    fig.savefig(OUT_DIR / f"linkage_compare_{stratum}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"linkage_compare_{stratum}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"图写入 {OUT_DIR}/linkage_compare_{stratum}.pdf")

# ---------------------------------------------------------------- 仅供参考
print(f"\n{'='*92}")
print("complete linkage 下的相邻 kappa (d_1) —— 仅供参考，无稳定性保证")
print(f"{'='*92}")
for stratum in STRATA:
    if (stratum, "complete") not in grids:
        continue
    g = grids[(stratum, "complete")]; dt = g[1] - g[0]
    ks = np.array([
        np.abs(fbar[(stratum, "complete", a)] - fbar[(stratum, "complete", a+1)]).sum() * dt
        if (stratum, "complete", a) in fbar and (stratum, "complete", a+1) in fbar
        else np.nan for a in range(20)])
    print(f"\n分层 {stratum}")
    top = int(np.nanargmax(ks))
    for a in range(20):
        print(f"  A{a:02d}->A{a+1:02d}  {ks[a]:>9.0f}" + ("   <-- 最大" if a == top else ""))

print(f"\n用时 {(time.time()-t0)/60:.1f} 分")