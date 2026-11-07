"""测量子抽样的覆盖半径 rho，用来补上「样本 -> 总体」那一步。

定义：S 是抽出的 N 条，P 是池。
    rho = max_{p in P}  min_{s in S}  d_infty(p, s)
即「池里最孤立的那条局，离最近的采样点多远」。

因为 S 包含于 P，Hausdorff 距离就等于 rho，而 GH <= d_H，所以
    d_interleave(f_S, f_P) <= 2 * rho
再用三角不等式：
    kappa_观测 <= 2*GH(X_a, X_a') + 2*(rho_a + rho_a')

同时扫几个 N，看 rho 随 N 怎么降 —— 这告诉你加大 N 能不能把缺口补上。

注意：rho 是对着 40,000 的池测的，不是对着真实总体（百万量级），
所以这是真实 rho 的一个下界。报告时要写明。
"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

POOL_DIR = Path("pools")
LEVELS   = [0, 10, 20]            # 够刻画量级了，全跑 21 级要十倍时间
STRATA   = ["all", "clear"]
NS       = [500, 1000, 2000, 4000]
SEED     = 7
CHUNK    = 2000                   # 池分块大小，控制内存

def covering_radius(P, S, chunk=CHUNK):
    """max over P of (min over S of d_infty)"""
    best = 0.0
    for i in range(0, P.shape[0], chunk):
        D = cdist(P[i:i + chunk], S, metric="chebyshev")
        m = D.min(axis=1).max()
        if m > best:
            best = float(m)
    return best

t0 = time.time()
print(f"{'级':<5}{'分层':<8}{'池':>8}" + "".join(f"{'N='+str(n):>10}" for n in NS))
rows = {}
for a in LEVELS:
    for stratum in STRATA:
        f = POOL_DIR / f"a{a:02d}_{stratum}.npy"
        if not f.exists():
            continue
        P = np.load(f).astype(np.float64)
        rng = np.random.default_rng(SEED * 1000 + a)
        out = []
        for n in NS:
            if P.shape[0] < n:
                out.append(float("nan")); continue
            idx = rng.choice(P.shape[0], size=n, replace=False)
            out.append(covering_radius(P, P[idx]))
        rows[(a, stratum)] = out
        print(f"A{a:02d}  {stratum:<8}{P.shape[0]:>8,}"
              + "".join(f"{v:>10.1f}" for v in out)
              + f"    ({(time.time()-t0)/60:.1f} 分)")

i2000 = NS.index(2000) if 2000 in NS else -1
vals = [v[i2000] for v in rows.values() if not np.isnan(v[i2000])]
if vals:
    rho = float(np.median(vals))
    print(f"""
{'='*78}
N = 2000 时 rho 的中位数 = {rho:.1f} 血

误差项  2*(rho_a + rho_a')  ≈ {4*rho:.0f} 血

对照：观测到的 kappa 换算成等效水平位移是 kappa / N。
      最大的那个 kappa(A13,A14) = 2594 (clear) -> 2594/2000 ≈ 1.3 血
      同级基线约 200 -> 0.1 血

如果 {4*rho:.0f} 远大于 1.3，说明确定性的误差项远大于信号，
那条「样本 -> 总体」的链在 N=2000 下不成立，论文里要照实写：
  样本层面的 kappa <= 2*GH 仍然严格；
  总体层面的推断依据的是实测的抽样波动（同级基线），不是这个界。

另外看 rho 随 N 下降的速度：如果几乎不降，说明加大 N 也补不上，
原因是点云本身稀疏（最近邻距离就有二十几血）。
{'='*78}""")
print(f"用时 {(time.time()-t0)/60:.1f} 分")