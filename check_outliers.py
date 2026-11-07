"""诊断：池子里每局的最大血量分布，以及 f_a 尾部由谁决定。

只读 pools/，几秒钟。用来决定两件事：
  1. HP_CAP = 400 是不是太松（合法上限到底在哪）
  2. 要不要把 filtration 截断在某个 t_max
"""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

POOL_DIR = Path("pools")
LEVELS   = list(range(21))
THRESH   = [120, 150, 200, 250, 300]
N_DEMO   = 2000          # 演示用的子抽样大小，与 2_kappa.py 一致
SEED     = 7

for stratum in ("all", "clear"):
    print(f"\n{'='*96}\n分层 {stratum}：每局最大血量 max(traj) 的分布\n{'='*96}")
    print(f"{'级':<5}{'池':>8}{'p50':>7}{'p90':>7}{'p99':>7}{'p99.9':>8}{'最大':>8}"
          + "".join(f"{'>'+str(t):>9}" for t in THRESH))
    for a in LEVELS:
        f = POOL_DIR / f"a{a:02d}_{stratum}.npy"
        if not f.exists():
            continue
        X = np.load(f).astype(np.int64)
        m = X.max(axis=1)
        q = np.percentile(m, [50, 90, 99, 99.9])
        print(f"A{a:02d} {len(m):>8,}{q[0]:>7.0f}{q[1]:>7.0f}{q[2]:>7.0f}{q[3]:>8.0f}"
              f"{m.max():>8,}"
              + "".join(f"{(m > t).sum():>9,}" for t in THRESH))

print(f"\n{'='*96}\n一次子抽样的最后 12 次合并高度（看尾巴是不是单点拖出来的）\n{'='*96}")
print(f"{'级':<5}{'分层':<7}" + "".join(f"{-i:>7}" for i in range(12, 0, -1)))
for a in (0, 9, 10, 20):
    for stratum in ("all", "clear"):
        f = POOL_DIR / f"a{a:02d}_{stratum}.npy"
        if not f.exists():
            continue
        X = np.load(f).astype(np.float64)
        rng = np.random.default_rng(SEED * 1000 + a)
        idx = rng.choice(X.shape[0], size=min(N_DEMO, X.shape[0]), replace=False)
        h = np.sort(linkage(pdist(X[idx], metric="chebyshev"), method="single")[:, 2])
        print(f"A{a:02d}  {stratum:<7}" + "".join(f"{v:>7.0f}" for v in h[-12:]))

print("""
读法：
  最后几个合并高度如果断崖式跳高（比如 ... 28 30 33 214），
  说明尾部是一两个孤立局拖出来的，交织距离会被它们完全主导。
  如果是平滑上升，说明尾巴是真实的分布边缘。
""")