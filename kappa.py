"""第 2 步：从 pools/ 做子抽样，算 H0 稳定秩 f_a 与 21x21 的 kappa 矩阵。

管线（每一步对应讲义 Point）：
  traj in R^n            -> 数据集 X_a                        (Point 1-3)
  d_infty (Chebyshev)    -> 伪度量 delta                      (Point 45)
  VR_t / 单连接          -> 划分 X_a/delta_t                  (Point 187-190, 95-100)
  H0 条形码 -> 稳定秩     -> f_a(t) = |X_a/delta_t|            (Point 152, 158-160)
  R 次子抽样取均值        -> f_bar_a                           (Point 62-63)
  d_interleave / L1      -> kappa(a, a')                       (Point 55-57, 60)
  单连接 on kappa         -> 21 个等级的二层树状图             (Point 112-118)

先用 R=1 跑一遍看形状（约 2 分钟），确认无误再改成 R=20（约 45 分钟）。
"""
import json, time
from pathlib import Path
import numpy as np
import stablebear as sb
from stablebear import persistence as P
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

POOL_DIR = Path("pools")
OUT_DIR  = Path("results")
N        = 2000          # 每次子抽样的局数
R        = 20            # 子抽样重复次数
SEED     = 7
TMAX     = None          # filtration 截断：delta_T = min(delta, TMAX)
                         # 80 = 铁甲战士基础最大生命值。截断后仍是伪度量，
                         # 且截断是 1-Lipschitz，所以 kappa <= 2*GH 对原空间仍成立。
                         # 不截断时 f_a 的尾部由每 2000 局里一两条怪局决定，
                         # 交织距离会退化成「两级的最大离群值之差」。
                         # 默认关闭：实测它对 L1 几乎没有影响（主结果用 L1），
                         # 只对 d_interleave 有用。稳健性对照时设成 80。
ACT1     = False         # True = 只用前 16 层（Act 1 口径）；False = 全程 51 层
STRATA   = ["all", "clear"]
LEVELS   = list(range(21))
SHOW_VAR = [0, 10, 20]   # 画「抽样波动」那张图时展示哪几级

TAG = ("act1" if ACT1 else "full") + (f"_T{TMAX}" if TMAX else "_noT")
NFL = 16 if ACT1 else 51
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 交织距离
def _ev(arr, t):
    """右连续阶梯函数求值：arr 是 (t_i, v_i) 断点表，v_i 在 [t_i, t_{i+1}) 上生效。"""
    i = np.searchsorted(arr[:, 0], t, side="right") - 1
    return arr[np.clip(i, 0, None), 1]

def d_interleave(f, g, tol=1e-7):
    r"""PCF 的交织距离
        d(f,g) = inf{ e >= 0 : f(t+e) <= g(t) 且 g(t+e) <= f(t) 对一切 t }
    对递减阶梯函数，条件只需在两者断点的并集上检验，故可二分。"""
    A, B = f.to_numpy(), g.to_numpy()
    ts = np.unique(np.concatenate([A[:, 0], B[:, 0]]))
    def ok(e):
        return bool((_ev(A, ts + e) <= _ev(B, ts) + 1e-9).all()
                    and (_ev(B, ts + e) <= _ev(A, ts) + 1e-9).all())
    if ok(0.0):
        return 0.0
    lo, hi = 0.0, float(max(A[:, 0].max(), B[:, 0].max())) + 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi

# ---------------------------------------------------------------- 单次稳定秩
def barcode_of(X):
    """X: (n, NFL) 浮点数组 -> H0 条形码。
    stablebear 0.4.4 的 DistanceMatrix 只接受稠密方阵，所以要 squareform。"""
    cd = pdist(X, metric="chebyshev")
    if TMAX:
        cd = np.minimum(cd, TMAX)        # 截断：delta_T = min(delta, TMAX)
    D  = squareform(cd)
    return P.compute_persistent_homology(
        sb.DistanceMatrix.from_dense(D), max_dim=0)[0], cd

def stable_rank(X):
    bc, _ = barcode_of(X)
    return P.barcode_to_stable_rank(bc)

# ---------------------------------------------------------------- 一次性核验
def verify(X):
    bc, cd = barcode_of(X)
    sr = P.barcode_to_stable_rank(bc).to_numpy()
    bt = P.barcode_to_betti_curve(bc).to_numpy()
    print("核验 1  H0 的 Betti 曲线 == 稳定秩            :",
          sr.shape == bt.shape and np.allclose(sr, bt))
    B = bc.to_numpy()
    deaths  = np.sort(B[np.isfinite(B[:, 1]), 1])
    heights = np.sort(linkage(cd, method="single")[:, 2])
    print("核验 2  VR 的 H0 死亡时刻 == 单连接合并高度   :",
          len(deaths) == len(heights) and np.allclose(deaths, heights))

# ================================================================ 主流程
t0 = time.time()
print(f"N={N}  R={R}  SEED={SEED}  口径={TAG}（{NFL} 层）\n")

store = {}
for stratum in STRATA:
    fbars, curves, reps_all, base = [], {}, {}, {}
    print(f"{'='*74}\n分层 {stratum}\n{'='*74}")
    for a in LEVELS:
        pool = np.load(POOL_DIR / f"a{a:02d}_{stratum}.npy").astype(np.float64)
        pool = pool[:, :NFL]
        if pool.shape[0] < N:
            raise SystemExit(f"A{a:02d}/{stratum} 池只有 {pool.shape[0]} 条，小于 N={N}")
        rng = np.random.default_rng(SEED * 1000 + a)
        reps = []
        for r in range(R):
            idx = rng.choice(pool.shape[0], size=N, replace=False)
            reps.append(stable_rank(pool[idx]))
        if a == 0 and stratum == "all":
            verify(pool[rng.choice(pool.shape[0], size=min(N, 800), replace=False)])
            print()
        fb = sb.mean(sb.tensor(reps, dtype=sb.pcf64), dim=0)[0] if R > 1 else reps[0]
        fbars.append(fb)
        # 同级基线：把 R 次重复对半分，两半各自取平均再比。
        # 这是「同一个总体的两份样本能差多少」，即零假设的尺度。
        if R >= 4:
            h = R // 2
            g1 = sb.mean(sb.tensor(reps[:h],    dtype=sb.pcf64), dim=0)[0]
            g2 = sb.mean(sb.tensor(reps[h:2*h], dtype=sb.pcf64), dim=0)[0]
            base[a] = (d_interleave(g1, g2), sb.lp_distance(g1, g2, p=1))
        curves[a] = fb.to_numpy()
        if a in SHOW_VAR:
            reps_all[a] = [p.to_numpy() for p in reps]
        arr = curves[a]
        print(f"A{a:02d}  池 {pool.shape[0]:>6,}  断点 {arr.shape[0]:>5}  "
              f"f(0)={arr[0,1]:>7.1f}  落到 1 的 t={arr[-1,0]:>6.2f}   "
              f"已用 {(time.time()-t0)/60:.1f} 分")

    # ---------- kappa ----------
    K_l1 = sb.pdist(sb.tensor(fbars, dtype=sb.pcf64), p=1).to_numpy()
    K_in = np.zeros((21, 21))
    for i in range(21):
        for j in range(i + 1, 21):
            K_in[i, j] = K_in[j, i] = d_interleave(fbars[i], fbars[j])

    store[stratum] = dict(curves=curves, K_l1=K_l1, K_in=K_in, reps=reps_all,
                          base=base)
    np.savez(OUT_DIR / f"kappa_{stratum}_{TAG}.npz", K_l1=K_l1, K_in=K_in,
             **{f"f{a:02d}": curves[a] for a in LEVELS})

    for tag, K in (("interleave", K_in), ("L1", K_l1)):
        np.savetxt(OUT_DIR / f"kappa_{stratum}_{TAG}_{tag}.csv", K,
                   delimiter=",", fmt="%.6f")
        adj = [K[a, a + 1] for a in range(20)]
        print(f"\nkappa({tag}) 相邻台阶 a -> a+1：")
        print("  " + "  ".join(f"{a:02d}:{v:.3f}" for a, v in enumerate(adj)))
        top = sorted(range(20), key=lambda a: -adj[a])[:3]
        print("  最大三处： " + ", ".join(f"A{a}->A{a+1} ({adj[a]:.3f})" for a in top))
        uniq = len(np.unique(np.round(K[np.triu_indices(21, 1)], 6)))
        print(f"  上三角不同取值个数: {uniq} / 210"
              + ("   <- 取值太少，这个度量分辨力不足" if uniq < 40 else ""))

    # ---------- 同级基线 vs 相邻台阶 ----------
    if base:
        print(f"\n{'-'*74}\n同级基线（同一等级两半样本之间的 kappa）vs 相邻台阶\n{'-'*74}")
        print(f"{'级':<5}{'基线_⋈':>10}{'κ⋈(a,a+1)':>12}{'倍数':>8}"
              f"{'基线_L1':>11}{'κL1(a,a+1)':>13}{'倍数':>8}")
        bi = np.array([base[a][0] for a in LEVELS])
        bl = np.array([base[a][1] for a in LEVELS])
        for a in LEVELS[:-1]:
            ki, kl = K_in[a, a + 1], K_l1[a, a + 1]
            ri = ki / bi[a] if bi[a] > 0 else float("inf")
            rl = kl / bl[a] if bl[a] > 0 else float("inf")
            flag = "  <<<" if rl >= 3 else ""
            print(f"A{a:02d} {bi[a]:>10.2f}{ki:>12.2f}{ri:>8.1f}"
                  f"{bl[a]:>11.1f}{kl:>13.1f}{rl:>8.1f}{flag}")
        print(f"\n  基线中位数:  d_interleave {np.median(bi):.2f}   "
              f"L1 {np.median(bl):.1f}")
        print("  倍数 >= 3 的台阶（标 <<<）才算差异明显超过抽样噪声。")

# ================================================================ 作图
for stratum in STRATA:
    S = store[stratum]

    # 图 1：21 条 f_bar 曲线
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    tmax = max(S["curves"][a][-1, 0] for a in LEVELS) * 1.05
    for a in LEVELS:
        arr = S["curves"][a]
        t = np.append(arr[:, 0], tmax)
        v = np.append(arr[:, 1], arr[-1, 1])
        ax.step(t, v, where="post", color=cmap(a / 20), lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$   (Chebyshev radius, absolute HP)")
    ax.set_ylabel(r"$\bar f_a(t) = |X_a/\delta_t|$")
    ax.set_title(f"H0 stable rank, stratum = {stratum}, {TAG}  (N={N}, R={R})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 20))
    fig.colorbar(sm, ax=ax, label="Ascension level $a$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fbar_{stratum}_{TAG}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"fbar_{stratum}_{TAG}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 图 2：抽样波动（R >= 5 才有意义）
    if R >= 5 and S["reps"]:
        fig, axes = plt.subplots(1, len(SHOW_VAR), figsize=(4 * len(SHOW_VAR), 3.6),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax_, a in zip(axes, SHOW_VAR):
            tm = max(c[-1, 0] for c in S["reps"][a]) * 1.05
            for c in S["reps"][a]:
                ax_.step(np.append(c[:, 0], tm), np.append(c[:, 1], c[-1, 1]),
                         where="post", color="#b0b8c4", lw=0.7, alpha=0.8)
            m = S["curves"][a]
            ax_.step(np.append(m[:, 0], tm), np.append(m[:, 1], m[-1, 1]),
                     where="post", color="#d4762a", lw=2.0)
            ax_.set_yscale("log"); ax_.set_title(f"Ascension {a}")
            ax_.set_xlabel(r"$t$")
        axes[0].set_ylabel(r"$f_a(t)$")
        fig.suptitle(f"Sampling variability: {R} subsamples (grey) and their "
                     f"mean (orange), stratum = {stratum}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"variability_{stratum}_{TAG}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"variability_{stratum}_{TAG}.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

    # 图 3：kappa 热图 + 二层树状图
    for tag, K in (("interleave", S["K_in"]), ("L1", S["K_l1"])):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        im = ax1.imshow(K, cmap="magma")
        ax1.set_xticks(LEVELS); ax1.set_yticks(LEVELS)
        ax1.set_xticklabels(LEVELS, fontsize=7); ax1.set_yticklabels(LEVELS, fontsize=7)
        ax1.set_title(rf"$\kappa$ ({tag}), {stratum}, {TAG}")
        fig.colorbar(im, ax=ax1)
        Z = linkage(squareform(K, checks=False), method="single")
        dendrogram(Z, labels=[f"A{a}" for a in LEVELS], ax=ax2, leaf_font_size=8)
        ax2.set_title(f"Dendrogram of the 21 levels (single linkage)")
        ax2.set_ylabel(r"$\kappa$")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"kappa_{stratum}_{TAG}_{tag}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"kappa_{stratum}_{TAG}_{tag}.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

    # 图 4：相邻台阶
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for tag, K, c in (("interleave", S["K_in"], "C0"), ("L1", S["K_l1"], "C1")):
        v = np.array([K[a, a + 1] for a in range(20)])
        ax.plot(range(20), v / v.max(), "o-", color=c, label=f"{tag} (normalised)")
    ax.set_xticks(range(20))
    ax.set_xticklabels([f"{a}→{a+1}" for a in range(20)], rotation=90, fontsize=7)
    ax.set_ylabel(r"$\kappa(a,a{+}1)$ / max")
    ax.set_title(f"Difficulty increment between adjacent levels ({stratum}, {TAG})")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT_DIR / f"kappa_adjacent_{stratum}_{TAG}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"kappa_adjacent_{stratum}_{TAG}.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分。结果与图写入 {OUT_DIR}/")