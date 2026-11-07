"""第 1 步：从 clean2/ 扫一遍，为每个进阶等级抽出两个样本池。

池 = 均匀随机样本（蓄水池抽样 Algorithm R）。之后所有子抽样都从池里取，
这样改 N / R / 分层都不必重扫 2,556 万行。

输出 pools/a{aa}_all.npy、pools/a{aa}_clear.npy （int16, 形状 (P, 51)）
     pools/_meta.json

只跑这一次，约 25 分钟。
"""
import gzip, json, random, time
from pathlib import Path
import numpy as np

IN_DIR    = Path("clean2")
OUT_DIR   = Path("pools")
POOL      = 40_000     # 每个池的目标容量
MAX_FLOOR = 51
SEED      = 12345
HP_CAP    = 400        # 绝对血量上限：超过即判为 mod / 作弊局，剔除
                       # 铁甲战士基础上限 80，靠遗物和事件最多到一百多，
                       # 400 已经极宽松。清洗阶段只查了 hp/max_hp 的比值，
                       # 查不出 hp 和 max_hp 被同时放大的局。

OUT_DIR.mkdir(exist_ok=True)
meta = {}
t0 = time.time()
grand_max = 0

for a in range(21):
    src = IN_DIR / f"ironclad_a{a:02d}_clean.jsonl.gz"
    if not src.exists():
        print(f"A{a:02d} 缺文件，跳过"); continue

    rng = random.Random(SEED + a)          # 每级独立且可复现
    res_all, res_clr = [], []
    n_all = n_clr = n_bad = 0
    lvl_max = 0

    with gzip.open(src, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            traj = r["traj"]
            if len(traj) != MAX_FLOOR:     # 防御：长度必须正好 51
                continue
            m = max(traj)
            if m > lvl_max:
                lvl_max = m
            if m > HP_CAP:                 # mod / 作弊局
                n_bad += 1
                continue

            n_all += 1
            if len(res_all) < POOL:
                res_all.append(traj)
            else:
                j = rng.randrange(n_all)
                if j < POOL:
                    res_all[j] = traj

            if r["success"]:
                n_clr += 1
                if len(res_clr) < POOL:
                    res_clr.append(traj)
                else:
                    j = rng.randrange(n_clr)
                    if j < POOL:
                        res_clr[j] = traj

    A = np.asarray(res_all, dtype=np.int32)
    C = np.asarray(res_clr, dtype=np.int32)
    np.save(OUT_DIR / f"a{a:02d}_all.npy",   A)
    np.save(OUT_DIR / f"a{a:02d}_clear.npy", C)
    grand_max = max(grand_max, lvl_max)
    meta[f"a{a:02d}"] = dict(n_all=n_all, n_clear=n_clr, n_over_cap=n_bad,
                             max_hp_seen=int(lvl_max),
                             pool_all=int(A.shape[0]), pool_clear=int(C.shape[0]))
    print(f"A{a:02d}  全部 {n_all:>9,} -> 池 {A.shape[0]:>6,}   "
          f"通关 {n_clr:>8,} -> 池 {C.shape[0]:>6,}   "
          f"超上限剔除 {n_bad:>6,}  该级最大血量 {lvl_max:>8,}   "
          f"已用 {(time.time()-t0)/60:.1f} 分")

(OUT_DIR / "_meta.json").write_text(json.dumps(
    {"per_level": meta,
     "params": dict(POOL=POOL, SEED=SEED, MAX_FLOOR=MAX_FLOOR, HP_CAP=HP_CAP)},
    ensure_ascii=False, indent=1), encoding="utf-8")

tot_bad = sum(v["n_over_cap"] for v in meta.values())
tot_all = sum(v["n_all"] for v in meta.values())
print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分。池写入 {OUT_DIR}/")
print(f"因绝对血量 > {HP_CAP} 被剔除：{tot_bad:,} 局"
      f"（占 {tot_bad/max(tot_all+tot_bad,1):.4%}），全库最大血量 {grand_max:,}")