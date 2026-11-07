import gzip, json, time
from pathlib import Path
from collections import Counter

IN_DIR, OUT_DIR = Path("by_ascension"), Path("clean2")
MAX_FLOOR   = 51      # 轨迹长度（第 1-51 层）
ABANDON_TH  = 0.80    # 退出时血量比例 >= 此值 -> 重开，剔除
CHEAT_TAIL  = 3       # 末尾连续「turns>=3 且 damage=0」战斗数阈值
CHEAT_MAX   = 6       # 任意位置最长连续数阈值
ACT4_FLOOR  = 52      # 第四幕起始层
MIN_WIN_FL  = 50      # 合法胜局的最低 floor_reached（第三幕 Boss 在第 50 层）
CONCEDE_PAD = 0       # 中途退出局的补值（0 = 等同失败；改为 None 则向前延续）

REASONS = ["special_mode", "mode_mismatch", "floor_range", "bad_hp",
           "odd_win", "ratio_gt1", "full_hp", "cheat_tail", "cheat_max",
           "abandon"]
ETYPES  = ["success", "heart_died", "died", "conceded"]

def tail_len(fl):
    n = 0
    for x in reversed(fl):
        if x: n += 1
        else: break
    return n

def max_run(fl):
    best = cur = 0
    for x in fl:
        cur = cur + 1 if x else 0
        if cur > best: best = cur
    return best

OUT_DIR.mkdir(exist_ok=True)

summary, sens_cheat, sens_aban, sens_winfl = {}, {}, {}, {}
t0 = time.time()

for a in range(21):
    src = IN_DIR / f"ironclad_a{a:02d}.jsonl.gz"
    if not src.exists():
        print(f"A{a:02d} 缺文件，跳过"); continue

    rej = Counter(); n_in = n_keep = n_succ = 0
    etype = Counter(); sc = Counter(); sa = Counter(); wfl = Counter()

    with gzip.open(src, "rt", encoding="utf-8") as fin, \
         gzip.open(OUT_DIR / f"ironclad_a{a:02d}_clean.jsonl.gz", "wt", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            r = json.loads(line)

            if r["is_daily"] or r["is_endless"] or r["chose_seed"] or r["is_trial"]:
                rej["special_mode"] += 1; continue
            if r["is_ascension_mode"] != (a > 0):
                rej["mode_mismatch"] += 1; continue

            f = r.get("floor_reached") or 0
            if f < 1 or f > 57:
                rej["floor_range"] += 1; continue

            hp, mx = r.get("current_hp_per_floor"), r.get("max_hp_per_floor")
            if not hp or not mx or len(hp) < f - 1 or len(mx) < f - 1:
                rej["bad_hp"] += 1; continue

            win = bool(r["victory"])
            if win:
                wfl[min(f, 57)] += 1
                if f < MIN_WIN_FL:          # victory 但楼层过低 -> 记录损坏
                    rej["odd_win"] += 1; continue

            # 已知血量记录覆盖第 1..n_real 层（败局末项是死亡值的重复，由 min 截掉）
            n_real = min(len(hp), len(mx), f)
            if n_real < 1:
                rej["bad_hp"] += 1; continue
            ratios = [hp[i] / mx[i] for i in range(n_real) if mx[i]]
            if not ratios:
                rej["bad_hp"] += 1; continue
            if max(ratios) > 1.001:
                rej["ratio_gt1"] += 1; continue
            if len(ratios) >= 16 and min(ratios) >= 0.99:
                rej["full_hp"] += 1; continue

            fights = r.get("damage_taken") or []
            flags = [(t >= 3 and d == 0) for _, d, t in fights]
            tl, mr = tail_len(flags), max_run(flags)
            sc[min(tl, 8)] += 1
            if tl >= CHEAT_TAIL:
                rej["cheat_tail"] += 1; continue
            if mr >= CHEAT_MAX:
                rej["cheat_max"] += 1; continue

            # ---------- 结局分类 ----------
            end_hp, end_mx = hp[n_real - 1], mx[n_real - 1]
            if win:
                et = "success"
            elif f >= ACT4_FLOOR:
                et = "heart_died"          # 打穿第三幕，死/退在第四幕
            elif end_hp <= 0:
                et = "died"
            else:
                rt = end_hp / end_mx if end_mx else 0.0
                sa[min(int(rt * 10), 9)] += 1
                if rt >= ABANDON_TH:
                    rej["abandon"] += 1; continue
                et = "conceded"

            success = win or f >= ACT4_FLOOR

            # ---------- 轨迹：按结局补尾 ----------
            base = hp[:min(n_real, MAX_FLOOR)]
            if len(base) < MAX_FLOOR:
                if et == "success":
                    pad = base[-1]                     # 通关者存活，向前延续
                elif et == "conceded" and CONCEDE_PAD is None:
                    pad = base[-1]
                else:
                    pad = 0                            # 死亡 / 退出
                base = base + [pad] * (MAX_FLOOR - len(base))
            traj = base[:MAX_FLOOR]

            r["end_type"] = et
            r["success"]  = success
            r["traj"]     = traj
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_keep += 1; n_succ += success; etype[et] += 1

    summary[a]    = dict(n_in=n_in, n_keep=n_keep, n_succ=n_succ,
                         rej=dict(rej), etype=dict(etype))
    sens_cheat[a] = dict(sc); sens_aban[a] = dict(sa); sens_winfl[a] = dict(wfl)
    print(f"A{a:02d}  读入 {n_in:>9,}  保留 {n_keep:>9,}  通关 {n_succ:>7,} "
          f"({n_succ/max(n_keep,1):6.2%})   已用 {(time.time()-t0)/60:.1f} 分")

(OUT_DIR / "_clean_stats.json").write_text(json.dumps(
    {"summary": summary, "sens_cheat": sens_cheat, "sens_abandon": sens_aban,
     "sens_win_floor": sens_winfl,
     "params": dict(ABANDON_TH=ABANDON_TH, CHEAT_TAIL=CHEAT_TAIL,
                    CHEAT_MAX=CHEAT_MAX, ACT4_FLOOR=ACT4_FLOOR,
                    MIN_WIN_FL=MIN_WIN_FL, CONCEDE_PAD=CONCEDE_PAD)},
    ensure_ascii=False), encoding="utf-8")

# ---------- 汇总 ----------
print(f"\n{'='*96}\n各等级剔除明细\n{'='*96}")
print(f"{'级':<4}{'读入':>10}" + "".join(f"{x[:9]:>10}" for x in REASONS)
      + f"{'保留':>10}{'通关率':>9}")
for a, s in summary.items():
    print(f"A{a:02d} {s['n_in']:>10,}"
          + "".join(f"{s['rej'].get(x,0):>10,}" for x in REASONS)
          + f"{s['n_keep']:>10,}{s['n_succ']/max(s['n_keep'],1):>9.2%}")

tot_in = sum(s["n_in"] for s in summary.values())
tot_k  = sum(s["n_keep"] for s in summary.values())
print(f"\n合计读入 {tot_in:,}，保留 {tot_k:,}（{tot_k/tot_in:.1%}）")

print(f"\n{'='*96}\n保留局的结束方式\n{'='*96}")
print(f"{'级':<4}" + "".join(f"{x:>13}" for x in ETYPES) + f"{'成功合计':>13}")
for a, s in summary.items():
    e = s["etype"]
    print(f"A{a:02d} " + "".join(f"{e.get(x,0):>13,}" for x in ETYPES)
          + f"{s['n_succ']:>13,}")

print(f"\n{'='*96}\n胜局的 floor_reached 分布（检查 odd_win 规模）\n{'='*96}")
BK = [(1,33,"1-33"),(34,49,"34-49"),(50,50,"50"),(51,51,"51"),(52,57,"52-57")]
print(f"{'级':<4}" + "".join(f"{l:>12}" for _,_,l in BK) + f"{'胜局合计':>12}")
for a, c in sens_winfl.items():
    c = {int(k): v for k, v in c.items()}
    tot = sum(c.values())
    print(f"A{a:02d} " + "".join(
        f"{sum(v for k,v in c.items() if lo<=k<=hi):>12,}" for lo,hi,_ in BK)
        + f"{tot:>12,}")

print(f"\n{'='*96}\n敏感性：作弊阈值（剔除局数）\n{'='*96}")
print(f"{'级':<4}{'tail>=2':>12}{'tail>=3':>12}{'tail>=4':>12}")
for a, c in sens_cheat.items():
    c = {int(k): v for k, v in c.items()}
    print(f"A{a:02d} " + "".join(
        f"{sum(v for k,v in c.items() if k>=th):>12,}" for th in (2,3,4)))

print(f"\n{'='*96}\n敏感性：放弃阈值（剔除局数）\n{'='*96}")
print(f"{'级':<4}{'>=0.7':>12}{'>=0.8':>12}{'>=0.9':>12}")
for a, c in sens_aban.items():
    c = {int(k): v for k, v in c.items()}
    print(f"A{a:02d} " + "".join(
        f"{sum(v for k,v in c.items() if k>=bb):>12,}" for bb in (7,8,9)))