"""Claude(チャット)による予想の補助スクリプト。判断はClaude、データ集めと記録はこのスクリプト。

  python scripts/claude_pick.py next              # 直前情報が出ていて締切5-15分前の未予想レースを1つ選び、判断材料を表示
  python scripts/claude_pick.py brief JCD RNO     # 指定レースの判断材料
  python scripts/claude_pick.py record JCD RNO 'WIN' 'PLACE' 'メモ'
        WIN/PLACE: 1号艇から6号艇の順に6つの数値(カンマ区切り)。WIN=1着確率, PLACE=2-3着に残る力
        → 3連単確率を計算し、締切前オッズで 期待値>THR の組を仮想購入(各100円)として記録
  python scripts/claude_pick.py settle            # 結果が出た記録を答え合わせ
  python scripts/claude_pick.py summary           # 累計成績
"""
import datetime as dt
import itertools
import json
import pathlib
import random
import re
import sys
import unicodedata
import zoneinfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boatpred import web  # noqa: E402
from boatpred.download import fetch, path_for  # noqa: E402
from boatpred.live import VENUES  # noqa: E402
from boatpred.parse import parse_b, parse_k  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PICKS = ROOT / "claude_picks"
CACHE = ROOT / "data" / "cache"
JST = zoneinfo.ZoneInfo("Asia/Tokyo")
THR = 1.2
# 2-3着の残る力を手で与えるHarville計算は、弱い艇の連絡みを過大に見積もりやすい(大穴に期待値が出る)。
# そのため確率の低い組・高配当の組は買わない
MIN_P = 0.02
# 単勝方式(2026-11-06 14:10〜): 私の1着確率 × 単勝オッズ > WIN_THR の艇を買う
WIN_THR = 1.15
MAX_ODDS = 100


def now():
    return dt.datetime.now(JST)


def today():
    return now().date()


def pick_path(d, jcd, rno):
    return PICKS / f"{d:%Y%m%d}_{jcd:02d}_{rno:02d}.json"


def history():
    """過去約1年の成績(決まり手つき)。日ごとにキャッシュ。"""
    cp = CACHE / f"claude_hist_{today():%Y%m%d}.pkl"
    if cp.exists():  # 自分で作ったローカルキャッシュのみ読む
        return pd.read_pickle(cp)
    start = today() - dt.timedelta(days=370)
    res, km = [], []
    for f in sorted((ROOT / "data" / "raw" / "K").glob("K*.txt")):
        if f.stem[1:] < f"{start:%Y%m%d}":
            continue
        res.append(parse_k(f)[0])
        jcd = rno = None
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(\d{2})KBGN", line)
            if m:
                jcd = int(m.group(1))
                continue
            n = unicodedata.normalize("NFKC", line)
            m = re.match(r"^\s+(\d{1,2})R\s+.*H\d+m", n)
            if m:
                rno = int(m.group(1))
                continue
            if "レースタイム" in n and jcd and rno:
                km.append((pd.Timestamp(f.stem[1:]), jcd, rno, n.split("レースタイム")[-1].strip()))
    r = pd.concat(res, ignore_index=True)
    r["date"] = pd.to_datetime(r.date)
    r = r.merge(pd.DataFrame(km, columns=["date", "jcd", "rno", "kimarite"]), on=["date", "jcd", "rno"], how="left")
    CACHE.mkdir(parents=True, exist_ok=True)
    for old in CACHE.glob("claude_hist_*.pkl"):
        old.unlink()
    r.to_pickle(cp)
    return r


def program():
    p = path_for("B", today())
    if not p.exists():
        fetch("B", today())
    return parse_b(p)


def candidates(lo=4, hi=11):  # 展示は締切の約10分前までに出ることが多い
    b = program()
    races = b[["jcd", "rno", "deadline", "race_name"]].drop_duplicates(["jcd", "rno"])
    out = []
    n = now()
    for r in races.itertuples():
        if not r.deadline or pick_path(today(), r.jcd, r.rno).exists():
            continue
        hh, mm = map(int, r.deadline.split(":"))
        mins = (dt.datetime.combine(today(), dt.time(hh, mm), JST) - n).total_seconds() / 60
        if lo <= mins <= hi:
            out.append((r.jcd, r.rno, r.deadline, r.race_name, round(mins, 1)))
    return out


def brief(jcd, rno):
    b = program()
    race = b[(b.jcd == jcd) & (b.rno == rno)].sort_values("lane")
    h = history()
    bi = web.beforeinfo(today(), jcd, rno)
    o = web.odds3t(today(), jcd, rno)
    print(f"■ {VENUES.get(jcd)}{rno}R {race.race_name.iloc[0]} 締切{race.deadline.iloc[0]} 現在{now():%H:%M}")
    w = bi["weather"]
    print(f"気象: {w.get('weather')} 風{w.get('wind')}m(向きコード{w.get('wind_dir_code')}) 波{w.get('wave')}cm")
    v = h[(h.jcd == jcd) & h.course.notna()]
    vs = v.assign(win=v.finish == 1).groupby("course").win.mean()
    print("場のコース別1着率:", " ".join(f"{int(c)}:{p:.2f}" for c, p in vs.items()))
    print("艇 選手 級 年齢 支部 全国勝率/2率 当地勝率 ﾓｰﾀｰ2率 ﾎﾞｰﾄ2率 今節 | 展示 進入 展示ST ﾁﾙﾄ")
    for r in race.itertuples():
        x = bi["boats"].get(r.lane, {})
        print(f"{r.lane} {r.name} {r.cls} {r.age:.0f} {r.branch} {r.nat_win:.2f}/{r.nat_2:.1f} {r.loc_win:.2f} "
              f"{r.motor_2:.1f} {r.boat_2:.1f} [{r.series_results.strip()}] | {x.get('exhibit')} {x.get('ex_course')} "
              f"{'F' if x.get('ex_st_flying') else ''}{x.get('ex_st')} {x.get('tilt')}")
    print("--- 過去1年(選手別) ---")
    for r in race.itertuples():
        x = h[h.toban == r.toban].sort_values(["date", "rno"])
        if not len(x):
            print(f"{r.lane}号艇 データなし")
            continue
        x = x.assign(win=x.finish == 1, t3=x.finish <= 3)
        rc = x[x.date >= x.date.max() - pd.Timedelta(days=45)]
        c = bi["boats"].get(r.lane, {}).get("ex_course") or r.lane
        xc = x[x.course == c]
        wk = x[x.win].kimarite.value_counts().head(3).to_dict()
        lx = x[x.course.notna()]
        print(f"{r.lane}号艇 {len(x)}走 1着{x.win.mean():.2f} 3連{x.t3.mean():.2f} ST{x.st.mean():.3f} "
              f"F{int(x.status.str.startswith('F').sum())} | {int(c)}ｺｰｽ {len(xc)}走 1着{xc.win.mean() if len(xc) else float('nan'):.2f} "
              f"3連{xc.t3.mean() if len(xc) else float('nan'):.2f} | 直近45日 1着{rc.win.mean():.2f} 3連{rc.t3.mean():.2f} "
              f"ST{rc.st.mean():.3f} | 前付け{(lx.course < lx.lane).mean():.2f} | 決まり手{wk} | 直近:{' '.join(x.status.tail(8)[::-1])}")
    if len(o) > 100:
        tot = sum(1 / v for k, v in o.items() if k != "_time")
        top = sorted(((v, k) for k, v in o.items() if k != "_time"))[:12]
        print(f"--- 3連単オッズ(人気順) 控除込み合計{tot:.3f} ---")
        print(" ".join(f"{k}:{v}" for v, k in top))
        mw = {i: 0.0 for i in range(1, 7)}
        for k, v in o.items():
            if k != "_time":
                mw[int(k[0])] += 1 / v / tot
        print("市場の1着確率:", " ".join(f"{i}:{p:.2f}" for i, p in mw.items()))
    else:
        print("オッズ取得できず")
    return bi, o


PLACE_POW = 2.0  # 手で付ける「残る力」は差が平坦になりがち(弱い艇の2-3着を過大評価)なので差を広げる


def trifecta(win, place):
    place = [x ** PLACE_POW for x in place]
    P = {}
    sp = sum(place)
    for i, j, k in itertools.permutations(range(6), 3):
        P[f"{i+1}-{j+1}-{k+1}"] = win[i] * place[j] / (sp - place[i]) * place[k] / (sp - place[i] - place[j])
    return P


def record(jcd, rno, win, place, memo):
    win = [float(x) for x in win.split(",")]
    place = [float(x) for x in place.split(",")]
    s = sum(win)
    win = [x / s for x in win]
    P = trifecta(win, place)
    o = web.odds3t(today(), jcd, rno)
    # 手で与えた「残る力」によるHarvilleは2-3着の並びの精度が低く、弱い艇の連絡みを市場の2-3倍に見積もる癖が
    # 出た(6R分の記録)。そこで買い判断は 市場確率とClaude確率の幾何平均(重みW) で行う
    tot = sum(1 / v for k, v in o.items() if k != "_time")
    mk = {c: (1 / o[c] / tot if o.get(c) else 0.0) for c in P}
    W = 0.5
    bl = {c: (mk[c] ** (1 - W)) * (P[c] ** W) for c in P}
    z = sum(bl.values())
    bl = {c: v / z for c, v in bl.items()}
    bets = []
    for c, p in sorted(bl.items(), key=lambda x: -x[1]):
        od = o.get(c)
        if od and p * od > THR and p >= MIN_P and od <= MAX_ODDS:
            bets.append(dict(combo=c, p=round(p, 4), p_claude=round(P[c], 4), odds=od, ev=round(p * od, 2)))
    tf = web.oddstf(today(), jcd, rno)
    win_bets = [dict(lane=i + 1, p=round(win[i], 3), odds=tf[i + 1], ev=round(win[i] * tf[i + 1], 2))
                for i in range(6) if tf.get(i + 1) and win[i] * tf[i + 1] > WIN_THR]
    top = sorted(P.items(), key=lambda x: -x[1])[:8]
    rec = dict(win_bets=win_bets, tf_odds=tf, mode="単勝",date=f"{today()}", venue=VENUES.get(jcd), jcd=jcd, rno=rno, made_at=f"{now():%H:%M}",
               p_win={str(i + 1): round(w, 3) for i, w in enumerate(win)},
               place={str(i + 1): p for i, p in enumerate(place)},
               top=[dict(combo=c, p=round(p, 4), odds=o.get(c)) for c, p in top],
               bets=bets, memo=memo, result=None,
               odds_all={k: v for k, v in o.items() if k != "_time"})
    PICKS.mkdir(exist_ok=True)
    pick_path(today(), jcd, rno).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps(dict(win_bets=win_bets, tf_odds=tf, top=rec["top"][:3]), ensure_ascii=False))


def settle():
    done = []
    for p in sorted(PICKS.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("result") or "jcd" not in rec or "p_win" not in rec:
            continue
        d = dt.date.fromisoformat(rec["date"])
        try:
            h = web._get("raceresult", d, rec["jcd"], rec["rno"])
        except Exception:  # noqa: BLE001
            continue
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", h, flags=re.S)))
        m = re.search(r"3連単 (\d) - (\d) - (\d) &yen;([\d,]+)", t)
        if not m:
            if "レース中止" in t or "不成立" in t:
                rec["result"] = dict(void=True)
                p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            continue
        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        mt = re.search(r"単勝 (\d) &yen;([\d,]+)", t)
        tansho = (int(mt.group(1)), int(mt.group(2).replace(",", ""))) if mt else (int(m.group(1)), 0)
        pay = int(m.group(4).replace(",", ""))
        P = trifecta([rec["p_win"][str(i)] for i in range(1, 7)], [rec["place"][str(i)] for i in range(1, 7)])
        hit = [b for b in rec["bets"] if b["combo"] == combo]
        oa = rec.get("odds_all") or {}
        tot = sum(1 / v for v in oa.values()) if oa else None
        mp = round(1 / oa[combo] / tot, 4) if oa.get(combo) else None
        wb = rec.get("win_bets", [])
        rec["result_win"] = dict(winner=tansho[0], payout=tansho[1], invest=100 * len(wb),
                                 ret=sum(tansho[1] for b in wb if b["lane"] == tansho[0]))
        rec["result"] = dict(combo=combo, payout=pay, claude_p=round(P[combo], 4), market_p=mp,
                             invest=100 * len(rec["bets"]), ret=pay if hit else 0)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        rw = rec["result_win"]
        done.append(f"{rec['venue']}{rec['rno']}R {combo} {pay}円 私の確率{P[combo]*100:.1f}% | "
                    f"単勝{tansho[0]}号艇 {tansho[1]}円 単勝買い{len(wb)}点 払戻{rw['ret']}")
    print("\n".join(done) or "新たに確定した結果なし")


def summary():
    rs = [json.loads(p.read_text()) for p in sorted(PICKS.glob("*.json"))]
    rs = [r for r in rs if r.get("result") and "invest" in r["result"]]
    if not rs:
        print("確定済みなし")
        return
    inv = sum(r["result"]["invest"] for r in rs)
    ret = sum(r["result"]["ret"] for r in rs)
    nll = -np.mean([np.log(max(r["result"]["claude_p"], 1e-6)) for r in rs])
    win1 = np.mean([r["p_win"][r["result"]["combo"][0]] for r in rs])
    mk = [r for r in rs if r["result"].get("market_p")]
    if mk:
        a = -np.mean([np.log(max(r["result"]["claude_p"], 1e-6)) for r in mk])
        b = -np.mean([np.log(r["result"]["market_p"]) for r in mk])
        print(f"市場との比較({len(mk)}R): 対数損失 Claude {a:.3f} / 市場 {b:.3f} (小さい方が良い)")
    ws = [r for r in rs if "result_win" in r and r.get("mode") == "単勝"]
    if ws:
        i2 = sum(r["result_win"]["invest"] for r in ws)
        r2 = sum(r["result_win"]["ret"] for r in ws)
        print(f"単勝方式(本番) {len(ws)}R 投資{i2} 払戻{r2} 回収率{(r2/i2*100 if i2 else float('nan')):.1f}%")
    print(f"確定{len(rs)}R 買ったR{sum(1 for r in rs if r['result']['invest'])} 投資{inv} 払戻{ret} "
          f"回収率{(ret/inv*100 if inv else float('nan')):.1f}% 平均対数損失{nll:.3f}(一様=4.787) 勝者の1着確率平均{win1:.3f}")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "next":
        c = candidates()
        if not c:
            print("対象レースなし(締切5-15分前の未予想レースがない)")
        else:
            random.shuffle(c)
            for jcd, rno, *_ in c:
                h = web._get("odds3t", today(), jcd, rno)
                if "中止" in h:  # 中止レースは記録して以後選ばない
                    pick_path(today(), jcd, rno).write_text(json.dumps(
                        dict(date=f"{today()}", jcd=jcd, rno=rno, void=True, result=dict(void=True))))
                    print(f"{VENUES.get(jcd)}{rno}R は中止")
                    continue
                if h.count("oddsPoint") < 120:
                    print(f"{VENUES.get(jcd)}{rno}R はオッズ未取得のため次回")
                    continue
                print(f"候補{len(c)}件から選択")
                brief(jcd, rno)
                break
            else:
                print("対象レースなし(直前情報・オッズのそろったレースがない)")
    elif cmd == "brief":
        brief(int(a[0]), int(a[1]))
    elif cmd == "record":
        record(int(a[0]), int(a[1]), a[2], a[3], a[4] if len(a) > 4 else "")
    elif cmd == "settle":
        settle()
    elif cmd == "summary":
        summary()
