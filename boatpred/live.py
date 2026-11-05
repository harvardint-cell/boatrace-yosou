"""当日の予想ループ。

1. 当日の番組表(B)と、過去365日の成績(K)から特徴量の土台を作る
2. 締切 BET_WINDOW 分前になったレースから順に、直前情報と現在のオッズを取得
3. 3連単の確率 → 市場とブレンド → 期待値 > 閾値 の買い目を出力(site/data/YYYYMMDD.json)
4. 翌日以降、成績(K)と突き合わせて仮想収支(フォワードテスト)を集計

本番で使うオッズは「締切数分前」のもので、バックテストの「締切時オッズ」とは異なる。
"""
import datetime as dt
import json
import pathlib
import time
import zoneinfo

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import model as M
from . import web
from .download import daterange, fetch, path_for
from .features import build, history_features
from .parse import parse_b, parse_k

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
JST = zoneinfo.ZoneInfo("Asia/Tokyo")
VENUES = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑",
          9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島",
          17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}


def now_jst():
    return dt.datetime.now(JST)


def load_params():
    p = json.loads((ROOT / "models" / "params.json").read_text())
    booster = lgb.Booster(model_file=str(ROOT / "models" / "win.txt"))
    return booster, p


def ensure_files(today: dt.date, days=370):
    for d in daterange(today - dt.timedelta(days=days), today - dt.timedelta(days=1)):
        if not path_for("K", d).exists():
            fetch("K", d)
    fetch("B", today)


def recent_results(today: dt.date, days=370) -> pd.DataFrame:
    fs = [path_for("K", d) for d in daterange(today - dt.timedelta(days=days), today - dt.timedelta(days=1))]
    return pd.concat([parse_k(f)[0] for f in fs if f.exists()], ignore_index=True)


class Day:
    def __init__(self, today: dt.date):
        self.today = today
        self.booster, self.p = load_params()
        ensure_files(today)
        self.prog = parse_b(path_for("B", today))
        self.res = recent_results(today)
        self.out_path = SITE / "data" / f"{today:%Y%m%d}.json"
        self.state = json.loads(self.out_path.read_text()) if self.out_path.exists() else {
            "date": f"{today:%Y-%m-%d}", "races": {}}

    def races(self):
        return self.prog[["jcd", "rno", "deadline", "race_name"]].drop_duplicates(["jcd", "rno"])

    def predict(self, jcd, rno, use_live=True):
        prog = self.prog[(self.prog.jcd == jcd) & (self.prog.rno == rno)].copy()
        if len(prog) != 6:
            return None
        pre = prog[["date", "jcd", "rno", "lane"]].copy()
        odds = {}
        bi = {"boats": {}, "weather": {}}
        if use_live:
            bi = web.beforeinfo(self.today, jcd, rno)
            odds = web.odds3t(self.today, jcd, rno)
        pre["exhibit"] = pre.lane.map(lambda l: bi["boats"].get(l, {}).get("exhibit"))
        pre["course"] = pre.lane.map(lambda l: bi["boats"].get(l, {}).get("ex_course"))
        pre["course"] = pre["course"].fillna(pre["lane"])
        pre["wind"] = bi["weather"].get("wind")
        pre["wave"] = bi["weather"].get("wave")
        asof = prog[["date", "toban", "jcd"]].assign(course=pre["course"].astype(int).values)
        hist = history_features(self.res[self.res.toban.isin(prog.toban)], asof=asof)
        df = build(prog, pre, hist)
        d = M.race_strengths(df, self.booster)
        s = d["s"].values.reshape(1, 6)
        pm = M.trifecta_probs(s, *self.p["lam"])[0]
        om = np.array([odds.get(c, np.nan) for c in M.COMBOS], dtype=float)
        lam1 = self.p["lam"][0]
        pwin = np.exp(lam1 * (s[0] - s[0].max())); pwin /= pwin.sum()
        rec = dict(
            at=now_jst().strftime("%H:%M"), live=use_live and len(odds) > 100,
            odds_time=odds.get("_time", ""),
            boats=[dict(lane=int(r.lane), name=r.name, cls=r.cls, toban=int(r.toban),
                        exhibit=None if pd.isna(r.exhibit) else float(r.exhibit),
                        course=int(r.course), p_win=round(float(pw), 4))
                   for r, pw in zip(df.sort_values("lane").itertuples(), pwin)],
            bets=[],
        )
        if rec["live"]:
            pk = M.market_probs(om[None])[0]
            pb = M.blend(pm[None], pk[None], self.p["a"], self.p["b"])[0]
            ev = pb * om
            mo = self.p.get("max_odds") or np.inf
            sel = np.where((ev > self.p["thr"]) & (om <= mo))[0]
            rec["bets"] = [dict(combo=M.COMBOS[i], p=round(float(pb[i]), 5), odds=float(om[i]),
                                ev=round(float(ev[i]), 3)) for i in sel[np.argsort(-ev[sel])]]
        top = np.argsort(-pm)[:5]
        rec["top_model"] = [dict(combo=M.COMBOS[i], p=round(float(pm[i]), 4)) for i in top]
        return rec

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated"] = now_jst().strftime("%Y-%m-%d %H:%M")
        self.out_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        days = sorted(p.stem for p in (SITE / "data").glob("2*.json"))
        (SITE / "data" / "index.json").write_text(json.dumps({"days": days}))

    def key(self, jcd, rno):
        return f"{jcd:02d}-{rno:02d}"

    def run(self, until: dt.datetime, window_min=(2, 7), on_update=None):
        """締切 window_min[1]分前～window_min[0]分前のレースを確定予想する。"""
        for r in self.races().itertuples():  # 朝の暫定予想(直前情報なし)
            k = self.key(r.jcd, r.rno)
            if k not in self.state["races"]:
                pr = self.predict(r.jcd, r.rno, use_live=False)
                if pr:
                    pr.update(venue=VENUES.get(r.jcd, str(r.jcd)), jcd=int(r.jcd), rno=int(r.rno),
                              deadline=r.deadline, race_name=r.race_name, stage="暫定")
                    self.state["races"][k] = pr
        self.save()
        on_update and on_update()
        while now_jst() < until:
            n = now_jst()
            pending = []
            for r in self.races().itertuples():
                k = self.key(r.jcd, r.rno)
                if not r.deadline or self.state["races"].get(k, {}).get("stage") == "確定":
                    continue
                hh, mm = map(int, r.deadline.split(":"))
                dl = dt.datetime.combine(self.today, dt.time(hh, mm), JST)
                mins = (dl - n).total_seconds() / 60
                if window_min[0] <= mins <= window_min[1]:
                    pending.append((r, k))
                elif mins > window_min[1]:
                    pending.append(None)
            todo = [x for x in pending if x]
            for r, k in todo:
                try:
                    pr = self.predict(r.jcd, r.rno, use_live=True)
                except Exception as e:  # noqa: BLE001
                    print("predict error", k, e, flush=True)
                    continue
                if pr:
                    pr.update(venue=VENUES.get(r.jcd, str(r.jcd)), jcd=int(r.jcd), rno=int(r.rno),
                              deadline=r.deadline, race_name=r.race_name, stage="確定")
                    self.state["races"][k] = pr
                    print(n.strftime("%H:%M"), k, "bets", len(pr["bets"]), flush=True)
            if todo:
                self.save()
                on_update and on_update()
            if not pending:
                print("本日の対象レースは終了", flush=True)
                break
            time.sleep(30)


def settle(day: dt.date):
    """成績(K)と突き合わせて仮想収支を記入する。"""
    p = SITE / "data" / f"{day:%Y%m%d}.json"
    kp = fetch("K", day)
    if not p.exists() or not kp:
        return False
    st = json.loads(p.read_text())
    _, _, pays = parse_k(kp)
    win = {f"{r.jcd:02d}-{r.rno:02d}": (r.combo, r.payout) for r in pays.itertuples()}
    inv = ret = 0
    for k, r in st["races"].items():
        if k not in win:
            continue
        combo, payout = win[k]
        r["result"] = dict(combo=combo, payout=int(payout))
        for b in r.get("bets", []):
            b["hit"] = b["combo"] == combo
            inv += 100
            ret += payout if b["hit"] else 0
    st["settled"] = dict(invest=inv, payout=int(ret), roi=(ret / inv if inv else None))
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return True
