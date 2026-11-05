"""学習→確率較正→市場ブレンド→期待値で買う、を時系列で検証する。

期間の切り方(リーク防止のため全て時系列順):
  train : ～ valid_start-1        (LightGBM学習)
  valid : valid_start ～ odds_start-1  (早期終了・Harville指数λの推定)
  blend : odds_start ～ test_start-1   (市場ブレンド係数a,bと買い条件の決定)
  test  : test_start ～            (評価のみ。ここの結果だけを信じる)

usage: python scripts/backtest.py 2026-08-01 2026-10-01 2026-10-21
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boatpred import model as M  # noqa: E402
from boatpred.features import FEATURES, training_frame  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_odds():
    fs = sorted((ROOT / "data" / "odds").glob("*.csv"))
    o = pd.concat([pd.read_csv(f) for f in fs if f.stat().st_size > 10], ignore_index=True)
    o["date"] = pd.to_datetime(o["date"])
    w = o.pivot_table(index=["date", "jcd", "rno"], columns="combo", values="odds")
    return w.reindex(columns=M.COMBOS)


def evaluate_bets(p, odds, win_idx, thr, max_odds=None, min_p=0.0):
    """期待値 p*odds > thr の組を各100円買う。戻り: 件数, 的中, 投資, 払戻, ROI, レース単位bootstrap95%CI"""
    ev = p * odds
    sel = (ev > thr) & np.isfinite(odds) & (p >= min_p)
    if max_odds:
        sel &= odds <= max_odds
    hit = np.zeros_like(sel)
    ok = win_idx >= 0
    hit[np.where(ok)[0], win_idx[ok]] = True
    stake = sel.sum(1) * 100.0
    ret = (sel & hit).astype(float) * np.nan_to_num(odds) * 100
    ret = ret.sum(1)
    n_bets, inv, pay = int(sel.sum()), stake.sum(), ret.sum()
    roi = pay / inv if inv else np.nan
    rng = np.random.default_rng(0)
    idx = np.where(stake > 0)[0]
    if len(idx) > 10:
        bs = [ret[s].sum() / stake[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(2000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return dict(thr=thr, max_odds=max_odds, races=int((stake > 0).sum()), bets=n_bets,
                hits=int((sel & hit).sum()), invest=int(inv), payout=int(pay), roi=roi, ci_lo=lo, ci_hi=hi)


def main(valid_start, odds_start, test_start):
    df, _ = training_frame()
    df = df[df.finish.notna() | df.lane.notna()]
    vs, os_, ts = map(pd.Timestamp, (valid_start, odds_start, test_start))
    tr = df[(df.date < vs) & df.finish.notna()]
    va = df[(df.date >= vs) & (df.date < os_) & df.finish.notna()]
    print(f"train rows={len(tr)} ({tr.date.min():%Y-%m-%d}..{tr.date.max():%Y-%m-%d}) valid rows={len(va)}")
    booster = M.train_win(tr, va)
    print("best_iter", booster.best_iteration)
    imp = pd.Series(booster.feature_importance("gain"), FEATURES).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k}" for k in imp.index[:12]))

    # λ推定(valid)
    dv = M.race_strengths(va, booster)
    _, sv, fv = M.to_matrix(dv)
    wv = M.winner_index(fv)
    lam, nll_v = M.fit_lambdas(sv, wv)
    print(f"lambda={lam.round(3)} valid trifecta NLL={nll_v:.4f} (一様=log120={np.log(120):.4f})")

    # オッズ期間
    odds = load_odds()
    do = M.race_strengths(df[(df.date >= os_)], booster)
    keys, so, fo = M.to_matrix(do)
    keys["date"] = pd.to_datetime(keys["date"])
    om = odds.reindex(pd.MultiIndex.from_frame(keys)).values
    has = np.isfinite(om).sum(1) >= 100
    keys, so, fo, om = keys[has].reset_index(drop=True), so[has], fo[has], om[has]
    wi = M.winner_index(fo)
    pm = M.trifecta_probs(so, *lam)
    pk = M.market_probs(om)
    isb = (keys.date < ts).values
    print(f"odds races: blend={isb.sum()} test={(~isb).sum()}")

    def nll(p, m):
        ok = (wi >= 0) & m
        return -np.log(p[ok][np.arange(ok.sum()), wi[ok]] + 1e-12).mean()

    (a, b), _ = M.fit_blend(pm[isb], pk[isb], wi[isb])
    pb = M.blend(pm, pk, a, b)
    print(f"blend a(model)={a:.3f} b(market)={b:.3f}")
    for name, m in [("blend期間", isb), ("test期間", ~isb)]:
        print(f"  {name}: NLL model={nll(pm, m):.4f} market={nll(pk, m):.4f} blend={nll(pb, m):.4f}")

    # 買い条件は blend期間で決め、test期間で評価
    rows = []
    for thr in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        for mo in [None, 100, 50, 20]:
            r = evaluate_bets(pb[isb], om[isb], wi[isb], thr, mo)
            r2 = evaluate_bets(pb[~isb], om[~isb], wi[~isb], thr, mo)
            rows.append({**{f"blend_{k}": v for k, v in r.items() if k in ("bets", "roi")},
                         "thr": thr, "max_odds": mo,
                         **{f"test_{k}": v for k, v in r2.items() if k not in ("thr", "max_odds")}})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    base = evaluate_bets(np.ones_like(pk), om[~isb], wi[~isb], 0)  # 全通り買い=控除率確認
    print(f"参考(test全120通り買い) ROI={base['roi']:.3f}")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "backtest.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
