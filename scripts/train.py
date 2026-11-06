"""本番用モデルの学習とパラメータ決定。models/win.txt と models/params.json を出力。

- LightGBM: オッズ期間より前の全データで学習(直近60日を早期終了用に使う)
- λ: 早期終了用期間で推定
- ブレンド係数 a,b と買い条件(閾値・オッズ上限): オッズがある期間で推定
  (期間の前半で決めて後半で確認し、両方の結果を params.json に残す)
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boatpred import model as M  # noqa: E402
from boatpred.features import training_frame  # noqa: E402
from scripts.backtest import evaluate_bets, load_odds  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    df, _ = training_frame()
    df = df[df.finish.notna()]
    odds = load_odds()
    os_ = odds.index.get_level_values("date").min()
    vs = os_ - pd.Timedelta(days=60)
    booster = M.train_win(df[df.date < vs], df[(df.date >= vs) & (df.date < os_)])
    dv = M.race_strengths(df[(df.date >= vs) & (df.date < os_)], booster)
    _, sv, fv = M.to_matrix(dv)
    lam, _ = M.fit_lambdas(sv, M.winner_index(fv))

    do = M.race_strengths(df[df.date >= os_], booster)
    keys, so, fo = M.to_matrix(do)
    keys["date"] = pd.to_datetime(keys["date"])
    om = odds.reindex(pd.MultiIndex.from_frame(keys)).values
    has = np.isfinite(om).sum(1) >= 100
    keys, so, fo, om = keys[has].reset_index(drop=True), so[has], fo[has], om[has]
    wi = M.winner_index(fo)
    pm, pk = M.trifecta_probs(so, *lam), M.market_probs(om)
    half = (keys.date < keys.date.quantile(0.5)).values
    (a, b), _ = M.fit_blend(pm[half], pk[half], wi[half])
    pb = M.blend(pm, pk, a, b)

    def nll(p, m):
        ok = (wi >= 0) & m
        return float(-np.log(p[ok][np.arange(ok.sum()), wi[ok]] + 1e-12).mean())

    print(f"races: 前半={half.sum()} 後半={(~half).sum()}")
    print(f"blend(前半で推定): a(model)={a:.3f} b(market)={b:.3f}")
    for nm, m in [("前半", half), ("後半", ~half)]:
        print(f"  {nm} 3連単NLL: model={nll(pm, m):.4f} market={nll(pk, m):.4f} blend={nll(pb, m):.4f}")
    # 1着だけの精度
    w1 = np.zeros((len(wi), 6))
    for i in range(6):
        w1[:, i] = pm[:, [k for k, c in enumerate(M.COMBOS) if c[0] == str(i + 1)]].sum(1)
    k1 = np.zeros((len(wi), 6))
    for i in range(6):
        k1[:, i] = pk[:, [k for k, c in enumerate(M.COMBOS) if c[0] == str(i + 1)]].sum(1)
    fin1 = np.array([int(M.COMBOS[x][0]) - 1 if x >= 0 else -1 for x in wi])
    ok = fin1 >= 0
    print(f"  1着NLL(全期間): model={-np.log(w1[ok, fin1[ok]] + 1e-12).mean():.4f} market={-np.log(k1[ok, fin1[ok]] + 1e-12).mean():.4f}")
    best, rows = None, []
    for thr in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        for mo in [None, 100, 50, 20]:
            r1 = evaluate_bets(pb[half], om[half], wi[half], thr, mo)
            r2 = evaluate_bets(pb[~half], om[~half], wi[~half], thr, mo)
            rows.append((thr, mo, r1["bets"], r1["roi"], r2["bets"], r2["roi"], r2["ci_lo"], r2["ci_hi"]))
            # 前半で選ぶ。買い数が少なすぎる条件は除外
            if r1["bets"] >= 200 and np.isfinite(r1["roi"]) and (best is None or r1["roi"] > best[0]["roi"]):
                best = (r1, r2, thr, mo)
    print("thr max_odds | 前半 bets roi | 後半 bets roi CI95")
    for r in rows:
        print(f"{r[0]:.1f} {str(r[1]):>4} | {r[2]:6d} {r[3]:.3f} | {r[4]:6d} {r[5]:.3f} [{r[6]:.3f},{r[7]:.3f}]")
    if best is None:
        print("買い条件: 前半で200点以上買う条件がない → 推奨なし(thr=999で実質買わない)")
        best = (dict(bets=0, roi=None), dict(bets=0, roi=None), 999.0, None)
    r1, r2, thr, mo = best
    # 最終的なブレンド係数は全オッズ期間で推定し直す
    (a, b), _ = M.fit_blend(pm, pk, wi)
    # 本番用は直近まで含めた全データで、同じ木の本数で学習し直す
    import lightgbm as lgb
    from boatpred.features import CATEGORICAL, FEATURES
    full = lgb.Dataset(df[FEATURES], (df.finish == 1).astype(int), categorical_feature=CATEGORICAL)
    booster = lgb.train(M.PARAMS, full, booster.best_iteration)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    booster.save_model(str(out / "win.txt"))
    params = dict(lam=list(map(float, lam)), a=float(a), b=float(b), thr=thr, max_odds=mo,
                  selected_on=r1, checked_on=r2,
                  odds_period=[str(keys.date.min().date()), str(keys.date.max().date())],
                  trained_until=str(df.date.max().date()))
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(params, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
