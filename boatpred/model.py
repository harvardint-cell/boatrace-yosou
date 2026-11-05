"""モデル: 1着確率(LightGBM) → 3連単120通りの確率(Harville+補正) → 市場オッズとのブレンド。

ブレンド(Benter型): log p = a*log p_model + b*log p_market を120通りで正規化。
a,b は「オッズがある検証期間」で対数尤度最大化して決める。
市場に勝てない(=a≈0)なら、モデルは市場以上の情報を持っていないことを意味する。
"""
import itertools
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import CATEGORICAL, FEATURES

COMBOS = [f"{i}-{j}-{k}" for i, j, k in itertools.permutations(range(1, 7), 3)]
_IDX = np.array([[i - 1, j - 1, k - 1] for i, j, k in itertools.permutations(range(1, 7), 3)])
MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def train_win(train: pd.DataFrame, valid: pd.DataFrame, rounds=3000):
    tr, va = train[train.finish.notna() | True], valid
    dtr = lgb.Dataset(tr[FEATURES], (tr.finish == 1).astype(int), categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(va[FEATURES], (va.finish == 1).astype(int), categorical_feature=CATEGORICAL)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def race_strengths(df: pd.DataFrame, booster) -> pd.DataFrame:
    """各レース6艇の強さ w (softmaxでp_winになる)を返す。6艇揃わないレースは除外。"""
    d = df.copy()
    p = np.clip(booster.predict(d[FEATURES]), 1e-6, 1 - 1e-6)
    d["s"] = np.log(p / (1 - p))
    d = d.sort_values(["date", "jcd", "rno", "lane"])
    n = d.groupby(["date", "jcd", "rno"]).lane.transform("size")
    return d[n == 6]


def trifecta_probs(s: np.ndarray, lam1=1.0, lam2=1.0, lam3=1.0) -> np.ndarray:
    """s: (R,6) 強さのlog。戻り (R,120)。
    Harville に着順ごとの温度補正(1着lam1, 2着lam2, 3着lam3)を入れたもの(Henery/Stern型の近似)。
    2値分類のlogitをそのままsoftmaxすると1着確率が尖りやすいので、lam1で較正する。"""
    s = s - s.max(axis=1, keepdims=True)
    w1, w2, w3 = np.exp(lam1 * s), np.exp(lam2 * s), np.exp(lam3 * s)
    i, j, k = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
    p1 = w1[:, i] / w1.sum(1, keepdims=True)
    p2 = w2[:, j] / (w2.sum(1, keepdims=True) - w2[:, i])
    p3 = w3[:, k] / (w3.sum(1, keepdims=True) - w3[:, i] - w3[:, j])
    return p1 * p2 * p3


def to_matrix(d: pd.DataFrame):
    keys = d[["date", "jcd", "rno"]].drop_duplicates().reset_index(drop=True)
    s = d["s"].values.reshape(-1, 6)
    fin = d["finish"].values.reshape(-1, 6) if "finish" in d else None
    return keys, s, fin


def winner_index(fin: np.ndarray) -> np.ndarray:
    """1-2-3着の組番index(COMBOS内)。同着・失格等で決まらない場合 -1。"""
    out = np.full(len(fin), -1)
    lut = {tuple(c): n for n, c in enumerate(_IDX)}
    for r, f in enumerate(fin):
        order = [np.where(f == x)[0] for x in (1, 2, 3)]
        if all(len(o) == 1 for o in order):
            out[r] = lut[(order[0][0], order[1][0], order[2][0])]
    return out


def fit_lambdas(s, win_idx):
    ok = win_idx >= 0

    def nll(x):
        p = trifecta_probs(s[ok], x[0], x[1], x[2])
        return -np.log(p[np.arange(ok.sum()), win_idx[ok]] + 1e-12).mean()

    r = minimize(nll, [1.0, 0.6, 0.4], method="Nelder-Mead")
    return r.x, r.fun


def market_probs(odds: np.ndarray) -> np.ndarray:
    """odds: (R,120) 欠損はnan。1/oddsを正規化(払戻率分を除いた市場の確率)。"""
    q = np.where(np.isfinite(odds) & (odds > 0), 1.0 / odds, 0.0)
    return q / q.sum(1, keepdims=True)


def blend(p_model, p_mkt, a, b):
    lp = a * np.log(p_model + 1e-12) + b * np.log(p_mkt + 1e-12)
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp) * (p_mkt > 0)
    return e / e.sum(1, keepdims=True)


def fit_blend(p_model, p_mkt, win_idx):
    ok = win_idx >= 0

    def nll(x):
        p = blend(p_model[ok], p_mkt[ok], x[0], x[1])
        return -np.log(p[np.arange(ok.sum()), win_idx[ok]] + 1e-12).mean()

    r = minimize(nll, [0.3, 0.8], method="Nelder-Mead")
    return r.x, r.fun
