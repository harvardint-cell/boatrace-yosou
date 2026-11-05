"""学習・予測用の特徴量を作る。1行=1艇。

情報の時点に注意:
- 番組表(B)由来: 前日には確定している
- 展示タイム・進入(course)・気象: 直前情報(締切の約10-15分前に公開)。
  学習時は成績(K)の値を代用する。進入は実際の進入を「スタート展示の進入」の代わりに使っており、
  前付け等で両者がずれるレースでは学習と本番に差が出る(要検証)。
- 選手の過去成績: 当日を含まない前日までの集計のみを使う(同日の先行レースは本番で使えないため)。
"""
import pathlib
import re

import numpy as np
import pandas as pd

from .parse import parse_b, parse_k

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "cache"
CLS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
RACE_TYPES = ["一般", "予選", "予選特賞", "準優勝戦", "優勝戦", "選抜", "特選", "準優", "ドリーム", "準優進出戦"]

FEATURES = [
    "lane", "course", "jcd", "rno", "distance", "cls_n", "age", "weight",
    "nat_win", "nat_2", "loc_win", "loc_2", "motor_2", "boat_2",
    "ser_n", "ser_avg", "ser_win",
    "exhibit", "exhibit_rank", "exhibit_diff",
    "nat_win_diff", "nat_win_rank", "motor_2_diff", "cls_diff",
    "wind", "wave", "race_type_n",
    "h_n", "h_win", "h_top2", "h_top3", "h_st", "h_fly",
    "hc_n", "hc_win", "hc_top2", "hc_top3",
    "v_course_win",
]
# 風向は直前情報ページの表記(画像コード)と成績の表記の対応が未検証のため、当面は使わない
CATEGORICAL = ["jcd", "race_type_n"]
WIND_DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
             "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]


def _series(s: str):
    ds = [int(c) for c in re.findall(r"[1-6]", s or "")]
    return len(ds), (np.mean(ds) if ds else np.nan), (ds.count(1) / len(ds) if ds else np.nan)


def load_raw(use_cache=True):
    """B/Kを全部読み、(program, results, races, payouts) を返す。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, kind in [("program", "B"), ("results", "K"), ("races", "K"), ("payouts", "K")]:
        out[name] = None
    files_b = sorted((RAW / "B").glob("B*.txt"))
    files_k = sorted((RAW / "K").glob("K*.txt"))
    key = f"{len(files_b)}_{len(files_k)}_{files_k[-1].stem if files_k else ''}"
    cp = CACHE / f"raw_{key}.pkl"
    # キャッシュは自分で生成したローカルファイルのみ(外部由来のpickleは読まない)
    if use_cache and cp.exists():
        return pd.read_pickle(cp)
    prog = pd.concat([parse_b(p) for p in files_b], ignore_index=True)
    ks = [parse_k(p) for p in files_k]
    res = pd.concat([k[0] for k in ks], ignore_index=True)
    races = pd.concat([k[1] for k in ks], ignore_index=True)
    pays = pd.concat([k[2] for k in ks if len(k[2])], ignore_index=True)
    data = dict(program=prog, results=res, races=races, payouts=pays)
    for old in CACHE.glob("raw_*.pkl"):
        old.unlink()
    pd.to_pickle(data, cp)
    return data


def history_features(results: pd.DataFrame, asof: pd.DataFrame | None = None):
    """選手別(180日)・選手×コース別(365日)・場×コース別(365日)の「その日の前日まで」の集計。
    戻り値は (toban,date) / (toban,course,date) / (jcd,course,date) をキーにした表。

    asof: 予測したい日の (date,toban,jcd,course) 。結果のない日の行(ダミー0件)を足して、
    その日時点の集計値を作る(本番予測用)。学習時は全レース当日に行があるので不要。"""
    r = results[["date", "toban", "jcd", "course", "finish", "st", "status"]].copy()
    r["one"] = 1.0
    if asof is not None:
        d = asof[["date", "toban", "jcd", "course"]].copy()
        d["one"] = 0.0
        r = pd.concat([r, d], ignore_index=True)
    r["date"] = pd.to_datetime(r["date"])
    w = r.one > 0
    r["win"] = ((r.finish == 1) & w).astype(float)
    r["top2"] = ((r.finish <= 2) & w).astype(float)
    r["top3"] = ((r.finish <= 3) & w).astype(float)
    r["st_ok"] = (r.st.notna() & w).astype(float)
    r["st_f"] = r.st.fillna(0)
    r["fly"] = (r.status.fillna("").astype(str).str.startswith("F") & w).astype(float)

    def roll(src, keys, cols, window):
        daily = src.groupby(keys + ["date"])[cols + ["one"]].sum().reset_index()
        daily = daily.sort_values(keys + ["date"])
        parts = []
        for _, g in daily.groupby(keys, sort=False):
            g = g.set_index("date")
            rolled = g[cols + ["one"]].rolling(window, closed="left").sum()
            rolled[keys] = g[keys].values
            parts.append(rolled.reset_index())
        out = pd.concat(parts, ignore_index=True)
        return out.rename(columns={"one": "n"})

    h = roll(r, ["toban"], ["win", "top2", "top3", "st_f", "st_ok", "fly"], "180D")
    h = h.assign(
        h_n=h.n, h_win=h.win / h.n, h_top2=h.top2 / h.n, h_top3=h.top3 / h.n,
        h_st=h.st_f / h.st_ok.replace(0, np.nan), h_fly=h.fly,
    )[["toban", "date", "h_n", "h_win", "h_top2", "h_top3", "h_st", "h_fly"]]

    rc = r[r.course.notna()].copy()
    rc["course"] = rc.course.astype(int)
    hc = roll(rc, ["toban", "course"], ["win", "top2", "top3"], "365D")
    hc = hc.assign(hc_n=hc.n, hc_win=hc.win / hc.n, hc_top2=hc.top2 / hc.n, hc_top3=hc.top3 / hc.n)[
        ["toban", "course", "date", "hc_n", "hc_win", "hc_top2", "hc_top3"]]
    vc = roll(rc, ["jcd", "course"], ["win"], "365D")
    vc = vc.assign(v_course_win=vc.win / vc.n)[["jcd", "course", "date", "v_course_win"]]
    return h, hc, vc


def _asof(left, right, by):
    """leftの各行に、right の (by一致 & right.date <= left.date) の最新行を付ける。
    rightの値はclosed='left'で集計済みなので「その日の前日まで」になる。"""
    left = left.sort_values("date")
    right = right.sort_values("date")
    return pd.merge_asof(left, right, on="date", by=by, direction="backward")


def build(program: pd.DataFrame, pre: pd.DataFrame, hist) -> pd.DataFrame:
    """program: 番組表(1行=1艇)。pre: 直前情報(date,jcd,rno,lane,exhibit,course,wind,wave,wind_dir)。
    hist: history_features の戻り値。"""
    h, hc, vc = hist
    df = program.merge(pre, on=["date", "jcd", "rno", "lane"], how="left", suffixes=("", "_pre"))
    for c in ["exhibit", "course", "wind", "wave"]:  # 直前情報が未公開(None)でも数値型にそろえる
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["course"] = df["course"].fillna(df["lane"]).astype(int)
    df["cls_n"] = df.cls.map(CLS)
    ser = df.series_results.map(_series)
    df["ser_n"] = [s[0] for s in ser]
    df["ser_avg"] = [s[1] for s in ser]
    df["ser_win"] = [s[2] for s in ser]
    g = df.groupby(["date", "jcd", "rno"])
    df["exhibit_rank"] = g.exhibit.rank(method="min")
    df["exhibit_diff"] = df.exhibit - g.exhibit.transform("mean")
    df["nat_win_diff"] = df.nat_win - g.nat_win.transform("mean")
    df["nat_win_rank"] = g.nat_win.rank(ascending=False, method="min")
    df["motor_2_diff"] = df.motor_2 - g.motor_2.transform("mean")
    df["cls_diff"] = df.cls_n - g.cls_n.transform("mean")
    df["wind_dir_n"] = df.wind_dir.map({d: i for i, d in enumerate(WIND_DIRS)}) if "wind_dir" in df else np.nan
    df["race_type_n"] = df.race_name.map(lambda s: next((i for i, t in enumerate(RACE_TYPES) if t == s), len(RACE_TYPES)))
    df = _asof(df, h, ["toban"])
    df["course"] = df["course"].astype(int)
    df = _asof(df, hc, ["toban", "course"])
    df = _asof(df, vc, ["jcd", "course"])
    for c in ["h_n", "hc_n"]:
        df[c] = df[c].fillna(0)
    return df.sort_values(["date", "jcd", "rno", "lane"]).reset_index(drop=True)


def training_frame(data=None):
    data = data or load_raw()
    res, races = data["results"], data["races"]
    pre = res[["date", "jcd", "rno", "lane", "exhibit", "course"]].merge(
        races[["date", "jcd", "rno", "wind", "wave", "wind_dir"]], on=["date", "jcd", "rno"], how="left")
    hist = history_features(res)
    df = build(data["program"], pre, hist)
    y = res[["date", "jcd", "rno", "lane", "finish"]].copy()
    y["date"] = pd.to_datetime(y["date"])
    df = df.merge(y, on=["date", "jcd", "rno", "lane"], how="left")
    return df, hist
