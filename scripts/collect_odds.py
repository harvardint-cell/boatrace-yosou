"""K(成績)ファイルにあるレースの締切時3連単オッズを取得して data/odds/YYYYMMDD.csv に保存。

usage: python scripts/collect_odds.py 2026-10-01 2026-11-05 [workers]
"""
import datetime as dt
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from boatpred import web  # noqa: E402
from boatpred.download import daterange, path_for  # noqa: E402
from boatpred.parse import parse_k  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "odds"
web.MIN_INTERVAL = 0  # 並列数で負荷を制御する


def one(args):
    d, jcd, rno = args
    try:
        o = web.odds3t(d, jcd, rno)
    except Exception as e:  # noqa: BLE001
        print("ERR", d, jcd, rno, e, flush=True)
        return []
    if rno == 12:
        print(" ", d, jcd, "done", flush=True)
    return [dict(date=d, jcd=jcd, rno=rno, combo=k, odds=v) for k, v in o.items() if k != "_time"]


def main(start, end, workers=6, *flags):
    from boatpred.download import fetch
    OUT.mkdir(parents=True, exist_ok=True)
    days = list(daterange(dt.date.fromisoformat(start), dt.date.fromisoformat(end)))
    if "--newest-first" in flags:
        days.reverse()
    for d in days:
        out = OUT / f"{d:%Y%m%d}.csv"
        if out.exists():
            continue
        kp = path_for("K", d)
        if not kp.exists():
            kp = fetch("K", d)  # レース一覧に成績ファイルが必要
            if not kp:
                continue
        _, races, _ = parse_k(kp)
        jobs = [(d, r.jcd, r.rno) for r in races.itertuples()]
        with ThreadPoolExecutor(int(workers)) as ex:
            rows = [x for res in ex.map(one, jobs) for x in res]
        pd.DataFrame(rows).to_csv(out, index=False)
        print(d, len(jobs), "races", len(rows), "rows", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    main(*args[:2], *(args[2:3] or [6]), *flags)
